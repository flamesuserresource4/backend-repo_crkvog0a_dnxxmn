import os
from typing import Any, Dict, List, Set, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------
# Logic representation
# -----------------------------
# We model simple predicates as tuples: (name, args)
# Example: ("MotionDetected", ()) or ("Temperature", ("LivingRoom", 17.0))
Predicate = Tuple[str, Tuple[Any, ...]]

class Rule(BaseModel):
    antecedents: List[str]  # textual predicates e.g., "MotionDetected", "NightTime", "Temperature(Room, T)"
    consequent: str         # textual predicate e.g., "TurnOn(Lights)"

# -----------------------------
# Parse textual predicates into internal tuples
# -----------------------------

def parse_predicate(text: str) -> Predicate:
    text = text.strip()
    if "(" not in text:
        # 0-arity predicate
        return (text, tuple())
    name, rest = text.split("(", 1)
    args_str = rest.rsplit(")", 1)[0].strip()
    if not args_str:
        return (name.strip(), tuple())
    # split by comma not handling nested for simplicity
    raw_args = [a.strip() for a in args_str.split(",")]
    parsed_args: List[Any] = []
    for a in raw_args:
        # numbers
        try:
            if a.endswith("°C"):
                a = a.replace("°C", "").strip()
            num = float(a)
            parsed_args.append(num)
            continue
        except ValueError:
            pass
        # quoted strings
        if (a.startswith("\"") and a.endswith("\"")) or (a.startswith("'") and a.endswith("'")):
            parsed_args.append(a[1:-1])
            continue
        # variables start uppercase or with ? (treated as symbols, not grounded here)
        parsed_args.append(a)
    return (name.strip(), tuple(parsed_args))


def pred_to_text(p: Predicate) -> str:
    name, args = p
    if not args:
        return name
    def fmt(x: Any) -> str:
        if isinstance(x, (int, float)):
            # Show as integer if whole number
            if float(x).is_integer():
                return str(int(x))
            return str(x)
        return str(x)
    return f"{name}({', '.join(fmt(a) for a in args)})"

# -----------------------------
# Knowledge Base (Rules)
# -----------------------------
# 1. If MotionDetected ∧ NightTime → TurnOn(Lights).
# 2. If Temperature(Room, T) ∧ T < 18°C → TurnOn(Heater).
# 3. If TurnOn(Heater) → IncreaseEnergyUsage.
# 4. If EnergyUsageHigh → Alert(User).

KB_RULES_TEXT = [
    Rule(antecedents=["MotionDetected", "NightTime"], consequent="TurnOn(Lights)"),
    Rule(antecedents=["Temperature(Room, T)", "T < 18"], consequent="TurnOn(Heater)"),
    Rule(antecedents=["TurnOn(Heater)"], consequent="IncreaseEnergyUsage"),
    Rule(antecedents=["EnergyUsageHigh"], consequent="Alert(User)"),
]

# -----------------------------
# Forward chaining engine
# -----------------------------

class ForwardResult(BaseModel):
    initial_facts: List[str]
    inferred_facts: List[str]
    actions: List[str]
    trace: List[str]


def forward_chain(initial: Set[Predicate], rules: List[Rule]) -> ForwardResult:
    facts: Set[Predicate] = set(initial)
    trace: List[str] = []
    applied = True
    while applied:
        applied = False
        for r in rules:
            # Special handling for rule 2 with inequality T < 18
            if r.consequent.startswith("TurnOn(Heater)") and "T < 18" in r.antecedents:
                # Find any Temperature(Room, T) with T < 18
                fired_rooms = []
                for f in list(facts):
                    if f[0] == "Temperature" and len(f[1]) == 2:
                        room, t = f[1]
                        try:
                            t_val = float(t)
                        except (TypeError, ValueError):
                            continue
                        if t_val < 18:
                            fired_rooms.append((room, t_val))
                if fired_rooms:
                    cons = ("TurnOn", ("Heater",))
                    if cons not in facts:
                        facts.add(cons)
                        applied = True
                        rooms_str = ", ".join(f"{r_}({t_}°C)" for r_, t_ in fired_rooms)
                        trace.append(f"Applied: Temperature(Room,T) & T<18 => TurnOn(Heater) because [{rooms_str}]")
                continue

            # Generic conjunction rule with 0-arity predicates or simple predicate-matching
            antecedent_preds = []
            numeric_guard: Tuple[str, float, str] | None = None
            ok = True
            for a in r.antecedents:
                a = a.strip()
                if "<" in a or ">" in a or "==" in a:
                    # numeric guard like "T < 18" already handled above
                    continue
                # Only parse concrete ground predicates
                ant = parse_predicate(a)
                antecedent_preds.append(ant)
            for ant in antecedent_preds:
                if ant not in facts:
                    ok = False
                    break
            if ok:
                cons = parse_predicate(r.consequent)
                if cons not in facts:
                    facts.add(cons)
                    applied = True
                    left = " ∧ ".join(r.antecedents)
                    trace.append(f"Applied: {left} => {r.consequent}")
    # Determine actions as any TurnOn(*) or Alert(User)
    actions = [pred_to_text(f) for f in facts if f[0] in ("TurnOn", "Alert")]
    return ForwardResult(
        initial_facts=[pred_to_text(f) for f in initial],
        inferred_facts=[pred_to_text(f) for f in facts if f not in initial],
        actions=sorted(actions),
        trace=trace,
    )

# -----------------------------
# Backward chaining (proof/explanation)
# -----------------------------

class ProofStep(BaseModel):
    goal: str
    satisfied_by: List[str] = Field(default_factory=list)
    rule_used: str | None = None

class BackwardResult(BaseModel):
    goal: str
    provable: bool
    proof: List[ProofStep]


def backward_explain(goal_text: str, facts: Set[Predicate], rules: List[Rule]) -> BackwardResult:
    goal = parse_predicate(goal_text)
    proof: List[ProofStep] = []

    def have_fact(p: Predicate) -> bool:
        return p in facts

    # For this KB, we offer crafted explanations for known consequents
    if goal[0] == "TurnOn" and goal[1] == ("Heater",):
        # Check if forward derivation would succeed (exists Temperature with T<18)
        cold_rooms = []
        for f in facts:
            if f[0] == "Temperature" and len(f[1]) == 2:
                room, t = f[1]
                try:
                    t_val = float(t)
                except (TypeError, ValueError):
                    continue
                if t_val < 18:
                    cold_rooms.append((room, t_val))
        if cold_rooms:
            step1 = ProofStep(
                goal="TurnOn(Heater)",
                satisfied_by=[f"Temperature({r}, {t})" for r, t in cold_rooms] + ["T < 18"],
                rule_used="If Temperature(Room, T) ∧ T < 18°C → TurnOn(Heater)"
            )
            proof.append(step1)
            return BackwardResult(goal="TurnOn(Heater)", provable=True, proof=proof)
        else:
            step1 = ProofStep(
                goal="TurnOn(Heater)",
                satisfied_by=[],
                rule_used="If Temperature(Room, T) ∧ T < 18°C → TurnOn(Heater)"
            )
            proof.append(step1)
            return BackwardResult(goal="TurnOn(Heater)", provable=False, proof=proof)

    if goal[0] == "TurnOn" and goal[1] == ("Lights",):
        need = [parse_predicate("MotionDetected"), parse_predicate("NightTime")]
        satisfied = [pred_to_text(p) for p in need if have_fact(p)]
        step = ProofStep(
            goal="TurnOn(Lights)",
            satisfied_by=satisfied,
            rule_used="If MotionDetected ∧ NightTime → TurnOn(Lights)"
        )
        proof.append(step)
        return BackwardResult(goal="TurnOn(Lights)", provable=len(satisfied) == 2, proof=proof)

    if goal[0] == "Alert" and goal[1] == ("User",):
        need = [parse_predicate("EnergyUsageHigh")]
        satisfied = [pred_to_text(p) for p in need if have_fact(p)]
        step = ProofStep(
            goal="Alert(User)",
            satisfied_by=satisfied,
            rule_used="If EnergyUsageHigh → Alert(User)"
        )
        proof.append(step)
        return BackwardResult(goal="Alert(User)", provable=len(satisfied) == 1, proof=proof)

    # Fallback: check if goal is already a fact
    if have_fact(goal):
        proof.append(ProofStep(goal=pred_to_text(goal), satisfied_by=[pred_to_text(goal)], rule_used=None))
        return BackwardResult(goal=pred_to_text(goal), provable=True, proof=proof)

    proof.append(ProofStep(goal=pred_to_text(goal), satisfied_by=[], rule_used=None))
    return BackwardResult(goal=pred_to_text(goal), provable=False, proof=proof)

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="Smart Home Logic Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FactsInput(BaseModel):
    motion_detected: bool = False
    night_time: bool = False
    temperatures: List[Tuple[str, float]] = Field(default_factory=list, description="List of [room, temperatureC]")
    energy_usage_high: bool = False

class ForwardRequest(BaseModel):
    facts: FactsInput

class BackwardRequest(BaseModel):
    facts: FactsInput
    goal: str = Field(example="TurnOn(Heater)")


def build_facts(fi: FactsInput) -> Set[Predicate]:
    facts: Set[Predicate] = set()
    if fi.motion_detected:
        facts.add(("MotionDetected", tuple()))
    if fi.night_time:
        facts.add(("NightTime", tuple()))
    for room, t in fi.temperatures:
        facts.add(("Temperature", (room, float(t))))
    if fi.energy_usage_high:
        facts.add(("EnergyUsageHigh", tuple()))
    return facts


@app.get("/")
def read_root():
    return {"message": "Smart Home Logic Assistant API"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/kb")
def get_kb():
    return {
        "rules": [r.dict() for r in KB_RULES_TEXT],
        "description": "Propositional + first-order rules with inequality guard",
    }

@app.post("/reason/forward", response_model=ForwardResult)
def reason_forward(req: ForwardRequest):
    facts = build_facts(req.facts)
    result = forward_chain(facts, KB_RULES_TEXT)
    return result

@app.post("/reason/backward", response_model=BackwardResult)
def reason_backward(req: BackwardRequest):
    facts = build_facts(req.facts)
    result = backward_explain(req.goal, facts, KB_RULES_TEXT)
    return result

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
