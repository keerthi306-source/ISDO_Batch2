"""
ISDO Lab C6 — LangGraph Orchestrator: Wire All Agents
A LangGraph StateGraph that routes a ticket through the ISDO agents:

    P2 / P3 / P4 :  Triage → Resolution → SLA → Communication
    P1 at risk   :  Triage → Resolution → SLA → HITL → Communication

Reuses the building blocks from earlier labs (single source of truth):
    agents/resolution_agent.py  (C4) → ChromaDB search_kb + code guardrail
    agents/sla_agent.py         (C5) → get_sla_status, HITL approval prompt, escalation teams

Run from the project root (or from inside orchestrator/ — both work):
    python orchestrator/supervisor.py
    python orchestrator/supervisor.py --show-graph     (also prints a Mermaid diagram)

Optional (scripted / non-interactive runs), pre-answer the HITL prompt:
    $env:HITL_ANSWERS="y"       (PowerShell)
"""

import json
import operator
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

# Windows consoles can choke on arrows/emoji — force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Project root resolved from this file (falls back to cwd in Interactive Window / Jupyter).
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "orchestrator" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

import anthropic
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Reuse the C4 and C5 agents' tools (importing them does not run their demos).
from resolution_agent import get_kb, search_kb, apply_guardrail, score_to_confidence
from sla_agent import get_sla_status, hitl_approve, team_for

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file (same one used in C2-C5).")

client = anthropic.Anthropic(api_key=API_KEY)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
TEMPERATURE = 0.0

CATEGORIES = ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"]
PRIORITIES = ["P1", "P2", "P3", "P4"]
ASSIGNMENT_GROUPS = ["Network-Ops", "App-Support", "Desktop-Support", "Service-Desk",
                     "Security-Ops", "Server-Ops", "Email-Support", "DBA-Team"]

# ── SHARED STATE (Step 2) ─────────────────────────────────────────────────────

class TicketState(TypedDict, total=False):
    """Single shared memory for the graph. Every field is optional at start;
    each node writes only the fields it owns."""
    # Input ticket
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    # Triage agent
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # Resolution agent
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    confidence_score: float
    # SLA agent
    sla_breach_risk: str
    sla_minutes_remaining: int
    escalation_required: bool
    escalation_team: str
    hitl_required: bool
    # HITL node
    hitl_approved: bool
    # Communication agent
    user_message: str
    final_status: str
    # Every node APPENDS here — operator.add merges lists instead of overwriting.
    audit_log: Annotated[list, operator.add]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def audit(agent: str, action: str, detail: str) -> list:
    """Build one audit entry (returned as a 1-item list so LangGraph appends it)."""
    print(f"  [AUDIT] {agent}: {action}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title: str):
    print(f"\n▶ {title}")


def most_severe(*priorities) -> str:
    """P1 beats P2 beats P3 ... — used so a triage downgrade can never skip a P1 safeguard."""
    valid = [p for p in priorities if p in PRIORITIES]
    return min(valid) if valid else "P3"


_temperature_mode = "extra_body"


def ask_claude(system: str, user: str, max_tokens: int = 800) -> str:
    """One plain-text model call (same temperature fallback as C3–C5)."""
    global _temperature_mode
    params = dict(model=MODEL, max_tokens=max_tokens, system=system,
                  messages=[{"role": "user", "content": user}])
    response = None
    if _temperature_mode == "extra_body":
        try:
            response = client.messages.create(**params, extra_body={"temperature": TEMPERATURE})
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            print("  (note: this model doesn't accept a custom temperature — using its default)")
            _temperature_mode = "off"
    if response is None:
        response = client.messages.create(**params)
    return "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()


def parse_json(text: str) -> dict:
    """Pull the first {...} object out of a model reply (tolerates ```json fences)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    return json.loads(match.group(0))


# Simple backstop for PII the model might miss (emails, employee IDs, IPv4 addresses).
PII_PATTERNS = [r"[\w.+-]+@[\w-]+\.[\w.]+", r"\b(?:emp[-_ ]?id|ZEN-\d{3,})", r"\b\d{1,3}(?:\.\d{1,3}){3}\b"]


def regex_pii(text: str) -> bool:
    return any(re.search(p, text, re.I) for p in PII_PATTERNS)

# ── NODE 1: TRIAGE ────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = f"""You are the ISDO Triage Agent for Zensar's IT Service Desk.
Classify the ticket and reply with ONLY a JSON object, no other text:
{{"category": one of {CATEGORIES},
  "priority": one of {PRIORITIES},
  "assignment_group": one of {ASSIGNMENT_GROUPS},
  "pii_detected": true/false (names, emails, employee IDs, IP addresses; "[REDACTED]" is not PII),
  "reasoning": "one sentence"}}

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)"""


def triage_node(state: TicketState) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")
    text = f"{state.get('short_description', '')}\n{state.get('description', '')}"
    try:
        result = parse_json(ask_claude(
            TRIAGE_SYSTEM,
            f"Ticket: {state['ticket_number']}\nSummary: {state.get('short_description')}\n"
            f"Details: {state.get('description')}"))
        note = result.get("reasoning", "")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  (triage JSON could not be parsed — keeping the ticket's own values: {e})")
        result, note = {}, "fallback to ticket values (unparseable model output)"

    # Validate every field against the allowed values; fall back to the ticket record.
    category = result.get("category") if result.get("category") in CATEGORIES else state.get("category", "Software")
    priority = result.get("priority") if result.get("priority") in PRIORITIES else state.get("priority", "P3")
    group = result.get("assignment_group") if result.get("assignment_group") in ASSIGNMENT_GROUPS else "Service-Desk"
    pii = bool(result.get("pii_detected")) or regex_pii(text)

    print(f"  Category: {category}")
    print(f"  Priority: {priority}")
    print(f"  Assign To: {group}")
    print(f"  PII: {pii}")
    return {
        "triage_category": category,
        "triage_priority": priority,
        "triage_assignment_group": group,
        "pii_detected": pii,
        "audit_log": audit("TriageAgent", "classify_ticket",
                           f"{category}/{priority} → {group}; PII={pii}. {note}"),
    }

# ── NODE 2: RESOLUTION ────────────────────────────────────────────────────────

RESOLUTION_SYSTEM = """You are the ISDO Resolution Agent for Zensar's IT Service Desk.
You are given a ticket and the best-matching KB article. Reply with ONLY a JSON object:
{"resolution_text": "3-4 numbered, plain-English steps copied from the article's Resolution Steps",
 "auto_resolve": true/false}

auto_resolve = true ONLY when ALL are true:
- confidence is HIGH
- priority is P2, P3 or P4 (never P1)
- the article's "Auto-Resolve Eligibility" section says this case is L1 auto-resolvable
If confidence is LOW, write a short L2 escalation note instead and set auto_resolve false."""


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    priority = most_severe(state.get("priority"), state.get("triage_priority"))
    query = f"{state.get('short_description', '')}. {state.get('description', '')}"
    hits = search_kb(query)["articles"]           # ChromaDB query (from Lab C4)

    top = hits[0] if hits else None
    score = top["confidence_score"] if top else None
    confidence = score_to_confidence(score) if top else "LOW"
    article = top["article"] if top and confidence != "LOW" else "none"

    try:
        draft = parse_json(ask_claude(
            RESOLUTION_SYSTEM,
            f"Ticket: {state['ticket_number']}\nCategory: {state.get('triage_category')}\n"
            f"Priority: {priority}\nSummary: {state.get('short_description')}\n"
            f"Details: {state.get('description')}\n\n"
            f"KB confidence: {confidence} (score {score})\n"
            f"KB article: {article}\n---\n{top['content'] if top else '(no article)'}"))
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  (resolution JSON could not be parsed: {e})")
        draft = {"resolution_text": "Resolution could not be drafted — route to L2 for review.",
                 "auto_resolve": False}

    draft.update(confidence=confidence, kb_article_used=article,
                 ticket_number=state["ticket_number"])
    final = apply_guardrail(draft, score, priority)   # code decides, not the model (C4)

    pct = f" ({score:.0%})" if score is not None else ""
    print(f"  KB Article: {article}")
    print(f"  Confidence: {final['confidence']}{pct}  |  Auto-resolve: {final['auto_resolve']}")
    for note in final.get("guardrail_notes", []):
        print(f"  Guardrail: {note}")
    return {
        "kb_article": article,
        "resolution_text": str(final.get("resolution_text", "")),
        "auto_resolve": bool(final["auto_resolve"]),
        "confidence": final["confidence"],
        "confidence_score": score if score is not None else 0.0,
        "audit_log": audit("ResolutionAgent", "search_kb",
                           f"{article} | {final['confidence']}{pct} | auto_resolve={final['auto_resolve']}"),
    }

# ── NODE 3: SLA ───────────────────────────────────────────────────────────────

def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline")
    # The SLA clock belongs to the priority recorded on the ticket (sla_due was set from it).
    sla = get_sla_status(state["ticket_number"], state["sla_due"], state.get("priority", "P3"))
    if "error" in sla:
        print(f"  SLA error: {sla['error']}")
        return {"sla_breach_risk": "UNKNOWN", "escalation_required": False, "hitl_required": False,
                "audit_log": audit("SLAAgent", "get_sla_status", sla["error"])}

    risk, minutes = sla["breach_risk"], sla["minutes_remaining"]
    is_p1 = most_severe(state.get("priority"), state.get("triage_priority")) == "P1"
    hitl_required = is_p1 and risk in ("CRITICAL", "BREACHED")
    escalation_required = bool(sla["requires_escalation"]) or hitl_required
    team = team_for(state.get("triage_category") or state.get("category", ""))

    print(f"  SLA Risk: {risk}  |  Minutes remaining: {minutes}")
    print(f"  Escalation required: {escalation_required}  |  HITL required: {hitl_required}")
    return {
        "sla_breach_risk": risk,
        "sla_minutes_remaining": minutes,
        "escalation_required": escalation_required,
        "escalation_team": team,
        "hitl_required": hitl_required,
        "audit_log": audit("SLAAgent", "get_sla_status",
                           f"{risk}, {minutes} min remaining; escalation={escalation_required}, "
                           f"hitl={hitl_required}"),
    }

# ── NODE 4: HITL ──────────────────────────────────────────────────────────────

def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    approved = hitl_approve(          # prompts "Approve escalation? [y/n]" (from Lab C5)
        state["ticket_number"],
        "Escalate P1 ticket",
        f"Escalate to {state.get('escalation_team')} — SLA {state.get('sla_breach_risk')}, "
        f"{state.get('sla_minutes_remaining')} min remaining")
    return {
        "hitl_approved": approved,
        "audit_log": audit("HITLGate", "human_approval",
                           f"{'APPROVED' if approved else 'REJECTED'} escalation to {state.get('escalation_team')}"),
    }

# ── NODE 5: COMMUNICATION ─────────────────────────────────────────────────────

COMMS_SYSTEM = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly, professional message to the end user (max 120 words).
Start with "Dear User, regarding <ticket number>". Plain text only, no markdown.
Never include personal data (names, emails, employee IDs, IP addresses).
Do not promise anything beyond the facts you are given."""


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    num = state["ticket_number"]
    group = state.get("triage_assignment_group", "the service desk")

    if state.get("auto_resolve"):
        case, status = "self_service_resolution", "RESOLVED"
        brief = (f"The issue can be fixed by the user. Include these steps:\n{state.get('resolution_text')}\n"
                 "Ask them to reply if the steps do not work, and the ticket will be reopened.")
    elif state.get("hitl_approved"):
        case, status = "escalation_confirmation", "ESCALATED"
        brief = (f"This is a critical incident. It has been escalated to {state.get('escalation_team')} "
                 "with priority handling. Reassure the user that the team is actively working on it.")
    else:
        case, status = "assignment_notification", "ASSIGNED"
        brief = f"The ticket has been assigned to the {group} team, who will contact the user."

    facts = f"Ticket: {num}\nIssue: {state.get('short_description')}\nSituation: {brief}"
    try:
        message = ask_claude(COMMS_SYSTEM, facts, max_tokens=400)
    except anthropic.APIError as e:
        print(f"  (message drafting failed: {e})")
        message = ""
    if not message:  # template fallback so the user always gets a message
        message = f"Dear User, regarding {num}: {brief}"

    print("  USER MESSAGE:")
    for line in message.splitlines():
        print(f"    {line}")
    print(f"\n✅ FINAL STATUS: {status}")
    return {
        "user_message": message,
        "final_status": status,
        "audit_log": audit("CommunicationAgent", "draft_user_message", f"{case} → {status}"),
    }

# ── ROUTING (Step 3) ──────────────────────────────────────────────────────────

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"

# ── BUILD THE GRAPH ───────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("sla", sla_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("communication", communication_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "sla")
    graph.add_conditional_edges("sla", route_after_sla,
                                {"hitl": "hitl", "communication": "communication"})
    graph.add_edge("hitl", "communication")      # fixed edge: HITL always flows to comms
    graph.add_edge("communication", END)
    return graph.compile()


def process_ticket(app, ticket: dict) -> dict:
    print(f"\n{'═' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'═' * 55}")
    return app.invoke({**ticket, "audit_log": []})

# ── TEST ON 2 TICKETS (Step 4) ────────────────────────────────────────────────

test_tickets = [
    # P2 VPN — expected: Triage → Resolution → SLA → Communication (auto-resolve)
    {"ticket_number": "INC0001001",
     "short_description": "VPN not connecting after password change",
     "description": "User reports VPN client fails to connect after AD password was reset. "
                    "Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 14:00:00"},
    # P1 SAP outage — expected: Triage → Resolution → SLA → HITL → Communication
    # sla_due is 10:40 (as in Lab C5): the CSV's 11:00 leaves exactly 50% → AT_RISK, no HITL.
    {"ticket_number": "INC0001002",
     "short_description": "SAP outage - Finance users cannot access ERP",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                    "Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
]

if __name__ == "__main__":
    app = build_graph()
    if "--show-graph" in sys.argv:
        print(app.get_graph().draw_mermaid())

    get_kb()  # load ChromaDB once, before the first ticket
    results = [process_ticket(app, t) for t in test_tickets]

    # Step 5 — full audit log for each ticket
    for r in results:
        path = " → ".join(e["agent"].replace("Agent", "").replace("Gate", "") for e in r["audit_log"])
        print(f"\n{'═' * 55}\nAUDIT LOG: {r['ticket_number']}  |  FINAL STATUS: {r.get('final_status')}")
        print(f"Route: {path}\n{'═' * 55}")
        for e in r["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19} {e['action']:<19} {e['detail']}")
