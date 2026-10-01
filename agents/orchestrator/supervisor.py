"""
ISDO Lab C6 — LangGraph Orchestrator: Wire All Agents
A LangGraph StateGraph that routes a ticket through the ISDO agents:

    P2 / P3 / P4 :  Triage → Resolution → SLA → Communication
    P1 at risk   :  Triage → Resolution → SLA → HITL → Communication

Reuses the building blocks from earlier labs (single source of truth):
    agents/resolution_agent.py  (C4) → ChromaDB search_kb + code guardrail
    agents/sla_agent.py         (C5) → get_sla_status, HITL approval prompt, escalation teams

Run from the project root (works from orchestrator/ or agents/orchestrator/):
    python orchestrator/supervisor.py
    python orchestrator/supervisor.py --show-graph     (also prints a Mermaid diagram)

Optional (scripted / non-interactive runs), pre-answer the HITL prompt:
    $env:HITL_ANSWERS="y"       (PowerShell)

Lab C9 — PII redaction + audit trail:
    • triage_node redacts short_description / description with guardrails.pii_redactor
      BEFORE anything reaches Claude, and stores the restore-mapping in TicketState.
      All later nodes only ever see the masked text.
    • One AuditLogger instance is passed into every node (build_graph(audit_logger)),
      and every action is appended to logs/audit_trail.jsonl as it happens.
    • communication_node calls restore() on the final user message just before it is
      written to the ServiceNow mock (mcp_server/snow_shim.py, port 5001).
    • After the run, the audit trail is printed per ticket with final_status, plus a
      check that no original PII value appears in any prompt sent to Claude.
"""

import json
import operator
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, TypedDict

# Windows consoles can choke on arrows/emoji — force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Project root = the first folder (walking up from this file, or from the cwd in
# Interactive Window / Jupyter) that contains both guardrails/ and agents/.
# Works whether this file lives in orchestrator/ or agents/orchestrator/.
def _find_project_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for folder in (start, *start.parents):
        if (folder / "guardrails").is_dir() and (folder / "agents").is_dir():
            return folder
    sys.exit(f"ERROR: could not find the project root (a folder containing guardrails/ "
             f"and agents/) above {start}")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "agents"))
sys.path.insert(0, str(PROJECT_ROOT))            # so `guardrails.` imports work

import anthropic
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Reuse the C4 and C5 agents' tools (importing them does not run their demos).
from resolution_agent import get_kb, search_kb, apply_guardrail, score_to_confidence
from sla_agent import get_sla_status, hitl_approve, team_for

# Lab C9 — PII guardrail + audit trail
from guardrails.pii_redactor import redact, restore, AuditLogger

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

SNOW_BASE_URL = os.environ.get("SNOW_BASE_URL", "http://localhost:5001")   # mcp_server/snow_shim.py
AUDIT_FILE = PROJECT_ROOT / "logs" / "audit_trail.jsonl"

# ── SHARED STATE (Step 2) ─────────────────────────────────────────────────────

class TicketState(TypedDict, total=False):
    """Single shared memory for the graph. Every field is optional at start;
    each node writes only the fields it owns."""
    # Input ticket. After triage_node runs, short_description / description hold the
    # REDACTED text, so no later node can accidentally send raw PII to Claude.
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
    pii_mapping: dict          # C9: token -> original value, e.g. {"[NAME_1]": "John Smith"}
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
    user_message: str          # restored (real) message, as written to ServiceNow
    final_status: str
    snow_update: str           # C9: result of the ServiceNow mock write
    # C9: every prompt sent to Claude, recorded so we can prove it was masked.
    claude_inputs: Annotated[list, operator.add]
    # Every node APPENDS here — operator.add merges lists instead of overwriting.
    audit_log: Annotated[list, operator.add]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def audit(logger: AuditLogger, state: TicketState, agent: str, action: str, detail: str,
          tool: str = "", approval_status: str = "Auto") -> list:
    """Write one entry through the shared AuditLogger (→ logs/audit_trail.jsonl) and
    return it as a 1-item list so LangGraph also appends it to state['audit_log'].
    AuditLogger.log() redacts the rationale itself, so PII never lands in the log file."""
    entry = logger.log(agent, action, state.get("ticket_number", ""), tool, detail, approval_status)
    return [entry]


def claude_input(agent: str, prompt: str) -> list:
    """Record exactly what a node sent to Claude (for the C9 masking check)."""
    return [{"agent": agent, "prompt": prompt}]


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



# ── NODE 1: TRIAGE ────────────────────────────────────────────────────────────

TRIAGE_SYSTEM = f"""You are the ISDO Triage Agent for Zensar's IT Service Desk.
Personal data in the ticket has already been masked with placeholders such as
[NAME_1], [EMAIL_1], [USERNAME_1], [EMPLOYEE_ID_1], [PHONE_1], [IP_ADDRESS_1].
Classify the ticket and reply with ONLY a JSON object, no other text:
{{"category": one of {CATEGORIES},
  "priority": one of {PRIORITIES},
  "assignment_group": one of {ASSIGNMENT_GROUPS},
  "pii_detected": true/false (true if any placeholder is present, or if you still see
                  unmasked names, emails, employee IDs or IP addresses),
  "reasoning": "one sentence, using placeholders rather than guessing real values"}}

Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)"""


def triage_node(state: TicketState, logger: AuditLogger) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")

    # C9 Step 1: mask PII BEFORE anything is sent to Claude.
    # Both fields share one mapping, so "John Smith" is [NAME_1] in both.
    # The "||" separator can't be part of any PII match, so no match spans both fields.
    sep = "\n||\n"
    combined = f"{state.get('short_description', '')}{sep}{state.get('description', '')}"
    clean, mapping = redact(combined)
    clean_short, _, clean_desc = clean.partition(sep)
    pii_types = sorted({tok.strip("[]").rsplit("_", 1)[0] for tok in mapping})
    print(f"  PII masked: {len(mapping)} item(s) {pii_types if mapping else ''}")
    redact_log = audit(logger, state, "PIIRedactor", "redact_ticket",
                       f"{len(mapping)} PII item(s) masked before LLM call: {pii_types}",
                       tool="pii_redactor.redact")

    prompt = (f"Ticket: {state['ticket_number']}\nSummary: {clean_short}\n"
              f"Details: {clean_desc}")
    try:
        result = parse_json(ask_claude(TRIAGE_SYSTEM, prompt))
        note = result.get("reasoning", "")
    except (ValueError, json.JSONDecodeError) as e:
        print(f"  (triage JSON could not be parsed — keeping the ticket's own values: {e})")
        result, note = {}, "fallback to ticket values (unparseable model output)"

    # Validate every field against the allowed values; fall back to the ticket record.
    category = result.get("category") if result.get("category") in CATEGORIES else state.get("category", "Software")
    priority = result.get("priority") if result.get("priority") in PRIORITIES else state.get("priority", "P3")
    group = result.get("assignment_group") if result.get("assignment_group") in ASSIGNMENT_GROUPS else "Service-Desk"
    pii = bool(mapping) or bool(result.get("pii_detected"))

    print(f"  Category: {category}")
    print(f"  Priority: {priority}")
    print(f"  Assign To: {group}")
    print(f"  PII: {pii}")
    return {
        # Overwrite with the masked text: every later node (and Claude call) sees only this.
        "short_description": clean_short,
        "description": clean_desc,
        "pii_mapping": mapping,
        "triage_category": category,
        "triage_priority": priority,
        "triage_assignment_group": group,
        "pii_detected": pii,
        "claude_inputs": claude_input("TriageAgent", prompt),
        "audit_log": redact_log + audit(logger, state, "TriageAgent", "classify_ticket",
                                        f"{category}/{priority} → {group}; PII={pii}. {note}",
                                        tool="claude:classify"),
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
If confidence is LOW, write a short L2 escalation note instead and set auto_resolve false.
Placeholders like [NAME_1] are masked personal data — keep them as-is, never guess values."""


def resolution_node(state: TicketState, logger: AuditLogger) -> dict:
    header("RESOLUTION AGENT — searching KB")
    priority = most_severe(state.get("priority"), state.get("triage_priority"))
    # state['short_description'] / ['description'] are already redacted by triage_node.
    query = f"{state.get('short_description', '')}. {state.get('description', '')}"
    hits = search_kb(query)["articles"]           # ChromaDB query (from Lab C4)

    top = hits[0] if hits else None
    score = top["confidence_score"] if top else None
    confidence = score_to_confidence(score) if top else "LOW"
    article = top["article"] if top and confidence != "LOW" else "none"

    prompt = (f"Ticket: {state['ticket_number']}\nCategory: {state.get('triage_category')}\n"
              f"Priority: {priority}\nSummary: {state.get('short_description')}\n"
              f"Details: {state.get('description')}\n\n"
              f"KB confidence: {confidence} (score {score})\n"
              f"KB article: {article}\n---\n{top['content'] if top else '(no article)'}")
    try:
        draft = parse_json(ask_claude(RESOLUTION_SYSTEM, prompt))
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
        "claude_inputs": claude_input("ResolutionAgent", prompt),
        "audit_log": audit(logger, state, "ResolutionAgent", "search_kb",
                           f"{article} | {final['confidence']}{pct} | auto_resolve={final['auto_resolve']}",
                           tool="search_kb"),
    }

# ── NODE 3: SLA ───────────────────────────────────────────────────────────────

def sla_node(state: TicketState, logger: AuditLogger) -> dict:
    header("SLA AGENT — checking deadline")
    # The SLA clock belongs to the priority recorded on the ticket (sla_due was set from it).
    sla = get_sla_status(state["ticket_number"], state["sla_due"], state.get("priority", "P3"))
    if "error" in sla:
        print(f"  SLA error: {sla['error']}")
        return {"sla_breach_risk": "UNKNOWN", "escalation_required": False, "hitl_required": False,
                "audit_log": audit(logger, state, "SLAAgent", "get_sla_status", sla["error"],
                                   tool="get_sla_status")}

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
        "audit_log": audit(logger, state, "SLAAgent", "get_sla_status",
                           f"{risk}, {minutes} min remaining; escalation={escalation_required}, "
                           f"hitl={hitl_required}",
                           tool="get_sla_status",
                           approval_status="PENDING" if hitl_required else "Auto"),
    }

# ── NODE 4: HITL ──────────────────────────────────────────────────────────────

def hitl_node(state: TicketState, logger: AuditLogger) -> dict:
    header("HITL GATE — human approval required")
    approved = hitl_approve(          # prompts "Approve escalation? [y/n]" (from Lab C5)
        state["ticket_number"],
        "Escalate P1 ticket",
        f"Escalate to {state.get('escalation_team')} — SLA {state.get('sla_breach_risk')}, "
        f"{state.get('sla_minutes_remaining')} min remaining")
    decision = "APPROVED" if approved else "REJECTED"
    return {
        "hitl_approved": approved,
        "audit_log": audit(logger, state, "HITLGate", "human_approval",
                           f"{decision} escalation to {state.get('escalation_team')}",
                           tool="hitl_approve", approval_status=decision),
    }

# ── NODE 5: COMMUNICATION ─────────────────────────────────────────────────────

COMMS_SYSTEM = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly, professional message to the end user (max 120 words).
Start with "Dear <user>, regarding <ticket number>" — if the facts include a person
placeholder like [NAME_1], use it as <user>, otherwise write "User". Plain text only, no markdown.
Personal data has been masked with placeholders ([NAME_1], [EMAIL_1], ...). You may repeat a
placeholder exactly as written; never invent, alter or guess the real value behind it, and never
include any other personal data. Do not promise anything beyond the facts you are given."""


def post_to_servicenow(ticket_number: str, message: str, status: str) -> str:
    """Write the final (restored) user message to the ServiceNow mock (snow_shim.py).
    Falls back to a simulated write if the mock isn't running, so the lab still completes."""
    payload = {"comments": message, "state": status.title()}
    try:
        r = requests.patch(f"{SNOW_BASE_URL}/api/now/table/incident/{ticket_number}",
                           json=payload, timeout=5)
        if r.status_code == 200:
            print(f"  [ServiceNow Mock] PATCH {ticket_number} → 200 OK (comments + state={payload['state']})")
            return "UPDATED"
        print(f"  [ServiceNow Mock] PATCH {ticket_number} → HTTP {r.status_code}: {r.text[:80]}")
        return f"HTTP_{r.status_code}"
    except requests.exceptions.ConnectionError:
        print(f"  [ServiceNow Mock] not reachable at {SNOW_BASE_URL} — simulated write "
              f"(start it with: python mcp_server/snow_shim.py)")
        return "SIMULATED"


def communication_node(state: TicketState, logger: AuditLogger) -> dict:
    header("COMMUNICATION AGENT")
    num = state["ticket_number"]
    group = state.get("triage_assignment_group", "the service desk")
    mapping = state.get("pii_mapping", {})

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

    # Still masked: short_description/description were redacted by triage_node.
    facts = (f"Ticket: {num}\nIssue: {state.get('short_description')}\n"
             f"Details: {state.get('description')}\nSituation: {brief}")
    try:
        masked_message = ask_claude(COMMS_SYSTEM, facts, max_tokens=400)
    except anthropic.APIError as e:
        print(f"  (message drafting failed: {e})")
        masked_message = ""
    if not masked_message:  # template fallback so the user always gets a message
        masked_message = f"Dear User, regarding {num}: {brief}"

    # C9 Step 3: put the real values back ONLY at the last step, right before the
    # message leaves the pipeline for the system of record.
    message = restore(masked_message, mapping)
    restored = [tok for tok in mapping if tok in masked_message]
    snow = post_to_servicenow(num, message, status)

    print("  MESSAGE FROM CLAUDE (masked):")
    for line in masked_message.splitlines():
        print(f"    {line}")
    if restored:
        print(f"  Restored before ServiceNow write: {restored}")
    print(f"\n✅ FINAL STATUS: {status}")
    return {
        "user_message": message,
        "final_status": status,
        "snow_update": snow,
        "claude_inputs": claude_input("CommunicationAgent", facts),
        "audit_log": (
            audit(logger, state, "PIIRedactor", "restore_message",
                  f"{len(restored)} placeholder(s) restored for ServiceNow: {restored}",
                  tool="pii_redactor.restore")
            + audit(logger, state, "CommunicationAgent", "post_comment",
                    f"{case} → {status}; ServiceNow write: {snow}",
                    tool="servicenow:patch_incident")
        ),
    }

# ── ROUTING (Step 3) ──────────────────────────────────────────────────────────

def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"

# ── BUILD THE GRAPH ───────────────────────────────────────────────────────────

def _with_logger(node: Callable, logger: AuditLogger) -> Callable:
    """Bind the shared AuditLogger to a node, keeping the (state) -> dict shape LangGraph expects."""
    def bound(state: TicketState) -> dict:
        return node(state, logger)
    bound.__name__ = node.__name__
    return bound


def build_graph(audit_logger: AuditLogger):
    """C9 Step 2: one AuditLogger instance is created by the caller and shared by every node."""
    graph = StateGraph(TicketState)
    graph.add_node("triage", _with_logger(triage_node, audit_logger))
    graph.add_node("resolution", _with_logger(resolution_node, audit_logger))
    graph.add_node("sla", _with_logger(sla_node, audit_logger))
    graph.add_node("hitl", _with_logger(hitl_node, audit_logger))
    graph.add_node("communication", _with_logger(communication_node, audit_logger))

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
    return app.invoke({**ticket, "audit_log": [], "claude_inputs": []})

# ── C9 STEP 4: AUDIT TRAIL + MASKING PROOF ────────────────────────────────────

def print_audit_report(result: dict, original: dict):
    """Print the audit trail with final_status, then prove Claude never saw raw PII."""
    num = result["ticket_number"]
    path = " → ".join(e["agent"].replace("Agent", "").replace("Gate", "")
                      for e in result["audit_log"] if e["agent"] != "PIIRedactor")
    print(f"\n{'═' * 72}\nAUDIT TRAIL: {num}  |  FINAL STATUS: {result.get('final_status')}"
          f"  |  ServiceNow: {result.get('snow_update')}")
    print(f"Route: {path}\n{'═' * 72}")
    for e in result["audit_log"]:
        print(f"  {e['timestamp'][:19]}  {e['agent']:<19} {e['action']:<17} "
              f"{e['approval_status']:<9} {e['rationale']}")

    mapping = result.get("pii_mapping", {})
    print(f"\n  PII MASKING CHECK — {num}")
    print(f"    Original description : {original.get('description')}")
    print(f"    Sent to Claude as    : {result.get('description')}")
    if not mapping:
        print("    No PII found in this ticket.")
        return
    for token, value in mapping.items():
        print(f"      {token:<16} ← {value}")
    leaks = [(c["agent"], v) for c in result.get("claude_inputs", [])
             for v in mapping.values() if v in c["prompt"]]
    calls = len(result.get("claude_inputs", []))
    if leaks:
        print(f"    ❌ LEAK: original PII found in Claude input(s): {leaks}")
    else:
        print(f"    ✅ {len(mapping)} PII value(s) masked in all {calls} Claude call(s); "
              f"none of the originals were sent.")
    print(f"    Final message to ServiceNow (restored): {result.get('user_message', '')[:160]}")

# ── TEST TICKETS (C9: PII added to the descriptions) ──────────────────────────

test_tickets = [
    # P2 VPN — expected: Triage → Resolution → SLA → Communication (auto-resolve)
    {"ticket_number": "INC0001001",
     "short_description": "VPN not connecting after password change",
     "description": "User John Smith (emp ID ZEN-9823) reports VPN client fails to connect after "
                    "AD password was reset. Contact: john.smith@zensar.com or +91-9876543210. "
                    "Laptop IP 10.20.30.41. Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 14:00:00"},
    # P1 SAP outage — expected: Triage → Resolution → SLA → HITL → Communication
    # sla_due is 10:40 (as in Lab C5): the CSV's 11:00 leaves exactly 50% → AT_RISK, no HITL.
    {"ticket_number": "INC0001002",
     "short_description": "SAP outage - Finance users cannot access ERP",
     "description": "Raised by Priya Nair (priya.nair@zensar.com, username: pnair01). Multiple users "
                    "in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
]

if __name__ == "__main__":
    # C9 Step 2: ONE AuditLogger for the whole run, shared by every node.
    audit_logger = AuditLogger(str(AUDIT_FILE))
    app = build_graph(audit_logger)
    if "--show-graph" in sys.argv:
        print(app.get_graph().draw_mermaid())

    get_kb()  # load ChromaDB once, before the first ticket
    results = [(t, process_ticket(app, t)) for t in test_tickets]

    for original, result in results:
        print_audit_report(result, original)

    print(f"\n{len(audit_logger.entries)} audit entries written to {AUDIT_FILE}")
