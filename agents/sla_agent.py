"""
ISDO Lab C5 — SLA & Escalation Agent
Monitors SLA deadlines, predicts breach risk, and escalates CRITICAL/BREACHED
P1/P2 tickets. Every P1 escalation pauses at a Human-in-the-Loop (HITL) gate.

Run from the project root (or from inside agents/ — both work):
    python agents/sla_agent.py

Optional (for scripted / non-interactive runs), pre-answer the HITL prompts:
    set HITL_ANSWERS=y,n        (Windows cmd)
    $env:HITL_ANSWERS="y,n"     (PowerShell)
"""

import anthropic
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Windows consoles can choke on arrows/emoji — force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paths resolved relative to this file, so it works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
HITL_LOG = LOG_DIR / "hitl_audit.jsonl"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file (same one used in C2-C4).")

client = anthropic.Anthropic(api_key=API_KEY)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
TEMPERATURE = 0.0
MAX_ROUNDS = 5  # hard safety cap on the agentic loop

# Simulated "now" for consistent, reproducible demo results (Step 1 of the lab)
SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)

# ── SLA RULES (Step 1) ────────────────────────────────────────────────────────

SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
CRITICAL_PCT = 0.20   # < 20% of SLA time left  -> CRITICAL
AT_RISK_PCT = 0.50    # < 50% of SLA time left  -> AT_RISK
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}
HITL_PRIORITIES = {"P1"}

ESCALATION_TEAMS = {
    "Network": "L2-Network-Ops",
    "Application": "L2-App-Support",
    "Server": "L2-Server-Ops",
    "Access": "L2-Security-Ops",
    "Security": "L2-Security-Ops",
}
DEFAULT_TEAM = "L2-Service-Desk"


def team_for(category: str) -> str:
    return ESCALATION_TEAMS.get(category, DEFAULT_TEAM)

# ── TOOL DEFINITIONS (Step 2) ─────────────────────────────────────────────────

tools = [
    {
        "name": "get_sla_status",
        "description": ("Check the SLA status of a ticket. Returns minutes remaining, breach risk "
                        "level (BREACHED / CRITICAL / AT_RISK / ON_TRACK) and a requires_escalation flag."),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {
                    "type": "string",
                    "description": "SLA due datetime in format YYYY-MM-DD HH:MM:SS"
                },
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]}
            },
            "required": ["ticket_number", "sla_due", "priority"]
        }
    },
    {
        "name": "update_ticket",
        "description": "Update a ticket in ServiceNow: escalate it, add a work note, or change its state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["escalate", "add_note", "update_state"],
                    "description": "Action to perform on the ticket"
                },
                "escalation_team": {
                    "type": "string",
                    "description": "Team to escalate to (required for escalate), e.g. L2-Network-Ops"
                },
                "note": {"type": "string", "description": "Work note text (required for add_note)"},
                "new_state": {
                    "type": "string",
                    "description": "New state (required for update_state), e.g. In Progress, Escalated, Resolved"
                }
            },
            "required": ["ticket_number", "action"]
        }
    }
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

def get_sla_status(ticket_number, sla_due, priority):
    """Calculate minutes remaining vs the priority's SLA target and classify breach risk."""
    try:
        due_dt = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format: {sla_due} (expected YYYY-MM-DD HH:MM:SS)"}
    if priority not in SLA_MINUTES:
        return {"error": f"Unknown priority: {priority}"}

    minutes_remaining = int((due_dt - SIMULATED_NOW).total_seconds() // 60)
    target = SLA_MINUTES[priority]
    pct_remaining = minutes_remaining / target

    if minutes_remaining < 0:
        risk = "BREACHED"
        msg = f"SLA BREACHED by {abs(minutes_remaining)} minutes"
    elif pct_remaining < CRITICAL_PCT:
        risk = "CRITICAL"
        msg = f"Only {minutes_remaining} minutes remaining — breach imminent"
    elif pct_remaining < AT_RISK_PCT:
        risk = "AT_RISK"
        msg = f"{minutes_remaining} minutes remaining — at risk"
    else:
        risk = "ON_TRACK"
        msg = f"{minutes_remaining} minutes remaining — on track"

    return {
        "ticket_number": ticket_number,
        "sla_due": sla_due,
        "priority": priority,
        "sla_target_minutes": target,
        "minutes_remaining": minutes_remaining,
        "pct_time_remaining": round(max(pct_remaining, 0) * 100, 1),
        "breach_risk": risk,
        "status_message": msg,
        "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES,
    }


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """Simulate a ServiceNow PATCH /api/now/table/incident call."""
    result = {
        "ticket_number": ticket_number,
        "action": action,
        "success": True,
        "timestamp": SIMULATED_NOW.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if action == "escalate":
        result.update(assignment_group=escalation_team, state="Escalated",
                      message=f"Ticket {ticket_number} escalated to {escalation_team}")
        print(f"  [ServiceNow Mock] ESCALATED {ticket_number} → {escalation_team}")
    elif action == "add_note":
        note = note or ""
        result.update(work_note=note, message=f"Work note added to {ticket_number}")
        print(f"  [ServiceNow Mock] NOTE ADDED to {ticket_number}: {note[:60]}{'...' if len(note) > 60 else ''}")
    elif action == "update_state":
        result.update(state=new_state, message=f"Ticket {ticket_number} state changed to {new_state}")
        print(f"  [ServiceNow Mock] STATE CHANGED {ticket_number} → {new_state}")
    else:
        return {"success": False, "error": f"Unsupported action: {action}"}
    return result


def handle_tool(name, inp):
    if name == "get_sla_status":
        return get_sla_status(inp["ticket_number"], inp["sla_due"], inp["priority"])
    if name == "update_ticket":
        return update_ticket(inp["ticket_number"], inp["action"], inp.get("escalation_team"),
                             inp.get("note"), inp.get("new_state"))
    return {"error": f"Unknown tool: {name}"}

# ── HITL GATE (Step 3) ────────────────────────────────────────────────────────

_scripted_answers = [a.strip().lower() for a in os.environ.get("HITL_ANSWERS", "").split(",") if a.strip()]


def log_hitl(ticket_number, action, detail, approved):
    """Append every HITL decision (approve AND reject) to an audit log."""
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "simulated_now": SIMULATED_NOW.isoformat(),
        "ticket_number": ticket_number,
        "action": action,
        "detail": detail,
        "decision": "APPROVED" if approved else "REJECTED",
    }
    with open(HITL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def hitl_approve(ticket_number, action, detail):
    """Pause and ask a human to approve before a P1 escalation. Anything but 'y' = reject."""
    print(f"\n  {'!!!  ' * 3}HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}")
    print(f"  Action:  {action}")
    print(f"  Detail:  {detail}")
    print(f"  {'!!!  ' * 3}")
    if _scripted_answers:
        decision = _scripted_answers.pop(0)
        print(f"  Approve escalation? [y/n]: {decision}   (from HITL_ANSWERS)")
    else:
        try:
            decision = input("  Approve escalation? [y/n]: ").strip().lower()
        except EOFError:          # no terminal attached -> fail safe
            decision = "n"
    approved = decision == "y"
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    log_hitl(ticket_number, action, detail, approved)
    return approved

# ── GUARDRAIL: the model proposes, code decides ──────────────────────────────

def check_escalation(ticket, inp, sla):
    """Validate an escalate call against the real ticket record and SLA result.
    Returns (allowed: bool, reason: str, corrected_input: dict)."""
    inp = dict(inp)
    if inp.get("ticket_number") != ticket["number"]:
        return False, f"ticket_number mismatch (expected {ticket['number']})", inp
    if sla is None:
        return False, "call get_sla_status before escalating", inp
    if not sla.get("requires_escalation"):
        return False, (f"policy: escalation only for {sorted(ESCALATE_PRIORITIES)} tickets that are "
                       f"CRITICAL/BREACHED (this one is {ticket['priority']} / {sla.get('breach_risk')})"), inp
    expected_team = team_for(ticket["category"])
    if inp.get("escalation_team") != expected_team:
        print(f"  [Guardrail] escalation_team corrected: {inp.get('escalation_team')} → {expected_team}")
        inp["escalation_team"] = expected_team
    return True, "", inp

# ── MODEL CALL (same temperature fallback as C3/C4) ──────────────────────────

_temperature_mode = "extra_body"


def call_claude(messages):
    global _temperature_mode
    params = dict(model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT, tools=tools, messages=messages)
    if _temperature_mode == "extra_body":
        try:
            return client.messages.create(**params, extra_body={"temperature": TEMPERATURE})
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            print("  (note: this model doesn't accept a custom temperature — using its default)")
            _temperature_mode = "off"
    return client.messages.create(**params)

# ── SLA AGENT ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status ONCE with the ticket's number, SLA due time and priority.
2. If requires_escalation is true (priority P1/P2 AND breach_risk CRITICAL or BREACHED),
   call update_ticket with action="escalate" and the correct escalation_team.
   P1 tickets that are CRITICAL/BREACHED must always be escalated — they cannot wait.
3. If breach_risk is AT_RISK, add a short work note (action="add_note") warning the
   assignment group of the remaining time. Do not escalate.
4. If breach_risk is ON_TRACK, take no action.
5. Finish with a 1-2 sentence summary: risk level, action taken, and escalation outcome.

Escalation teams by category:
- Network → L2-Network-Ops
- Application → L2-App-Support
- Server → L2-Server-Ops
- Access / Security → L2-Security-Ops
- Anything else → {DEFAULT_TEAM}

A human must approve every P1 escalation. If a tool result says the escalation was
rejected by the human approver, do NOT retry it — add a work note recording that the
escalation was declined and that the ticket stays with the current assignment group."""


def monitor_ticket(ticket):
    """Run SLA monitoring for one ticket. Returns a summary dict."""
    num, pri, cat = ticket["number"], ticket["priority"], ticket["category"]
    print(f"\n{'=' * 55}")
    print(f"SLA Check: {num} | {pri} | Category: {cat}")
    print(f"{'=' * 55}")

    messages = [{
        "role": "user",
        "content": (f"Monitor SLA for this ticket and escalate if needed:\n\n"
                    f"Ticket: {num}\nDescription: {ticket['short_description']}\n"
                    f"Category: {cat}\nPriority: {pri}\nSLA Due: {ticket['sla_due']}")
    }]
    summary = {"ticket_number": num, "priority": pri, "breach_risk": None,
               "escalation": "not required", "final_text": ""}
    sla = None

    for _ in range(MAX_ROUNDS):
        response = call_claude(messages)

        if response.stop_reason != "tool_use":
            text = "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
            if text:
                print(f"  Agent: {text}")
            summary["final_text"] = text
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input

            if block.name == "get_sla_status":
                # Always use the real ticket record, not whatever the model typed.
                result = get_sla_status(num, ticket["sla_due"], pri)
                sla = result
                summary["breach_risk"] = result.get("breach_risk")
                print(f"  → Risk Level: {result.get('breach_risk')}")
                print(f"  → Status:     {result.get('status_message')}")

            elif block.name == "update_ticket" and inp.get("action") == "escalate":
                allowed, reason, inp = check_escalation(ticket, inp, sla)
                if not allowed:
                    print(f"  [Guardrail] escalation blocked — {reason}")
                    result = {"success": False, "message": f"Escalation blocked: {reason}"}
                    summary["escalation"] = "blocked by guardrail"
                # HITL gate keys off the ticket's REAL priority, so the model can't bypass it.
                elif pri in HITL_PRIORITIES and not hitl_approve(
                        num, "Escalate ticket", f"Escalate to {inp['escalation_team']}"):
                    result = {"success": False,
                              "message": "Escalation rejected by human approver. Do not retry."}
                    print("  Escalation cancelled and logged.")
                    summary["escalation"] = "REJECTED by human"
                else:
                    result = handle_tool(block.name, inp)
                    summary["escalation"] = (f"APPROVED → {inp['escalation_team']}" if pri in HITL_PRIORITIES
                                             else f"auto → {inp['escalation_team']}")
            else:
                result = handle_tool(block.name, inp)

            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})
    else:
        print("  [Safety] MAX_ROUNDS reached — stopping agent loop.")

    return summary

# ── RUN SLA MONITORING (Step 4) ───────────────────────────────────────────────

# Simulated "now" = 2024-01-15 10:30. One ticket per risk level.
# NOTE: sla_due for INC0001002 and INC0001001 is adjusted from data/incidents.csv
# (11:00 and 14:00) — with the Step 1 thresholds those CSV values give
# 30/60 = 50% (AT_RISK) and 210/240 = 87.5% (ON_TRACK), not CRITICAL / AT_RISK.
test_tickets = [
    # P1, 10 of 60 min left (16.7%) → CRITICAL → HITL gate (type 'y' in the lab)
    {"number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # P1, due 09:30 → BREACHED by 60 min → HITL gate (type 'n' in the lab)
    {"number": "INC0001010", "short_description": "Exchange server high CPU alert",
     "category": "Server", "priority": "P1", "sla_due": "2024-01-15 09:30:00"},
    # P2, 90 of 240 min left (37.5%) → AT_RISK → work note only, no escalation
    # Step 5: change to '2024-01-15 10:00:00' → BREACHED → auto-escalate to L2-Network-Ops
    {"number": "INC0001001", "short_description": "VPN not connecting after password change",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
    # P3, due 17 Jan 09:00 → 2790 of 480 min left → ON_TRACK, monitor only
    {"number": "INC0001003", "short_description": "Laptop running very slowly",
     "category": "Hardware", "priority": "P3", "sla_due": "2024-01-17 09:00:00"},
]

if __name__ == "__main__":
    print(f"ISDO SLA Agent | model={MODEL} | simulated now={SIMULATED_NOW:%Y-%m-%d %H:%M}")
    results = [monitor_ticket(t) for t in test_tickets]

    print(f"\n{'=' * 55}\nSLA MONITORING SUMMARY\n{'=' * 55}")
    for r in results:
        print(f"  {r['ticket_number']} | {r['priority']} | {str(r['breach_risk']):<9} | {r['escalation']}")
    print(f"\n  HITL decisions logged to: {HITL_LOG.relative_to(PROJECT_ROOT)}")
