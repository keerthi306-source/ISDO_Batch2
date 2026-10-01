"""
ISDO Capstone - Lab C6/C7/C8
orchestrator/supervisor.py

LangGraph StateGraph supervisor with an extended HITL (Human-in-the-Loop) gate.

Flow:
    triage -> resolution -> sla -> (hitl?) -> communication -> END

The HITL gate fires for ANY of these conditions (Lab C7):
    1. P1 ticket where SLA status is CRITICAL or BREACHED        (from C5/C6)
    2. Resolution Agent KB confidence is LOW (any priority)       (new in C7)
    3. Access Grant request: category == 'Access' and
       request_type == 'Access Grant' (REQ- tickets)              (new in C7)

Lab C8: when the local KB returns LOW confidence, resolution_node calls the
A2A Knowledge Specialist (POST /tasks, then GET /tasks/{task_id}) and uses its
resolution and confidence. If the A2A server is not running, confidence stays
LOW and the ticket falls back to the HITL gate.

If the HITL decision is REJECTED, the graph still continues to the
Communication node, but sends a 'pending approval' message instead of a
resolution.

NOTE: If your C1-C5 agents live in separate modules, point the four
`run_*` helper functions below at them. The node functions, state, routing
and HITL logic do not need to change.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, TypedDict

import requests
from langgraph.graph import END, StateGraph

# A2A Knowledge Specialist (Lab C8) -- a2a/knowledge_specialist.py
A2A_BASE_URL = "http://localhost:8001"
A2A_TIMEOUT_SECONDS = 60   # POST /tasks runs the LLM synchronously, so allow time


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    # Ticket input
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str          # e.g. 'Access Grant' for REQ- tickets

    # Agent outputs
    triage_summary: str
    assignment_group: str
    resolution_steps: List[str]
    kb_article: Optional[str]
    confidence: str            # HIGH / MEDIUM / LOW
    resolution_text: str       # NEW in C8 - resolution drafted by A2A specialist
    confidence_source: str     # NEW in C8 - LOCAL_KB / A2A / A2A_UNAVAILABLE
    a2a_task_id: str           # NEW in C8 - task id returned by POST /tasks
    sla_status: str            # OK / AT_RISK / CRITICAL / BREACHED
    minutes_remaining: int

    # HITL
    hitl_required: bool
    hitl_reason: str           # NEW in C7 - why the gate fired
    hitl_decision: str         # APPROVED / REJECTED / NOT_REQUIRED

    # Output
    user_message: str
    audit_log: List[dict]


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------
def audit(state: TicketState, agent: str, action: str, detail: str) -> List[dict]:
    """Append an audit entry and print it. Returns the updated log."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ticket": state.get("ticket_number", "?"),
        "agent": agent,
        "action": action,
        "detail": detail,
    }
    print(f"  [AUDIT] {agent}: {action} -- {detail}")
    return list(state.get("audit_log", [])) + [entry]


# ---------------------------------------------------------------------------
# Agent logic (swap these for your C1-C5 agent calls if needed)
# ---------------------------------------------------------------------------
ASSIGNMENT_GROUPS = {
    "Network": "Network Operations",
    "Application": "Application Support",
    "Access": "Identity & Access Management",
    "Hardware": "End User Computing",
    "Software": "End User Computing",
}

# Tiny local KB used by the Resolution Agent to score confidence.
KNOWLEDGE_BASE = [
    {
        "id": "KB0010001",
        "title": "VPN connection drops / cannot connect to GlobalProtect VPN",
        "keywords": {"vpn", "globalprotect", "connect", "connection", "tunnel", "drops", "remote"},
        "steps": [
            "Confirm internet connectivity without VPN.",
            "Restart the GlobalProtect client and re-authenticate with MFA.",
            "Clear cached VPN credentials and reconnect to the nearest gateway.",
            "If still failing, collect client logs and escalate to Network Operations.",
        ],
    },
    {
        "id": "KB0020045",
        "title": "SAP production system down / SAP login failures",
        "keywords": {"sap", "production", "down", "outage", "erp", "login", "users"},
        "steps": [
            "Check SAP application server status in SM51.",
            "Verify database connectivity and HANA service health.",
            "Restart the affected dialog instance if hung.",
            "Engage SAP Basis on-call for P1 bridge.",
        ],
    },
    {
        "id": "KB0030012",
        "title": "Provisioning VPN access for new users and contractors",
        "keywords": {"vpn", "access", "contractor", "new", "user", "provision", "grant"},
        "steps": [
            "Verify the requester's manager approval and contract end date.",
            "Add the user to the VPN-Contractors AD group.",
            "Send VPN onboarding instructions to the user's email.",
        ],
    },
]


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def run_triage(state: TicketState) -> dict:
    category = state.get("category", "General")
    group = ASSIGNMENT_GROUPS.get(category, "Service Desk")
    summary = f"{state['priority']} {category} ticket: {state['short_description']}"
    return {"triage_summary": summary, "assignment_group": group}


def run_resolution(state: TicketState) -> dict:
    """Score the ticket against the KB. Confidence = keyword overlap."""
    words = _tokens(state["short_description"] + " " + state.get("description", ""))
    best, best_score = None, 0
    for article in KNOWLEDGE_BASE:
        score = len(words & article["keywords"])
        if score > best_score:
            best, best_score = article, score

    if best_score >= 3:
        confidence = "HIGH"
    elif best_score == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if best is None or confidence == "LOW":
        return {
            "kb_article": None,
            "resolution_steps": ["No clear KB match -- manual investigation required."],
            "confidence": "LOW",
        }
    return {"kb_article": best["id"], "resolution_steps": best["steps"], "confidence": confidence}


def run_sla(state: TicketState, now: Optional[datetime] = None) -> dict:
    now = now or datetime.now()
    due = datetime.strptime(state["sla_due"], "%Y-%m-%d %H:%M:%S")
    minutes = int((due - now).total_seconds() // 60)
    if minutes < 0:
        status = "BREACHED"
    elif minutes <= 30:
        status = "CRITICAL"
    elif minutes <= 120:
        status = "AT_RISK"
    else:
        status = "OK"
    return {"sla_status": status, "minutes_remaining": minutes}


def run_communication(state: TicketState) -> str:
    tn = state["ticket_number"]
    decision = state.get("hitl_decision", "NOT_REQUIRED")
    is_access = _is_access_grant(state)

    if decision == "REJECTED":
        return (
            f"Dear User, your ticket {tn} has been received and is pending approval. "
            f"A support engineer will review it and update you shortly. "
            f"No changes have been made yet."
        )

    if is_access:
        return (
            f"Dear Requester, your access grant request {tn} has been approved. "
            f"The {state.get('assignment_group')} team will provision access and "
            f"send onboarding instructions to the email on the request."
        )

    steps = "\n    ".join(f"{i}. {s}" for i, s in enumerate(state.get("resolution_steps", []), 1))
    if state.get("confidence") == "LOW":
        return (
            f"Dear User, your ticket {tn} has been reviewed by an engineer. "
            f"We could not find a known fix, so it has been assigned to "
            f"{state.get('assignment_group')} for investigation. We will update you shortly."
        )

    prefix = "Approved by on-call engineer. " if decision == "APPROVED" else ""
    if state.get("confidence_source") == "A2A" and state.get("resolution_text"):
        return (
            f"Dear User. {prefix}Our Knowledge Specialist found a resolution for ticket {tn} "
            f"(source: {state.get('kb_article')}, confidence {state.get('confidence')}):\n"
            f"    {state['resolution_text']}"
        )
    return (
        f"Dear User. {prefix}Here is the resolution for ticket {tn} "
        f"(KB {state.get('kb_article')}):\n    {steps}"
    )


def call_a2a_knowledge_specialist(state: TicketState) -> dict:
    """
    Lab C8: ask the A2A Knowledge Specialist for a deeper KB lookup.
      1) POST /tasks            -> {task_id, status}
      2) GET  /tasks/{task_id}  -> {..., result: {resolution, confidence, ...}}
    Raises requests.exceptions.ConnectionError if the server is not running.
    """
    payload = {
        "query": f"{state['short_description']}. {state.get('description', '')}".strip(),
        "ticket_number": state["ticket_number"],
        "context": f"category={state.get('category')}, priority={state.get('priority')}",
    }
    post = requests.post(f"{A2A_BASE_URL}/tasks", json=payload, timeout=A2A_TIMEOUT_SECONDS)
    post.raise_for_status()
    task_id = post.json()["task_id"]
    print(f"  [A2A] Task submitted: {task_id} (status: {post.json().get('status')})")

    get = requests.get(f"{A2A_BASE_URL}/tasks/{task_id}", timeout=A2A_TIMEOUT_SECONDS)
    get.raise_for_status()
    result = get.json().get("result", {})
    return {
        "task_id": task_id,
        "resolution_text": result.get("resolution", ""),
        "confidence": (result.get("confidence") or "LOW").upper(),
        "best_match": result.get("best_match"),
        "confidence_score": result.get("confidence_score"),
    }


def _is_access_grant(state: TicketState) -> bool:
    return (
        state.get("category") == "Access"
        and state.get("request_type") == "Access Grant"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------
def triage_node(state: TicketState) -> dict:
    print("\n> TRIAGE AGENT")
    out = run_triage(state)
    print(f"  {out['triage_summary']} -> {out['assignment_group']}")
    return {**out, "audit_log": audit(state, "TriageAgent", "classified", out["assignment_group"])}


def resolution_node(state: TicketState) -> dict:
    print("\n> RESOLUTION AGENT")
    out = run_resolution(state)
    out["confidence_source"] = "LOCAL_KB"
    out["resolution_text"] = ""
    print(f"  KB: {out['kb_article'] or 'none'} | Confidence: {out['confidence']}")
    log = audit(
        state, "ResolutionAgent", "kb_match", f"{out['kb_article'] or 'none'} ({out['confidence']})"
    )

    # Lab C8: LOW confidence -> delegate to the A2A Knowledge Specialist
    if out["confidence"] == "LOW":
        print(f"  Confidence LOW -- calling A2A Knowledge Specialist at {A2A_BASE_URL}")
        try:
            a2a = call_a2a_knowledge_specialist(state)
            out["a2a_task_id"] = a2a["task_id"]
            out["resolution_text"] = a2a["resolution_text"]
            out["confidence"] = a2a["confidence"]
            out["confidence_source"] = "A2A"
            if a2a.get("best_match"):
                out["kb_article"] = a2a["best_match"]
            print(
                f"  [A2A] Result: {a2a['best_match']} | Confidence: {a2a['confidence']}"
                f" (score {a2a['confidence_score']})"
            )
            log = audit(
                {**state, "audit_log": log}, "ResolutionAgent", "a2a_result",
                f"task {a2a['task_id']} -> {a2a['confidence']} ({a2a['best_match']})",
            )
        except requests.exceptions.ConnectionError:
            out["confidence_source"] = "A2A_UNAVAILABLE"
            print("  [A2A] Knowledge Specialist not reachable -- falling back to HITL")
            log = audit(
                {**state, "audit_log": log}, "ResolutionAgent", "a2a_unavailable",
                "ConnectionError -- confidence stays LOW, HITL fallback",
            )
        except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
            out["confidence_source"] = "A2A_UNAVAILABLE"
            print(f"  [A2A] Call failed ({type(exc).__name__}) -- falling back to HITL")
            log = audit(
                {**state, "audit_log": log}, "ResolutionAgent", "a2a_error",
                f"{type(exc).__name__}: {exc} -- HITL fallback",
            )

    return {**out, "audit_log": log}


def sla_node(state: TicketState) -> dict:
    """Compute SLA status and decide whether the HITL gate is required."""
    print("\n> SLA AGENT")
    out = run_sla(state)
    print(f"  SLA status: {out['sla_status']} ({out['minutes_remaining']} min remaining)")

    merged = {**state, **out}
    reasons = []

    # Trigger 1: P1 + SLA CRITICAL/BREACHED (existing C6 behaviour)
    if merged["priority"] == "P1" and merged["sla_status"] in ("CRITICAL", "BREACHED"):
        reasons.append(f"P1 SLA {merged['sla_status']} -- escalation requires on-call approval")

    # Trigger 2: LOW KB confidence, regardless of priority
    if merged.get("confidence") == "LOW":
        if merged.get("confidence_source") == "A2A_UNAVAILABLE":
            reasons.append(
                "LOW KB CONFIDENCE -- A2A Knowledge Specialist unavailable, no clear fix found"
            )
        elif merged.get("confidence_source") == "A2A":
            reasons.append(
                "LOW KB CONFIDENCE -- even the A2A Knowledge Specialist could not find a clear fix"
            )
        else:
            reasons.append("LOW KB CONFIDENCE -- Resolution Agent could not find a clear fix")

    # Trigger 3: Access Grant request, regardless of priority
    if _is_access_grant(merged):
        reasons.append(
            f"ACCESS GRANT -- {merged['short_description'].lower()} requires security approval"
        )

    hitl_required = bool(reasons)
    hitl_reason = "; ".join(reasons) if reasons else ""

    log = audit(state, "SLAAgent", "sla_check", out["sla_status"])
    if hitl_required:
        log = audit({**state, "audit_log": log}, "SLAAgent", "hitl_triggered", hitl_reason)

    return {**out, "hitl_required": hitl_required, "hitl_reason": hitl_reason, "audit_log": log}


def hitl_node(state: TicketState) -> dict:
    print("\n> HITL GATE -- human approval required")
    print("  " + "WARNING " * 8)
    print(f"  Ticket: {state['ticket_number']} | Priority: {state['priority']}")
    print(f"  Reason: {state.get('hitl_reason', 'unspecified')}")
    print("  " + "WARNING " * 8)

    answer = ""
    while answer not in ("y", "n"):
        answer = input("  Approve action? [y/n]: ").strip().lower()

    decision = "APPROVED" if answer == "y" else "REJECTED"
    log = audit(
        state,
        "HITLGate",
        "approval_decision",
        f"{decision} (reason: {state.get('hitl_reason', '')})",
    )
    print(f"  Decision: {decision}")
    return {"hitl_decision": decision, "audit_log": log}


def communication_node(state: TicketState) -> dict:
    print("\n> COMMUNICATION AGENT")
    decision = state.get("hitl_decision") or "NOT_REQUIRED"
    state = {**state, "hitl_decision": decision}
    msg = run_communication(state)
    print(f"  USER MESSAGE: {msg}")
    return {
        "hitl_decision": decision,
        "user_message": msg,
        "audit_log": audit(state, "CommunicationAgent", "user_notified", decision),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("triage", triage_node)
    graph.add_node("resolution", resolution_node)
    graph.add_node("sla", sla_node)
    graph.add_node("hitl", hitl_node)
    graph.add_node("communication", communication_node)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "resolution")
    graph.add_edge("resolution", "sla")   # sla_node needs resolution confidence
    graph.add_conditional_edges(
        "sla", route_after_sla, {"hitl": "hitl", "communication": "communication"}
    )
    graph.add_edge("hitl", "communication")
    graph.add_edge("communication", END)
    return graph.compile()


def print_audit_trail(state: TicketState) -> None:
    print(f"\n=== AUDIT TRAIL: {state['ticket_number']} ===")
    for e in state.get("audit_log", []):
        print(f"  {e['timestamp']} | {e['agent']:<18} | {e['action']:<17} | {e['detail']}")


# ---------------------------------------------------------------------------
# Test tickets
# ---------------------------------------------------------------------------
TEST_TICKETS: List[TicketState] = [
    # 1) P1 SAP outage -> HITL via P1 SLA CRITICAL/BREACHED (C6 path)
    TicketState(
        ticket_number="INC-1001",
        short_description="SAP production system down for all users",
        description="Users cannot log in to SAP ERP. Production outage affecting finance.",
        category="Application",
        priority="P1",
        sla_due="2024-01-15 10:30:00",
    ),
    # 2) P3 ticket -> HITL via LOW KB confidence (C7 Step 3)
    #    Swap the short_description back to the VPN text to see the no-HITL path:
    #    short_description="Unable to connect to VPN from home",
    TicketState(
        ticket_number="INC-1003",
        short_description="Cisco Webex not launching on MacBook M2 after Sonoma update",
        description="App bounces in the dock and closes. Reinstall did not help.",
        category="Software",
        priority="P3",
        sla_due="2099-12-31 17:00:00",
    ),
    # 3) Access Grant -> HITL regardless of priority (C7 Step 4)
    TicketState(
        ticket_number="REQ-1002",
        short_description="VPN access for new contractor",
        description="Contractor needs VPN access. Email: contractor@client.com",
        category="Access",
        request_type="Access Grant",
        priority="P2",
        sla_due="2024-01-15 15:00:00",
    ),
    # 4) Webex on MacBook M2 -> HITL via LOW KB confidence even though P3
    #    (SLA is past due, but the P1 SLA trigger does not apply to P3)
    TicketState(
        ticket_number="TEST-004",
        short_description="Cisco Webex not launching on MacBook M2 after Sonoma update",
        description="Cisco Webex not launching on MacBook M2 after Sonoma update.",
        category="Software",
        priority="P3",
        sla_due="2024-01-17 09:00:00",
    ),
]


def main() -> None:
    app = build_graph()
    for ticket in TEST_TICKETS:
        print("\n" + "=" * 72)
        print(f"PROCESSING {ticket['ticket_number']}: {ticket['short_description']}")
        print("=" * 72)
        final = app.invoke({**ticket, "audit_log": [], "hitl_required": False, "hitl_reason": ""})
        print_audit_trail(final)


if __name__ == "__main__":
    main()