"""
ISDO Lab C3 — Triage Agent
Reads a ticket and assigns: category, priority, assignment group, and PII flag.
Uses the Anthropic SDK with tool calling (ReAct loop: Reason -> Act -> Observe).
 
Run from the project root:
    python agents/triage_agent.py
"""
 
import anthropic
import csv
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
 
# Windows consoles/redirects can choke on arrows and box characters — force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
 
# Paths resolved relative to this file, so the script works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "incidents.csv"
 
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # also pick up a .env in the current folder, if any
 
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file (same one used in Lab C2).")
 
client = anthropic.Anthropic(api_key=API_KEY)
 
# Model from the lab guide; override in .env with CLAUDE_MODEL=... if your key uses a different one.
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
MAX_TURNS = 5  # safety cap so the agentic loop can never spin forever
 
# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────
 
tools = [
    {
        "name": "classify_ticket",
        "description": "Classify an IT support ticket. Returns category, priority, assignment_group, and whether PII was detected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"],
                    "description": "The ticket category"
                },
                "priority": {
                    "type": "string",
                    "enum": ["P1", "P2", "P3", "P4"],
                    "description": "P1=Critical/many users affected, P2=High/some users, P3=Medium/single user, P4=Low/request"
                },
                "assignment_group": {
                    "type": "string",
                    "description": "Team to assign the ticket to e.g. Network-Ops, App-Support, Desktop-Support, Service-Desk, Security-Ops, Server-Ops, Email-Support, DBA-Team"
                },
                "pii_detected": {
                    "type": "boolean",
                    "description": "True if the ticket contains names, email addresses, employee IDs, or IP addresses"
                },
                "reasoning": {
                    "type": "string",
                    "description": "One sentence explaining the classification decision"
                }
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"]
        }
    },
    {
        "name": "get_open_tickets",
        "description": "Get a summary count of currently open tickets by category from the incidents CSV.",
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to incidents.csv file"
                }
            },
            "required": ["csv_path"]
        }
    }
]
 
# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────
 
def get_open_tickets(csv_path=CSV_PATH):
    """Read incidents.csv and return count of Open tickets by category."""
    path = Path(csv_path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path  # model may pass "data/incidents.csv"
    counts = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("state", "").strip() == "Open":
                    cat = row.get("category", "Unknown").strip()
                    counts[cat] = counts.get(cat, 0) + 1
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}
    return counts
 
 
def handle_tool_call(tool_name, tool_input):
    """Route tool calls to their implementations."""
    if tool_name == "get_open_tickets":
        return get_open_tickets(tool_input.get("csv_path", CSV_PATH))
    elif tool_name == "classify_ticket":
        return tool_input  # the structured classification IS the tool's output
    return {"error": f"Unknown tool: {tool_name}"}
 
# ── TRIAGE AGENT ─────────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.
 
Your job is to classify incoming IT support tickets. For each ticket:
1. Use the classify_ticket tool to assign category, priority, and assignment group
2. Flag if any PII (names, emails, employee IDs, IP addresses) is present
 
Priority rules:
- P1: Service down, many users affected, or security breach
- P2: Significant impact, single department or function affected
- P3: Single user impacted, workaround exists
- P4: Request (new software, access, equipment)
 
Assignment groups: Network-Ops, App-Support, Desktop-Support, Service-Desk,
Security-Ops, Server-Ops, Email-Support, DBA-Team.
 
Always use temperature=0 logic: consistent, rule-based classification.
After the tool result comes back, reply with one short confirmation line."""
 
 
TEMPERATURE = 0.0
_temperature_mode = "extra_body"  # "extra_body" -> "off" if the API rejects it
 
 
def call_claude(messages):
    """One model call. Asks for temperature=0 (per the lab) in a way that works
    across SDK versions: some SDK builds don't accept `temperature=` as a
    keyword, so it is sent in extra_body instead. If the model itself rejects
    a custom temperature, the call is retried without it."""
    global _temperature_mode
    params = dict(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
    )
    if _temperature_mode == "extra_body":
        try:
            return client.messages.create(**params, extra_body={"temperature": TEMPERATURE})
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            print("  (note: this model doesn't accept a custom temperature — using its default)")
            _temperature_mode = "off"
    return client.messages.create(**params)
 
 
def triage_ticket(ticket_number, short_description, description):
    """Run the triage agent on a single ticket. Returns the classification dict."""
    print(f"\n{'='*55}")
    print(f"Triaging: {ticket_number}")
    print(f"{'='*55}")
    print(f"Description: {short_description}")
 
    messages = [
        {
            "role": "user",
            "content": f"Please triage this ticket:\n\nTicket: {ticket_number}\nSummary: {short_description}\nDetails: {description}"
        }
    ]
    classification = None
    turns = 0
 
    # Agentic while loop (ReAct): Reason -> Act (tool) -> Observe (tool_result) -> Reason ...
    # Runs until Claude returns stop_reason == "end_turn".
    while True:
        turns += 1
        if turns > MAX_TURNS:  # safety net — never loop forever
            print(f"  (loop stopped after {MAX_TURNS} turns)")
            break
 
        response = call_claude(messages)
 
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
 
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  -> Tool called: {block.name}")
                    result = handle_tool_call(block.name, block.input)
 
                    if block.name == "classify_ticket":
                        classification = result
                        print(f"  -> Category:    {result.get('category')}")
                        print(f"  -> Priority:    {result.get('priority')}")
                        print(f"  -> Assign To:   {result.get('assignment_group')}")
                        print(f"  -> PII Found:   {result.get('pii_detected')}")
                        print(f"  -> Reason:      {result.get('reasoning')}")
 
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
 
            messages.append({"role": "user", "content": tool_results})
            continue
 
        # end_turn (or max_tokens / anything else) -> print any text and stop
        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"  Agent: {block.text.strip()}")
        if response.stop_reason != "end_turn":
            print(f"  (stopped: {response.stop_reason})")
        break
 
    if classification is None:
        print("  WARNING: agent did not call classify_ticket for this ticket")
    return classification
 
# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────
 
if __name__ == "__main__":
    # Step 4: 5 tickets from incidents.csv
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed."),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today."),
        ("INC0001008", "Network switch down - Building C",
         "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
        ("REQ-1002", "VPN access for new contractor joining project Phoenix",
         "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. Email: contractor@client.com"),
 
        # Step 5: uncomment to add your own 6th ticket
        # ("TEST-0006", "Salesforce CRM access issue",
        #  "User cannot access Salesforce CRM from company laptop since this morning."),
 
        # Step 5 (experiment): uncomment to see if priority rises when many users are hit
        # ("TEST-0007", "Salesforce CRM access issue - Sales team",
        #  "Entire Sales team (25 users) cannot access Salesforce CRM from company laptops since this morning."),
    ]
 
    results = {}
    for number, short_desc, desc in test_tickets:
        results[number] = triage_ticket(number, short_desc, desc)
 
    # Summary table
    print("\n" + "="*55)
    print("TRIAGE SUMMARY")
    print("="*55)
    print(f"  {'Ticket':<12}{'Category':<13}{'Pri':<5}{'Assign To':<17}PII")
    for number, r in results.items():
        if r:
            print(f"  {number:<12}{r['category']:<13}{r['priority']:<5}{r['assignment_group']:<17}{r['pii_detected']}")
        else:
            print(f"  {number:<12}(not classified)")
 
    print("\n" + "="*55)
    print("OPEN TICKET COUNTS BY CATEGORY")
    print("="*55)
    counts = get_open_tickets(CSV_PATH)
    if "error" in counts:
        print(f"  {counts['error']}")
    else:
        for cat, count in sorted(counts.items()):
            print(f"  {cat:<20} {count} open")