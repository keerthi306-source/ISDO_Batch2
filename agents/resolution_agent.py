"""
ISDO Lab C4 — Resolution / KB Agent
Searches ChromaDB for matching KB articles and drafts a resolution.
HIGH confidence + non-P1 ticket  -> auto-resolve (L1).
MEDIUM / LOW confidence or P1    -> Human-in-the-Loop (HITL) flag.

Run from the project root (or from inside agents/ — both work):
    python agents/resolution_agent.py
"""

import anthropic
import chromadb
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Windows consoles can choke on arrows/emoji — force UTF-8 output.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paths resolved relative to this file, so it works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "kb"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    sys.exit("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env file (same one used in C2/C3).")

client = anthropic.Anthropic(api_key=API_KEY)
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
TEMPERATURE = 0.0
MAX_ROUNDS = 4  # hard safety cap on the agentic loop

# Confidence thresholds (Step 3 of the lab)
HIGH_THRESHOLD = 0.60
MEDIUM_THRESHOLD = 0.35

# ── LOAD CHROMADB KB (same chunking as Lab C1) ───────────────────────────────

def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a markdown article at ## headings. Each section = one chunk."""
    chunks, current_lines, current_heading = [], [], "Introduction"
    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({"content": "\n".join(current_lines).strip(),
                           "heading": current_heading, "filename": filename})
            current_lines = []
            current_heading = line[3:].strip()
        current_lines.append(line)
    if current_lines:
        chunks.append({"content": "\n".join(current_lines).strip(),
                       "heading": current_heading, "filename": filename})
    return chunks


def build_kb():
    """Load KB articles into an in-memory ChromaDB collection called 'isdo_kb'.
    Uses cosine distance so that (1 - distance) is a true 0-1 similarity score."""
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"ERROR: no KB articles found in {KB_DIR}. Copy the data/kb folder into your project.")

    db = chromadb.Client()
    try:
        db.delete_collection("isdo_kb")  # fresh copy each run — no stale data
    except Exception:
        pass
    kb = db.create_collection("isdo_kb", metadata={"hnsw:space": "cosine"})

    docs, ids, metas = [], [], []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        for chunk in chunk_article(text, md_file.name):
            ids.append(f"kb_{len(ids)}")
            docs.append(chunk["content"])
            metas.append({"filename": md_file.name, "heading": chunk["heading"]})

    kb.add(documents=docs, ids=ids, metadatas=metas)
    print(f"KB loaded: {len(docs)} chunks from {len(md_files)} articles")
    return kb


_KB = None

def get_kb():
    """Build the KB once, on first use (keeps imports fast for Lab C6)."""
    global _KB
    if _KB is None:
        _KB = build_kb()
    return _KB

# ── TOOL DEFINITIONS ──────────────────────────────────────────────────────────

tools = [
    {
        "name": "search_kb",
        "description": "Search the knowledge base for articles matching the ticket description. Returns the top 2 matching articles with confidence scores (1 - distance).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query based on the ticket's short description and symptoms"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "draft_resolution",
        "description": "Draft the resolution for the ticket based on the KB article content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {
                    "type": "string",
                    "description": "3-4 numbered, plain-English steps taken from the KB article, to send to the requester"
                },
                "auto_resolve": {
                    "type": "boolean",
                    "description": "True only if confidence is HIGH, the ticket is not P1, and the KB article says the issue is L1 auto-resolvable"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                    "description": "HIGH = top score > 0.60; MEDIUM = 0.35-0.60; LOW = < 0.35"
                },
                "kb_article_used": {"type": "string", "description": "File name of the KB article used, or 'none'"}
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve", "confidence", "kb_article_used"]
        }
    }
]

# ── TOOL IMPLEMENTATION ───────────────────────────────────────────────────────

def score_to_confidence(score: float) -> str:
    if score > HIGH_THRESHOLD:
        return "HIGH"
    if score > MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def search_kb(query: str) -> dict:
    """Query ChromaDB, group chunks by article, return the top 2 articles.
    The full article text is returned so the model has the complete fix steps."""
    kb = get_kb()
    raw = kb.query(query_texts=[query], n_results=min(10, kb.count()))

    best_per_file = {}
    for i in range(len(raw["documents"][0])):
        fname = raw["metadatas"][0][i].get("filename", f"unknown_{i}")
        distance = raw["distances"][0][i]
        if fname not in best_per_file or distance < best_per_file[fname]:
            best_per_file[fname] = distance

    ranked = sorted(best_per_file.items(), key=lambda kv: kv[1])[:2]
    articles = []
    for fname, distance in ranked:
        score = round(max(0.0, 1 - distance), 2)
        path = KB_DIR / fname
        articles.append({
            "article": fname,
            "confidence_score": score,
            "confidence": score_to_confidence(score),
            "content": path.read_text(encoding="utf-8") if path.exists() else "",
        })
    return {"query": query, "articles": articles}


def handle_tool(name, inp):
    if name == "search_kb":
        return search_kb(inp["query"])
    elif name == "draft_resolution":
        return {"status": "draft recorded"}
    return {"error": f"Unknown tool: {name}"}

# ── GUARDRAIL (code-enforced HITL boundary) ──────────────────────────────────

def apply_guardrail(draft: dict, top_score, priority: str) -> dict:
    """The model proposes; code decides. Confidence is recomputed from the real
    KB score, and auto_resolve can only be turned OFF here, never ON."""
    final = dict(draft)
    reasons = []

    if top_score is None:
        final["confidence"] = "LOW"
        reasons.append("no KB search was performed")
    else:
        actual = score_to_confidence(top_score)
        if actual != draft.get("confidence"):
            reasons.append(f"confidence corrected {draft.get('confidence')} -> {actual} (top score {top_score:.0%})")
        final["confidence"] = actual

    if final["auto_resolve"]:
        if final["confidence"] != "HIGH":
            final["auto_resolve"] = False
            reasons.append(f"{final['confidence']} confidence cannot auto-resolve")
        if priority == "P1":
            final["auto_resolve"] = False
            reasons.append("P1 incidents always need a human")

    final["guardrail_notes"] = reasons
    return final

# ── MODEL CALL (works across SDK versions — same fix as C3) ──────────────────

_temperature_mode = "extra_body"

def call_claude(messages):
    global _temperature_mode
    params = dict(model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
                  tools=tools, messages=messages)
    if _temperature_mode == "extra_body":
        try:
            return client.messages.create(**params, extra_body={"temperature": TEMPERATURE})
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e).lower():
                raise
            print("  (note: this model doesn't accept a custom temperature — using its default)")
            _temperature_mode = "off"
    return client.messages.create(**params)

# ── RESOLUTION AGENT ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb ONCE, using the ticket's summary and details as the query.
2. Then call draft_resolution.

Confidence (use the top article's confidence_score):
- HIGH   : score > {HIGH_THRESHOLD}  — KB article covers the issue
- MEDIUM : {MEDIUM_THRESHOLD} to {HIGH_THRESHOLD} — partial match, human should review
- LOW    : score < {MEDIUM_THRESHOLD} — no clear match, escalate to L2

auto_resolve = True ONLY when ALL are true:
- confidence is HIGH
- priority is P2, P3 or P4 (never P1)
- the KB article's "Auto-Resolve Eligibility" section says this case is L1 auto-resolvable
Otherwise auto_resolve = False.

resolution_text: 3-4 numbered steps copied from the matched KB article's
Resolution Steps — specific, not generic advice. If confidence is LOW, write
a short L2 escalation note instead and set kb_article_used to "none".

A LOW-confidence result is a correct, expected outcome for topics the KB does
not cover. Do NOT search again with different wording."""


def resolve_ticket(ticket_number, short_description, description, category, priority="P3"):
    """Run the Resolution Agent on one ticket. Returns the final (guard-railed) resolution dict."""
    print(f"\n{'='*55}")
    print(f"Resolving: {ticket_number} | Category: {category} | Priority: {priority}")
    print(f"{'='*55}")
    print(f"Issue: {short_description}")

    messages = [{
        "role": "user",
        "content": (f"Find a resolution for this ticket:\n\nTicket: {ticket_number}\n"
                    f"Category: {category}\nPriority: {priority}\n"
                    f"Summary: {short_description}\nDetails: {description}")
    }]

    top_score = None
    draft = None
    rounds = 0

    # Agentic while loop: runs until Claude returns end_turn
    while True:
        rounds += 1
        if rounds > MAX_ROUNDS:
            print(f"  (stopped after {MAX_ROUNDS} rounds)")
            break

        response = call_claude(messages)

        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  Agent: {block.text.strip()}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []

        for block in response.content:
            if block.type != "tool_use":
                continue
            result = handle_tool(block.name, block.input)

            if block.name == "search_kb":
                print(f"  -> KB search: '{block.input.get('query')}'")
                for art in result["articles"]:
                    print(f"     [{art['confidence_score']:.0%}] {art['article']}")
                if result["articles"]:
                    top_score = result["articles"][0]["confidence_score"]
            elif block.name == "draft_resolution":
                draft = dict(block.input)

            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})

        messages.append({"role": "user", "content": tool_results})
        if draft is not None:
            break  # draft is the final answer — no need for another model call

    if draft is None:
        draft = {"ticket_number": ticket_number, "resolution_text": "Agent did not produce a draft.",
                 "auto_resolve": False, "confidence": "LOW", "kb_article_used": "none"}

    final = apply_guardrail(draft, top_score, priority)

    print(f"\n  -> Confidence: {final['confidence']}  |  Auto-resolve: {final['auto_resolve']}")
    print(f"  -> KB Article: {final['kb_article_used']}")
    for note in final["guardrail_notes"]:
        print(f"  -> Guardrail: {note}")
    print("\n  RESOLUTION DRAFT:")
    for line in str(final["resolution_text"]).splitlines():
        print(f"  {line}")

    if not final["auto_resolve"]:
        why = "P1 incident" if priority == "P1" and final["confidence"] == "HIGH" else f"{final['confidence'].capitalize()} confidence"
        print(f"\n  ⚠️  HITL FLAG: {why} — human review required before sending.")
    else:
        print("\n  ✅ AUTO-RESOLVE: L1 fix can be sent to the requester.")

    return final

# ── RUN ON SAMPLE TICKETS ─────────────────────────────────────────────────────

if __name__ == "__main__":
    get_kb()

    # (number, short_description, description, category, priority from Triage / C3)
    test_tickets = [
        ("INC0001001", "VPN not connecting after password change",
         "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
         "Network", "P2"),
        ("INC0001006", "Password reset request",
         "User locked out of AD account after 5 failed attempts. Needs immediate reset.",
         "Access", "P2"),
        ("INC0001002", "Cannot access ERP system - login error",
         "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.",
         "Application", "P1"),

        # Step 5: uncomment to test a ticket with no KB match
        # ("TEST-0005", "Cisco Webex not launching on Mac M2",
        #  "Cisco Webex not launching on Mac M2", "Software", "P3"),
    ]

    results = []
    for number, short_desc, desc, cat, pri in test_tickets:
        results.append((number, pri, resolve_ticket(number, short_desc, desc, cat, pri)))

    print(f"\n{'='*55}")
    print("RESOLUTION SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Confidence':<12}{'Auto':<7}KB Article")
    for number, pri, r in results:
        print(f"  {number:<12}{pri:<5}{r['confidence']:<12}{str(r['auto_resolve']):<7}{r['kb_article_used']}")