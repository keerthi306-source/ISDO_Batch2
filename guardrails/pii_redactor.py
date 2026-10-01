"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.
Covers: person names (spaCy NER + regex fallback), usernames / login IDs,
DOMAIN\\user accounts, email addresses, employee IDs, IP addresses, phone numbers.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import re
import json
from datetime import datetime

# Try to import spaCy. If it's missing, fall back to regex (names are still caught).
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("⚠  spaCy not available — using regex-only PII detection "
          "(install: pip install spacy && python -m spacy download en_core_web_sm)")

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
# Order = priority. When two matches overlap, the one listed first wins
# (e.g. an email is never partly re-tagged as a username).

PATTERNS = [
    ("EMAIL",       r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b', 0),
    # DOMAIN\user  e.g. ZENSAR\keerthi.m1
    ("USERNAME",    r'\b[A-Za-z][A-Za-z0-9\-]{1,15}\\[A-Za-z0-9](?:[A-Za-z0-9._\-]*[A-Za-z0-9])?', 0),
    # "Username: rkumar01", "user_id=akash123", "login id - jdoe", "samaccountname: x"
    ("USERNAME",    r'(?i:\b(?:user[\s_\-]?name|user[\s_\-]?id|login[\s_\-]?id|log[\s_\-]?in|'
                    r'account(?:[\s_\-]?name)?|samaccountname|upn|uid)\s*[:=\-]\s*)'
                    r'([A-Za-z0-9](?:[A-Za-z0-9._\-]*[A-Za-z0-9])?)', 1),
    # "for user rkumar01", "user j.doe" — free-text handles with a digit, dot or underscore
    ("USERNAME",    r'(?i:\buser\s+)([A-Za-z][A-Za-z0-9]*[._\d](?:[A-Za-z0-9._\-]*[A-Za-z0-9])?)', 1),
    ("EMPLOYEE_ID", r'\b(?:EMP|ZEN)-?\d{3,6}\b', 0),
    ("PHONE",       r'(?<!\w)(?:\+91[\-\s]?)?\d{10}\b|\b\d{3}[\-\s]\d{3}[\-\s]\d{4}\b', 0),
    ("IP_ADDRESS",  r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 0),
]

# Ticket refs are NOT PII; they are kept as-is.
TICKET_REF = r'\b(?:INC|REQ|CHG)-?\d{4,7}\b'

# Regex name fallback: a capitalised name (1-3 words) right after a cue word.
# Cue words are case-insensitive; the name itself must be Title Case or ALL CAPS.
_NAME_WORD = r"(?:[A-Z][a-z][A-Za-z'\-]*|[A-Z]{2,}|[A-Z]'[A-Z][a-z]+)"
NAME_CUE_RE = re.compile(
    r"(?i:\b(?:user|employee|contractor|customer|caller|requester|requestor|"
    r"reported\s+by|raised\s+by|assigned\s+to|for|by|from|contact|name|"
    r"mr|mrs|ms|dr|dear|hi|hello|thanks|regards)\.?)[:,]?\s+"
    rf"({_NAME_WORD}(?:\s+{_NAME_WORD}){{0,2}})"
)

# Words that are never names. Stops "for VPN Access" or "by Outlook" being tagged.
NOT_NAMES = {
    "VPN", "SLA", "KB", "PII", "IT", "HR", "SAP", "AD", "MFA", "SSO", "OTP", "API",
    "Outlook", "Windows", "Teams", "Office", "Excel", "Azure", "Citrix", "Okta",
    "Password", "Access", "Reset", "Error", "Ticket", "Laptop", "Printer", "Email",
    "Network", "Server", "Login", "Account", "Admin", "Support", "Team", "Manager",
    "The", "This", "That", "All", "Please", "Urgent", "Contractor", "Employee", "User",
    "Contact", "ServiceNow", "Claude", "Finance", "Jira", "Webex", "Zensar", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}

# ── AUDIT LOGGER ──────────────────────────────────────────────────────────────

audit_log = []

def _audit(action, detail):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "module": "PIIRedactor",
        "action": action,
        "detail": detail,
    }
    audit_log.append(entry)
    return entry

# ── DETECTION ─────────────────────────────────────────────────────────────────

def _name_candidates(text):
    """Return (start, end, value) spans for person names."""
    spans = []

    # 1. spaCy NER
    if SPACY_AVAILABLE:
        for ent in nlp(text).ents:
            if ent.label_ != "PERSON":
                continue
            # Skip single ALL-CAPS acronyms (PII, SLA...) that spaCy sometimes tags
            # as PERSON. Multi-word all-caps like "RAJESH KUMAR" is still kept.
            if ent.text.isupper() and " " not in ent.text:
                continue
            if ent.text in NOT_NAMES:
                continue
            # Real names don't contain digits ("L2-Network-Ops", "Windows 11", "INC-1001")
            if any(ch.isdigit() for ch in ent.text):
                continue
            spans.append((ent.start_char, ent.end_char, ent.text))

    # 2. Regex fallback (always runs: catches names spaCy misses, e.g. ALL CAPS)
    for m in NAME_CUE_RE.finditer(text):
        words = m.group(1).split()
        # Trim trailing non-name words ("John Smith Password" -> "John Smith")
        while words and words[-1] in NOT_NAMES:
            words.pop()
        if not words or words[0] in NOT_NAMES:
            continue
        if any(ch.isdigit() for w in words for ch in w):
            continue
        # A single ALL-CAPS word is far more likely an acronym than a name
        if len(words) == 1 and words[0].isupper():
            continue
        value = " ".join(words)
        start = m.start(1)
        spans.append((start, start + len(value), value))

    return spans


def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    Example:
      clean, m = redact("User John Smith (username: jsmith01) - john.smith@corp.com")
      # clean = "User [NAME_1] (username: [USERNAME_1]) - [EMAIL_1]"
    """
    # Protect ticket refs so no pattern ever touches them
    protected = [(m.start(), m.end()) for m in re.finditer(TICKET_REF, text)]

    def overlaps(a, b, spans):
        return any(a < e and s < b for s, e, *_ in spans)

    # Collect every candidate span: (start, end, label, value)
    found = []
    for label, pattern, group in PATTERNS:
        for m in re.finditer(pattern, text):
            s, e = m.span(group)
            if not overlaps(s, e, protected) and not overlaps(s, e, found):
                found.append((s, e, label, m.group(group)))

    for s, e, value in _name_candidates(text):
        if not overlaps(s, e, protected) and not overlaps(s, e, found):
            found.append((s, e, "NAME", value))

    # Mask every other occurrence of a detected name/username too
    # (e.g. "John Smith ... John Smith called again").
    for label in ("NAME", "USERNAME"):
        for value in {v for _, _, l, v in found if l == label}:
            for m in re.finditer(rf'(?<!\w){re.escape(value)}(?!\w)', text):
                if not overlaps(m.start(), m.end(), found) and not overlaps(m.start(), m.end(), protected):
                    found.append((m.start(), m.end(), label, value))

    # Assign tokens in reading order; the same value always gets the same token
    found.sort()
    mapping, value_to_token, counters = {}, {}, {}
    for s, e, label, value in found:
        key = (label, value)
        if key not in value_to_token:
            counters[label] = counters.get(label, 0) + 1
            token = f"[{label}_{counters[label]}]"
            value_to_token[key] = token
            mapping[token] = value

    # Replace right-to-left so earlier offsets stay valid
    clean = text
    for s, e, label, value in sorted(found, reverse=True):
        clean = clean[:s] + value_to_token[(label, value)] + clean[e:]

    if mapping:
        # Log counts by type only. Never write the raw values to the audit trail.
        by_type = {k: v for k, v in sorted(counters.items())}
        _audit("redact", f"{len(mapping)} PII item(s) masked: {by_type} "
                         f"(engine={'spacy+regex' if SPACY_AVAILABLE else 'regex'})")
    else:
        _audit("redact", "No PII detected")

    return clean, mapping


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        # Agent rationales often quote ticket text. Redact before persisting
        # so the audit trail itself doesn't become a PII store.
        safe_rationale = redact(rationale)[0][:200] if rationale else ""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale,
            "approval_status": approval_status,
        }
        self.entries.append(entry)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'='*55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'='*55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(f"PII REDACTION DEMO  (spaCy: {'ON' if SPACY_AVAILABLE else 'OFF'})")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        "User RAJESH KUMAR cannot login. Username: rkumar01. Account locked for ZENSAR\\rkumar01.",
        "Login failed for user priya.n (user_id=akash123). Raised by Priya Nair, follow-up: Priya Nair called again.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")
            assert restore(clean, mapping) == ticket, "round-trip failed"

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 210 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation for user John Smith requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user — auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: logs/demo_audit.jsonl")