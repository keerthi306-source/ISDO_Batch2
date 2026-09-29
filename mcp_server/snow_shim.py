"""
ISDO Lab C2 — Mock ServiceNow REST API (Flask Shim)
Mimics the ServiceNow Table API so the MCP server can make real HTTP calls
without touching a production system.

Endpoints:
  GET  /api/now/table/incident          — list all incidents
  GET  /api/now/table/incident/<number> — get one incident
  PATCH /api/now/table/incident/<number>— update a field (e.g. state, notes)
  GET  /api/now/table/incident?category=Network — filter by category

Run with:  python snow_shim.py
Default port: 5001
"""

from flask import Flask, jsonify, request
import csv
import os

app = Flask(__name__)

# Load mock data from CSV at startup
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "incidents.csv")

def load_incidents():
    """Load incidents from CSV.

    Guards against rows where an unquoted comma in the free-text
    'description' column splits it into extra columns. csv.DictReader
    stores those overflow values under the key None, which later crashes
    jsonify (it sorts keys, and None cannot be compared with str).
    """
    incidents = {}
    try:
        with open(DATA_FILE, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            desc_idx = header.index("description")
            for line_no, values in enumerate(reader, start=2):
                if not values or not any(v.strip() for v in values):
                    continue  # skip blank lines
                extra = len(values) - len(header)
                if extra > 0:
                    # Re-join the pieces of the description that were split on commas
                    merged = ",".join(values[desc_idx:desc_idx + extra + 1])
                    values = values[:desc_idx] + [merged] + values[desc_idx + extra + 1:]
                    print(f"[ServiceNow Mock] Warning: line {line_no} had {extra} unquoted "
                          f"comma(s) in 'description' - merged back together.")
                elif extra < 0:
                    values = values + [""] * (-extra)  # pad missing trailing fields
                row = {k: v.strip().strip('"') for k, v in zip(header, values)}
                incidents[row["number"]] = row
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
    return incidents

# In-memory store (simulates ServiceNow DB for the session)
INCIDENTS = load_incidents()

@app.route("/api/now/table/incident", methods=["GET"])
def list_incidents():
    """Return all incidents, optionally filtered by query params."""
    results = list(INCIDENTS.values())

    # Apply simple filters: ?category=Network or ?state=Open or ?priority=P1
    for key in ["category", "state", "priority", "assignment_group"]:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]

    # Mimic ServiceNow envelope
    return jsonify({"result": results, "total": len(results)})

@app.route("/api/now/table/incident/<number>", methods=["GET"])
def get_incident(number):
    """Return a single incident by number."""
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})

@app.route("/api/now/table/incident/<number>", methods=["PATCH"])
def update_incident(number):
    """Update fields on an incident (e.g. state, work_notes)."""
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404

    updates = request.get_json()
    if not updates:
        return jsonify({"error": "No update body provided"}), 400

    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})

@app.route("/api/now/table/incident", methods=["POST"])
def create_incident():
    """Create a new incident."""
    data = request.get_json()
    if not data or "number" not in data:
        return jsonify({"error": "Missing required field: number"}), 400

    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})

if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:5001")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    print("Endpoints: GET /api/now/table/incident  |  GET /health")
    app.run(port=5001, debug=True)
