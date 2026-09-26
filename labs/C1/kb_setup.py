"""
ISDO Lab C1 — Knowledge Base Setup with ChromaDB
Loads the KB markdown articles into a ChromaDB vector store for semantic search.
Run this script ONCE before Labs C4, C6, C8.

What it does:
  1. Reads all .md files from data/kb/
  2. Chunks each article into sections (split at ## headings)
  3. Stores sections in ChromaDB with metadata
  4. Tests the setup with 4 sample queries

Requirements:
  pip install chromadb anthropic python-dotenv
"""

import chromadb
import os
from pathlib import Path

# Find the project root (the folder that contains data/kb), so the script
# works whether you run it from C:\ISDO_Batch2 or from labs\C1
def find_project_root(start: Path) -> Path:
    for folder in [start, *start.parents]:
        if (folder / "data" / "kb").is_dir():
            return folder
    return start

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"   # saved on disk for later labs

# ── STEP 1: Load all KB markdown files ───────────────────────────────────────

print("=" * 55)
print("ISDO Lab C1 — KB Setup with ChromaDB")
print("=" * 55)
print()

md_files = sorted(KB_DIR.glob("*.md"))
if not md_files:
    print(f"ERROR: No .md files found in {KB_DIR}")
    print("Make sure the KB articles are in <project>/data/kb/.")
    exit(1)

print(f"Step 1: Found {len(md_files)} KB articles")
for f in md_files:
    print(f"  ✓ {f.name}")
print()

# ── STEP 2: Chunk each article by ## heading ──────────────────────────────────

def chunk_article(text: str, filename: str) -> list[dict]:
    """Split a markdown article at ## headings. Each section = one chunk."""
    chunks = []
    current_lines = []
    current_heading = "Introduction"

    for line in text.split("\n"):
        if line.startswith("## ") and current_lines:
            chunks.append({
                "content": "\n".join(current_lines).strip(),
                "heading": current_heading,
                "filename": filename
            })
            current_lines = []
            current_heading = line[3:].strip()
        current_lines.append(line)

    if current_lines:
        chunks.append({
            "content": "\n".join(current_lines).strip(),
            "heading": current_heading,
            "filename": filename
        })

    return chunks

all_chunks = []
for md_file in md_files:
    text = md_file.read_text(encoding="utf-8")
    chunks = chunk_article(text, md_file.name)
    all_chunks.extend(chunks)

print(f"Step 2: Split into {len(all_chunks)} chunks")
print()

# ── STEP 3: Store in ChromaDB ─────────────────────────────────────────────────

# PersistentClient saves to disk; chromadb.Client() is in-memory only and
# the collection would be lost as soon as this script ends.
db = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Remove existing collection if re-running
try:
    db.delete_collection("isdo_kb")
except Exception:
    pass

# Cosine distance (0 = identical) so that "1 - distance" is a sensible confidence
kb = db.create_collection("isdo_kb", metadata={"hnsw:space": "cosine"})

docs = [c["content"] for c in all_chunks]
ids = [f"chunk_{i}" for i in range(len(all_chunks))]
metas = [{"filename": c["filename"], "heading": c["heading"]} for c in all_chunks]

kb.add(documents=docs, ids=ids, metadatas=metas)

print(f"Step 3: Stored {len(docs)} chunks in ChromaDB")
print(f"        Collection: isdo_kb  |  Chunks: {kb.count()}")
print(f"        Saved to: {CHROMA_DIR}")
print()

# ── STEP 4: Test with sample queries ─────────────────────────────────────────

print("Step 4: Testing KB with sample queries")
print("-" * 45)

test_queries = [
    "VPN not connecting after password change",
    "How do I reset an Active Directory password?",
    "SAP ERP login failure for multiple users",
    "Network switch down, building offline",
]

for query in test_queries:
    results = kb.query(query_texts=[query], n_results=1)
    if results["documents"][0]:
        meta = results["metadatas"][0][0]
        distance = results["distances"][0][0]
        confidence = max(0, 1 - distance)
        print(f"  Q: {query[:50]}")
        print(f"     → {meta['filename']}  [{confidence:.0%} confidence]")
    else:
        print(f"  Q: {query[:50]} → No match found")
    print()

print("✅ KB setup complete! ChromaDB is ready for Labs C4, C6, and C8.")
print()
print("Next step: Run labs/C2/snow_shim.py and jira_shim.py to start mock APIs.")
