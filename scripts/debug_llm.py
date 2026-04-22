"""Debug script to inspect raw LLM response and schema loading."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(".env")

import ollama
from src.schema.parser import SchemaParser
from src.lrg.builder import _build_extraction_prompt, _EXTRACTION_SCHEMA

# 1. Check schema loading
parser = SchemaParser()
schema = parser.auto_parse(pathlib.Path("data/spider"), "college_2")
print("=== SCHEMA ===")
print("Tables found:", schema.table_names())
print()
print(schema.format_for_prompt()[:400])
print()

# 2. Build the actual prompt used by the LRG builder
prompt = _build_extraction_prompt(
    "How many students are enrolled in each department?", schema, None
)
print("=== PROMPT (first 600 chars) ===")
print(prompt[:600])
print()

# 3. Raw call to Ollama
client = ollama.Client(host="http://localhost:11434")
print("=== RAW LLM RESPONSE (format=json) ===")
resp = client.generate(
    model="llama3.2:3b",
    prompt=prompt,
    format="json",
    options={"temperature": 0.0},
)
raw = resp["response"]
print(raw[:1500])
print()

# 4. Try to parse
try:
    parsed = json.loads(raw)
    print("=== PARSED KEYS ===")
    print(list(parsed.keys()))
    print("main_entities:", parsed.get("main_entities"))
except Exception as e:
    print("JSON PARSE ERROR:", e)
