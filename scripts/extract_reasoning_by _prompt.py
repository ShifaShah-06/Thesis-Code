#!/usr/bin/env python3


import os, csv, json, re
from pathlib import Path
from openai import AzureOpenAI

# ─── configuration ───
ESSAY_DIR  =Path("cleaned")
CSV_OUT    = Path("gpt_reasoning_counts.csv")

API_KEY=''
AZURE_ENDPOINT=""
API_VERSION="2024-12-01-preview"
DEPLOYMENT_NAME = "gpt-4o-mini01"


client = AzureOpenAI(
    api_key       = API_KEY,
    azure_endpoint= AZURE_ENDPOINT,
    api_version   = API_VERSION
)

DEFS = {
    "deductive":  "It refers to the deduction of a particular effect of a general premise before it is postulated (cause and rule).",
    "inductive":  "It refers to the inference of a general proposition (rule) or generalizing a proposition about a limited set of particular propositions (cause and effect).",
    "abductive":  "It refers to the production of reasoning that develops (deduces) consequences of a specific premise taken as hypothesis.",
    "analogical": "It refers to reasoning based on other realities with structural similarity (A is to B as X is to Y)."
}

def safe_int(val):
    if val in (None, '', 'null', 'N/A'):
        return 0
    if isinstance(val, (list, tuple, set, dict)):
        return len(val)
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(str(val).strip())
        except ValueError:
            return 0



PROMPT_TEMPLATE = """You are an argument-analysis assistant.
Below are the only four reasoning types we care about:

{definitions}

TASK:
1. Read the essay in full.
2. Count every distinct reasoning instance and assign exactly one of the four labels.
3. **FORMAT REQUIREMENT** – respond **only** with

{{"deductive": <int>,
  "inductive": <int>,
  "abductive": <int>,
  "analogical": <int>}}

(no Markdown, no explanation, no extra keys).

Essay:
\"\"\"{essay_text}\"\"\"
"""

def run_azure_gpt(prompt: str) -> dict:
    response = client.chat.completions.create(
        model            = DEPLOYMENT_NAME,
        messages         = [
            {"role": "system", "content": "You are a strict JSON-only reasoning-analysis assistant."},
            {"role": "user",   "content": prompt}
        ],
        temperature      = 0.0,
        max_tokens       = 300,
        response_format  = {"type": "json_object"}   # enforce JSON
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*?\}", raw, re.S)
        if m:
            return json.loads(m.group())
        raise

# ─── main loop ────
def main():
    fieldnames = ["Essay_ID", "Type",
                  "Deductive", "Inductive", "Abductive", "Analogical",
                  "Total Types", "Distinct Types"]
    rows = []

    for essay_path in sorted(ESSAY_DIR.glob("*.txt")):
        essay_id = essay_path.stem
        print("Processing:", essay_id, flush=True)

        essay_text = essay_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not essay_text:
            print(f"{essay_id}: empty file, skipped")
            continue

        definitions = "\n".join(f"- **{k.capitalize()}**: {v}" for k, v in DEFS.items())
        prompt = PROMPT_TEMPLATE.format(definitions=definitions, essay_text=essay_text)

        counts_raw = run_azure_gpt(prompt)
        counts = {k: safe_int(counts_raw.get(k)) for k in DEFS}

        total    = sum(counts.values())
        distinct = sum(1 for v in counts.values() if v > 0)

        rows.append({
            "Essay_ID": essay_id,
            "Type":     "Matched Corpus 4o-mini",
            **{k.capitalize(): v for k, v in counts.items()},
            "Total Types":    total,
            "Distinct Types": distinct
        })

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {CSV_OUT} with {len(rows)} rows.")

if __name__ == "__main__":
    if not (API_KEY and AZURE_ENDPOINT and DEPLOYMENT_NAME):
        raise SystemExit("Missing Azure OpenAI credentials or deployment name.")
    main()