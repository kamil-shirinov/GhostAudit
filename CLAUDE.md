The organisers' agent instructions (AGENTS.md) are in the official challenge repo and aren't included here.

## Project notes

Goal: an auditor that decides COMPLETE / INCOMPLETE / NOT_APPLICABLE for every request in dataset/deletion_requests.csv and writes ordered, evidence-cited steps to output.csv.

Principle: models read, code judges. Code applies deletion_rules.csv; the LLM (claude-haiku-5-5) only classifies request intent and reads screenshots/PDFs. Every call is cached, with a spend cap.

Modules (planned):
- code/evidence.py: every evidence file -> one table of sightings (file_id, time, system, resource, category, state, identifiers)
- code/intent.py: does request_text authorise deletion? (cached LLM, keyword fallback)
- code/identity.py: email -> user_id -> all current and historical identifiers, exact matches only
- code/agent.py: state at audit_at per linked resource -> deletion_rules.csv -> ordered steps
- code/evaluation/main.py: scores against the 32 gold samples

Rules:
- Python 3.9-compatible; run everything from the repo root
- Only the evaluator reads dataset/sample_output.csv; the agent never does
- Secrets live only in .env; never print or log them
- Record decisions in code/DECISIONS.md, one line each

Status: M3 in progress. Done: engine (samples status 32/32, P 98.8%, R 86.0%) and LLM intent (10/10 probes, 10 NOT_APPLICABLE in the 518). Next: workflow account linking, then vision.