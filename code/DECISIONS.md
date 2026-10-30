# Decisions

One line per decision: what we chose, and why.

1. Work in the official repo and start Claude Code at its root: AGENTS.md logs every turn to log.txt, the submitted transcript.
2. Models read, code judges: a deletion audit must be reproducible, so code applies deletion_rules.csv and the LLM only reads language and images.
3. Baseline = trust reported_status: it is the failure the challenge describes, and it scores 53.1% on the samples, an honest floor.
4. Standard library for the scaffold: no install friction, runs on any Python 3.9+.
5. Sort output numerically by case_id: matches the input order; a string sort would put case_100 first.
6. Exact identifier matching only: the data plants name decoys (Tomasz vs Nadia Novak).
7. Text evidence first, vision last: none of the 32 gold answers cites a screenshot.
8. Claude Haiku 5.5 for all LLM calls, cached, with a spend cap: a full run should cost cents, and the cap protects the $7.41 budget from a looping bug.
9. Design checked before building: a throwaway prototype in planning matched 70 of the 75 evidence citations in the gold samples; the other 5 needed absence-by-coverage and PDF text, so the spec includes both.
10. One schema-agnostic extractor: any record with a resource_id becomes a sighting, whatever the file format, so there is no parser per file family to maintain.
11. Absence needs proof: a snapshot that lists a system's full contents at a moment proves a missing resource ABSENT then (gold case_31 cites an empty account_service snapshot).
12. system_id comes from the resource_id prefix: it is one-to-one across all gold answers, while the activity stream's target_system varies.
13. PDFs are read with pypdf, not a model: all four have text layers, so it is free and deterministic.
14. Develop on the system Python 3.10 that Claude Code also uses, so there is one environment; a clean venv install from requirements.txt is verified before submission.
15. SQL tables count as complete listings only for systems named in their snapshot_scope: out-of-scope tables are empty and would falsely prove everything absent.
16. PDFs without resource IDs (account history, legal register) are left out for now: no gold answer cites them; revisit if evaluation shows retention misses.
17. Explicit sightings beat omission: a snapshot proves absence only for a copy with no direct observation (gold cases 21 and 24).
18. Replicas are separate copies: if any copy still holds data, the record remains (gold case_05: a failover replica kept a deleted profile).
19. Policy is read from deletion_rules.csv, not hardcoded, so a policy change needs no code change.
20. Category comes from the record's own record_type, then a small commented table of semantic hints, then the system default. The hints came from the samples, so they are a known overfitting risk, reported in the evaluation.
21. Engine rules prototyped in planning before building: 32/32 status, step precision 0.99, recall 0.86 on the samples; remaining misses are resources linked only through deletion-worker activities.
22. Ownership is timeless, state is not: linking may use evidence captured after audit_at, but state and citations use only evidence available at audit_at. Eight gold steps need the later link; this is a deliberate, narrow reading of "later evidence must not change the answer".
23. Backups are retained with restore blocked (rule_14's condition). Data flows are processes, not data, so they are checked against "no active sync can recreate the data" rather than a rules row.
24. Resources with no category, no rule, or no admissible evidence are skipped, with the reason shown in --explain, rather than guessed.
25. One LLM wrapper owns every API call: cached, logged to a usage ledger, and capped at $1, so re-runs are free and a bug can't drain the $7.41 budget.
26. The LLM decides only intent (and later reads screenshots); it never decides a case's status or steps.
27. A request to verify, audit or confirm an erasure counts as a deletion request: gold treats three such samples as deletions, and a person checking their erasure is still exercising the right to erasure.
28. Label-free check: all 10 NOT_APPLICABLE verdicts link to an account that the activity stream's privacy intake also classified as non-deletion, a signal the classifier never sees.
29. Primary accounts with no identifier link are matched to the deletion worker's own DELETE_RECORD activity. The worker's result must agree with the reported status, each activity goes to at most one case, and the nearest to the report wins. In planning this raised exact sequences from 10/32 to 17/32; the same idea for analytics was right 3 times out of 9, so it was rejected.
30. Every request is classified in its own LLM call: a hostile request text can only affect its own verdict, never someone else's, and at ~375 unique texts this costs cents.
31. The run fails loudly if any intent verdict came from the keyword fallback: a silent fallback would turn German or Polish deletions into NOT_APPLICABLE without anyone noticing.
32. Screenshots are read by Claude Haiku 5.5 once each, literally, into the same sighting schema; the rules engine judges them like any file. This supersedes 7: each screenshot was captured just before one evaluation case's audit. 11 reads cost $0.01 and are cached by image hash.
33. "Not shown" is dropped, never stored as null, so a field a screen doesn't display can't read as "identifier removed"; a resource-shaped ID printed as an alias is promoted to the record's ID, and a ticket or order number is never trusted as one.
34. Evidence without resource IDs (screenshot rows, the two PDFs left out in 16) reaches a resource only through the data catalog's published aliases, e.g. "object_19 (delivery_confirmation_19.pdf)". A user_<n> alias matches only an item's own key, never free text, so a register listing many users can't tie them all to one tombstone.
35. State evidence must fall in the audit window (requested_at to audit_at): an older screenshot cannot create a finding. Linking still uses all evidence (22). Neutral on the samples; 9 evaluation cases lose a verification step for an account whose only evidence predates the request.
36. Unproven safety is unsafe: a backup with no explicit restore state, or a data flow with no explicit enabled state, counts as active.
37. A linked resource whose in-window evidence shows only other people's direct identifiers is kept but flagged "identity mismatch" in --explain: the catalog link is the ownership claim; the flag makes a possible mislink reviewable.
38. A tombstone showing only the subject's user reference counts as redacted: that is the policy's own definition of a tombstone. We deliberately did not adopt a broader rule for all screenshots, which could hide real exposure.
39. The evaluation and usage reports are generated by build_reports.py from the evaluator, the custom cases, the label-free checks and the usage ledger, never typed by hand. The ledger is cumulative (probes, earlier prompts and custom cases included), so the usage report says its totals overstate the final run rather than presenting them as a measurement.
40. Every LLM call is tagged with a run_id and entrypoint, and each finished main.py run is recorded in code/cache/llm_runs.jsonl even when it made no call, so the usage report describes exactly the latest main run and lists all other calls as development; this supersedes the cumulative-ledger caveat in 39. Temperature is left unset: the API rejects it for claude-haiku-5-5 ("temperature: Extra inputs are not permitted", tested with and without effort), so repeatability comes from the cache.
41. Reproducibility was tested, not assumed: with the cache deleted, a fresh full run reproduced output.csv byte-for-byte.
42. The usage report's final run is the latest main run that made API calls (the latest cold run), not simply the latest main run: a later run served entirely from the cache is listed as a cache replay and never replaces it, and the upper-bound paragraph appears only when no main run made calls. This refines 40.
43. Unit tests (stdlib unittest, code/tests/) run offline: sockets are patched to fail, a model call that misses the cache fails the test, and the spend-cap and keyword-fallback tests run on a scratch copy of the cache, ledger and run log, so they cannot spend money or touch code/cache/.
44. Evidence files are untrusted input, so a bad one raises UnsafeEvidenceError instead of being skipped: SQL dumps load under a sqlite3 authorizer that denies ATTACH and DETACH (and so VACUUM INTO, which SQLite reports as ATTACH), and a manifest path must resolve, symlinks included, inside dataset/evidence/. agent.step_sort_key() was extracted so step order is testable; output.csv is unchanged.
