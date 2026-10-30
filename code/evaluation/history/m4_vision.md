# Latest evaluation metrics

`code/agent.py` run on `dataset/sample_deletion_requests.csv` and scored against `dataset/sample_output.csv`.

| Metric | Value |
| --- | --- |
| Cases | 32 |
| Gold steps / predicted steps | 193 / 182 |
| Status accuracy | 32/32 (100.0%) |
| Status accuracy, gold COMPLETE | 10/10 (100.0%) |
| Status accuracy, gold INCOMPLETE | 22/22 (100.0%) |
| Status accuracy, gold NOT_APPLICABLE | 0/0 (0.0%) |
| Step precision | 177/182 (97.3%) |
| Step recall | 177/193 (91.7%) |
| Step F1 | 94.4% |
| Field accuracy, phase | 180/195 (92.3%) |
| Field accuracy, operation | 168/195 (86.2%) |
| Field accuracy, state | 168/195 (86.2%) |
| Field accuracy, system_id | 166/195 (85.1%) |
| Field accuracy, resource_id | 163/195 (83.6%) |
| Field accuracy, evidence_id | 179/195 (91.8%) |
| Sequence accuracy | 17/32 (53.1%) |
| Workflow links | 18 needed an account: 11 assigned correctly, 3 assigned wrongly, 4 unmatched |

- Steps are matched as a multiset of (phase, operation, state, system_id, resource_id, evidence_id); step_no is ignored. Precision, recall and F1 are micro-averaged over all cases.
- Field accuracy compares steps position by position up to the longer list; a position present in only one list counts as wrong.
- Sequence accuracy is the share of cases whose whole steps array equals gold exactly and in order.
- Workflow links: sample deletions with no identifier-linked account; an assigned account is correct if gold's steps name that resource.
- A metric with an empty denominator is reported as 0.0%.
