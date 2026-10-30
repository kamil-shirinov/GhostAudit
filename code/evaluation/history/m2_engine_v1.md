# Latest evaluation metrics

`code/agent.py` run on `dataset/sample_deletion_requests.csv` and scored against `dataset/sample_output.csv`.

| Metric | Value |
| --- | --- |
| Cases | 32 |
| Gold steps / predicted steps | 193 / 168 |
| Status accuracy | 32/32 (100.0%) |
| Status accuracy, gold COMPLETE | 10/10 (100.0%) |
| Status accuracy, gold INCOMPLETE | 22/22 (100.0%) |
| Status accuracy, gold NOT_APPLICABLE | 0/0 (0.0%) |
| Step precision | 166/168 (98.8%) |
| Step recall | 166/193 (86.0%) |
| Step F1 | 92.0% |
| Field accuracy, phase | 166/195 (85.1%) |
| Field accuracy, operation | 135/195 (69.2%) |
| Field accuracy, state | 135/195 (69.2%) |
| Field accuracy, system_id | 124/195 (63.6%) |
| Field accuracy, resource_id | 124/195 (63.6%) |
| Field accuracy, evidence_id | 166/195 (85.1%) |
| Sequence accuracy | 10/32 (31.2%) |

- Steps are matched as a multiset of (phase, operation, state, system_id, resource_id, evidence_id); step_no is ignored. Precision, recall and F1 are micro-averaged over all cases.
- Field accuracy compares steps position by position up to the longer list; a position present in only one list counts as wrong.
- Sequence accuracy is the share of cases whose whole steps array equals gold exactly and in order.
- A metric with an empty denominator is reported as 0.0%.
