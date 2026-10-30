# Latest evaluation metrics

`code/agent.py` run on `dataset/sample_deletion_requests.csv` and scored against `dataset/sample_output.csv`.

| Metric | Value |
| --- | --- |
| Cases | 32 |
| Gold steps / predicted steps | 193 / 0 |
| Status accuracy | 17/32 (53.1%) |
| Status accuracy, gold COMPLETE | 6/10 (60.0%) |
| Status accuracy, gold INCOMPLETE | 11/22 (50.0%) |
| Status accuracy, gold NOT_APPLICABLE | 0/0 (0.0%) |
| Step precision | 0/0 (0.0%) |
| Step recall | 0/193 (0.0%) |
| Step F1 | 0.0% |
| Field accuracy, phase | 0/193 (0.0%) |
| Field accuracy, operation | 0/193 (0.0%) |
| Field accuracy, state | 0/193 (0.0%) |
| Field accuracy, system_id | 0/193 (0.0%) |
| Field accuracy, resource_id | 0/193 (0.0%) |
| Field accuracy, evidence_id | 0/193 (0.0%) |
| Sequence accuracy | 0/32 (0.0%) |

- Steps are matched as a multiset of (phase, operation, state, system_id, resource_id, evidence_id); step_no is ignored. Precision, recall and F1 are micro-averaged over all cases.
- Field accuracy compares steps position by position up to the longer list; a position present in only one list counts as wrong.
- Sequence accuracy is the share of cases whose whole steps array equals gold exactly and in order.
- A metric with an empty denominator is reported as 0.0%.
