## Failures found and improvements made

Each row is a failure we measured, what we changed, and the effect on the 32 labelled samples.

| Stage | Failure found | Change | Effect |
|---|---|---|---|
| Baseline | Trusting the workflow's `reported_status` | Measured as the floor to beat | 17/32 statuses correct (53.1%), no steps |
| Evidence | A first prototype found 70 of the 75 evidence citations in the gold answers. The misses were absences proven by snapshots, plus one PDF | Added coverage (absence by omission from a complete snapshot) and PDF text extraction | 75/75 citations supported |
| Engine | A snapshot omission overrode a direct observation (cases 21 and 24) | Explicit sightings beat omission | Both cases fixed |
| Engine | A failover replica still held a profile the primary had deleted (case_05) | Replicas are tracked as separate copies; any violating copy decides | Correct INCOMPLETE finding |
| Engine | Restricting identity links to evidence before audit_at lost 8 gold steps | Ownership may use any evidence; state and citations use only evidence available at audit_at | Recall 81.9% → 86.0% |
| Intent | The first prompt classified three "please verify my deletion" requests as non-deletions | Asking to verify or confirm an erasure counts as a deletion request | Status 29/32 → 32/32 |
| Intent | A security review found that batching 25 requests per call let one text influence another's verdict | One request per call | A hostile text can affect only its own verdict |
| Linking | Some primary accounts were linked only through the deletion worker's activity, and nearest-in-time matching was right 8 times and wrong 9 | The worker's result must agree with the reported status, and each activity goes to at most one case | 11 right, 3 wrong; exact sequences 10/32 → 17/32 |
| Linking | We tried the same matching for anonymised analytics records | Rejected: right only 3 times in 9 | Not shipped |
| Vision | The first vision prompt dropped the page's own subject on detail screens and misread a ticket number as a resource ID | The page subject is extracted first, and aliases are resolved through the data catalog | All expected resource IDs extracted |
| Vision | A reference-only tombstone screenshot was reported as exposed PII (case_46) | A tombstone showing only the user reference counts as redacted, per the policy's own definition | False finding removed |
| Safety | A missing API key or spent budget would silently fall back to keyword rules | The run refuses to write output if any verdict came from a fallback | Fails loudly instead of being quietly wrong |