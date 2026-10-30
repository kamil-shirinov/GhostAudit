"""One end-to-end case, served from the committed LLM cache."""
import unittest

import common
import agent

CASE_ID = "case_05"


@common.requires_dataset
class SampleCase05Tests(common.OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        common.forbid_api_calls(self)  # a cache miss must fail the test, not call the model
        self.case = common.read_case(common.DATASET_DIR / "sample_deletion_requests.csv", CASE_ID)

    def test_predict_reports_the_failover_replica_as_incomplete(self) -> None:
        status, steps = agent.predict(self.case)
        self.assertEqual(status, "INCOMPLETE")
        self.assertEqual(
            steps,
            [
                {"step_no": 1, "phase": "FINDING", "system_id": "account_service", "resource_id": "account_005",
                 "operation": "REPORT_REMAINING_RECORD", "state": "PRESENT", "evidence_id": "evidence_54"},
                {"step_no": 2, "phase": "REMEDIATION", "system_id": "account_service", "resource_id": "account_005",
                 "operation": "DELETE_RECORD", "state": "ABSENT", "evidence_id": ""},
                {"step_no": 3, "phase": "VERIFICATION", "system_id": "account_service", "resource_id": "account_005",
                 "operation": "VERIFY_RECORD_ABSENT", "state": "ABSENT", "evidence_id": ""},
            ],
        )

    def test_the_cited_copy_is_the_failover_replica_and_the_intent_came_from_the_cache(self) -> None:
        trail = agent.audit_case(self.case)
        self.assertEqual(trail["intent"]["source"], "llm")
        self.assertTrue(trail["intent"]["deletion"])
        (resource,) = trail["resources"]
        self.assertEqual(resource["decided"]["copy"], "customer_profiles_failover_replica")
        self.assertEqual(resource["decided"]["evidence_id"], "evidence_54")


if __name__ == "__main__":
    unittest.main()
