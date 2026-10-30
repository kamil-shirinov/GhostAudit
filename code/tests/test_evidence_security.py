"""Evidence files are untrusted: a SQL dump may not ATTACH a file, a manifest path may not leave evidence/."""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import common
import evidence

ENTRY = {"file_id": "evidence_test", "file_path": "evidence/databases/x.sql", "captured_at": "2026-09-01T00:00:00Z"}
CAPTURED = "-- Captured at: 2026-09-01T00:00:00Z\n"
HANDOVER_LINE = "- 2026-09-01T00:00:00Z | account_service | account_001 | profile | FOUND | user_001\n"


class TempDirTestCase(common.OfflineTestCase):
    def setUp(self) -> None:
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name).resolve()


def sql_literal(path: Path) -> str:
    return "'%s'" % str(path).replace("'", "''")


class SqlDumpGuardTests(TempDirTestCase):
    def load(self, script: str):
        path = self.dir / "dump.sql"
        path.write_text(script, encoding="utf-8")
        return evidence._sql_evidence(ENTRY, path)

    def test_control_unguarded_sqlite_does_create_the_file(self) -> None:
        # Without the guard the attack works, so the refusals below are not passing vacuously.
        planted = self.dir / "control.db"
        sqlite3.connect(":memory:").executescript("ATTACH DATABASE %s AS planted;" % sql_literal(planted))
        self.assertTrue(planted.exists())

    def test_attach_is_refused_and_creates_no_file(self) -> None:
        planted = self.dir / "planted.db"
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load(CAPTURED + "ATTACH DATABASE %s AS planted;\nCREATE TABLE planted.t (x);\n" % sql_literal(planted))
        self.assertFalse(planted.exists())

    def test_detach_is_refused(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load(CAPTURED + "DETACH DATABASE planted;\n")

    @unittest.skipIf(sqlite3.sqlite_version_info < (3, 27), "VACUUM INTO needs SQLite 3.27")
    def test_vacuum_into_is_refused_and_creates_no_file(self) -> None:
        planted = self.dir / "copy.db"
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load(CAPTURED + "CREATE TABLE t (x);\nVACUUM INTO %s;\n" % sql_literal(planted))
        self.assertFalse(planted.exists())

    def test_an_ordinary_dump_still_loads(self) -> None:
        sightings, coverage = self.load(
            CAPTURED
            + "PRAGMA foreign_keys=OFF;\nBEGIN TRANSACTION;\n"
            + "CREATE TABLE customer_profiles (resource_id TEXT, email TEXT);\n"
            + "INSERT INTO customer_profiles VALUES ('account_001', 'a@example.com');\n"
            + "COMMIT;\n"
        )
        self.assertEqual([s["resource_id"] for s in sightings], ["account_001"])
        self.assertEqual(sightings[0]["evidence_id"], "evidence_test")
        self.assertEqual([(c["system_id"], c["resource_ids"]) for c in coverage], [("account_service", ["account_001"])])

    def test_other_sql_errors_are_not_reported_as_unsafe(self) -> None:
        with self.assertRaises(sqlite3.DatabaseError) as caught:
            self.load(CAPTURED + "THIS IS NOT SQL;\n")
        self.assertNotIsInstance(caught.exception, evidence.UnsafeEvidenceError)


class ManifestPathTests(TempDirTestCase):
    """A scratch dataset/ whose manifest has one entry, so evidence.load_sightings() reads only that."""

    def setUp(self) -> None:
        super().setUp()
        self.dataset = self.dir / "dataset"
        (self.dataset / "evidence" / "records").mkdir(parents=True)
        (self.dataset / "system_activity.csv").write_text("activity_id,event_type,details\n", encoding="utf-8")
        # The same readable record in three places: inside evidence/, beside it, and outside dataset/.
        (self.dataset / "evidence" / "records" / "ok.txt").write_text(HANDOVER_LINE, encoding="utf-8")
        (self.dataset / "user_identities.csv").write_text(HANDOVER_LINE, encoding="utf-8")
        (self.dir / "outside.txt").write_text(HANDOVER_LINE, encoding="utf-8")

    def load(self, file_path: str):
        row = "evidence_01,%s,test_origin,2026-09-01T00:00:00Z\n" % file_path
        (self.dataset / "evidence_files.csv").write_text("file_id,file_path,origin,captured_at\n" + row, encoding="utf-8")
        return evidence.load_sightings(self.dataset)

    def test_a_path_inside_evidence_is_read(self) -> None:
        self.assertEqual([s["resource_id"] for s in self.load("evidence/records/ok.txt")], ["account_001"])

    def test_dot_dot_out_of_dataset_is_rejected(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load("../outside.txt")

    def test_dot_dot_out_of_evidence_is_rejected(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load("evidence/../user_identities.csv")

    def test_a_sibling_of_evidence_is_rejected(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load("user_identities.csv")

    def test_an_absolute_path_is_rejected(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load(str(self.dir / "outside.txt"))

    def test_the_evidence_directory_itself_is_rejected(self) -> None:
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load("evidence")

    def test_a_symlink_leading_out_of_evidence_is_rejected(self) -> None:
        link = self.dataset / "evidence" / "records" / "link.txt"
        try:
            os.symlink(str(self.dir / "outside.txt"), str(link))
        except (OSError, NotImplementedError):
            self.skipTest("symlinks are not available here")
        with self.assertRaises(evidence.UnsafeEvidenceError):
            self.load("evidence/records/link.txt")


if __name__ == "__main__":
    unittest.main()
