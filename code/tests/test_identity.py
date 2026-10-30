"""Identity resolution: current and historical emails resolve; nothing else does."""
import unittest

import common
from identity import resolve_identity


@common.requires_dataset
class ResolveIdentityTests(unittest.TestCase):
    def test_historical_email_resolves_to_its_user(self) -> None:
        identity = resolve_identity("minjun.kim90@gmail.com")
        self.assertEqual(identity.user_ids, ("user_12",))
        # The identifier set covers the current email and the user reference as well.
        self.assertIn("minjun.kim@gmail.com", identity.identifiers)
        self.assertIn("user_12", identity.identifiers)

    def test_current_email_resolves_to_the_same_user(self) -> None:
        self.assertEqual(resolve_identity("minjun.kim@gmail.com").user_ids, ("user_12",))

    def test_case_and_surrounding_spaces_do_not_matter(self) -> None:
        self.assertEqual(resolve_identity("  MinJun.Kim90@GMAIL.com ").user_ids, ("user_12",))

    def test_unknown_email_resolves_to_nothing(self) -> None:
        identity = resolve_identity("nobody.at.all@example.org")
        self.assertEqual(identity.user_ids, ())
        # Only the address itself: nothing is borrowed from anyone else.
        self.assertEqual(identity.identifiers, frozenset({"nobody.at.all@example.org"}))

    def test_matching_is_exact_so_a_similar_name_never_matches(self) -> None:
        near_misses = [
            "minjun.kim91@gmail.com",  # digit changed
            "minjun.kim9@gmail.com",  # prefix of the historical address
            "minjun.kim900@gmail.com",  # historical address as a prefix
            "minjun.kim90@gmail.co",  # truncated domain
            "minjun.kim90@gmail.com.au",  # extended domain
            "kim.minjun90@gmail.com",  # same words, other order
            "min-jun.kim@gmail.com",  # spelling of the name in user_identities.csv
            "Min-jun Kim",  # the name itself
        ]
        for email in near_misses:
            with self.subTest(email=email):
                self.assertEqual(resolve_identity(email).user_ids, ())

    def test_another_person_with_the_same_surname_is_a_different_user(self) -> None:
        self.assertEqual(resolve_identity("kim.hyunwoo@gmail.com").user_ids, ("user_194",))


if __name__ == "__main__":
    unittest.main()
