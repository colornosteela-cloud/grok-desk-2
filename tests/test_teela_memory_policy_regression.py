"""Recall questions must inject memory even on a FAST turn."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deskd"))

from teela_cl.deliberation import choose_turn_policy
from teela_cl.skill_store import SkillStore


class MemoryPolicyRegressionTests(unittest.TestCase):
    def test_name_recall_sets_needs_memory_without_leaving_fast(self):
        policy = choose_turn_policy("what is my name", skills=SkillStore(seed=True).all())
        self.assertTrue(policy.needs_memory)
        self.assertEqual(policy.reasoning_mode, "fast")
        self.assertFalse(policy.needs_learning)

    def test_greeting_does_not_search_memory(self):
        policy = choose_turn_policy("Hi Teela.", skills=SkillStore(seed=True).all())
        self.assertFalse(policy.needs_memory)
        self.assertEqual(policy.reasoning_mode, "fast")


if __name__ == "__main__":
    unittest.main()
