"""Regression coverage for executive tool results, not canned posture replies."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deskd'))
import deskd as d
import robot_sim


class ToolLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        class Bot:
            id = 'b_tool_loop_regression'
            kind = 'teela-brain'
            model = 'qwen38-27b-q5'
            surface = 'preview'
            desktop_cursor = {}
            messages = []
            robot_state = robot_sim.default_state()
            def record_local_generation(self, *a, **kw):
                pass
        self.bot = Bot()
        self.bot.workspace = Path(self.tmp.name)
        (self.bot.workspace / 'proof.txt').write_text('ACTUAL_FILE_PROOF_7921')

    def test_fast_turn_reads_tool_result_before_answering(self):
        seen = []
        def complete(payload):
            seen.append(payload)
            if len(seen) == 1:
                return {'choices': [{'message': {'tool_calls': [{'id': 'read1', 'function': {
                    'name': 'read_file', 'arguments': '{"path":"proof.txt"}'}}]}}]}
            self.assertIn('ACTUAL_FILE_PROOF_7921', payload['messages'][-1]['content'])
            self.assertEqual(payload.get('tool_choice'), 'none')
            return {'choices': [{'message': {'content': 'The file contains ACTUAL_FILE_PROOF_7921.'}}]}
        line = d._teela_minios_continue(self.bot, d.assemble_teela_executive_payload(self.bot, 'Read proof.txt'),
                                       intent='Read proof.txt', acted=False, completer=complete, rounds=1)
        self.assertEqual(line, 'The file contains ACTUAL_FILE_PROOF_7921.')
        self.assertEqual(len(seen), 2)


    def test_final_answer_round_cannot_execute_more_tools(self):
        calls = []
        def complete(payload):
            return {'choices': [{'message': {'tool_calls': [{'id': 'again', 'function': {
                'name': 'read_file', 'arguments': '{"path":"proof.txt"}'}}]}}]}
        real = d.execute_teela_allowed_tool
        def execute(*args):
            calls.append(args[1])
            return real(*args)
        with patch.object(d, 'execute_teela_allowed_tool', side_effect=execute):
            line = d._teela_minios_continue(self.bot, d.assemble_teela_executive_payload(self.bot, 'Read proof.txt'),
                                           intent='Read proof.txt', acted=False, completer=complete, rounds=1)
        self.assertEqual(calls, ['read_file'])
        self.assertIn('limit', line.lower())
        self.assertNotIn('Standing straight', line)


if __name__ == '__main__':
    unittest.main()
