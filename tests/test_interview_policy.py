"""Unit tests for the deterministic interview policy + pure helpers.

Run:  python -m unittest tests.test_interview_policy -v
(no pytest in the venv; stdlib unittest keeps it zero-dependency).

These cover the reliability-critical rules the design doc puts in code rather
than in a prompt: budget, min interval, confidence floor, time-sensitivity,
dedup, quiet mode, pending-ask exclusivity, debrief ordering, ducking filter
rebasing, and the notes formatter's "silent when empty" contract.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GEMINI_API_KEY", "test-key")

from models.interview import Fact, Gap, InterviewLog, PlannerDecision, QAEntry  # noqa: E402
from services.clip_extractor import DUCK_GAIN, duck_audio_filter  # noqa: E402
from services.expert_interview import policy  # noqa: E402
from services.expert_interview.notes import format_interview_notes  # noqa: E402
from services.expert_interview.planner import is_wav, pcm16_rms, pcm16_to_wav  # noqa: E402
from services.expert_interview.store import parse_upload  # noqa: E402


def _gap(key="lug_nut_socket_size", prio="spec", ts=True, q="What size socket is that?"):
    return Gap(topic_key=key, priority=prio, question=q, time_sensitive=ts)


def _decision(propose=None, conf=0.9, gaps=(), filled=()):
    return PlannerDecision(
        new_gaps=list(gaps), propose_ask=propose, confidence=conf, filled=list(filled),
    )


def _budget(live_asks=0, last_close=None):
    return {"live_asks": live_asks, "last_ask_at": None, "last_window_closed_at": last_close}


def _decide(gaps, decision, budget, *, now=1000.0, phase="recording", quiet=False, pending=False):
    return policy.decide_live_ask(
        gaps, decision, budget, now=now, phase=phase, quiet_mode=quiet, pending_ask=pending,
    )


class NormalizeAndDedup(unittest.TestCase):
    def test_normalize_topic_key(self):
        self.assertEqual(policy.normalize_topic_key(" Lug-Nut Socket Size! "), "lug_nut_socket_size")
        self.assertEqual(policy.normalize_topic_key(""), "")

    def test_duplicate_question_by_tokens(self):
        self.assertTrue(policy.is_duplicate_question(
            "What size socket is that?", ["Which socket size are you using?"]))
        self.assertFalse(policy.is_duplicate_question(
            "What size socket is that?", ["How tight should the lug nuts be?"]))

    def test_merge_dedups_by_key_and_text(self):
        gaps = {}
        added = policy.merge_decision(gaps, _decision(gaps=[_gap()]), video_t=10.0)
        self.assertEqual(added, ["lug_nut_socket_size"])
        # Same key again → ignored. Different key, same question → ignored.
        added = policy.merge_decision(gaps, _decision(gaps=[
            _gap(), _gap(key="socket_size_lug", q="Which socket size is that?")]), video_t=12.0)
        self.assertEqual(added, [])
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps["lug_nut_socket_size"]["first_seen_t"], 10.0)

    def test_merge_marks_filled_from_narration(self):
        gaps = {}
        policy.merge_decision(gaps, _decision(gaps=[_gap()]), video_t=10.0)
        policy.merge_decision(gaps, _decision(filled=["lug_nut_socket_size"]), video_t=15.0)
        self.assertEqual(gaps["lug_nut_socket_size"]["status"], "answered")

    def test_safety_is_always_time_sensitive(self):
        gaps = {}
        policy.merge_decision(gaps, _decision(gaps=[_gap(key="jack_point", prio="safety", ts=False)]), video_t=1.0)
        self.assertTrue(gaps["jack_point"]["time_sensitive"])


class LiveAskGates(unittest.TestCase):
    def setUp(self):
        self.gaps = {}
        policy.merge_decision(self.gaps, _decision(gaps=[_gap()]), video_t=10.0)

    def test_happy_path_approves_and_mutates_ledger(self):
        budget = _budget()
        ask = _decide(self.gaps, _decision(propose="lug_nut_socket_size"), budget)
        self.assertIsNotNone(ask)
        self.assertEqual(ask.topic_key, "lug_nut_socket_size")
        self.assertIn("[ASK]", ask.inject_text)
        self.assertEqual(self.gaps["lug_nut_socket_size"]["status"], "asked")
        self.assertEqual(budget["live_asks"], 1)

    def test_confidence_floor(self):
        self.assertIsNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size", conf=0.5), _budget()))
        self.assertEqual(self.gaps["lug_nut_socket_size"]["status"], "open")  # gap kept for debrief

    def test_not_time_sensitive_is_deferred(self):
        gaps = {}
        policy.merge_decision(gaps, _decision(gaps=[_gap(key="why_this_brand", prio="tip", ts=False)]), video_t=1.0)
        self.assertIsNone(_decide(gaps, _decision(propose="why_this_brand"), _budget()))

    def test_budget_max(self):
        self.assertIsNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size"),
                                  _budget(live_asks=policy.DEFAULT_POLICY.max_live_asks)))

    def test_min_interval_from_window_close(self):
        cfg = policy.DEFAULT_POLICY
        budget = _budget(last_close=1000.0 - cfg.min_seconds_between_live_asks + 1)
        self.assertIsNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size"), budget))
        budget = _budget(last_close=1000.0 - cfg.min_seconds_between_live_asks)
        self.assertIsNotNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size"), budget))

    def test_quiet_pending_phase(self):
        d = _decision(propose="lug_nut_socket_size")
        self.assertIsNone(_decide(self.gaps, d, _budget(), quiet=True))
        self.assertIsNone(_decide(self.gaps, d, _budget(), pending=True))
        self.assertIsNone(_decide(self.gaps, d, _budget(), phase="preroll"))
        self.assertIsNone(_decide(self.gaps, d, _budget(), phase="debrief"))

    def test_never_reasks_live(self):
        budget = _budget()
        self.assertIsNotNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size"), budget))
        # Expert ignored it → unanswered; a later proposal must not re-ask live.
        policy.apply_answer(self.gaps, topic_key="lug_nut_socket_size", outcome="unanswered", transcript="")
        budget["last_window_closed_at"] = 0.0
        self.assertIsNone(_decide(self.gaps, _decision(propose="lug_nut_socket_size"), budget, now=5000.0))

    def test_unknown_or_unproposed_key(self):
        self.assertIsNone(_decide(self.gaps, _decision(propose="nope"), _budget()))
        self.assertIsNone(_decide(self.gaps, _decision(propose=None), _budget()))


class Debrief(unittest.TestCase):
    def test_ordering_and_eligibility(self):
        gaps = {}
        policy.merge_decision(gaps, _decision(gaps=[
            _gap(key="tip_a", prio="tip", ts=False, q="Any tip?"),
            _gap(key="spec_b", prio="spec", ts=False, q="What torque spec?"),
            _gap(key="safety_c", prio="safety", q="Where do you put the jack?"),
            _gap(key="answered_d", prio="spec", q="Which fluid grade?"),
            _gap(key="dropped_e", prio="spec", q="What bolt pattern?"),
        ]), video_t=1.0)
        policy.apply_answer(gaps, topic_key="answered_d", outcome="answered", transcript="5W-30")
        policy.apply_answer(gaps, topic_key="dropped_e", outcome="skipped", transcript="")
        asks = policy.pick_debrief_questions(gaps, max_questions=5)
        self.assertEqual([a.topic_key for a in asks], ["safety_c", "spec_b", "tip_a"])
        self.assertTrue(all(g["status"] == "asked" for g in (gaps["safety_c"], gaps["spec_b"], gaps["tip_a"])))

    def test_max_questions_and_one_debrief_per_topic(self):
        gaps = {}
        topics = ["torque spec", "jack placement", "wheel chock", "lug pattern",
                  "valve cap", "hubcap clip", "spare pressure", "breaker bar"]
        policy.merge_decision(gaps, _decision(gaps=[
            _gap(key=f"k{i}", q=f"What about the {t}?") for i, t in enumerate(topics)
        ]), video_t=1.0)
        self.assertEqual(len(gaps), 8)
        self.assertEqual(len(policy.pick_debrief_questions(gaps, 3)), 3)
        # Previously live-asked + unanswered gets one debrief shot; asked twice never again.
        gaps["k7"]["status"] = "unanswered"; gaps["k7"]["asked_count"] = 1
        gaps["k6"]["status"] = "unanswered"; gaps["k6"]["asked_count"] = 2
        keys = {a.topic_key for a in policy.pick_debrief_questions(gaps, 10)}
        self.assertIn("k7", keys)
        self.assertNotIn("k6", keys)

    def test_inject_text(self):
        self.assertIn("no open questions", policy.format_debrief_inject([]))
        self.assertIn("[BRIEFING]", policy.format_briefing_inject("changing a tyre"))


class Ducking(unittest.TestCase):
    def test_none_when_empty_or_outside(self):
        self.assertIsNone(duck_audio_filter(None))
        self.assertIsNone(duck_audio_filter([]))
        self.assertIsNone(duck_audio_filter([(50.0, 53.0)], offset=0.0, duration=10.0))
        self.assertIsNone(duck_audio_filter([(5.0, 5.0)]))

    def test_rebases_and_clamps_to_clip(self):
        f = duck_audio_filter([(8.0, 12.0), (41.0, 44.2)], offset=10.0, duration=20.0)
        # (8,12) → clamped to (0,2); (41,44.2) → outside (10..30) → dropped
        self.assertEqual(f, f"volume=enable='between(t,0.000,2.000)':volume={DUCK_GAIN}")

    def test_multiple_windows_are_ored(self):
        f = duck_audio_filter([(1.0, 2.0), (3.0, 4.0)])
        self.assertIn("between(t,1.000,2.000)+between(t,3.000,4.000)", f)


class NotesAndAudio(unittest.TestCase):
    def test_upload_parse_is_strict(self):
        # A session-start response (or any stray JSON object) must NOT pass as
        # an empty interview — that was a live bug: has_interview=true for nothing.
        with self.assertRaises(ValueError):
            parse_upload(b'{"session_id": "abc", "system_prompt": "x", "tools": []}')
        with self.assertRaises(ValueError):
            parse_upload(b'{}')  # version is required
        with self.assertRaises(ValueError):
            parse_upload(b'not json')
        self.assertEqual(parse_upload(b'{"version": 1}').version, 1)

    def test_notes_silent_when_nothing_to_say(self):
        self.assertEqual(format_interview_notes(None), "")
        self.assertEqual(format_interview_notes(InterviewLog(version=1)), "")
        # Unanswered Q/A only → still nothing worth telling the extractor.
        log = InterviewLog(version=1, qa=[QAEntry(question_id="x", topic_key="k", priority="spec",
                                                  question="Q?", outcome="unanswered", phase="recording")])
        self.assertEqual(format_interview_notes(log), "")

    def test_notes_prefers_facts_and_lists_windows(self):
        log = InterviewLog(
            version=1,
            facts=[Fact(kind="spec", value="19 mm socket", part="lug nut socket", t=42.1)],
            ai_speech_windows=[[41.0, 44.2]],
        )
        text = format_interview_notes(log)
        self.assertIn("EXPERT INTERVIEW NOTES", text)
        self.assertIn("[t=42.1] spec: 19 mm socket (part: lug nut socket)", text)
        self.assertIn("[41.0, 44.2]", text)

    def test_wav_wrap_and_rms(self):
        pcm = b"\x00\x00" * 1600
        self.assertEqual(pcm16_rms(pcm), 0.0)
        wav = pcm16_to_wav(pcm, 16_000)
        self.assertTrue(is_wav(wav))
        loud = (b"\x00\x40" * 1600)  # 0x4000 = 16384
        self.assertGreater(pcm16_rms(loud), 10_000)


if __name__ == "__main__":
    unittest.main()
