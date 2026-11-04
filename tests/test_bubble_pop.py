#!/usr/bin/env python3
"""Desktop checks for Event Bubble Pop (no board needed).

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window during tests
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bubble_pop as bp
import event_stream as es


def make_events(times_us, xs=None, ys=None, types=None):
    """Build processed rows (type, sec, msec, usec, x, y) from microsecond times."""
    t = np.asarray(times_us, dtype=np.int64)
    n = len(t)
    rows = np.zeros((n, 6), dtype=np.uint16)
    rows[:, 0] = 1 if types is None else types
    rows[:, 1] = t // 1_000_000
    rows[:, 2] = (t // 1_000) % 1_000
    rows[:, 3] = t % 1_000
    rows[:, 4] = 10 if xs is None else xs
    rows[:, 5] = 10 if ys is None else ys
    return rows


class HelperTests(unittest.TestCase):
    def test_decode_batch_little_endian(self):
        rows = make_events([1_234_567], xs=[300], ys=[2])
        decoded = es.decode_batch(rows.astype("<u2").tobytes())
        np.testing.assert_array_equal(decoded, rows)
        with self.assertRaises(ValueError):
            es.decode_batch(b"\x00" * 13)

    def test_full_timestamp_does_not_overflow(self):
        rows = make_events([70_123_456])
        self.assertEqual(es.timestamps_us(rows)[0], 70_123_456)

    def test_pixel_filter_drops_triggers_and_bad_coordinates(self):
        rows = make_events([0, 1, 2, 3], xs=[5, 5, 320, 5], types=[0, 1, 1, 2])
        self.assertEqual(len(es.pixel_events(rows)), 2)


class CameraScriptTests(unittest.TestCase):
    def test_patch_sets_buffers_and_event_rows(self):
        script = es.patch_camera_script(es.CAMERA_SCRIPT.read_text(), fifo_depth=2, evt_res=4096)
        self.assertIn("CSI_FIFO_DEPTH = 2", script)
        self.assertIn("EVENT_FIFO_DEPTH = 2", script)
        self.assertIn("np.zeros((4096, 6)", script)
        self.assertNotIn("np.zeros((EVT_RES, 6)", script)


class RawFormatTests(unittest.TestCase):
    """Decode hand-made sensor words with known answers."""
    T = 1_000_123   # microseconds

    def check(self, events, expected):
        np.testing.assert_array_equal(es.timestamps_us(events), [self.T] * len(expected))
        self.assertEqual([tuple(int(v) for v in (e[0], e[4], e[5])) for e in events], expected)

    def test_evt20(self):
        t = self.T
        words = np.array([(0x8 << 28) | ((t >> 6) & 0x0FFFFFFF),
                          (0x1 << 28) | ((t & 0x3F) << 22) | (100 << 11) | 200,
                          (0x0 << 28) | ((t & 0x3F) << 22) | (5 << 11) | 319], dtype="<u4")
        self.check(es.make_decoder("evt20")(words.tobytes()), [(1, 100, 200), (0, 5, 319)])

    def test_evt21_expands_row_mask(self):
        t = self.T
        th_hi = (0x8 << 28) | ((t >> 6) & 0x0FFFFFFF)
        cd_hi = (0x1 << 28) | ((t & 0x3F) << 22) | (64 << 11) | 10
        words = np.array([0, th_hi, 0b1011, cd_hi], dtype="<u4")       # (lo, hi) pairs
        self.check(es.make_decoder("evt21")(words.tobytes()), [(1, 64, 10), (1, 65, 10), (1, 67, 10)])

    def test_evt30_state_survives_batches(self):
        t = self.T
        decode = es.make_decoder("evt30")
        first = np.array([(0x8 << 12) | ((t >> 12) & 0xFFF), (0x6 << 12) | (t & 0xFFF),
                          (0x0 << 12) | 42], dtype="<u2")
        self.assertEqual(len(decode(first.tobytes())), 0)               # time and y only
        second = np.array([(0x2 << 12) | (1 << 11) | 7], dtype="<u2")  # x in the next batch
        self.check(decode(second.tobytes()), [(1, 7, 42)])

    def test_misaligned_raw_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            es.make_decoder("evt20")(b"\x00" * 6)

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            es.CameraStream("port", fmt="aer")


class RawScriptTests(unittest.TestCase):
    def test_patch_raw_script_sets_format_and_buffers(self):
        script = es.patch_camera_script(es.RAW_CAMERA_SCRIPT.read_text(), fifo_depth=2,
                                        evt_res=4096, sensor_format="EVT21")
        self.assertIn('EVT_FORMAT = "EVT21"', script)
        self.assertIn("EVT_RES = 4096", script)
        self.assertIn("CSI_FIFO_DEPTH = 2", script)


class TimeWindowTests(unittest.TestCase):
    def test_windows_span_batches(self):
        w = es.TimeWindows(10)
        self.assertEqual(w.add(make_events([0, 3_000])), [])
        self.assertEqual(w.add(make_events([6_000, 9_999])), [])     # same window
        done = w.add(make_events([10_000, 25_000]))
        self.assertEqual([(s, len(e)) for s, e in done], [(0, 4), (10_000, 1)])
        self.assertEqual([(s, len(e)) for s, e in w.flush()], [(20_000, 1)])
        self.assertEqual(w.flush(), [])

    def test_small_backward_step_joins_current_window(self):
        w = es.TimeWindows(10)
        w.add(make_events([10_005]))
        self.assertEqual(w.add(make_events([9_980])), [])            # 25 us back
        self.assertEqual(len(w.flush()[0][1]), 2)

    def test_large_backward_jump_restarts(self):
        w = es.TimeWindows(10)
        w.add(make_events([5_000_000]))
        done = w.add(make_events([100]))                             # board restarted
        self.assertEqual(done[0][0], 5_000_000)
        self.assertEqual(w.flush()[0][0], 0)


class ProcessingTests(unittest.TestCase):
    def test_denoise_keeps_edges_and_removes_isolated_pixels(self):
        edge = make_events(range(20), xs=[100] * 20, ys=range(100, 120))
        noise = make_events([0, 0], xs=[5, 300], ys=[5, 300])
        kept = bp.denoise(np.concatenate([edge, noise]))
        self.assertEqual(len(kept), 20)

    def test_events_inside_circle(self):
        rows = make_events([0, 0, 0], xs=[100, 110, 150], ys=[100, 100, 100])
        self.assertEqual(bp.events_inside(rows, 100, 100, 20), 2)

    def test_active_fraction(self):
        rows = make_events([0] * 4, xs=[1, 1, 2, 3], ys=[1, 1, 1, 1])
        self.assertAlmostEqual(bp.active_fraction(rows), 3 / (320 * 320))


class GameTests(unittest.TestCase):
    def setUp(self):
        import pygame
        pygame.init()
        self.game = bp.Game(pygame, 50, 40, mirror=False)
        self.game.filter_on = False
        bubble = self.game.bubbles[0]
        self.game.bubbles = [bubble]
        bubble.gold, bubble.r, bubble.x, bubble.y = False, 25, 160, 160
        self.bubble = bubble

    def burst(self, n, x, y):
        rng = np.random.default_rng(0)
        xs = np.clip(x + rng.integers(-5, 6, n), 0, 319)
        ys = np.clip(y + rng.integers(-5, 6, n), 0, 319)
        return make_events(np.zeros(n), xs, ys)

    def test_moving_hand_pops_bubble_and_scores(self):
        self.game.start_round()
        self.game.bubbles = [self.bubble]
        self.game.process_window(self.burst(60, int(self.bubble.cx), 160))
        self.assertEqual(self.game.score, 1)
        self.assertNotIn(self.bubble, self.game.bubbles)

    def test_few_events_do_not_pop(self):
        self.game.process_window(self.burst(10, int(self.bubble.cx), 160))
        self.assertIn(self.bubble, self.game.bubbles)

    def test_light_change_is_ignored(self):
        everywhere = make_events(np.zeros(320 * 320), np.tile(np.arange(320), 320),
                                 np.repeat(np.arange(320), 320))
        self.game.process_window(everywhere)
        self.assertIn(self.bubble, self.game.bubbles)
        self.assertGreater(self.game.flash_timer, 0)

    def test_bubble_reaching_top_ends_round(self):
        self.game.start_round()
        self.game.update(1 / 60)
        self.assertEqual(self.game.state, "playing")      # new round starts safely
        self.game.bubbles[0].y = self.game.bubbles[0].r - 1
        self.game.update(1 / 60)
        self.assertEqual(self.game.state, "over")

    def test_escape_outside_a_round_is_not_game_over(self):
        self.bubble.y = 0
        self.game.update(1 / 60)
        self.assertEqual(self.game.state, "ready")
        self.assertEqual(len(self.game.bubbles), bp.MAX_BUBBLES)

    def test_threshold_scales_with_window(self):
        self.game.windows.set_window(10)
        self.assertEqual(self.game.pop_events, 8)
        self.game.windows.set_window(100)
        self.assertEqual(self.game.pop_events, 80)

    def test_draw_runs(self):
        import pygame
        screen = pygame.display.set_mode((bp.SENSOR_SIZE * bp.SCALE + bp.PANEL_W, bp.SENSOR_SIZE * bp.SCALE))
        fonts = (pygame.font.Font(None, 44), pygame.font.Font(None, 30), pygame.font.Font(None, 22))
        self.game.update(1 / 60)
        self.game.draw(screen, fonts, {"source": "test", "rate": 0, "dropped": 0})


class ReplayTests(unittest.TestCase):
    def write_csv(self, header, rows):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        f.write(header + "\n")
        np.savetxt(f, rows, fmt="%d", delimiter=",")
        f.close()
        self.addCleanup(os.unlink, f.name)
        return Path(f.name)

    def test_load_csv_and_viewer_headers(self):
        rows = make_events([0, 1_000])
        for header in ("type,sec,msec,usec,x,y", "type,sec,ms,us,x,y"):
            np.testing.assert_array_equal(es.load_csv(self.write_csv(header, rows)), rows)
        with self.assertRaises(ValueError):
            es.load_csv(self.write_csv("a,b,c", rows))

    def test_replay_streams_all_rows_in_time_order(self):
        path = self.write_csv("type,sec,msec,usec,x,y", make_events([20_000, 0, 10_000]))
        with es.ReplayStream(path, loop=False) as stream:
            deadline = time.time() + 2
            while stream.running and time.time() < deadline:
                time.sleep(0.01)
        self.assertIsNone(stream.error)
        got = np.concatenate(stream.batches())
        np.testing.assert_array_equal(es.timestamps_us(got), [0, 10_000, 20_000])

    def test_shift_time_carries_into_seconds(self):
        shifted = es._shift_time(make_events([999_999]), 1)
        self.assertEqual(es.timestamps_us(shifted)[0], 1_000_000)


if __name__ == "__main__":
    unittest.main()
