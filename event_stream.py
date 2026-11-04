#!/usr/bin/env python3
"""Event sources and time windows for GENX320 programs.

You normally do NOT need to edit this file. It gives your project three tools:

    CameraStream(port)   live GENX320 events from an OpenMV H7 Plus over USB
    ReplayStream(csv)    replay a recorded CSV (type,sec,msec,usec,x,y), no board needed
    TimeWindows(ms)      group events into fixed sensor-time windows

Both streams run in a background thread and hand you NumPy batches with
shape (N, 6) and columns: type, sec, msec, usec, x, y.

CameraStream can use the Processed format (12 bytes per event, decoded on the
board) or a raw sensor format (EVT2.0, EVT2.1, EVT3.0), which is decoded on
the PC into the same (N, 6) layout.

The USB part uses the openmv Python package and OpenMV's official camera
scripts and raw decoders (third_party/openmv/, MIT license).
patch_camera_script() and wait_for_script_stopped() are adapted from OpenMV's
genx320_event_mode_streaming_on_pc.py (openmv/openmv-projects,
revision fd10126a6acf9e1eeffb5bed8fd14c79c1943947, MIT license).
"""
from pathlib import Path
import queue
import re
import sys
import threading
import time

import numpy as np

SENSOR_SIZE = 320
OPENMV_DIR = Path(__file__).resolve().parent / "third_party" / "openmv"
CAMERA_SCRIPT = OPENMV_DIR / "genx320_event_mode_streaming_on_cam.py"
RAW_CAMERA_SCRIPT = OPENMV_DIR / "genx320_raw_event_mode_streaming_on_cam.py"

# Stream formats: name -> (sensor format, bytes per word on USB)
FORMATS = {
    "processed": (None, 12),   # six uint16 fields per event, decoded on the board
    "evt20": ("EVT20", 4),     # one 32-bit word per event
    "evt21": ("EVT21", 8),     # one 64-bit word for up to 32 events on a row
    "evt30": ("EVT30", 2),     # compressed 16-bit words
}


# ---------------------------------------------------------------- helpers

def decode_batch(data):
    """Processed mode only: each event is six little-endian uint16 values."""
    if len(data) % 12:
        raise ValueError(f"Invalid processed batch: {len(data)} bytes (expected multiple of 12)")
    return np.frombuffer(data, dtype="<u2").reshape(-1, 6).copy()


def pixel_events(events):
    """Keep ON/OFF pixel events (types 0 and 1) with valid coordinates."""
    mask = (events[:, 0] <= 1) & (events[:, 4] < SENSOR_SIZE) & (events[:, 5] < SENSOR_SIZE)
    return events[mask]


def timestamps_us(events):
    """Full sensor time in microseconds. Cast first: uint16 would overflow."""
    fields = events[:, 1:4].astype(np.int64)
    return fields[:, 0] * 1_000_000 + fields[:, 1] * 1_000 + fields[:, 2]


def find_openmv_port():
    """Return the only connected OpenMV USB port, or None."""
    import serial.tools.list_ports
    ports = [p.device for p in serial.tools.list_ports.comports()
             if "openmv" in f"{p.description} {p.manufacturer} {p.product}".lower()]
    return ports[0] if len(ports) == 1 else None


# ---------------------------------------------------------------- OpenMV helpers
# Adapted from OpenMV's genx320_event_mode_streaming_on_pc.py (MIT license).

def patch_camera_script(script, fifo_depth, evt_res, sensor_format=None):
    """Set buffer sizes (and the raw format) in a camera script before uploading it."""
    script = re.sub(r"CSI_FIFO_DEPTH\s*=\s*\d+", f"CSI_FIFO_DEPTH = {fifo_depth}", script)
    if sensor_format:   # raw script
        script = re.sub(r"EVT_RES\s*=\s*\d+", f"EVT_RES = {evt_res}", script)
        script = re.sub(r"EVT_FORMAT\s*=\s*[\"']\w+[\"']", f'EVT_FORMAT = "{sensor_format}"', script)
    else:               # processed script
        script = re.sub(r"EVENT_FIFO_DEPTH\s*=\s*\d+", f"EVENT_FIFO_DEPTH = {fifo_depth}", script)
        script = re.sub(r"np\.zeros\(\((?:\d+|EVT_RES),\s*6\)", f"np.zeros(({evt_res}, 6)", script)
    if sys.platform != "win32":
        # Zero-copy channel reads on macOS/Linux, as upstream does.
        script = script.replace("def read(self, offset, size):", "def readp(self, offset, size):")
    return script


def wait_for_script_stopped(camera, timeout):
    """Wait until the previous board script has really stopped.

    camera.stop() only sends a request; starting a new script while the old
    one still holds the sensor can crash sensor initialisation.
    """
    stopped = threading.Event()
    original = camera._handle_event

    def handle(channel_id, event):
        original(channel_id, event)
        if camera.channels_by_id.get(channel_id, {}).get("name") == "stdin" and event == 0:
            stopped.set()

    camera._handle_event = handle
    try:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline and not stopped.is_set():
            try:
                camera.poll_events()
                while camera.read_stdout():
                    pass
            except Exception:
                pass
            time.sleep(0.02)
        return stopped.is_set()
    finally:
        camera._handle_event = original


# ---------------------------------------------------------------- streams

class _Stream:
    """Background thread that puts event batches into a bounded queue."""

    def __init__(self, max_batches=256):
        self.queue = queue.Queue(maxsize=max_batches)
        self.received_rows = 0
        self.dropped_rows = 0      # rows discarded because the program fell behind
        self.error = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_safely, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def running(self):
        return self._thread.is_alive()

    def batches(self):
        """Return every batch that arrived since the last call (may be [])."""
        out = []
        while True:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                return out

    def _put(self, events):
        self.received_rows += len(events)
        try:
            self.queue.put_nowait(events)
        except queue.Full:
            # The program fell behind: drop the OLDEST batch and count it.
            try:
                self.dropped_rows += len(self.queue.get_nowait())
            except queue.Empty:
                pass
            self.queue.put_nowait(events)

    def _run_safely(self):
        try:
            self._run()
        except BaseException as error:  # reported to the main thread
            self.error = error


def _load_raw_decoders():
    from third_party.openmv import raw_decoders   # imports numba (slow the first time)
    return raw_decoders


def make_decoder(fmt):
    """Return decode(data) -> (N, 6) uint16 for a stream format.

    Raw decoders keep state (the running timestamp) between batches, so use
    one decoder per connection.
    """
    sensor_format, word_bytes = FORMATS[fmt]
    if sensor_format is None:
        return decode_batch
    raw = _load_raw_decoders()
    if fmt == "evt30":
        decoder, state = raw.decode_raw_events_evt3, np.zeros(6, dtype=np.int64)
    elif fmt == "evt21":
        decoder, state = raw.decode_raw_events_evt21, [0]
    else:
        decoder, state = raw.decode_raw_events, [0]

    def decode(data):
        if len(data) % word_bytes:
            raise ValueError(f"Misaligned {fmt} batch: {len(data)} bytes")
        return decoder(data, state)
    return decode


class CameraStream(_Stream):
    """Live events from the H7 Plus (official OpenMV protocol).

    fmt: "processed" (default), "evt20", "evt21", or "evt30".
    fifo_depth: board-side buffers; fewer means less delay but more loss.
    """

    def __init__(self, port, evt_res=4096, fifo_depth=8, fmt="processed", verbose=True):
        super().__init__()
        if fmt not in FORMATS:
            raise ValueError(f"Unknown format {fmt!r}; choose from {', '.join(FORMATS)}")
        self.port = port
        self.evt_res = evt_res
        self.fifo_depth = fifo_depth
        self.fmt = fmt
        self.verbose = verbose
        self.received_bytes = 0    # bytes read from USB
        self.batches_read = 0
        self.misaligned = 0        # raw batches skipped because of a partial word

    def _run(self):
        from openmv.camera import Camera

        sensor_format = FORMATS[self.fmt][0]
        decode = make_decoder(self.fmt)
        channel = "raw_events" if sensor_format else "events"
        source = RAW_CAMERA_SCRIPT if sensor_format else CAMERA_SCRIPT
        script = patch_camera_script(source.read_text(), self.fifo_depth, self.evt_res, sensor_format)
        with Camera(
            self.port, baudrate=921600, crc=(sys.platform == "win32"),
            seq=True, ack=False, events=True, timeout=1.0,
            max_retry=3, max_payload=4096,
        ) as camera:
            try:
                camera.stop()
                wait_for_script_stopped(camera, 1.0)
                camera.exec(script)  # the PC uploads and starts the board script
                start = time.perf_counter()
                channel_seen = False
                while not self._stop.is_set():
                    status = camera.read_status() or {}
                    if status.get("stdout"):
                        text = camera.read_stdout()
                        if text and self.verbose:
                            print(text, end="" if text.endswith("\n") else "\n")
                        if text and "Traceback (most recent call last)" in text:
                            raise RuntimeError("Camera script failed; read the board traceback above")
                    channel_seen |= camera.has_channel(channel)
                    if not channel_seen and time.perf_counter() - start > 10:
                        raise RuntimeError(f"No {channel} channel after 10 s; check camera stdout and firmware")
                    if channel_seen and status.get(channel):
                        size = camera.channel_size(channel)
                        if size > 0:
                            data = camera.channel_read(channel, size)
                            if len(data) != size:
                                raise RuntimeError(f"Incomplete batch: {len(data)} of {size} bytes")
                            self.received_bytes += len(data)
                            self.batches_read += 1
                            try:
                                events = decode(data)
                            except ValueError:
                                self.misaligned += 1   # skip a partial raw batch
                                continue
                            self._put(events)
                            continue
                    time.sleep(0.001)
            finally:
                camera.stop()


def load_csv(path):
    """Read a CSV recording (type,sec,msec,usec,x,y; OpenMV viewer ms/us also OK)."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().replace(" ", "").lower()
    if header not in ("type,sec,msec,usec,x,y", "type,sec,ms,us,x,y"):
        raise ValueError(f"{path}: expected header type,sec,msec,usec,x,y, got {header!r}")
    rows = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, 6), dtype=np.uint16)
    if rows.shape[1] != 6:
        raise ValueError(f"{path}: expected 6 columns, got {rows.shape[1]}")
    return rows.astype(np.uint16)


class ReplayStream(_Stream):
    """Replay a recorded CSV at its recorded speed (loops by default)."""

    def __init__(self, path, loop=True, chunk_ms=5):
        super().__init__()
        self.events = load_csv(path)
        if not len(self.events):
            raise ValueError(f"{path}: no events to replay")
        self.loop = loop
        self.chunk_us = int(chunk_ms * 1000)

    def _run(self):
        t = timestamps_us(self.events)
        t = t - t.min()
        # Recordings are batch-ordered; sort by time for smooth playback.
        order = np.argsort(t, kind="stable")
        events, t = self.events[order], t[order]
        duration_us = int(t[-1]) + self.chunk_us
        offset_us = 0  # shifts sensor time forward on each loop
        while not self._stop.is_set():
            start = time.perf_counter()
            for chunk_start in range(0, duration_us, self.chunk_us):
                if self._stop.is_set():
                    return
                lo, hi = np.searchsorted(t, [chunk_start, chunk_start + self.chunk_us])
                wait = start + (chunk_start + self.chunk_us) / 1e6 - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                if hi > lo:
                    self._put(_shift_time(events[lo:hi], offset_us))
            if not self.loop:
                return
            offset_us += duration_us


def _shift_time(events, offset_us):
    """Copy of events with offset_us added to the sec/msec/usec fields."""
    if offset_us == 0:
        return events.copy()
    t = timestamps_us(events) + offset_us
    out = events.copy()
    out[:, 1] = (t // 1_000_000) % 65536
    out[:, 2] = (t // 1_000) % 1_000
    out[:, 3] = t % 1_000
    return out


# ---------------------------------------------------------------- windows

class TimeWindows:
    """Group pixel events into fixed windows of SENSOR time (not arrival time).

    A window can span several USB batches, so unfinished events are kept
    until a later event shows that the window has ended. Small backward
    timestamp steps (tens of microseconds) are joined to the current window.
    Windows without any events are skipped.
    """

    def __init__(self, window_ms):
        self.set_window(window_ms)

    def set_window(self, window_ms):
        self.window_ms = window_ms
        self.window_us = int(window_ms * 1000)
        self._current = None    # id of the window being filled
        self._pending = []      # event arrays belonging to that window

    def add(self, events):
        """Add a batch; return a list of finished windows as (start_us, events)."""
        if not len(events):
            return []
        ids = timestamps_us(events) // self.window_us
        finished = []
        if self._current is None:
            self._current = int(ids.min())
        elif ids.max() * self.window_us < (self._current * self.window_us) - 1_000_000:
            # Time jumped back more than 1 s (board restarted): start over.
            finished += self.flush()
            self._current = int(ids.min())
        ids = np.maximum(ids, self._current)  # late events join the current window
        for window_id in np.unique(ids):      # ascending order
            part = events[ids == window_id]
            if window_id != self._current:
                finished += self.flush()
                self._current = int(window_id)
            self._pending.append(part)
        return finished

    def flush(self):
        """Finish the current window now (e.g. when the scene goes quiet)."""
        if not self._pending:
            return []
        events = np.concatenate(self._pending)
        self._pending = []
        return [(self._current * self.window_us, events)]
