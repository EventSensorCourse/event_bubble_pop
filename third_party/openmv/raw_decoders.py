# Raw GENX320 event decoders (EVT2.0, EVT2.1, EVT3.0).
#
# The functions below are copied unchanged from OpenMV's
# genx320_event_mode_streaming_on_pc.py (lines 514-839) at
# https://github.com/openmv/openmv-projects/tree/fd10126a6acf9e1eeffb5bed8fd14c79c1943947/tools/genx320-event-streaming
# Copyright (c) 2013-2026 OpenMV LLC. All rights reserved. MIT license; see LICENSE in this folder.
# Only this header and the imports were added.

import numba
import numpy as np


def decode_raw_events(buf, time_high_ref):
    """Vectorized EVT 2.0 decoder.

    Decodes a buffer of raw 32-bit little-endian EVT 2.0 words into an (N, 6)
    uint16 array matching the ec_event_t layout from the processed cam script:
      [0] type   — EC_PIX_OFF_EVENT(0), EC_PIX_ON_EVENT(1),
                   EC_EXT_TRIGGER_FALLING(2), EC_EXT_TRIGGER_RISING(3),
                   EC_RST_TRIGGER_FALLING(4), EC_RST_TRIGGER_RISING(5)
      [1] ts_s   — whole seconds of timestamp
      [2] ts_ms  — milliseconds within the second (0-999)
      [3] ts_us  — microseconds within the millisecond (0-999)
      [4] x      — pixel column (0 for trigger events)
      [5] y      — pixel row    (0 for trigger events)

    Handles TD pixel events (0x0/0x1) and EXT_TRIGGER events (0xA).
    EV_TIME_HIGH (0x8) words update the running timestamp accumulator.

    time_high_ref is a one-element list carrying the running EV_TIME_HIGH
    value across consecutive calls (updated in-place).
    """
    n_words = len(buf) // 4
    if n_words == 0:
        return np.zeros((0, 6), dtype=np.uint16)

    words = np.frombuffer(buf, dtype=np.uint32)
    types = (words >> 28) & 0xF

    # --- EV_TIME_HIGH (0x8): update running timestamp accumulator ---
    th_mask = types == 0x8
    th_idx  = np.where(th_mask)[0]
    th_vals = ((words[th_mask] & 0x0FFFFFFF).astype(np.uint64)) << 6

    # --- all output events: TD pixel (0x0, 0x1) + EXT_TRIGGER (0xA) ---
    evt_mask = (types <= 0x1) | (types == 0xA)
    evt_idx  = np.where(evt_mask)[0]

    if evt_idx.size == 0:
        if th_vals.size > 0:
            time_high_ref[0] = int(th_vals[-1])
        return np.zeros((0, 6), dtype=np.uint16)

    # --- assign the correct time_high to every event (vectorized) ---
    # The applicable time_high for event at position i is the last EV_TIME_HIGH
    # whose stream position is strictly before i.
    if th_idx.size > 0:
        ins = np.searchsorted(th_idx, evt_idx, side='right') - 1
        th_for_evt = np.empty(evt_idx.size, dtype=np.uint64)
        before_first = ins < 0
        th_for_evt[before_first]  = time_high_ref[0]
        th_for_evt[~before_first] = th_vals[ins[~before_first]]
        time_high_ref[0] = int(th_vals[-1])
    else:
        th_for_evt = np.full(evt_idx.size, time_high_ref[0], dtype=np.uint64)

    # --- reconstruct full microsecond timestamp ---
    evt_words = words[evt_idx]
    evt_types = types[evt_idx]
    ts_low = ((evt_words >> 22) & 0x3F).astype(np.uint64)
    t_us   = th_for_evt | ts_low

    ts_s  = (t_us // 1_000_000).astype(np.uint16)
    ts_ms = ((t_us // 1_000) % 1_000).astype(np.uint16)
    ts_us = (t_us % 1_000).astype(np.uint16)

    # --- pixel events: type is 0 (off) or 1 (on), x/y from bit fields ---
    out_type = evt_types.astype(np.uint16)   # 0 or 1 for TD events
    x = ((evt_words >> 11) & 0x7FF).astype(np.uint16)
    y = (evt_words & 0x7FF).astype(np.uint16)

    # --- trigger events: remap type to EC_TRIGGER_EVENT(id, polarity) ---
    # EC_TRIGGER_EVENT = EC_EXT_TRIGGER_FALLING(2) + ((id & 1) << 1) + polarity
    # x and y are 0 for trigger events (no pixel coordinate).
    is_trig = evt_types == 0xA
    if is_trig.any():
        tw       = evt_words[is_trig]
        trig_id  = ((tw >> 8) & 0x1F).astype(np.uint16)
        trig_pol = (tw & 0x1).astype(np.uint16)
        out_type[is_trig] = 2 + ((trig_id & 1) << 1) + trig_pol
        x[is_trig] = 0
        y[is_trig] = 0

    return np.stack([out_type, ts_s, ts_ms, ts_us, x, y], axis=1)


def decode_raw_events_evt21(buf, time_high_ref):
    """Vectorized EVT 2.1 decoder.

    EVT 2.1 packs events into 64-bit little-endian word pairs. The high 32-bit
    word has the same bit layout as an EVT 2.0 word (type/ts/x/y), and the low
    32-bit word is a 32-bit `valid` bitmask. For CD events the x coordinate is
    aligned on 32 and bit n of the mask marks a valid event at (x+n, y), so a
    single pair expands to up to 32 pixel events.

    Returns the same (N, 6) uint16 layout as decode_raw_events:
      [type, ts_s, ts_ms, ts_us, x, y]

    time_high_ref is a one-element list carrying the running EV_TIME_HIGH value
    across consecutive calls (updated in-place).
    """
    words = np.frombuffer(buf, dtype=np.uint32)
    if words.shape[0] < 2:
        return np.zeros((0, 6), dtype=np.uint16)

    # Determine the lo/hi phase. An EVT2.1 event is a 64-bit pair stored
    # little-endian as (lo = low/valid word, hi = type/ts/x/y header). Normally a
    # frame starts on a pair boundary (hi = words[1::2]), but the capture can
    # begin one 32-bit word off, swapping lo/hi and turning the frame to garbage
    # — most visibly when idle, where the swap decodes TIME_HIGH pairs as CD
    # events sprayed across the canvas.
    #
    # A TIME_HIGH pair is a header of type 0x8 whose low word is exactly 0 — a
    # signature a CD valid-bitmask can't fake (masks are non-zero, and counting
    # 0x8 nibbles alone mis-votes because busy masks often have bit 31 set, i.e.
    # a 0x8.. top nibble). TIME_HIGH is emitted regularly even when idle, so the
    # phase that lines those headers up with zero low words is the correct one.
    n = words.shape[0]
    mA = n & ~1
    loA, hiA = words[0:mA:2], words[1:mA:2]
    sigA = int(np.count_nonzero(((hiA >> 28) == 0x8) & (loA == 0)))
    sigB = -1
    loB = hiB = None
    if n >= 3:
        wB = words[1:]
        mB = wB.shape[0] & ~1
        loB, hiB = wB[0:mB:2], wB[1:mB:2]
        sigB = int(np.count_nonzero(((hiB >> 28) == 0x8) & (loB == 0)))
    if sigB > sigA:
        lo, hi = loB, hiB           # frame started one 32-bit word late
    else:
        lo, hi = loA, hiA
    if hi.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.uint16)
    types = (hi >> 28) & 0xF

    # --- EV_TIME_HIGH (0x8): update running timestamp accumulator ---
    th_mask = types == 0x8
    th_idx  = np.where(th_mask)[0]
    th_vals = ((hi[th_mask] & 0x0FFFFFFF).astype(np.uint64)) << 6

    # --- output events: CD pixel (0x0, 0x1) + EXT_TRIGGER (0xA) ---
    evt_mask = (types <= 0x1) | (types == 0xA)
    evt_idx  = np.where(evt_mask)[0]

    if evt_idx.size == 0:
        if th_vals.size > 0:
            time_high_ref[0] = int(th_vals[-1])
        return np.zeros((0, 6), dtype=np.uint16)

    # --- assign the applicable time_high to every event (vectorized) ---
    if th_idx.size > 0:
        ins = np.searchsorted(th_idx, evt_idx, side='right') - 1
        th_for_evt = np.empty(evt_idx.size, dtype=np.uint64)
        before_first = ins < 0
        th_for_evt[before_first]  = time_high_ref[0]
        th_for_evt[~before_first] = th_vals[ins[~before_first]]
        time_high_ref[0] = int(th_vals[-1])
    else:
        th_for_evt = np.full(evt_idx.size, time_high_ref[0], dtype=np.uint64)

    evt_hi    = hi[evt_idx]
    evt_types = types[evt_idx]
    ts_low    = ((evt_hi >> 22) & 0x3F).astype(np.uint64)
    t_us      = th_for_evt | ts_low

    is_cd   = evt_types <= 0x1
    is_trig = evt_types == 0xA

    out_chunks = []

    # --- CD events: expand the valid bitmask into individual pixel events ---
    if is_cd.any():
        cd_hi   = evt_hi[is_cd]
        cd_pol  = evt_types[is_cd].astype(np.uint16)        # 0=off, 1=on
        cd_xbase = ((cd_hi >> 11) & 0x7FF).astype(np.uint32)
        cd_y     = (cd_hi & 0x7FF).astype(np.uint16)
        cd_t     = t_us[is_cd]
        cd_valid = lo[evt_idx][is_cd]                       # 32-bit mask per word

        # (M, 32) bool: bit k set -> event at (xbase+k, y)
        offsets = np.arange(32, dtype=np.uint32)
        bits = ((cd_valid[:, None] >> offsets) & 1).astype(bool)
        rows, cols = np.nonzero(bits)                       # row-major: parent order, then x

        if rows.size:
            ev_t  = cd_t[rows]
            cd_out = np.empty((rows.size, 6), dtype=np.uint16)
            cd_out[:, 0] = cd_pol[rows]
            cd_out[:, 1] = (ev_t // 1_000_000).astype(np.uint16)
            cd_out[:, 2] = ((ev_t // 1_000) % 1_000).astype(np.uint16)
            cd_out[:, 3] = (ev_t % 1_000).astype(np.uint16)
            cd_out[:, 4] = (cd_xbase[rows] + cols).astype(np.uint16)
            cd_out[:, 5] = cd_y[rows]
            out_chunks.append(cd_out)

    # --- trigger events: remap type to EC_TRIGGER_EVENT(id, polarity) ---
    # EC_TRIGGER_EVENT = EC_EXT_TRIGGER_FALLING(2) + ((id & 1) << 1) + polarity
    if is_trig.any():
        tw       = evt_hi[is_trig]
        trig_t   = t_us[is_trig]
        trig_id  = ((tw >> 8) & 0x1F).astype(np.uint16)
        trig_pol = (tw & 0x1).astype(np.uint16)
        trig_out = np.zeros((tw.size, 6), dtype=np.uint16)
        trig_out[:, 0] = 2 + ((trig_id & 1) << 1) + trig_pol
        trig_out[:, 1] = (trig_t // 1_000_000).astype(np.uint16)
        trig_out[:, 2] = ((trig_t // 1_000) % 1_000).astype(np.uint16)
        trig_out[:, 3] = (trig_t % 1_000).astype(np.uint16)
        out_chunks.append(trig_out)

    if not out_chunks:
        return np.zeros((0, 6), dtype=np.uint16)
    if len(out_chunks) == 1:
        return out_chunks[0]
    return np.concatenate(out_chunks, axis=0)


@numba.njit(nogil=True, cache=True)
def _decode_evt3_core(words, state, out):
    """Sequential EVT 3.0 state-machine decode (compiled, nogil).

    EVT 3.0 is a 16-bit compressed format: y, x, polarity, timestamp and type
    are only re-sent when they change, so the decoder must carry state. `state`
    is a 6-element int64 array preserved across calls:
        [0] th        — current time_high, already shifted (<<12)
        [1] tl        — current time_low (12 bits)
        [2] y         — current y coordinate
        [3] bx        — current vector base x
        [4] bp        — current vector base polarity
        [5] wrapbase  — accumulated 2^24 us offsets (time_high is only 24-bit)

    `out` is a preallocated (M, 6) int32 buffer; the function writes
    [type, ts_s, ts_ms, ts_us, x, y] rows and returns the number written.
    """
    th = state[0]; tl = state[1]; y = state[2]
    bx = state[3]; bp = state[4]; wrapbase = state[5]
    n = 0
    for i in range(words.shape[0]):
        w = words[i]
        t = (w >> 12) & 0xF
        p = w & 0xFFF
        if t == 0x0:            # EVT_ADDR_Y
            y = p & 0x7FF
        elif t == 0x2:          # EVT_ADDR_X — single valid event
            pol = (p >> 11) & 1
            x   = p & 0x7FF
            bx  = (x >> 5) << 5             # VECT_BASE_X.x = x & ~0x1F (align 32)
            bp  = pol
            t_us = wrapbase + th + tl
            out[n, 0] = pol
            out[n, 1] = t_us // 1_000_000
            out[n, 2] = (t_us // 1_000) % 1_000
            out[n, 3] = t_us % 1_000
            out[n, 4] = x
            out[n, 5] = y
            n += 1
        elif t == 0x3:          # VECT_BASE_X — sets base for following vectors
            bp = (p >> 11) & 1
            bx = p & 0x7FF
        elif t == 0x4:          # VECT_12 — 12 validity bits from bx
            t_us = wrapbase + th + tl
            for k in range(12):
                if (p >> k) & 1:
                    out[n, 0] = bp
                    out[n, 1] = t_us // 1_000_000
                    out[n, 2] = (t_us // 1_000) % 1_000
                    out[n, 3] = t_us % 1_000
                    out[n, 4] = bx + k
                    out[n, 5] = y
                    n += 1
            bx += 12
        elif t == 0x5:          # VECT_8 — 8 validity bits from bx
            valid = p & 0xFF
            t_us = wrapbase + th + tl
            for k in range(8):
                if (valid >> k) & 1:
                    out[n, 0] = bp
                    out[n, 1] = t_us // 1_000_000
                    out[n, 2] = (t_us // 1_000) % 1_000
                    out[n, 3] = t_us % 1_000
                    out[n, 4] = bx + k
                    out[n, 5] = y
                    n += 1
            bx += 8
        elif t == 0x6:          # EVT_TIME_LOW
            tl = p & 0xFFF
        elif t == 0x8:          # EVT_TIME_HIGH (24-bit timer; wraps ~every 16.7s)
            new_th = (p & 0xFFF) << 12
            if new_th < th:
                wrapbase += (1 << 24)
            th = new_th
        elif t == 0xA:          # EXT_TRIGGER
            tid = (p >> 8) & 0xF
            val = p & 1
            t_us = wrapbase + th + tl
            out[n, 0] = 2 + ((tid & 1) << 1) + val
            out[n, 1] = t_us // 1_000_000
            out[n, 2] = (t_us // 1_000) % 1_000
            out[n, 3] = t_us % 1_000
            out[n, 4] = 0
            out[n, 5] = 0
            n += 1
        # OTHERS (0xE) / CONTINUED (0x7, 0xF) carry no CD data — ignored.
    state[0] = th; state[1] = tl; state[2] = y
    state[3] = bx; state[4] = bp; state[5] = wrapbase
    return n


def decode_raw_events_evt3(buf, state):
    """EVT 3.0 decoder. Returns the (N, 6) uint16 layout of decode_raw_events.

    state is a 6-element int64 numpy array carrying the decoder's running state
    across consecutive calls (see _decode_evt3_core).
    """
    words = np.frombuffer(buf, dtype=np.uint16)
    if words.size == 0:
        return np.zeros((0, 6), dtype=np.uint16)
    # Exact upper bound on emitted events: ADDR_X / TRIGGER are 1 each, VECT_12
    # up to 12, VECT_8 up to 8. Cheap to compute and keeps the buffer tight.
    types = (words >> 12) & 0xF
    n_max = int(np.count_nonzero((types == 0x2) | (types == 0xA))
                + 12 * np.count_nonzero(types == 0x4)
                + 8 * np.count_nonzero(types == 0x5))
    out = np.empty((n_max, 6), dtype=np.int32)
    n = _decode_evt3_core(words, state, out)   # updates state even when n == 0
    if n == 0:
        return np.zeros((0, 6), dtype=np.uint16)
    return out[:n].astype(np.uint16)
