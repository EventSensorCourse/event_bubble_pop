#!/usr/bin/env python3
"""Compare GENX320 stream formats over USB on a real H7 Plus.

For each format, this streams for a fixed time and reports events/s,
USB bytes, delay from the sensor to the PC, and sanity checks. Results go
to recordings/formats-YYYYMMDD-HHMMSS/ (report.md, results.json, and a short
CSV per format with --csv).

    python benchmark_formats.py --seconds 10 --rounds 3
    python benchmark_formats.py --formats processed evt20 --fifo-depth 2

Keep the scene the same for all formats. For a busy scene, keep a hand
waving in front of the camera during the whole run.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

from event_stream import FORMATS, CameraStream, find_openmv_port, pixel_events, timestamps_us


def measure(port, fmt, seconds, fifo_depth, evt_res, csv_path=None, csv_seconds=3):
    arrivals, rows, last_t = [], [], []
    pixels = zero_xy = backward = 0
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, "x", encoding="utf-8")
        csv_file.write("type,sec,msec,usec,x,y\n")
    prev_t = None
    with CameraStream(port, evt_res=evt_res, fifo_depth=fifo_depth, fmt=fmt, verbose=False) as s:
        # Wait for the first batch so startup time is not counted.
        deadline = time.perf_counter() + 15
        while not s.batches_read and time.perf_counter() < deadline and not s.error:
            time.sleep(0.01)
        if s.error:
            raise RuntimeError(f"{fmt}: {s.error}")
        s.batches()
        start_bytes, start_batches, start_mis = s.received_bytes, s.batches_read, s.misaligned
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            for batch in s.batches():
                now = time.perf_counter() - start
                px = pixel_events(batch)
                pixels += len(px)
                rows.append(len(batch))
                arrivals.append(now)
                if len(px):
                    t = timestamps_us(px)
                    last_t.append((now, int(t.max())))
                    zero_xy += int(np.count_nonzero((px[:, 4] == 0) & (px[:, 5] == 0)))
                    steps = np.diff(t)
                    backward += int(np.count_nonzero(steps < 0))
                    if prev_t is not None and t[0] < prev_t:
                        backward += 1
                    prev_t = int(t[-1])
                    if csv_file and now < csv_seconds:
                        np.savetxt(csv_file, px, fmt="%d", delimiter=",")
            if s.error:
                raise RuntimeError(f"{fmt}: {s.error}")
            time.sleep(0.002)
        elapsed = time.perf_counter() - start
        usb_bytes = s.received_bytes - start_bytes
        batches = s.batches_read - start_batches
        misaligned = s.misaligned - start_mis
        dropped = s.dropped_rows
    if csv_file:
        csv_file.close()
    delay = np.array([a * 1e6 - t for a, t in last_t]) / 1000 if last_t else np.zeros(1)
    delay -= delay.min()   # relative: 0 = the fastest batch in this run
    return {
        "format": fmt,
        "seconds": round(elapsed, 2),
        "pixel_events_per_s": round(pixels / elapsed),
        "usb_kb_per_s": round(usb_bytes / elapsed / 1024, 1),
        "bytes_per_event": round(usb_bytes / pixels, 2) if pixels else None,
        "batches_per_s": round(batches / elapsed, 1),
        "delay_ms_p50": round(float(np.percentile(delay, 50)), 1),
        "delay_ms_p90": round(float(np.percentile(delay, 90)), 1),
        "delay_ms_max": round(float(delay.max()), 1),
        "misaligned_batches": misaligned,
        "dropped_rows_pc": dropped,
        "events_at_0_0": zero_xy,
        "backward_time_steps": backward,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="OpenMV USB port (default: find it automatically)")
    parser.add_argument("--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    parser.add_argument("--seconds", type=float, default=10, help="seconds per format per round")
    parser.add_argument("--rounds", type=int, default=1,
                        help="repeat all formats in turn; the report shows medians (scenes drift)")
    parser.add_argument("--fifo-depth", type=int, default=8, choices=range(2, 9), metavar="2-8")
    parser.add_argument("--evt-res", type=int, default=4096, help="events per board buffer (power of two)")
    parser.add_argument("--scene", default="", help="label for the report, e.g. 'quiet' or 'waving'")
    parser.add_argument("--csv", action="store_true", help="save the first 3 s of pixel events per format")
    args = parser.parse_args()
    port = args.port or find_openmv_port()
    if not port:
        parser.error("No OpenMV board found; pass --port")

    out = Path("recordings") / f"formats-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    results = []
    for rnd in range(1, args.rounds + 1):
        for fmt in args.formats:
            print(f"round {rnd}, {fmt}: streaming {args.seconds:g} s ...", flush=True)
            csv_path = out / f"{fmt}.csv" if args.csv and rnd == 1 else None
            r = measure(port, fmt, args.seconds, args.fifo_depth, args.evt_res, csv_path)
            r["round"] = rnd
            print("   ", r, flush=True)
            results.append(r)
            time.sleep(1.0)   # let the board script stop before the next format

    info = {"date": datetime.now().isoformat(timespec="seconds"), "port": port,
            "scene": args.scene, "fifo_depth": args.fifo_depth, "evt_res": args.evt_res,
            "seconds": args.seconds, "rounds": args.rounds, "python": sys.version.split()[0]}
    (out / "results.json").write_text(json.dumps({"info": info, "results": results}, indent=2))
    lines = [f"# Format benchmark {info['date']}", "",
             f"Scene: {args.scene or '-'}; fifo depth {args.fifo_depth}; evt_res {args.evt_res}; "
             f"{args.seconds:g} s per format, {args.rounds} round(s); medians over rounds "
             "(events/s range in brackets).", "",
             "| Format | Pixel events/s | USB KB/s | Bytes/event | Batches/s | Delay p50 / p90 / max (ms) | Misaligned | (0,0) events |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fmt in args.formats:
        runs = [r for r in results if r["format"] == fmt]
        med = {k: float(np.median([r[k] for r in runs if r[k] is not None] or [0]))
               for k in runs[0] if isinstance(runs[0][k], (int, float)) and k != "round"}
        rates = [r["pixel_events_per_s"] for r in runs]
        rng = f" [{min(rates):,}–{max(rates):,}]" if len(runs) > 1 else ""
        lines.append(f"| {fmt} | {med['pixel_events_per_s']:,.0f}{rng} | {med['usb_kb_per_s']:.0f} | "
                     f"{med['bytes_per_event']:.1f} | {med['batches_per_s']:.0f} | "
                     f"{med['delay_ms_p50']:.0f} / {med['delay_ms_p90']:.0f} / {med['delay_ms_max']:.0f} | "
                     f"{sum(r['misaligned_batches'] for r in runs)} | {sum(r['events_at_0_0'] for r in runs)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
