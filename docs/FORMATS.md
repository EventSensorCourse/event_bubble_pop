# USB Stream Formats: Measurements

`bubble_pop.py --format` and `CameraStream(port, fmt=...)` choose how events travel from the H7 Plus to the PC. All formats give your program the same `(N, 6)` array: `type, sec, msec, usec, x, y`.

| Format | What goes over USB | Decoded on |
|---|---|---|
| `processed` (default) | Six 16-bit fields per event (12 bytes) | Board |
| `evt20` | One 32-bit sensor word per event, plus timing words | PC (OpenMV decoder) |
| `evt21` | One 64-bit word for up to 32 events on the same row | PC (OpenMV decoder) |
| `evt30` | Compressed 16-bit words that repeat only what changed | PC (OpenMV decoder, uses numba) |

AER is not offered: it has no timestamps, so time windows would not work.

## Setup

- OpenMV H7 Plus + GENX320, firmware 5.0.1, USB Full-Speed; MacBook (Apple Silicon), Python 3.12.
- Measured on 2026-11-04 with `benchmark_formats.py`, `evt_res` 4096.
- The scene was the room as it was: no hand was waving, and its activity changed over time (about 19k–76k events/s when nothing moved on purpose). Formats were therefore run **in turn**, several rounds, so each format saw similar conditions.
- **Delay** is how late events reach the PC compared with the fastest batch of the same run (sensor clock against PC clock).

## Results

### Busy room, 8 board buffers (default)

Three rounds of 6 s per format. Rounds 1 and 2 were busy; round 3 was quieter.

| Format | Pixel events/s | USB KB/s | Bytes per event | Delay p50 (ms) |
|---|---:|---:|---:|---:|
| processed | 72,600–76,000 | 850–890 (USB full) | 12.0 | 280–300 |
| **evt20** | **123,800–127,800** | 650–660 | about 5 | **43–64** |
| evt21 | 50,300–59,500 | 680–710 (USB full) | about 14 | 65–73 |
| evt30 | 25,900–50,600 | 390–500 | about 10 | 2–3 |

A second check with this repository (one round, 5 s) gave the same order: processed 52,300 events/s with 154 ms median delay; evt20 86,100 events/s with 3 ms; evt21 28,100; evt30 20,700.

### Quiet room, 2 board buffers

Three rounds of 6 s per format; the room was producing about 19k events/s.

| Format | Pixel events/s | USB KB/s | Bytes per event | Delay p50 / max (ms) |
|---|---:|---:|---:|---:|
| processed | 18,600 | 218 | 12.0 | 3 / 35 |
| evt20 | 20,700 | 320 | 15.8 | 2 / 5 |
| evt21 | 18,900 | 624 | 33.7 | 3 / 6 |
| evt30 | 19,400 | 354 | 18.7 | 2 / 6 |

When USB is not full, every format delivers the same events with a few milliseconds of delay.

### Sanity checks

Short recordings from each format had no duplicate events, all coordinates were inside 0–319, and the ON/OFF ratio (0.56–0.62 ON) and the spatial spread matched across formats. No raw batch was misaligned.

## What the numbers mean

- **Processed fills USB first.** At about 850–890 KB/s it carries only about 75k events/s; the rest wait in the board's buffers, which shows up as hundreds of milliseconds of delay.
- **EVT2.0 was the best format in this room.** It delivered about 1.7× more events than Processed while using less USB bandwidth, with much less delay.
- **EVT2.1 and EVT3.0 compress rows of neighbouring events.** The events in this room were mostly scattered single pixels, so their compression did not help: they spent more bytes per event and delivered fewer events. With large, dense moving edges (a hand close to the camera) they may do better; this was not measured.
- **Raw formats always send full buffers.** The board sends a fixed 16 KB raw buffer each time (4096 × 4 bytes), filled with events plus timing words. In a quiet scene this costs more bytes per event than Processed, but it does no harm.

## Recommendation

For live, interactive programs (games, gestures):

```sh
python bubble_pop.py --format evt20 --fifo-depth 2
```

EVT2.0 carries the most events, and 2 board buffers keep the delay low when USB is still full. Processed stays the default because it is the simplest path and needs no PC-side decoder.

For complete data (counting, measuring), keep 8 buffers, prefer `evt20`, record short clips, and check the delay and rate with the benchmark.

## Not measured yet

- A controlled busy scene (a hand waving for the whole run) for each format.
- 2 board buffers under heavy motion; in a live test, Processed with 2 buffers kept up with fast waving.
- Larger raw buffers (`--evt-res 8192`).

## Reproduce

Disconnect OpenMV IDE, keep the scene the same, and from the repository folder:

```sh
python benchmark_formats.py --seconds 6 --rounds 3 --scene "quiet room"
python benchmark_formats.py --seconds 6 --rounds 3 --scene "waving" --fifo-depth 2
```

Each run writes `report.md` and `results.json` to `recordings/formats-YYYYMMDD-HHMMSS/` (`--csv` also saves 3 s of events per format). Rounds alternate the formats because the scene changes over time.
