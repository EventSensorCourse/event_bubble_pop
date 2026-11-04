# USB Stream Formats: Measurements

`bubble_pop.py --format` and `CameraStream(port, fmt=...)` choose how events travel from the H7 Plus to the PC. All formats give your program the same `(N, 6)` array: `type, sec, msec, usec, x, y`.

| Format | What goes over USB | Decoded on |
|---|---|---|
| `processed` (`CameraStream` default) | Six 16-bit fields per event (12 bytes) | Board |
| `evt20` (game default) | One 32-bit sensor word per event, plus timing words | PC (OpenMV decoder) |
| `evt21` | One 64-bit word for up to 32 events on the same row | PC (OpenMV decoder) |
| `evt30` | Compressed 16-bit words that repeat only what changed | PC (OpenMV decoder, uses numba) |

AER is not offered: it has no timestamps, so time windows would not work.

## Setup

- OpenMV H7 Plus + GENX320, firmware 5.0.1, USB Full-Speed; MacBook (Apple Silicon), Python 3.12.
- Measured on 2026-11-04 with `benchmark_formats.py`, `evt_res` 4096.
- Two kinds of scene: a hand waving for the whole run, and the room as it was (its activity changed over time, about 19k–76k events/s when nothing moved on purpose). Formats were run **in turn**, several rounds, so each format saw similar conditions.
- **Delay** is how late events reach the PC compared with the fastest batch of the same run (sensor clock against PC clock).

## Results

### Waving hand, 2 board buffers

Measured by waving a hand in front of the camera for the whole run; three rounds of 6 s per format. Round 1 had less movement; rounds 2 and 3 were steady.

| Format | Pixel events/s (median, range) | USB KB/s | Bytes per event | Delay p50 / max (ms) |
|---|---:|---:|---:|---:|
| processed | 69,000 (64,400–75,100) | 810 (USB full) | 12.0 | 31 / 40 |
| **evt20** | **174,000** (88,000–176,100) | 720 (USB full) | **4.2** | **3 / 6** |
| evt21 | 87,400 (31,400–87,800) | 714 (USB full) | 8.4 | 4 / 6 |
| evt30 | 158,300 (118,600–161,300) | 741 (USB full) | 4.8 | 3 / 7 |

With a moving hand, EVT2.0 delivered about **2.5×** more events than Processed, and EVT3.0 about 2.3×. All formats filled USB; with 2 buffers the board drops what does not fit, so the delay stayed small and these rates are what arrived, not everything the sensor produced.

### Busy room without waving, 8 board buffers

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

- **Processed fills USB first.** At about 800–890 KB/s it carries only about 70–75k events/s. With 8 buffers the rest wait on the board (hundreds of milliseconds of delay); with 2 buffers they are dropped (31 ms median delay while waving).
- **EVT2.0 was the best format in every busy test.** About 2.5× more events than Processed with a waving hand, and about 1.7× in the busy room, with only a few milliseconds of delay at 2 buffers.
- **EVT3.0 depends on the scene.** It packs neighbouring events, so it was close to EVT2.0 with a waving hand (dense edges, 4.8 bytes per event) but poor in the room without waving (scattered single pixels, about 10 bytes per event).
- **EVT2.1 was never better than EVT2.0.** Even with a waving hand it needed about 8 bytes per event.
- **Raw formats also reach a ceiling.** They topped out at about 45 buffers per second (720–750 KB/s), which is about 175k events/s for EVT2.0.
- **Raw formats always send full buffers.** The board sends a fixed 16 KB raw buffer each time (4096 × 4 bytes), filled with events plus timing words. In a quiet scene this costs more bytes per event than Processed, but it does no harm.

## Recommendation

For live, interactive programs (games, gestures):

```sh
python bubble_pop.py --format evt20 --fifo-depth 2
```

EVT2.0 carried the most events in every busy test (about 174k events/s with a waving hand, 3 ms median delay), and 2 board buffers keep the delay low when USB is full. EVT3.0 is a close second for scenes with large moving objects. The game uses this setting by default. `CameraStream` keeps Processed with 8 buffers as its default (the Session 04 setup), so your own programs choose the faster settings explicitly.

For complete data (counting, measuring), keep 8 buffers, prefer `evt20`, record short clips, and check the delay and rate with the benchmark.

## Not measured yet

- A waving hand with 8 board buffers (how much delay builds up in each format).
- How many events the board drops at 2 buffers (the sensor's full event rate is unknown).
- Larger raw buffers (`--evt-res 8192`).

## Reproduce

Disconnect OpenMV IDE, keep the scene the same, and from the repository folder:

```sh
python benchmark_formats.py --seconds 6 --rounds 3 --scene "quiet room"
python benchmark_formats.py --seconds 6 --rounds 3 --scene "waving" --fifo-depth 2
```

Each run writes `report.md` and `results.json` to `recordings/formats-YYYYMMDD-HHMMSS/` (`--csv` also saves 3 s of events per format). Rounds alternate the formats because the scene changes over time.
