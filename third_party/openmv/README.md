# OpenMV camera scripts and raw decoders

From [openmv/openmv-projects](https://github.com/openmv/openmv-projects/tree/fd10126a6acf9e1eeffb5bed8fd14c79c1943947/tools/genx320-event-streaming),
revision `fd10126a6acf9e1eeffb5bed8fd14c79c1943947`, MIT license (`LICENSE`).

| File | Source | Changes |
|---|---|---|
| `genx320_event_mode_streaming_on_cam.py` | Processed camera script | None |
| `genx320_raw_event_mode_streaming_on_cam.py` | Raw camera script (EVT2.0/2.1/3.0) | None |
| `raw_decoders.py` | Raw decoders from `genx320_event_mode_streaming_on_pc.py`, lines 514–839 | Header and imports added; functions unchanged |

The camera scripts run on the OpenMV H7 Plus. `event_stream.py` uploads one over USB
and sets its buffer sizes and raw format at upload time; the files here are not edited.
