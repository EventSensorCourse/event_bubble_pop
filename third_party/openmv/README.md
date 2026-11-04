# OpenMV camera script

`genx320_event_mode_streaming_on_cam.py` and `LICENSE` are unmodified copies from
[openmv/openmv-projects](https://github.com/openmv/openmv-projects/tree/fd10126a6acf9e1eeffb5bed8fd14c79c1943947/tools/genx320-event-streaming),
revision `fd10126a6acf9e1eeffb5bed8fd14c79c1943947`, MIT license.

The script runs on the OpenMV H7 Plus. `event_stream.py` uploads it over USB and
sets its buffer sizes (`CSI_FIFO_DEPTH`, `EVENT_FIFO_DEPTH`, event rows) at upload
time; this file is not edited.
