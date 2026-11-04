#!/usr/bin/env python3
"""EVENT BUBBLE POP: a live game for the GENX320 event camera.

Bubbles float up over the live event view. Wave your hand through a bubble
to pop it. If any bubble reaches the top, the game is over.
A still hand pops nothing: the sensor only reports CHANGES.

Pipeline (one example of a complete event-camera project):

    GENX320 -> H7 Plus -> USB batches -> TimeWindows -> denoise -> game logic -> screen

Run from this folder (see README.md for setup):

    python bubble_pop.py                          # find the board automatically
    python bubble_pop.py --port /dev/cu.usbmodem...
    python bubble_pop.py --replay my_recording.csv

Keys: SPACE start a round (until a bubble escapes)
      1/2/3 window 10/50/100 ms  F filter  M mirror
      +/- pop threshold  P pause  S screenshot  ESC/Q quit
"""
import argparse
from datetime import datetime
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

from event_stream import (FORMATS, SENSOR_SIZE, CameraStream, ReplayStream, TimeWindows,
                          find_openmv_port, pixel_events)

# ===================================================================
# TRY THIS: game and processing parameters
# ===================================================================
WINDOW_CHOICES_MS = {1: 10, 2: 50, 3: 100}   # keys 1/2/3
POP_EVENTS_PER_50MS = 200      # events inside a bubble needed to pop it (per 50 ms)
THRESHOLD_STEP = 20            # change per +/- key press
MIN_NEIGHBOURS = 1             # denoise: events needed in the 3x3 neighbourhood
FLASH_FRACTION = 0.20          # ignore windows where >20% of pixels fire (light change)
SPEED_UP_PER_MINUTE = 0.5      # bubbles get 50% faster every minute you survive
MAX_BUBBLES = 6
GOLD_CHANCE = 0.12             # small fast golden bubbles are worth 3 points

SCALE = 2                      # 320x320 sensor -> 640x640 on screen
PANEL_W = 280
FPS = 60


# ===================================================================
# Event processing: the part to replace with YOUR project's algorithm
# ===================================================================

def denoise(events, min_neighbours=MIN_NEIGHBOURS):
    """Remove isolated events: keep an event only if nearby pixels fired too.

    Real motion makes connected edges; sensor noise is scattered single pixels.
    """
    if not len(events):
        return events
    x = events[:, 4].astype(np.intp)
    y = events[:, 5].astype(np.intp)
    counts = np.bincount(y * SENSOR_SIZE + x, minlength=SENSOR_SIZE * SENSOR_SIZE)
    counts = counts.reshape(SENSOR_SIZE, SENSOR_SIZE)
    padded = np.pad(counts, 1)
    area = sum(padded[dy:dy + SENSOR_SIZE, dx:dx + SENSOR_SIZE]
               for dy in range(3) for dx in range(3))      # 3x3 sum around each pixel
    neighbours = area[y, x] - 1                              # do not count the event itself
    return events[neighbours >= min_neighbours]


def active_fraction(events):
    """Fraction of all sensor pixels that fired in this window."""
    if not len(events):
        return 0.0
    idx = events[:, 5].astype(np.intp) * SENSOR_SIZE + events[:, 4]
    return len(np.unique(idx)) / (SENSOR_SIZE * SENSOR_SIZE)


def events_inside(events, cx, cy, r):
    """Count events inside a circle (sensor coordinates)."""
    dx = events[:, 4].astype(np.float32) - cx
    dy = events[:, 5].astype(np.float32) - cy
    return int(np.count_nonzero(dx * dx + dy * dy <= r * r))


# ===================================================================
# Game objects
# ===================================================================

class Bubble:
    def __init__(self, first=False, y=None):
        self.gold = random.random() < GOLD_CHANCE
        self.r = random.uniform(12, 16) if self.gold else random.uniform(20, 32)
        self.x = random.uniform(self.r + 10, SENSOR_SIZE - self.r - 10)
        # At the start, spread bubbles over the screen; later spawn at the bottom.
        self.y = random.uniform(80, SENSOR_SIZE) if first else SENSOR_SIZE + self.r
        if y is not None:
            self.y = y
        self.speed = random.uniform(55, 75) if self.gold else random.uniform(20, 40)
        self.phase = random.uniform(0, 2 * math.pi)
        self.hits = 0          # events inside in the latest window (for the glow)

    @property
    def points(self):
        return 3 if self.gold else 1

    @property
    def cx(self):
        return self.x + 8 * math.sin(self.phase + self.y / 25)   # gentle wobble

    def update(self, dt, speed_up=1.0):
        self.y -= self.speed * speed_up * dt

    @property
    def escaped(self):
        return self.y <= self.r          # touched the top of the screen


class Pop:
    """Short expanding ring and particles where a bubble popped."""

    def __init__(self, x, y, r, gold):
        self.x, self.y, self.r, self.gold, self.age = x, y, r, gold, 0.0
        self.sparks = [(random.uniform(0, 2 * math.pi), random.uniform(40, 120)) for _ in range(12)]

    def update(self, dt):
        self.age += dt
        return self.age < 0.5


def make_pop_sound(pygame):
    """Synthesize a short 'pop' so no sound files are needed (optional)."""
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=1)
        freq, channels = pygame.mixer.get_init()[0], pygame.mixer.get_init()[2]
        t = np.linspace(0, 0.08, int(freq * 0.08), endpoint=False)
        wave = np.sin(2 * np.pi * (900 - 6000 * t) * t) * np.exp(-t * 45)
        samples = (wave * 12000).astype(np.int16)
        if channels > 1:
            samples = np.repeat(samples[:, None], channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(samples))
    except Exception:
        return None   # no audio device: play silently


# ===================================================================
# The game
# ===================================================================

class Game:
    def __init__(self, pygame, window_ms, threshold, mirror):
        self.pg = pygame
        self.windows = TimeWindows(window_ms)
        self.threshold_50ms = threshold
        self.mirror = mirror
        self.filter_on = True
        self.paused = False
        self.state = "ready"            # ready -> playing -> over
        self.score = self.best = 0
        self.survived = 0.0             # seconds in the current round
        self.bubbles = [Bubble(first=True) for _ in range(MAX_BUBBLES)]
        self.pops = []
        self.glow = np.zeros((SENSOR_SIZE, SENSOR_SIZE, 3), np.float32)  # indexed [x, y]
        self.flash_timer = 0.0
        self.last_window_events = 0
        self.sound = make_pop_sound(pygame)

    # -- thresholds scale with window length so difficulty stays similar
    @property
    def pop_events(self):
        return max(3, round(self.threshold_50ms * self.windows.window_ms / 50))

    def start_round(self):
        self.state, self.score, self.survived = "playing", 0, 0.0
        # Start every bubble below the screen, spaced out, so nobody loses at once.
        self.bubbles = [Bubble(y=SENSOR_SIZE + 40 + i * 50) for i in range(MAX_BUBBLES)]

    # -- called for every finished time window of events
    def process_window(self, events):
        if self.filter_on:
            events = denoise(events)
        self.last_window_events = len(events)
        self.draw_events(events)
        if self.paused:
            return
        if active_fraction(events) > FLASH_FRACTION:
            self.flash_timer = 1.0      # a light change, not a hand: do not pop
            return
        for bubble in list(self.bubbles):
            bubble.hits = events_inside(events, bubble.cx, bubble.y, bubble.r)
            if bubble.hits >= self.pop_events:
                self.pop(bubble)

    def pop(self, bubble):
        self.bubbles.remove(bubble)
        self.pops.append(Pop(bubble.cx, bubble.y, bubble.r, bubble.gold))
        if self.state == "playing":
            self.score += bubble.points
            self.best = max(self.best, self.score)
        if self.sound:
            self.sound.play()

    def draw_events(self, events):
        """Add ON events (cyan) and OFF events (magenta) to a fading glow image."""
        on = events[:, 0] == 1
        for mask, color in ((on, (40, 230, 255)), (~on, (255, 60, 200))):
            idx = events[mask, 4].astype(np.intp) * SENSOR_SIZE + events[mask, 5]
            hits = np.bincount(idx, minlength=SENSOR_SIZE * SENSOR_SIZE)
            fired = (hits > 0).reshape(SENSOR_SIZE, SENSOR_SIZE)[:, :, None]
            self.glow += fired * np.array(color, np.float32)

    def update(self, dt):
        self.glow *= 0.85 ** (dt * FPS)              # fade old events
        np.minimum(self.glow, 255, out=self.glow)
        self.flash_timer = max(0.0, self.flash_timer - dt)
        self.pops = [p for p in self.pops if p.update(dt)]
        if self.paused:
            return
        speed_up = 1 + SPEED_UP_PER_MINUTE * self.survived / 60 if self.state == "playing" else 1
        for b in self.bubbles:
            b.update(dt, speed_up)
        if self.state == "playing":
            self.survived += dt
            if any(b.escaped for b in self.bubbles):
                self.state = "over"     # a bubble reached the top: game over
        # Escaped bubbles leave; new ones rise from the bottom.
        self.bubbles = [b for b in self.bubbles if not b.escaped]
        while len(self.bubbles) < MAX_BUBBLES:
            self.bubbles.append(Bubble())

    # -- drawing
    def draw(self, screen, fonts, stats):
        pg = self.pg
        screen.fill((10, 12, 24))
        surface = pg.surfarray.make_surface(self.glow.astype(np.uint8))
        screen.blit(pg.transform.scale(surface, (SENSOR_SIZE * SCALE, SENSOR_SIZE * SCALE)), (0, 0))

        for b in self.bubbles:
            center = (int(b.cx * SCALE), int(b.y * SCALE))
            base = (255, 210, 60) if b.gold else (120, 200, 255)
            heat = min(1.0, b.hits / self.pop_events)   # brighter when almost popped
            color = tuple(int(c + (255 - c) * heat) for c in base)
            pg.draw.circle(screen, color, center, int(b.r * SCALE), 3)
            pg.draw.circle(screen, (255, 255, 255),
                           (center[0] - int(b.r * 0.8), center[1] - int(b.r * 0.8)), max(3, int(b.r / 4)))
        for p in self.pops:
            k = p.age / 0.5
            color = (255, 220, 80) if p.gold else (180, 230, 255)
            pg.draw.circle(screen, color, (int(p.x * SCALE), int(p.y * SCALE)),
                           int((p.r + 30 * k) * SCALE), max(1, int(4 * (1 - k))))
            for angle, speed in p.sparks:
                d = (p.r + speed * p.age) * SCALE
                pg.draw.circle(screen, color, (int(p.x * SCALE + d * math.cos(angle)),
                                               int(p.y * SCALE + d * math.sin(angle))), 3)

        big, mid, small = fonts
        w = SENSOR_SIZE * SCALE
        if self.state == "ready":
            self.banner(screen, big, small, "WAVE TO POP!", "press SPACE to start a round")
        elif self.state == "over":
            self.banner(screen, big, small, f"GAME OVER  score {self.score}",
                        "a bubble escaped! press SPACE to play again")
        if self.flash_timer > 0:
            text = mid.render("Light change! Not counted", True, (255, 120, 120))
            screen.blit(text, text.get_rect(center=(w // 2, w - 30)))
        if self.paused:
            text = big.render("PAUSED", True, (255, 255, 255))
            screen.blit(text, text.get_rect(center=(w // 2, w // 2)))

        # side panel
        pg.draw.rect(screen, (22, 26, 48), (w, 0, PANEL_W, w))
        x, y = w + 18, 16
        screen.blit(mid.render("EVENT BUBBLE POP", True, (120, 200, 255)), (x, y))
        y += 44
        for label, value in (("Score", self.score), ("Best", self.best),
                             ("Time", f"{self.survived:4.1f} s")):
            screen.blit(big.render(f"{label} {value}", True, (255, 255, 255)), (x, y))
            y += 46
        y += 8
        lines = [
            f"Source   {stats['source']}",
            f"Rate     {stats['rate']:,.0f} ev/s",
            f"Window   {self.windows.window_ms} ms   [1/2/3]",
            f"In window {self.last_window_events:,} events",
            f"Pop at   {self.pop_events} events  [+/-]",
            f"Filter   {'ON' if self.filter_on else 'off'}   [F]",
            f"Mirror   {'ON' if self.mirror else 'off'}   [M]",
            f"Dropped  {stats['dropped']:,} rows",
        ]
        for line in lines:
            screen.blit(small.render(line, True, (200, 210, 230)), (x, y))
            y += 24
        y = w - 4 * 22 - 12
        for line in ("SPACE start   P pause", "S screenshot", "ESC / Q quit",
                     "Still hand = no events!"):
            screen.blit(small.render(line, True, (130, 140, 170)), (x, y))
            y += 22

    def banner(self, screen, big, small, title, subtitle):
        w = SENSOR_SIZE * SCALE
        box = self.pg.Surface((w - 120, 120), self.pg.SRCALPHA)
        box.fill((0, 0, 0, 150))
        screen.blit(box, (60, w // 2 - 60))
        t = big.render(title, True, (255, 230, 90))
        screen.blit(t, t.get_rect(center=(w // 2, w // 2 - 18)))
        s = small.render(subtitle, True, (230, 230, 230))
        screen.blit(s, s.get_rect(center=(w // 2, w // 2 + 30)))


# ===================================================================
# Main loop
# ===================================================================

def run(stream, args, source_name):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((SENSOR_SIZE * SCALE + PANEL_W, SENSOR_SIZE * SCALE))
    pygame.display.set_caption("Event Bubble Pop: GENX320")
    fonts = (pygame.font.Font(None, 44), pygame.font.Font(None, 30), pygame.font.Font(None, 22))
    game = Game(pygame, args.window_ms, args.threshold, mirror=not args.no_mirror)
    with stream:   # start receiving only after the window is ready
        try:
            game_loop(pygame, screen, fonts, game, stream, args, source_name)
        finally:
            pygame.quit()


def game_loop(pygame, screen, fonts, game, stream, args, source_name):
    clock = pygame.time.Clock()
    rate, rate_rows, rate_start = 0.0, 0, time.perf_counter()
    last_data = time.perf_counter()
    end_time = time.perf_counter() + args.seconds if args.seconds else None

    while True:
        dt = clock.tick(FPS) / 1000
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type != pygame.KEYDOWN:
                continue
            key = event.unicode.lower()
            if event.key == pygame.K_ESCAPE or key == "q":
                return
            if event.key == pygame.K_SPACE and game.state != "playing":
                game.start_round()
            elif key in ("1", "2", "3"):
                game.windows.set_window(WINDOW_CHOICES_MS[int(key)])
            elif key == "f":
                game.filter_on = not game.filter_on
            elif key == "m":
                game.mirror = not game.mirror
            elif key in ("+", "="):
                game.threshold_50ms += THRESHOLD_STEP
            elif key == "-":
                game.threshold_50ms = max(THRESHOLD_STEP, game.threshold_50ms - THRESHOLD_STEP)
            elif key == "p":
                game.paused = not game.paused
            elif key == "s":
                Path("recordings").mkdir(exist_ok=True)
                name = Path("recordings") / f"bubble_pop_{datetime.now():%Y%m%d_%H%M%S}.png"
                pygame.image.save(screen, str(name))
                print(f"Saved {name}")

        if stream.error:
            raise RuntimeError(f"Event stream stopped: {stream.error}")

        # 1) take all USB batches, 2) group into time windows, 3) process each window
        for batch in stream.batches():
            last_data = time.perf_counter()
            rate_rows += len(batch)
            events = pixel_events(batch)
            if game.mirror:   # selfie view: your right hand appears on the right
                events = events.copy()
                events[:, 4] = SENSOR_SIZE - 1 - events[:, 4]
            for _start_us, window in game.windows.add(events):
                game.process_window(window)
        # A quiet scene sends no events, so finish a waiting window ourselves.
        if time.perf_counter() - last_data > 2 * game.windows.window_ms / 1000:
            for _start_us, window in game.windows.flush():
                game.process_window(window)

        now = time.perf_counter()
        if now - rate_start >= 1:
            rate, rate_rows, rate_start = rate_rows / (now - rate_start), 0, now
        game.update(dt)
        game.draw(screen, fonts, {"source": source_name, "rate": rate, "dropped": stream.dropped_rows})
        pygame.display.flip()
        if end_time and now >= end_time:
            return


def main():
    parser = argparse.ArgumentParser(description="Event Bubble Pop for the GENX320 event camera.")
    parser.add_argument("--port", help="OpenMV USB port (default: find it automatically)")
    parser.add_argument("--replay", type=Path, help="play a recorded CSV instead of the live camera")
    parser.add_argument("--window-ms", type=int, default=50, choices=sorted(WINDOW_CHOICES_MS.values()))
    parser.add_argument("--threshold", type=int, default=POP_EVENTS_PER_50MS,
                        help="events inside a bubble per 50 ms needed to pop it")
    parser.add_argument("--no-mirror", action="store_true", help="show the raw sensor orientation")
    parser.add_argument("--format", default="evt20", choices=list(FORMATS),
                        help="USB stream format (default evt20: most events in our tests; "
                             "processed = Session 04 format)")
    parser.add_argument("--fifo-depth", type=int, default=2, choices=range(2, 9), metavar="2-8",
                        help="board buffers (default 2: low delay; 8 = fewer events dropped, more delay)")
    parser.add_argument("--seconds", type=float, default=0, help="quit automatically after N seconds")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error("Use Python 3.12 or newer")

    if args.replay:
        stream, source = ReplayStream(args.replay), "replay"
    else:
        port = args.port or find_openmv_port()
        if not port:
            parser.error("No OpenMV board found. Plug it in, disconnect OpenMV IDE, "
                         "or pass --port (list ports: python -m serial.tools.list_ports -v)")
        stream = CameraStream(port, fifo_depth=args.fifo_depth, fmt=args.format)
        source = f"camera {args.format}"
        print(f"Connecting to {port}; the PC uploads the camera script...")

    try:
        run(stream, args, source)
    except KeyboardInterrupt:
        pass
    print(f"Received {stream.received_rows:,} rows; dropped {stream.dropped_rows:,} rows.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
