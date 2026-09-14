"""Transient pacing for the instance-based CAVE display protocol.

There is no verified request-correlated load acknowledgement in this protocol.
The settling delay is a working estimate, never evidence of remote completion.
"""

from collections import OrderedDict
import math
import time

CAVE_COLOR_SEND_INTERVAL_MS = 16
CAVE_COLOR_NAME_CHUNK = 128


def cave_load_settle_ms(count):
    """Allow cluster loading time: 45 seconds to 15 minutes, 1.2s/object."""
    return min(900_000, max(45_000, max(0, int(count)) * 1200))


class CaveColorQueue:
    """Coalesce unsent visuals by remote instance ID without dropping other loads."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._pending = OrderedDict()
        self._ready_at = 0.0
        self._loading_count = 0

    def __bool__(self):
        return bool(self._pending)

    def __len__(self):
        return len(self._pending)

    def clear(self):
        self._pending.clear()
        self._ready_at = 0.0
        self._loading_count = 0

    def begin_load(self, count):
        now = self._clock()
        if now >= self._ready_at:
            self._loading_count = 0
        self._loading_count += max(0, int(count))
        delay = cave_load_settle_ms(self._loading_count)
        self._ready_at = max(self._ready_at, now + delay / 1000.0)
        return delay

    def enqueue(self, groups):
        for visual, names in groups.items():
            for name in names:
                if name:
                    self._pending[str(name)] = tuple(visual)

    def delay_ms(self):
        return max(0, math.ceil((self._ready_at - self._clock()) * 1000))

    def pop_ready(self):
        if not self._pending or self.delay_ms():
            return None
        name, visual = self._pending.popitem(last=False)
        names = [name]
        # Limit each payload and each event-loop turn, even for unique colors.
        while self._pending and len(names) < CAVE_COLOR_NAME_CHUNK:
            next_name = next(iter(self._pending))
            if self._pending[next_name] != visual:
                break
            names.append(next_name)
            del self._pending[next_name]
        return (names, *visual)
