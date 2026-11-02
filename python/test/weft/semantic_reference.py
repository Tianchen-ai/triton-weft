"""Independent scalar reference for source-state and snapshot observations."""
from dataclasses import dataclass, field
import hashlib
import numpy as np

@dataclass
class LogicalLevel:
    extent: int
    partition: int
    events: list = field(default_factory=list)

    def instances(self):
        if self.extent < 0 or self.partition <= 0:
            raise ValueError('invalid domain')
        for start in range(0, self.extent, self.partition):
            stop = min(start + self.partition, self.extent)
            self.events.append(('initialize', start, stop))
            yield slice(start, stop)
            self.events.append(('handoff', start, stop))

def narrow16(value):
    value = int(value) % 65536
    return value - 65536 if value >= 32768 else value

def composition_reference(source, initial, steps, rows, block, physical_width=None):
    memory = source.copy()
    result = np.zeros((len(initial), 3), np.int32)
    logical = LogicalLevel(len(initial), rows)
    for region in logical.instances():
        state = initial[region].astype(np.int64).copy()
        cumulative = np.zeros(len(state), np.int64)
        snapshot = memory[region].copy()
        visits = 0
        for iteration in range(steps):
            for column in LogicalLevel(source.shape[1], block).instances():
                delta = snapshot[:, column].sum(axis=1, dtype=np.int64)
                width = physical_width or len(state)
                for start in range(0, len(state), width):
                    for r in range(start, min(start + width, len(state))):
                        state[r] = narrow16(state[r] + delta[r])
                        cumulative[r] += state[r]
                memory[region, column] += 1 if iteration % 2 == 0 else -1
                visits += 1
        result[region, 0] = state
        result[region, 1] = cumulative
        result[region, 2] = visits
    return (memory, result, logical.events)
