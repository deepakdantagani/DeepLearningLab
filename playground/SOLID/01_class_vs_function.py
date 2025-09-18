"""Playground: Class vs Function decision guide examples.

Run this file directly to execute all demos:

    python playground/SOLID/01_class_vs_function.py
"""

from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


# 1) Class with state: moving average cache across calls
class MovingAverage:
    """Maintain a simple moving average over the last N values."""

    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.values: List[float] = []
        self._sum: float = 0.0

    def add(self, value: float) -> float:
        if len(self.values) == self.window_size:
            oldest = self.values.pop(0)
            self._sum -= oldest
        self.values.append(value)
        self._sum += value
        return self.current()

    def current(self) -> float:
        if not self.values:
            return 0.0
        return self._sum / len(self.values)


# 2) Class hierarchy (Strategy pattern): plug-replace behavior
class Sampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, logits: List[float]) -> int:
        """Return an index given logits."""


class GreedySampler(Sampler):
    def sample(self, logits: List[float]) -> int:
        return max(range(len(logits)), key=lambda i: logits[i])


class TemperatureSampler(Sampler):
    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def sample(self, logits: List[float]) -> int:
        # Softmax-like reweighting by temperature, then greedy on scaled
        scaled = [x / self.temperature for x in logits]
        return max(range(len(scaled)), key=lambda i: scaled[i])


# 3) Lifecycle with context manager: open→use→close resource
class ExpensiveResource:
    """Simulates a resource with setup/teardown that we want to reuse."""

    def __init__(self, setup_delay_s: float = 0.2) -> None:
        self.setup_delay_s = setup_delay_s
        self._opened = False

    def __enter__(self) -> "ExpensiveResource":
        time.sleep(self.setup_delay_s)
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        self._opened = False
        return None

    def compute(self, x: int) -> int:
        if not self._opened:
            raise RuntimeError("Resource not opened; use as a context manager")
        return x * x


# 4) Class with config (dataclass + behavior)
@dataclass
class AugmentConfig:
    add: float = 0.0
    mul: float = 1.0


class Augmenter:
    def __init__(self, config: AugmentConfig) -> None:
        self.config = config

    def apply(self, values: Iterable[float]) -> List[float]:
        return [v * self.config.mul + self.config.add for v in values]


# 5) Pure function: stateless transformation


def normalize(values: Iterable[float]) -> List[float]:
    xs = list(values)
    if not xs:
        return []
    min_v, max_v = min(xs), max(xs)
    span = max_v - min_v
    if span == 0:
        return [0.0 for _ in xs]
    return [(x - min_v) / span for x in xs]


# 6) Module of helpers: related small functions grouped together


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mean(values: Iterable[float]) -> float:
    xs = list(values)
    return sum(xs) / len(xs) if xs else 0.0


# 7) Thread safety: shared state guarded by locks
class Counter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self._value += 1

    def value(self) -> int:
        with self._lock:
            return self._value


# Demos


def demo_moving_average() -> None:
    print("=== MovingAverage (stateful class) ===")
    ma = MovingAverage(window_size=3)
    for v in [10, 20, 30, 40]:
        avg = ma.add(v)
        print(f"add({v}) -> current avg: {avg:.2f}")
    print()


def demo_strategies() -> None:
    print("=== Strategy (Sampler interface) ===")
    logits = [0.1, 0.4, 0.3, 0.2]
    print("logits:", logits)
    greedy = GreedySampler()
    temp = TemperatureSampler(temperature=0.5)
    print("greedy -> index:", greedy.sample(logits))
    print("temp=0.5 -> index:", temp.sample(logits))
    print()


def demo_context_manager() -> None:
    print("=== Context Manager (resource lifecycle) ===")
    with ExpensiveResource() as r:
        vals = [r.compute(i) for i in range(5)]
    print("squared values:", vals)
    print()


def demo_dataclass_with_behavior() -> None:
    print("=== Dataclass + Behavior ===")
    config = AugmentConfig(add=2.0, mul=3.0)
    aug = Augmenter(config)
    print("input:", [1, 2, 3])
    print("augmented:", aug.apply([1, 2, 3]))
    print()


def demo_pure_function_and_helpers() -> None:
    print("=== Pure Function + Helpers ===")
    xs = [5, 10, 15]
    print("input:", xs)
    print("normalized:", normalize(xs))
    print("mean:", mean(xs))
    print("clamp(20, lo=0, hi=12):", clamp(20, 0, 12))
    print()


def demo_thread_safety() -> None:
    print("=== Thread Safety with Lock ===")
    counter = Counter()

    def worker(n: int) -> None:
        for _ in range(n):
            counter.increment()

    threads = [threading.Thread(target=worker, args=(10000,)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("expected:", 4 * 10000, "actual:", counter.value())
    print()


if __name__ == "__main__":
    demo_moving_average()
    demo_strategies()
    demo_context_manager()
    demo_dataclass_with_behavior()
    demo_pure_function_and_helpers()
    demo_thread_safety()
