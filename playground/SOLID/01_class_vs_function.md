# Class vs Function — Playground

This playbook mirrors the doc and provides small runnable snippets to internalize the decision flow. See `01_class_vs_function.py` to run all examples.

## Flowchart (reference)

```mermaid
flowchart TD
  Q1["Is there evolving data across calls?<br/>(cache, counters, moving averages, handles)"]
  Q1 -- Yes --> C1["Class with state<br/>(keep cache/resource; small API)"]
  Q1 -- No --> Q2["Need polymorphism / plug-replace behavior?<br/>(samplers, schedulers, backends)"]
  Q2 -- Yes --> C2["Class hierarchy<br/>(ABC interface + subclasses)"]
  Q2 -- No --> Q3["Does it have a lifecycle?<br/>(open→use→close; heavy init then reuse)"]
  Q3 -- Yes --> C3["Class (context manager)<br/>(__enter__/__exit__)"]
  Q3 -- No --> Q4["Is configuration bound to behavior?<br/>(config + methods)"]
  Q4 -- Yes --> C4["Class with config<br/>(dataclass + behavior)"]
  Q4 -- No --> Q5["Is it a pure transformation?<br/>(input → output; no hidden state)"]
  Q5 -- Yes --> F1["Function<br/>(stateless, easily testable)"]
  Q5 -- No --> Q6["Is it a bag of unrelated helpers?"]
  Q6 -- Yes --> F2["Module of functions<br/>(group by theme; no classes)"]
  Q6 -- No --> Q7["Will you reuse expensive setup across calls?"]
  Q7 -- Yes --> C5["Class (cache/resource)<br/>(reuse init; small API)"]
  Q7 -- No --> Q8["Need thread/process safety for shared mutable state?"]
  Q8 -- Yes --> C6["Class (guard with locks)<br/>—or avoid sharing"]
  Q8 -- No --> F3["Default to Function<br/>(YAGNI)"]
```

## What's inside the Python file?

- Stateful cache: Class maintaining a moving average
- Strategy interface: ABC with two implementations
- Context manager: Simulated resource with open/use/close
- Dataclass + behavior: Config drives behavior
- Pure function: Stateless transform
- Module helpers: Group of small related functions
- Thread safety demo: Shared state guarded by locks

Run:

```bash
python playground/SOLID/01_class_vs_function.py
```
