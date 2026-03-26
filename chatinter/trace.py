from dataclasses import dataclass, field
import time

from zhenxun.services.log import logger

SLOW_TRACE_THRESHOLD = 1.2


@dataclass
class StageTrace:
    name: str
    tags: dict[str, str] = field(default_factory=dict)
    _start: float = field(default_factory=time.perf_counter)
    _last: float = field(default_factory=time.perf_counter)
    _stages: list[tuple[str, float]] = field(default_factory=list)

    def stage(self, label: str) -> None:
        now = time.perf_counter()
        self._stages.append((label, now - self._last))
        self._last = now

    def finish(self) -> float:
        total = time.perf_counter() - self._start
        summary = " | ".join(
            f"{name}={cost * 1000:.1f}ms" for name, cost in self._stages
        )
        tag_text = " ".join(f"{k}={v}" for k, v in self.tags.items())
        msg = (
            f"[{self.name}] {summary} | total={total * 1000:.1f}ms {tag_text}"
        ).strip()
        if total >= SLOW_TRACE_THRESHOLD:
            logger.warning(msg)
        else:
            logger.info(msg)
        return total
