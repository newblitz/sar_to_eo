"""Small dependency-free progress and ETA reporting helpers."""

from __future__ import annotations

import sys
import time


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class Progress:
    def __init__(self, total: int, label: str, report_every: int = 1):
        self.total = max(0, total)
        self.label = label
        self.report_every = max(1, report_every)
        self.current = 0
        self.started = time.monotonic()
        self.last_report = 0

    def update(self, current: int | None = None) -> None:
        self.current = self.current + 1 if current is None else current
        if (
            self.current != self.total
            and self.current - self.last_report < self.report_every
        ):
            return
        self.last_report = self.current
        elapsed = time.monotonic() - self.started
        rate = self.current / elapsed if self.current else 0
        remaining = (self.total - self.current) / rate if rate else None
        percent = 100 * self.current / self.total if self.total else 100
        message = (
            f"\r{self.label}: {self.current}/{self.total} ({percent:5.1f}%) "
            f"{rate:.2f}/s elapsed {format_duration(elapsed)} "
            f"ETA {format_duration(remaining)}"
        )
        print(message, end="", file=sys.stderr, flush=True)
        if self.current >= self.total:
            print(file=sys.stderr, flush=True)
