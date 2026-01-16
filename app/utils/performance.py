"""
Performance Monitoring Utilities

Provides timing decorators and context managers for measuring
performance of various operations.
"""

import time
import logging
import functools
import inspect
from typing import Optional, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def timer(name: Optional[str] = None, log_level: str = "INFO"):
    """
    Decorator to time function execution.

    Usage:
        @timer("my_function")
        def my_function():
            pass

    Args:
        name: Custom name for the timer (defaults to function name)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    def decorator(func: Callable) -> Callable:
        timer_name = name or func.__name__
        log_func = getattr(logger, log_level.lower())

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                log_func(f"⏱️  {timer_name} took {elapsed*1000:.2f}ms")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                log_func(f"⏱️  {timer_name} took {elapsed*1000:.2f}ms")

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@contextmanager
def timer_context(name: str, log_level: str = "INFO"):
    """
    Context manager to time a code block.

    Usage:
        with timer_context("my_operation"):
            # code to time
            pass

    Args:
        name: Name for the timer
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    log_func = getattr(logger, log_level.lower())
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        log_func(f"⏱️  {name} took {elapsed*1000:.2f}ms")


class PerformanceTracker:
    """
    Track multiple timing measurements and report statistics.

    Usage:
        tracker = PerformanceTracker()
        tracker.start("operation_1")
        # ... do work ...
        tracker.stop("operation_1")
        tracker.report()
    """

    def __init__(self):
        """Initialize the performance tracker."""
        self._timers = {}
        self._measurements = {}

    def start(self, name: str) -> None:
        """
        Start timing an operation.

        Args:
            name: Name of the operation
        """
        self._timers[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """
        Stop timing an operation and record the duration.

        Args:
            name: Name of the operation

        Returns:
            Elapsed time in seconds
        """
        if name not in self._timers:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0

        elapsed = time.perf_counter() - self._timers[name]

        # Store measurement
        if name not in self._measurements:
            self._measurements[name] = []
        self._measurements[name].append(elapsed)

        # Clean up
        del self._timers[name]

        return elapsed

    def measure(self, name: str) -> float:
        """
        Get current elapsed time without stopping the timer.

        Args:
            name: Name of the operation

        Returns:
            Elapsed time in seconds
        """
        if name not in self._timers:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0

        return time.perf_counter() - self._timers[name]

    def report(self, log_level: str = "INFO") -> dict:
        """
        Generate and log a performance report.

        Args:
            log_level: Logging level for the report

        Returns:
            Dictionary with performance statistics
        """
        log_func = getattr(logger, log_level.lower())

        report = {}
        log_func("=" * 60)
        log_func("Performance Report")
        log_func("=" * 60)

        for name, measurements in sorted(self._measurements.items()):
            count = len(measurements)
            total = sum(measurements)
            avg = total / count if count > 0 else 0
            min_time = min(measurements) if measurements else 0
            max_time = max(measurements) if measurements else 0

            report[name] = {
                "count": count,
                "total_ms": total * 1000,
                "avg_ms": avg * 1000,
                "min_ms": min_time * 1000,
                "max_ms": max_time * 1000
            }

            log_func(
                f"{name}: "
                f"count={count}, "
                f"avg={avg*1000:.2f}ms, "
                f"min={min_time*1000:.2f}ms, "
                f"max={max_time*1000:.2f}ms, "
                f"total={total*1000:.2f}ms"
            )

        log_func("=" * 60)
        return report

    def clear(self) -> None:
        """Clear all measurements."""
        self._timers.clear()
        self._measurements.clear()


# Global performance tracker for the application
_global_tracker = PerformanceTracker()


def get_tracker() -> PerformanceTracker:
    """
    Get the global performance tracker.

    Returns:
        Global PerformanceTracker instance
    """
    return _global_tracker
