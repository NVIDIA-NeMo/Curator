import time
from typing import Any, Callable, Generic, TypeVar

from loguru import logger

from nemo_curator.tasks.image import SingleDataTask

T = TypeVar("T")


class StatsCollector:
    """Base class for stats collectors."""

    def update_tokens(self, tokens_generated: int) -> None:
        pass
    
    def update_task_start(self) -> None:
        pass
    
    def update_task_done(self) -> None:
        pass
    
    def step(self, force: bool = False) -> dict[str, Any]:
        """Step the stats collector and return the current stats."""
        return {}
    
    def __enter__(self) -> "StatsCollector":
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: Any | None) -> None:
        pass


class BaseStatsCollector(StatsCollector):
    """Base class for stats collectors."""
    tokens_generated: int = 0
    active_tasks: int = 0
    done_tasks: int = 0
    last_update: float = 0.0
    last_tokens_generated: int = 0

    def update_tokens(self, tokens_generated: int) -> None:
        self.tokens_generated += tokens_generated
    
    def update_task_start(self) -> None:
        self.active_tasks += 1
    
    def update_task_done(self) -> None:
        self.active_tasks -= 1
        self.done_tasks += 1
    
    def step(self, force: bool = False) -> dict[str, Any]:
        """Step the stats collector and return the current stats."""
        dt = time.time() - self.last_update
        update_tokens = self.tokens_generated - self.last_tokens_generated
        self.last_update = time.time()
        self.last_tokens_generated = self.tokens_generated
        return {
            "tokens_generated": self.tokens_generated,
            "active_tasks": self.active_tasks,
            "done_tasks": self.done_tasks,
            "tps": update_tokens / dt,
            "tps_per_task": update_tokens / dt / max(self.active_tasks, 1),
        }
    
    def __enter__(self) -> "StatsCollector":
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: Any | None) -> None:
        pass


class TqdmStatsCollector(BaseStatsCollector):
    """Stats collector that updates a progress bar."""
    last_done_tasks: int = 0

    def __init__(self, total: int, *, frequency: float = 5.0) -> None:
        super().__init__()
        self.frequency = frequency

        from tqdm.notebook import tqdm

        self.pbar = tqdm(total=total, desc="Tasks")

        self.last_update = time.time()

        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except ImportError:
            print("pynvml not installed, cannot display GPU utilization.")
            self.gpu_handle = None
        except Exception as e:
            print(f"Could not retrieve GPU utilization: {e}")
            self.gpu_handle = None
    
    def __enter__(self) -> "TqdmStatsCollector":
        self.pbar.__enter__()
        return self
    
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: Any | None) -> None:
        self.pbar.__exit__(exc_type, exc_value, tb)
    
    def step(self, force: bool = False) -> dict[str, Any]:
        """Step the stats collector and update the progress bar."""
        if time.time() - self.last_update < self.frequency and not force:
            return {}
        base_stats = super().step()
        final_stats = {
            "tokens_generated": f"{base_stats['tokens_generated']}tokens",
            "active_tasks": base_stats["active_tasks"],
            "tps": f"{base_stats['tps']:.2f}tokens/s",
            "tps_per_task": f"{base_stats['tps_per_task']:.2f}tokens/s/task",
        }
        if self.gpu_handle is not None:
            gpu_util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            final_stats["gpu"] = f"{gpu_util.gpu}%"
            final_stats["gpu_mem"] = f"{gpu_util.memory}%"
        update_tasks = self.done_tasks - self.last_done_tasks
        self.last_done_tasks = self.done_tasks
        self.pbar.update(update_tasks)
        self.pbar.set_postfix(final_stats)
        self.pbar.refresh()
        self.last_update = time.time()
        return base_stats


class TextResultProcessor(Generic[T]):
    """Processor for general text output."""

    last_text = ""
    last_tokens = 0

    def __init__(self, task: SingleDataTask[T], processor: Callable[[SingleDataTask[T], str], SingleDataTask[T]], *, stats_collector: StatsCollector | None = None) -> None:
        self.task = task
        self.processor = processor
        self.stats_collector = stats_collector
        if stats_collector is not None:
            stats_collector.update_task_start()

    def __call__(self, result: str | None, num_tokens: int | None) -> SingleDataTask[T] | None:
        try:
            if result is None:
                result = self.processor(self.task, self.last_text)
                if self.stats_collector is not None:
                    self.stats_collector.update_task_done()
                return result

            self.last_text = result
            if self.stats_collector is not None:
                self.stats_collector.update_tokens(num_tokens - self.last_tokens)
            self.last_tokens = num_tokens
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            self.task.data.error = str(e)
            self.task.data.is_valid = False
            if self.stats_collector is not None:
                self.stats_collector.update_task_done()
            return self.task
        return None
