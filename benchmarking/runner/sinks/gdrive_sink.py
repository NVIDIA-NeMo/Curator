from typing import Any
import traceback

from loguru import logger

from runner.sinks.sink import Sink


class GdriveSink(Sink):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.enabled = self.config.get("enabled", True)
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.env_data: dict[str, Any] = None

    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        self.session_name = session_name
        self.env_data = env_data

    def process_result(self, result: dict[str, Any]) -> None:
        self.results.append(result)

    def finalize(self) -> None:
        if self.enabled:
            try:
                self._push(self.results)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"GdriveSink: Error posting to Google Drive: {e}\n{tb}")
        else:
            logger.warning("GdriveSink: Not enabled, skipping post.")

    def _push(self, results: list[dict[str, Any]]) -> None:
        pass
