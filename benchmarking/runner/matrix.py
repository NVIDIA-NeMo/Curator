from __future__ import annotations

import re
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .datasets import DatasetResolver
from .sinks.sink import Sink


@dataclass
class MatrixEntry:
    name: str
    script: str | None = None
    args: str | None = None
    script_base_dir: Path = Path(__file__).parent.parent / "scripts"
    timeout_s: int | None = None
    ray: dict[str, Any] = field(default_factory=dict)  # supports only single node: num_cpus,num_gpus,object_store_gb
    # If set, overrides the session-level delete_scratch setting for this entry
    delete_scratch: bool | None = None

    def get_command_to_run(self, session_entry_path: Path, resolver: DatasetResolver) -> str:
        if self.script:
            script_path = self.script_base_dir / self.script
            # FIXME: should --benchmark-results-path always be passed?
            cmd = f"python {script_path} {self.args or ''} --benchmark-results-path" + " {session_entry_dir}/benchmark_results"

            cmd = self.substitute_datasets_in_cmd(cmd, resolver)
            cmd = self.substitute_template_placeholders(cmd, session_entry_path)
        else:
            msg = f"Entry {self.name} must specify either cmd or script"
            raise ValueError(msg)

        return cmd

    @staticmethod
    def substitute_datasets_in_cmd(cmd: str, resolver: DatasetResolver) -> str:
        pattern = re.compile(r"\{dataset:([^,}]+),([^}]+)\}")

        def _replace(match: re.Match[str]) -> str:
            dataset_name = match.group(1).strip()
            dataset_format = match.group(2).strip()
            return resolver.resolve(dataset_name, dataset_format)

        return pattern.sub(_replace, cmd)

    @staticmethod
    def substitute_template_placeholders(cmd: str, session_entry_path: Path) -> str:
        """Substitute template placeholders in command.

        Supports {session_entry_dir}/dir patterns where anything after {session_entry_dir}/ becomes
        a directory under the generated session entry directory.

        Examples:
        - {session_entry_dir}/results.json -> /path/to/session/entry/results.json
        - {session_entry_dir}/tempdir/output -> /path/to/session/entry/tempdir/output
        - {session_entry_dir}/logs -> /path/to/session/entry/logs
        """
        session_entry_pattern = re.compile(r"\{session_entry_dir\}/([^}\s]+)")

        def replace_session_entry_path(match: re.Match[str]) -> str:
            subpath = match.group(1)
            return str(session_entry_path / subpath)

        return session_entry_pattern.sub(replace_session_entry_path, cmd)


@dataclass(frozen=True, kw_only=True)
class MatrixConfig:
    results_dir: str
    entries: list[MatrixEntry]
    sinks: list[Sink] = field(default_factory=list)
    default_timeout_s: int = 7200
    mlflow: dict[str, Any] = field(default_factory=dict)
    wandb: dict[str, Any] = field(default_factory=dict)
    slack: dict[str, Any] = field(default_factory=dict)
    # Whether to delete the entry's scratch directory after completion by default
    delete_scratch: bool = True

    @classmethod
    def create_from_dict(cls, data: dict) -> MatrixConfig:
        """
        Factory method to create a MatrixConfig from a dictionary.

        The dictionary is typically created from reading one or more YAML files.
        This method resolves environment variables and converts the list of
        entry dicts to MatrixEntry objects, and returns a new MatrixConfig
        object.
        """
        mc_field_names = {f.name for f in fields(cls)}
        mc_data = {k: v for k, v in data.items() if k in mc_field_names}
        mc_data = _resolve_env_vars(mc_data)
        sinks = cls.load_sinks(mc_data["sinks"])
        mc_data["sinks"] = sinks
        entries = [MatrixEntry(**e) for e in mc_data["entries"]]
        mc_data["entries"] = entries
        return cls(**mc_data)

    @classmethod
    def load_sinks(cls, sink_configs: list[dict]) -> list[Sink]:
        """Load sinks from the list of sink configuration dictionaries."""
        sinks = []
        for sink_config in sink_configs:
            sink_name = sink_config["name"]
            if sink_name == "mlflow":
                from runner.sinks.mlflow_sink import MlflowSink
                sinks.append(MlflowSink(config=sink_config))
            elif sink_name == "slack":
                from runner.sinks.slack_sink import SlackSink
                sinks.append(SlackSink(config=sink_config))
            elif sink_name == "gdrive":
                from runner.sinks.gdrive_sink import GdriveSink
                sinks.append(GdriveSink(config=sink_config))
            else:
                logger.warning(f"Unknown sink: {sink_name}, skipping")
        return sinks

    def __post_init__(self) -> None:
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)):
            duplicates = set([name for name in names if names.count(name) > 1])
            raise ValueError(f"Duplicate entry name(s) found: {', '.join(duplicates)}")
        
        # Update delete_scratch for each entry that has not been set to the session-level delete_scratch setting
        for entry in self.entries:
            if entry.delete_scratch is None:
                entry.delete_scratch = self.delete_scratch

        # Update timeout_s for each entry that has not been set to the session-level default_timeout_s
        for entry in self.entries:
            if entry.timeout_s is None:
                entry.timeout_s = self.default_timeout_s


def _resolve_env_vars(data: dict | list | str) -> dict | list | str:
    """Recursively resolve environment variables in dictionary data.

    Supports ${VAR_NAME} syntax. If the environment variable is not found,
    the original string is left unchanged.
    """
    if isinstance(data, dict):
        return {key: _resolve_env_vars(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Pattern to match ${VAR_NAME}
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace_env_var(match: re.Match[str]) -> str:
            env_var_name = match.group(1)
            env_value = os.getenv(env_var_name)
            if env_value is not None:
                return env_value
            else:
                msg = f"Environment variable {env_var_name} not found in the environment"
                raise ValueError(msg)

        return pattern.sub(replace_env_var, data)
    else:
        return data
