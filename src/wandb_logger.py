"""
WandbLogger
-----------
Lightweight Weights & Biases wrapper for ICBHI AST-SAM training.

Mirrors the env-var-driven config style from DomainBed
(domainbed/lib/wandb.py): credentials and project metadata are read from
environment variables / a .env file, so the same training script runs in
local / Colab / DGX environments without code changes.

Configuration via environment variables (see .env.example):
    WANDB_PROJECT   — wandb project name        (placeholder: TODO)
    WANDB_ENTITY    — wandb team/user           (placeholder: TODO)
    WANDB_API_KEY   — auth token; if missing we fall back to offline mode
    WANDB_MODE      — online / offline / disabled  (default: online)
    WANDB_TAGS      — comma-separated extra tags
    WANDB_NOTES     — free-form notes attached to every run

If `wandb` is not installed or `WANDB_MODE=disabled`, every method becomes
a no-op so the training script can be run without wandb at all.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

try:
    import wandb
except ImportError:
    wandb = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class WandbLogger:
    def __init__(
        self,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
    ):
        self.run = None

        if wandb is None:
            return

        # Placeholder defaults — override via env / .env
        project = os.getenv("WANDB_PROJECT", "icbhi-ast-sam")    # TODO: confirm project name
        entity  = os.getenv("WANDB_ENTITY", None)                # TODO: set team/user
        mode    = os.getenv("WANDB_MODE", "online")

        if mode == "disabled":
            return

        api_key = os.getenv("WANDB_API_KEY")
        if not api_key and mode == "online":
            print("[wandb] No WANDB_API_KEY found — switching to offline mode.")
            mode = "offline"
        elif api_key and mode != "offline":
            try:
                wandb.login(key=api_key)
            except Exception as e:
                print(f"[wandb] Login failed ({e}) — switching to offline mode.")
                mode = "offline"

        all_tags = list(tags or [])
        if os.getenv("WANDB_TAGS"):
            all_tags.extend(t.strip() for t in os.getenv("WANDB_TAGS").split(","))

        # Under `wandb agent`, omit `name` so wandb generates a unique one;
        # we only override the auto-name when an explicit run_name was given.
        init_kwargs: Dict[str, Any] = dict(
            project=project,
            entity=entity,
            group=group,
            tags=all_tags or None,
            notes=os.getenv("WANDB_NOTES", None),
            config=config,
            mode=mode,
        )
        if run_name:
            init_kwargs["name"] = run_name

        try:
            self.run = wandb.init(**init_kwargs)
            print(f"[wandb] Initialized run: {self.run.name} (id={self.run.id}, mode={mode})")
        except Exception as e:
            print(f"[wandb] Failed to initialize ({e}) — continuing without wandb.")
            self.run = None

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.run is None:
            return
        if step is not None:
            self.run.log({"step": step, **metrics})
        else:
            self.run.log(metrics)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        if self.run is None or not os.path.exists(path):
            return
        try:
            artifact = wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)
        except Exception as e:
            print(f"[wandb] log_artifact({name}) failed: {e}")

    def finish(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self.run is None:
            return
        if summary:
            self.run.summary.update(summary)
        self.run.finish()
