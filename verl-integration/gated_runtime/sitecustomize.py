"""Runtime-only wiring for the tau2 binary-outcome gated GRPO experiment.

Python imports ``sitecustomize`` automatically when this directory is first on
``PYTHONPATH``.  The gated launcher sets that path only for the seed-123 gated
arm, so ordinary GRPO launches remain untouched.

Why this lives here instead of in the shell launcher:

* veRL's legacy ``main_ppo`` runs ``RayPPOTrainer.fit`` inside a Ray process;
  registering an estimator only in the shell's driver process is not enough.
* the dynamic-sampling mask must be applied immediately before that process
  computes advantages and updates the actor.

The patch is deliberately fail-closed.  If the expected veRL seam or the raw
binary outcome is missing, training raises rather than silently becoming an
ordinary GRPO run.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
from pathlib import Path
from typing import Any


def _enabled() -> bool:
    return os.environ.get("USE_DYNAMIC_SAMPLING") == "1"


def _prepend_integration_root() -> None:
    integration_root = str(Path(__file__).resolve().parent.parent)
    if integration_root not in sys.path:
        sys.path.insert(0, integration_root)


def _estimator_name(adv_estimator: Any) -> str:
    return str(getattr(adv_estimator, "value", adv_estimator))


def _ensure_binary_outcome(data: Any) -> None:
    """Require the raw tau2 success/failure label exported by Tau2AgentLoop.

    ``token_level_rewards`` may become shaped in future experiments, so this
    intentionally does not derive the gate label from it.  The current binary
    tau2-airline run exports ``outcome_binary`` from the unshaped evaluator
    reward instead.
    """

    if "uid" not in data.non_tensor_batch:
        raise RuntimeError("gated run refused: veRL batch has no uid grouping key")
    if "outcome_binary" not in data.non_tensor_batch:
        raise RuntimeError(
            "gated run refused: Tau2AgentLoop did not export raw outcome_binary; "
            "refusing to silently skip dynamic sampling"
        )


def _install_patch() -> None:
    _prepend_integration_root()

    from grpo_gated.dynamic_sampling import apply_dynamic_sampling
    from grpo_gated.grpo_gated import register_grpo_gated
    from verl.trainer.ppo import ray_trainer

    if getattr(ray_trainer, "_tau2_gated_patch", False):
        return

    if not hasattr(ray_trainer, "compute_advantage"):
        raise RuntimeError("gated run refused: installed veRL has no ray_trainer.compute_advantage")

    trainer_cls = getattr(ray_trainer, "RayPPOTrainer", None)
    if trainer_cls is None:
        raise RuntimeError("gated run refused: installed veRL has no RayPPOTrainer")
    try:
        fit_source = inspect.getsource(trainer_cls.fit)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "gated run refused: could not inspect the installed RayPPOTrainer.fit seam"
        ) from exc
    if "compute_advantage(" not in fit_source:
        raise RuntimeError(
            "gated run refused: installed veRL does not expose the expected "
            "RayPPOTrainer.fit compute_advantage seam"
        )

    register_grpo_gated()
    original_compute_advantage = ray_trainer.compute_advantage

    @functools.wraps(original_compute_advantage)
    def compute_advantage_with_dynamic_sampling(data: Any, adv_estimator: Any, *args: Any, **kwargs: Any):
        if _estimator_name(adv_estimator) == "grpo_gated":
            _ensure_binary_outcome(data)
            # Older veRL variants build this mask inside compute_advantage.
            # Dynamic sampling runs immediately before it, so materialize the
            # same mask first instead of relying on a version-specific order.
            if "response_mask" not in data.batch:
                make_response_mask = getattr(ray_trainer, "compute_response_mask", None)
                if make_response_mask is None:
                    raise RuntimeError(
                        "gated run refused: veRL lacks response_mask and "
                        "compute_response_mask"
                    )
                data.batch["response_mask"] = make_response_mask(data)
            data, stats = apply_dynamic_sampling(data)
            if stats.get("total_rows", 0) and stats.get("dropped_rows") == stats.get("total_rows"):
                raise RuntimeError(
                    "gated run refused: dynamic sampling masked the entire batch; "
                    "this would create an undefined masked loss"
                )
            print(
                "[gated-bootstrap] GATED_DYNAMIC_SAMPLING "
                f"live_frac={stats['live_frac']} "
                f"dropped_rows={stats['dropped_rows']}/{stats['total_rows']} "
                f"dropped_groups={stats['dropped_groups']}/{stats['total_groups']}",
                flush=True,
            )
        return original_compute_advantage(data, adv_estimator, *args, **kwargs)

    ray_trainer.compute_advantage = compute_advantage_with_dynamic_sampling
    ray_trainer._tau2_gated_patch = True
    print("[gated-bootstrap] GATED_BOOTSTRAP_READY estimator=grpo_gated", flush=True)


if _enabled():
    _install_patch()
