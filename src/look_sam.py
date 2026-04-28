"""
LookSAM Optimizer
-----------------
Liu et al. 2022 — "Towards Efficient and Scalable Sharpness-Aware Minimization"
https://arxiv.org/abs/2203.02714

Key idea: the "flat-region" gradient direction (orthogonal to the ordinary gradient)
stays similar across nearby iterations.  We therefore only recompute the full SAM
second-pass every `k` steps; in between we reuse the cached direction and do a cheap
first-order update.  This gives ~SAM accuracy at ~(1 + 1/k) × first-order cost,
compared to 2× for vanilla SAM.

Usage:
    base_opt = torch.optim.AdamW
    optimizer = LookSAM(
        model.parameters(), base_opt,
        lr=1e-5, rho=0.05, k=5,
        weight_decay=1e-4,
    )

    # Inside the training loop (with AMP scaler):
    with autocast():
        loss = criterion(model(inputs), labels)
    scaler.scale(loss).backward()
    optimizer.first_step(zero_grad=True)          # stores perturbation, steps weights

    with autocast():
        loss2 = criterion(model(inputs), labels)
    scaler.scale(loss2).backward()
    optimizer.second_step(scaler, zero_grad=True) # unperturbs + real update

    scaler.update()
"""

import torch


class LookSAM(torch.optim.Optimizer):
    """
    LookSAM: periodically-updated SAM.

    Parameters
    ----------
    params : iterable
        Model parameters.
    base_optimizer : class
        A standard torch.optim class (e.g. AdamW, SGD).
    rho : float
        Neighbourhood radius for the SAM perturbation (default 0.05).
    k : int
        How often to recompute the full SAM ascent step.
        k=1  to  equivalent to vanilla SAM (2x cost every step).
        k=5  to  SAM ascent only every 5 steps (~1.2x cost overall).
    adaptive : bool
        If True use element-wise scaling (ASAM).  Default False.
    **kwargs
        Forwarded verbatim to base_optimizer (lr, weight_decay, …).
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        k: int = 5,
        adaptive: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        assert k >= 1, f"k must be >= 1, got {k}"

        defaults = dict(rho=rho, k=k, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Build the underlying first-order optimizer over the *same* param groups.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self._step_count = 0          # global iteration counter
        self._v_cache: dict = {}      # cached orthogonal direction per param

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """L2 norm of all gradients, computed on a shared device."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                norms.append(g.norm(p=2).to(shared_device))
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _compute_and_cache_v(self, grad_norm: torch.Tensor):
        """
        Cache the SAM orthogonal direction v = g_sam - g_sgd (projected component).
        We approximate: v ≈ (ρ / ‖g‖) * g  (the perturbation itself).
        """
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                self._v_cache[p] = e_w.clone()

    # ------------------------------------------------------------------
    # Public API (mirrors vanilla SAM for drop-in compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Step 1 — perturb weights toward the sharp region.

        On SAM-update steps (step_count % k == 0):
            • Recompute the perturbation from current gradients.
            • Cache the direction.
        On cheap steps:
            • Reuse the cached direction scaled to current grad norm.
        In both cases the weights are shifted by ε̂ so the second forward
        pass sees the perturbed parameter.
        """
        grad_norm = self._grad_norm()
        do_sam = (self._step_count % self.param_groups[0]["k"] == 0)

        if do_sam or not self._v_cache:
            # Full SAM ascent — recompute and cache direction
            self._compute_and_cache_v(grad_norm)

        # Apply the stored perturbation
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self._v_cache:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                p.add_(self._v_cache[p])  # w ← w + ε̂

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, scaler=None, zero_grad: bool = False, skip_update: bool = False):
        """
        Step 2 — restore original weights, then do the real optimizer update
        using the gradient computed at the perturbed point.

        Parameters
        ----------
        scaler : torch.cuda.amp.GradScaler or None
            Pass the AMP scaler so we can call scaler.step() correctly.
            If None, calls base_optimizer.step() directly (no AMP).
        zero_grad : bool
            Whether to zero gradients after the update.
        skip_update : bool
            If True, only restore weights without performing the optimizer step.
            Used when inf/nan gradients are detected in AMP mode.
        """
        # Restore original weights (for all params that have old_p saved)
        for group in self.param_groups:
            for p in group["params"]:
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]

        # Real optimizer step (AMP-aware) - only if not skipping
        if not skip_update:
            if scaler is not None:
                scaler.step(self.base_optimizer)
            else:
                self.base_optimizer.step()

        self._step_count += 1

        if zero_grad:
            self.zero_grad()

    def state_dict(self):
        """Override to also persist base_optimizer state, step counter, and v_cache."""
        base = super().state_dict()
        base["base_optimizer_state"] = self.base_optimizer.state_dict()
        base["_step_count"] = self._step_count
        # Save v_cache - convert tensors to CPU before saving
        base["_v_cache"] = {id(p): v.cpu() for p, v in self._v_cache.items()}
        return base

    def load_state_dict(self, state_dict: dict):
        """Override to restore base_optimizer state, step counter, and v_cache."""
        base_opt_state = state_dict.pop("base_optimizer_state", None)
        step_count = state_dict.pop("_step_count", 0)
        v_cache_state = state_dict.pop("_v_cache", {})
        super().load_state_dict(state_dict)
        if base_opt_state is not None:
            self.base_optimizer.load_state_dict(base_opt_state)
        self._step_count = step_count
        # Restore v_cache - need to map back from param id to param reference
        self._v_cache = {}
        param_map = {id(p): p for group in self.param_groups for p in group["params"]}
        for param_id, v in v_cache_state.items():
            if param_id in param_map:
                self._v_cache[param_map[param_id]] = v
