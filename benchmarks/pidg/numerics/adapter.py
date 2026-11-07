"""Task adapter: the only task-specific code the precision judge needs.

A task implements a thin subclass that builds its real training objects, restores state, replays
fixed inputs and runs ONE real training step by calling the task's own code. Everything else
(gradients, parameter updates, optimizer states, EMA, buffers, counters, RNG fingerprints, dtypes,
cross-rank relations) is captured by generic hooks in numerics.record.
"""
from dataclasses import dataclass, field


@dataclass
class Objects:
    # name -> torch.nn.Module. Use the module that owns the parameters (e.g. ddp.module).
    models: dict
    # name -> torch.optim.Optimizer. Every optimizer.step() call is observed.
    optimizers: dict = field(default_factory=dict)
    # name -> torch.nn.Module holding EMA weights.
    ema: dict = field(default_factory=dict)
    # model name -> 'replicated' (params/grads must be identical on all ranks) or 'local'
    # (each rank may differ). The candidate must reproduce the reference's cross-rank pattern.
    sync: dict = field(default_factory=dict)
    # fnmatch patterns over '<model>.<submodule path>' whose forward outputs are recorded.
    capture_outputs: tuple = ()
    # Count forward calls of each model as an exact check. Turn off when candidates may legitimately
    # wrap models (e.g. torch.compile) in ways that change how module hooks fire; use invariants instead.
    count_forwards: bool = True


class Adapter:
    """Override the methods marked 'required'."""

    name = 'adapter'

    def configure(self, case):
        """Optional. Receives the whole case dict before build(), for task-specific case options
        (e.g. a per-case config override that places the run in a given training phase)."""

    def build(self, seed, golden=False):
        """Required. Construct models, optimizers, EMA and data for this process.

        golden=True builds the fp64 reference ("golden") variant: models and inputs in float64,
        with any hard-coded low-precision casts in the task code patched here."""
        raise NotImplementedError

    def objects(self):
        """Required. Return an Objects instance describing what to observe."""
        raise NotImplementedError

    def save_state(self, path):
        """Full state: models, optimizers, schedulers, EMA, buffers, every RNG stream, step/phase counters."""
        raise NotImplementedError

    def load_state(self, path):
        raise NotImplementedError

    def batch(self, step, rank):
        """Required. Return the fixed input for this step and rank (replayed, never freshly sampled)."""
        raise NotImplementedError

    def step(self, step, batch):
        """Required. Run one REAL training step and return {loss name: tensor or float}."""
        raise NotImplementedError

    def phases(self):
        """Training phases in order, e.g. ['recon', 'disc', 'gan']."""
        return ['default']

    def set_phase(self, phase, step):
        """Move the task's own step/epoch counters so that `step` runs in `phase`."""

    def invariants(self):
        """Scalars that must match the reference exactly (learning rates, phase flags, task counters)."""
        return {}

    def canonical(self, kind, name, tensor):
        """Map a candidate tensor to the reference layout, e.g. split a packed qkv weight.

        Return {reference name: tensor}. A candidate that changes parameter layout must override this;
        otherwise its records will not line up with the reference and the judge reports ERROR."""
        return {name: tensor}
