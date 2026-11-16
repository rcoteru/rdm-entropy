from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

""" Small utilities for the scripts -- run durations, cache naming, cycle windows, device
memory -- rather than for the model. The file exists so that an array utility or a path rule does not
have to live in a physics module to be shared between scripts. """


def grab_closest_idxs(vector: torch.Tensor, values: Sequence[float]) -> list[int]:
    """ Indices of the entries of `vector` closest to each of `values`. """
    return [int(torch.argmin((vector - value).abs())) for value in values]


def cycle_window(x: torch.Tensor, rel_tol: float = 1e-9) -> tuple[int, int, int]:
    """ (start, stop, n_cycles) spanning a whole number of periods of `x`.

    For averaging a periodic signal: a window that is not a whole number of periods leaves a
    phase error of order amplitude/n_periods, which on a burst whose amplitude is several times
    its mean is larger than most things worth measuring. Delimited by upward crossings of the
    mean rather than by peaks -- a crossing of a smooth signal is located to within one step,
    while the peak of a flat-topped burst is not.

    A settled state has no crossings to find and returns the whole window with n_cycles = 0,
    which is the honest answer: there are no cycles to align to and none are needed. So is a
    NOISY signal's answer meaningful only when the oscillation is deterministic -- a finite-N
    trajectory crosses its mean many times per period, and this will happily report those as
    cycles. Use it on the mean field, and a plain long average on the samples.

    `rel_tol` is the peak-to-peak, relative to the mean, below which the signal counts as
    settled. """
    if float(x.max() - x.min()) < rel_tol * float(x.mean().abs()):
        return 0, len(x), 0
    y  = x - x.mean()
    up = ((y[:-1] <= 0) & (y[1:] > 0)).nonzero().flatten()
    if len(up) < 2:
        return 0, len(x), 0
    return int(up[0]), int(up[-1]), len(up) - 1


# Durations
# ~~~~~~~~~

def n_steps(ms: float, dt: float, *, name: str = "duration") -> int:
    """ Steps for a physical duration of `ms` milliseconds at solver step `dt`.

    Runs are specified in ms, not in steps. A step count is a statement about the solver and
    says nothing about the system: the same 20000 steps is 20000 ms at dt = 1 and 4000 ms at
    dt = 0.2, so a dt refinement done on a fixed step count silently shortens the window it
    measures, and two scripts at different dt cannot be compared by the numbers at the top of
    them. In ms, equilibration is "long enough for the slowest timescale in the model" and the
    recording is "this many periods", which are the statements actually meant.

    Rounds to the nearest step, so a duration that is not a multiple of dt costs at most half a
    step; the realized duration is `n_steps(ms, dt) * dt`, which is what a dump should report
    if the difference ever matters. Raises below one step, which is a duration that was meant
    in ms and given in steps, or a dt that is coarser than the window. """
    if not ms > 0 or not dt > 0:
        raise ValueError(f"{name} and dt must be positive, got {name}={ms}, dt={dt}")
    n = int(round(ms / dt))
    if n < 1:
        raise ValueError(f"{name}={ms} ms is shorter than one step of dt={dt} ms")
    return n


# Cache naming
# ~~~~~~~~~~~~

def _stable(v):
    """ A value's identity for hashing, normalized so that equal runs hash equal.

    Numbers collapse to float, because J=4 and J=4.0 are the same model and would otherwise
    get separate caches. Tensors hash by their contents rather than by their id, which is what
    makes this work for the swept axes of the grid scripts. Mappings sort by key, so a dict
    rebuilt in a different order is still the same run. """
    if torch.is_tensor(v):
        return ("tensor", tuple(v.flatten().tolist()), tuple(v.shape))
    if isinstance(v, Mapping):
        return tuple((k, _stable(v[k])) for k in sorted(v))
    if isinstance(v, (list, tuple)):
        return tuple(_stable(x) for x in v)
    if isinstance(v, bool):          # before int: True == 1, but it does not mean 1
        return ("bool", v)
    if isinstance(v, (int, float)):
        return float(v)
    return v


def cache_path(script: str, key: Mapping, *, tag: str = "",
               root: str | Path = "cache", suffix: str = ".pt") -> Path:
    """ Cache file for `script` run at parameters `key`, named so the parameters pick the file.

    Changing any parameter changes the hash, so a run never finds a cache another parameter
    set wrote and there is no `overwrite` flag to remember -- the old file simply stays where
    it is, which also means switching between parameter sets costs nothing after the first
    run of each. `tag` is a readable prefix (a regime name, say) so the directory can still be
    read by eye.

    What this does NOT catch is a change to the code: the same parameters through a fixed
    model still hash the same. That is the one case still worth an explicit re-run. """
    h = hashlib.sha1(repr(_stable(key)).encode()).hexdigest()[:8]
    name = "_".join(p for p in (script, tag, h) if p)
    return Path(root) / f"{name}{suffix}"


def save_atomic(obj, path: str | Path) -> None:
    """ torch.save through a temporary and a rename: a reader, or the resume after a crash,
    sees the previous version or the new one, never a half-written file. Call it after every
    unit of work so an interrupted run keeps what it measured. """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# Device memory
# ~~~~~~~~~~~~~

def device_budget(device: str | torch.device, fraction: float = 0.8) -> int:
    """ Bytes a run may use on `device`: a fraction of what is actually free right now.

    Read rather than declared, so the same script runs on a different card without a constant
    to edit, and so a budget set while something else holds half the card does not OOM. """
    if torch.device(device).type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(fraction * free)
    with open("/proc/meminfo") as f:                      # MemAvailable, in kB
        avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
    return int(fraction * avail * 1024)
