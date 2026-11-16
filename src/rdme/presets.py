from __future__ import annotations

import math
from collections.abc import Mapping

import rdme.hazards as hzd

""" The named parameter sets every run starts from.

A run is a PRESET plus explicit overrides, and nothing else. The point is not convenience --
seven numbers are not hard to type -- but that a number quoted in a dump or a note can be
traced back to a parameter set with a name, instead of to a dict that happened to be at the
top of a script on the day. If a run differs from a preset in anything other than its swept
axes and its listed overrides, it is a new preset and belongs in this file (and in the table
in CLAUDE.md, which is the same table in prose).

What a preset holds is the MODEL: the couplings, the kernels, the rate unit `nu0` and the
absolute refractory window `t_abs`, both in physical units. What it deliberately does not
hold:

  the hazard, which is its own axis. Three kernels share these parameters, and `bernoulli` is
  a different model from the two route-B ones, so a preset that named one would quietly fold
  that choice into every comparison. `params()` takes `hazard` as a keyword and falls back to
  hazards.DEFAULT_HAZARD ("poisson_exact"); scripts state it anyway, and it ends up in the
  dump's frontmatter because the script had to write it down.

  the solver step `dt`, the age grid `Qm` and the tolerance, which are numerics rather than
  model. `dt` is REQUIRED per run -- under route B there is no time quantum to default to --
  and `Qm` is fitted to a tolerance by helpers.fit_age_grid rather than declared.

  N, for the spin runs. The same preset drives the mean field and the microscopic model, and
  N is a property of the simulation, not of the system being simulated.

`nu0` is the rate unit of the Poisson reference, 1 ms^-1, and is the same in every preset: it
is not identifiable on its own, (nu0, theta) -> (nu0 e^{beta delta}, theta + delta) being an
exact reparametrization. `t_abs` is a PHYSICAL dead time in ms, 1 ms by default, and the only
thing that caps the realised rate under an exponential hazard. A study that is ABOUT the
window overrides it explicitly; `t_abs = 0` (no ceiling) has to be asked for.

Status: the two-population sets are not here yet, see PENDING. The Hopf points and the named
cycle points move with the hazard and are recorded where they are known for a stated one. """


# The presets
# ~~~~~~~~~~~
#
# Single-population (Ising) sets. Every entry is kwargs for RDMIsingModel / RDMIsingModelBatch
# / SpinIsingModel, which take the same names -- so one preset drives the mean field, the batch
# sweeps and the spin simulations without a translation layer in between.

PRESETS: dict[str, dict] = {

    # The main one-population system (formerly "the Ising set"): no drive, self-sustained
    # activity. J is NOT part of it -- it is the axis this system is studied along, and a run
    # either sweeps it or names a point below.
    "S1": dict(
        E       = 0.0,
        beta    = 6.0,
        theta   = 0.5,
        tau_int = 15.0,
        tau_ref = 15.0,
        K_ref   = 15.0,
        nu0     = 1.0,
        t_abs   = 1.0,
    ),

    # The rate-ceiling test (formerly "the low-beta point"): strong coupling into saturation,
    # where the legacy sigmoid kernel peaks at 100% of nu0. Under route B the ceiling is the
    # dead time instead -- the rate is unbounded and exponential -- so this is where the
    # realised rate should sit at 1/t_abs. J is part of this one: the point IS that state.
    "S1-sat": dict(
        E       = 0.0,
        beta    = 8.0,
        theta   = 0.5,
        tau_int = 10.0,
        tau_ref = 25.0,
        K_ref   = 25.0,
        J       = 40.0,
        nu0     = 1.0,
        t_abs   = 1.0,
    ),

    # Driven, no refractory kernel, strong gain: the fourth of the four parameter sets that
    # S1-mf_vs_sm.py carried inline (as "noref_driven_strong"). Named here because it matches
    # no other preset in any parameter -- the other three resolved to S1 and S1-sat with
    # overrides, this one did not -- and presets.py's own rule is that a set which is a system
    # gets a name rather than a dict at the top of a script. J IS part of it: the point is that
    # state. It runs at dt = 0.1, far finer than S1's, because beta = 40 makes the hazard stiff.
    #
    # PROVISIONAL NAME. The parameters are the ones that were run; "D2" is this port's choice
    # and was never a vault decision. Rename freely -- only S1-mf_vs_sm.py refers to it.
    "D2": dict(
        E       = 1.0,
        beta    = 40.0,
        theta   = 0.5,
        tau_int = 15.0,
        tau_ref = 1.0,
        K_ref   = 0.0,
        J       = 6.0,
        nu0     = 1.0,
        t_abs   = 1.0,
    ),

    # Legacy driven high-beta set. Here to reproduce old figures and for nothing else: no
    # refractory kernel (K_ref = 0), a drive that does the work instead. Its old runs were at
    # dt0 = 0.5 under route A, i.e. hazard="bernoulli" at dt = 0.5 with nu0 = 2; the preset
    # carries the route-B unit instead, so reproducing a figure means stating both.
    #
    # LEGACY-ONLY, and deliberately kept with NO live caller (2026-11-07): the paper figures
    # that used it are frozen at 378bf2a^ and reproduce from their own commit. It stays here
    # because it is the record of those figures' parameters, not because anything runs it.
    "D1": dict(
        E       = 0.9,
        beta    = 30.0,
        theta   = 1.0,
        tau_int = 20.0,
        tau_ref = 3.0,
        K_ref   = 0.0,
        nu0     = 1.0,
        t_abs   = 1.0,
    ),
}

# Named points on a preset's axis: a preset plus the one coupling that makes it a system
# rather than a family. `params("S1/fp")` resolves them, and they read as presets everywhere
# else, so a script and a dump can name the point instead of quoting the number.
#
# The cycle point is deliberately absent, pending a vault decision. The Hopf is MEASURED at
# J_c = 11.534 +- 0.012 under `poisson_exact` as a dt -> 0 limit (11.4646 under `poisson`,
# 12.3525 under `bernoulli`; S1-cycle_point.py, S1-hopf_dt.py). J = 17 is the working cycle
# point -- 5.47 past onset, confirmed workable -- and the scripts carry it as a literal until
# this dict does. The J = 14.4-14.6 that stood here was route-A `poisson`, never a route-B
# prediction; 12.06 was the legacy sigmoid kernel.
POINTS: dict[str, dict] = {
    "S1/fp": {"J": 10.0},                   # the fixed point below the Hopf
}

# Populations per preset, so a caller can tell which constructor a name belongs to without
# counting parameters.
POPS: dict[str, int] = {"S1": 1, "S1-sat": 1, "D1": 1, "D2": 1, "S1/fp": 1}

# Every model parameter a preset of M populations may carry, which is what an override is
# checked against: a preset leaves J out when J is its swept axis, and that absence should not
# make `J=...` an error. Mirrors the signature of RDMIsingModel; tests/test_presets.py checks
# the two against each other so a parameter added to the model cannot be forgotten here.
MODEL_KEYS: dict[int, frozenset[str]] = {
    1: frozenset({"J", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref", "nu0", "t_abs"}),
}

# Two-population sets named in CLAUDE.md but NOT encoded yet, with what is missing. Asking for
# one raises with this text rather than returning an invented parameter set: the EI row is
# short of w_EE and w_EI and is marked "to be confirmed", and the Painchaud cell is built by a
# mapping whose tau_m step may still change.
PENDING: dict[str, str] = {
    "EI":   "the CLAUDE.md row is incomplete (no w_EE, w_EI) and marked to be confirmed; "
            "wc-mf_vs_sm.py carries two further E-I sets that have to be reconciled first",
    "EE-P": "built by the Painchaud et al. (2022) mapping in wc-painchaud.py, whose tau_m "
            "step is not settled; encode it here once the mapping is fixed",
}

# The hazard is its own axis, and scripts state it; this is the fallback for one that does
# not. It is defined in rdme.hazards, beside the kernels it names, and is "poisson_exact"
# (route B, 2026-11-07): the exact-survival solver for the Poisson-reference model, whose
# convergence is monotone and so is the one a dt -> 0 limit is taken with.
DEFAULT_HAZARD: str | None = hzd.DEFAULT_HAZARD

# Keys a run may pass that are not model parameters: the solver, the grid, the hardware and,
# for the spin runs, the population size. Separated from the model keys so that `params()` can
# reject a typo in a model parameter without also rejecting these.
SOLVER_KEYS = frozenset({"dt", "Qm", "eps", "device", "N", "ic"})


# Resolving a run
# ~~~~~~~~~~~~~~~

def resolve(name: str) -> dict:
    """ The parameter dict a preset name stands for, including the named points (`S1/fp`).

    A copy, so a caller mutating what it gets back cannot edit the preset for the rest of the
    session -- the bug that makes two runs in one process disagree for no visible reason. """
    if name in PRESETS:
        return dict(PRESETS[name])
    if name in POINTS:
        base = name.split("/")[0]
        return dict(PRESETS[base]) | dict(POINTS[name])
    if name in PENDING:
        raise KeyError(f"preset {name!r} is not encoded yet: {PENDING[name]}")
    raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS) + sorted(POINTS)}")


def params(name: str, *, hazard: str | None = None, **overrides) -> dict:
    """ Model kwargs for `name`, with `hazard` and `overrides` applied: the one call a script
    makes to build its parameter dict.

        PARAMS = presets.params("S1", hazard="poisson", J=torch.linspace(0, 20, 300), dt=0.2)
        mf = RDMIsingModelBatch(**PARAMS, Qm=Qm, device=device)

    `hazard` falls back to hazards.DEFAULT_HAZARD ("poisson_exact"). A script states it
    anyway: it is a modelling choice of the same weight as the couplings, `bernoulli` is a
    different model from the two route-B kernels, and the dump's frontmatter has to carry
    it.

    An override of a name the preset does not have, and that is not a solver key, raises:
    swept axes (J here) are model parameters and pass, while `tau_m=...` or `Jay=...` is a typo
    or a parameter this model does not have, and either way is not something to discover in the
    figure. The deliberate exception is a key the constructor takes but the preset leaves out,
    which is what SOLVER_KEYS lists. """
    p = resolve(name)

    hazard = hazard if hazard is not None else DEFAULT_HAZARD
    if hazard is None:
        raise ValueError(
            "hazard must be given: it is its own axis, not part of a preset. Pass "
            "hazard='poisson_exact' (exact survival, the solver a dt -> 0 limit is taken "
            "with), hazard='poisson' (the MaxCal kernel at each bin, for binned data) or "
            "hazard='bernoulli' (route A, comparison only).")

    if "dt" not in overrides:
        raise ValueError(
            f"params({name!r}) needs dt=: under route B there is no time quantum to default "
            "to, so the solver step is a choice the run has to state (and the dump has to "
            "record). Pass the coarse member of the Richardson pair, e.g. dt=0.5.")

    model_keys = MODEL_KEYS[POPS[name]]
    unknown = [k for k in overrides if k not in model_keys and k not in SOLVER_KEYS]
    if unknown:
        raise TypeError(f"{sorted(unknown)} is not a parameter of preset {name!r} "
                        f"(it takes {sorted(model_keys)}) nor a solver key "
                        f"({sorted(SOLVER_KEYS)}). "
                        f"If the run really needs it, it is a NEW preset: add it here and to "
                        f"the table in CLAUDE.md.")

    return p | {"hazard": hazard} | overrides


def diff(name: str, p: Mapping) -> dict:
    """ What `p` overrides in preset `name`: the `overrides:` field of a dump's frontmatter.

    Written from the run's own parameter dict rather than typed out, so the frontmatter cannot
    disagree with what was computed. Solver keys and the hazard are reported separately in the
    frontmatter and are left out here; so are the swept axes, which are not scalars. """
    base = resolve(name)
    return {k: v for k, v in p.items()
            if k not in SOLVER_KEYS and k != "hazard"
            and not hasattr(v, "shape")               # a swept axis, not an override
            and (k not in base or base[k] != v)}


def banner(name: str, p: Mapping, *, keys: tuple[str, ...] | None = None) -> str:
    """ The one-line parameter string for a figure title: preset, hazard, dt, parameters.

    CLAUDE.md asks every figure to state its preset, hazard and dt, and this is that line, so
    a figure cannot claim a set it was not run at -- it is built from the dict that built the
    model. """
    head = f'{name}  ·  hazard={p.get("hazard", "?")}'
    dt, t_abs = p.get("dt"), p.get("t_abs")
    if dt is not None:
        head += f'  ·  dt={dt:g}'
    if t_abs is not None:
        head += f'  ·  t_abs={t_abs:g} ms'

    keys = keys or ("J", "E", "beta", "theta", "tau_int", "tau_ref", "K_ref")
    shown = {k: p[k] for k in keys if k in p and not hasattr(p[k], "shape")}
    pretty = {"beta": "β", "theta": "θ", "tau_int": "τ_int", "tau_ref": "τ_ref"}
    body = ',  '.join(f'{pretty.get(k, k)}={v:g}' for k, v in shown.items())
    return f'{head}  —  {body}' if body else head


# The gauge-invariant combinations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# (nu0, theta) -> (nu0 e^{beta delta}, theta + delta) leaves the hazard nu0 e^{beta(S+R-theta)}
# unchanged, so nu0 and theta are not separately identifiable and a number quoted for either
# alone says nothing. These five combinations are: they are what two parameter sets have to
# agree on to be the same system, and what they differ in when one is a re-anchoring rather
# than a relabel. REPORTING ONLY -- nothing here changes a preset.

INVARIANT_KEYS = ("beta_theta", "nu0_exp", "Kref_theta", "nu0_tabs", "lam_op_nu0")


def invariants(p: Mapping | str, *, lam_op: float | None = None) -> dict:
    """ The gauge-invariant combinations of a preset name or a parameter dict.

        beta*theta       threshold in units of the gain
        nu0 e^{-beta*theta}   the escape rate at zero field, 1/ms -- the shift-invariant pair
        K_ref/theta      the refractory kernel in units of the threshold
        nu0*t_abs        the dead time in units of the rate unit (cortically ~0.04)
        lam_op/nu0       the operating hazard in units of the rate unit, = e^{beta*h_op}

    The last one is an output of a run, not of a parameter set, so it is None unless the
    caller passes the operating hazard `lam_op` (1/ms) it read off an attractor. """
    p = resolve(p) if isinstance(p, str) else p
    beta, theta = p["beta"], p["theta"]
    nu0, t_abs, K_ref = p["nu0"], p["t_abs"], p["K_ref"]
    return {
        "beta_theta": beta * theta,
        "nu0_exp":    nu0 * math.exp(-beta * theta),
        "Kref_theta": K_ref / theta,
        "nu0_tabs":   nu0 * t_abs,
        "lam_op_nu0": None if lam_op is None else lam_op / nu0,
    }


def invariants_line(p: Mapping | str, *, lam_op: float | None = None) -> str:
    """ `invariants` as one line, for a figure caption, a dump or the summary below. """
    inv = invariants(p, lam_op=lam_op)
    lam = inv["lam_op_nu0"]
    return (f'β·θ={inv["beta_theta"]:g}  ·  ν₀e^(−βθ)={inv["nu0_exp"]:.4g}/ms  ·  '
            f'K_ref/θ={inv["Kref_theta"]:g}  ·  ν₀·t_abs={inv["nu0_tabs"]:g}  ·  '
            f'λ_op/ν₀=' + ("n/a (needs a run)" if lam is None else f"{lam:.4g}"))


def summary() -> str:
    """ Every preset with its parameters and its gauge invariants: `python -m rdme.presets`. """
    out = []
    for name in list(PRESETS) + list(POINTS):
        p = resolve(name)
        body = ',  '.join(f'{k}={v:g}' for k, v in p.items())
        out += [f'{name}', f'    {body}', f'    {invariants_line(p)}', '']
    out += ['λ_op/ν₀ = e^(β·h_op) is read off an attractor, not off a preset; pass lam_op=.', '']
    return '\n'.join(out)


if __name__ == "__main__":
    print(summary())
