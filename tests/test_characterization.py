""" Characterization tests: what the CURRENT RDM library computes, pinned as float64 literals.

Step 0 of `P1 2026-11-13 code rebuild`, first bullet. These are the acceptance criterion for the
rebuild's later steps: Step 1 (equations as pure functions) must reproduce every literal here
bit for bit, except the single-system ones its arithmetic decision moved (SINGLE_RTOL below);
Step 2 (the O(Q) EPR observer) changes the EPR diagonal arithmetic and may loosen
the EPR quantities, and only those, through EPR_RTOL / EPR_ATOL below; Step 3 (one driver
class) must keep both the single-system and the batch literals: since Step 3 "single" is the
unbatched RDMNetwork (no leading axis) and "batch" the batched one, and the joint-method cases
run the oracle in tests/epr_reference.py through the driver's loop.

No oracle here is a closed form: each literal is what the code at `8e8a724` produced, recorded
with `repr`. A tensor is pinned by a fingerprint (shape, exact `math.fsum` of its entries, an
exact fsum weighted by (flat index mod 7) + 1, and four entries at fixed flat positions), which
any change to the arithmetic moves. The fsums are order-independent, so the fingerprint itself
does not depend on the reduction order of the code under test.

What the existing suite already pins, and is not repeated here: closed-form checks of the
`rdm.py` free functions (test_rdm.py), sliding == joint and inplace == out-of-place for the
EPR, the joint oracle over long runs, the live-state rewind (test_epr.py), batch == single to
1e-14 and the batch save/load round trip to 1e-15 (test_srm.py), fixed points being stationary
and the tangent matching the Jacobian (test_fixed_points.py, test_srm.py). Those are relations
between paths; this file pins the values the paths produce.

Configurations, all `poisson_exact` at dt = 0.5 ms with a short age window:
  M1_fp   S1, J = 10, placed on its fixed point (set_state at LIT["fp_root"])
  M1_cyc  S1, J = 17, 400 ms from the default initial condition: on the cycle (asserted)
  M2_fp   two S1-like populations, explicit w (see below), placed on the fixed point
  M2_cyc  the same with a stronger w, 400 ms in: oscillating (asserted)
Batch runs are B = 2: element 0 is exactly the single system's parameters, element 1 differs
in the coupling, so a batch element can be compared with the single system on the same input.
"""

import io
import math

import pytest
import torch

import rdme.epr as epr
import rdme.presets as pre
import rdme.rdm as rdm
from rdme.network import RDMIsingModel, RDMIsingModelBatch, RDMNetwork

from epr_reference import joint_entropy_trajectory


# Tolerance on the EPR quantities ONLY (sigma, H_fwd, H_rev, their _tot aggregates and their
# Poisson-referenced variants). 0.0 means bit for bit, which is what holds at Step 0 and must
# hold through Step 1. Step 2 may set these to a stated value; nothing else reads them.
# Step 2 (the O(Q) observer) rebuilds the age-0 diagonals from the input window in closed form
# and accumulates its initial state lag by lag, which moves the last bits of the sliding path's
# H_rev and sigma; H_fwd and the joint path are unchanged bit for bit. These are the smallest
# values the literals pass under: the rtol is set by a batch H_rev (2.1e-16 relative), the atol
# by the M2_fp batch sigma fingerprint, an fsum of values at the fixed point's 1.8e-13
# truncation floor (3.03e-15 absolute), where no relative tolerance means anything.
EPR_RTOL = 2.5e-16
EPR_ATOL = 3.1e-15
EPR_KEYS = ("sigma", "H_fwd", "H_rev", "sigma_tot", "H_fwd_tot", "H_rev_tot",
            "H_fwd_ref", "H_rev_ref")

# Tolerance on the SINGLE-SYSTEM literals ONLY, and on the two 1-D free-function literals that
# share their cause. It exists because of the 2026-11-13 arithmetic decision (Step 1): the
# equations became pure functions with the BATCH arithmetic, so RDMNetwork's step switched
# (a) from torch.dot to the batch's product for m and (b) from the in-place age update (last
# bin 1 - m - interior.sum()) to the out-of-place one (1 - P_new[:-1].sum()). The literals
# below are still the 8e8a724 ones; these are the smallest (rtol, atol) they pass under. The
# atol is needed because a fingerprint's fsums can sit at roundoff (a sum-zero tangent block,
# the fixed point's sigma floor), where no relative tolerance means anything.
# The stronger statement is test_single_equals_batch_element0: the unbatched RDMNetwork
# reproduces element 0 of the batched one BIT FOR BIT, and the batch literals are pinned exactly.
SINGLE_RTOL = 1.4e-15
SINGLE_ATOL = 1e-14


def _moved(name: str, key: str) -> bool:
    """ The literals SINGLE_RTOL/SINGLE_ATOL apply to. """
    return "/single" in name or (name == "rdm_free_functions" and key in ("rate", "age.inplace"))

HAZARD = "poisson_exact"
DT = 0.5                      # ms; divides t_abs = 1 ms
T_EQUI = 400.0                # ms, to get onto the cycle from the default initial condition
T_REC = 70.0                  # ms, recorded window: about one period of the S1 cycle
# The windows are short for speed, and short enough to be felt: at Q1 the fixed point's sigma
# reads 2.7e-13 per step (age-grid truncation, 0.8% truncation error), not machine zero, and
# falls to ~1e-18 by Q = 300. A characterization, not a converged number.
Q1 = 200                      # M = 1 age window: 100 ms
QM2 = [200, 150]              # M = 2, unequal on purpose: 100 ms and 75 ms

N_EQUI = round(T_EQUI / DT)
N_REC = round(T_REC / DT)

# The *_fp configurations are placed on LIT["fp_root"]: the S1 root at J = 10 found by
# RDMNetwork.fixed_points (bracket) and the M2_fp root (Newton), held as literals so the
# dynamics cases do not depend on the root finders. The fixed_points/* cases pin that the
# finders still return exactly these.


@pytest.fixture(autouse=True)
def float64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# Building the systems
# ~~~~~~~~~~~~~~~~~~~~

def _s1(J: float) -> dict:
    p = pre.params("S1", hazard=HAZARD, J=J, dt=DT)
    return p


def _single_m1(J: float) -> RDMNetwork:
    return RDMIsingModel(**_s1(J), Qm=[Q1])


def _batch_m1(J: float) -> RDMNetwork:
    # element 0 is the single system; element 1 has J + 1
    return RDMIsingModelBatch(**(_s1(J) | {"J": torch.tensor([J, J + 1.0])}), Qm=[Q1])


# Two populations. The two-population presets (EI, EE-P) are PENDING in rdme.presets, so the
# M = 2 system is built here explicitly: both populations carry S1's single-population
# parameters except tau_int (15 and 10 ms), with an excitatory population 0 and an inhibitory
# population 1. This is a test fixture, not a preset, and no run should quote it.
W2_FP = [[8.0, -2.0], [6.0, -1.0]]
W2_CYC = [[17.0, -2.0], [10.0, -1.0]]
_M2 = dict(N=[800, 200], E=[0.0, 0.0], beta=[6.0, 6.0], theta=[0.5, 0.5],
           tau_int=[15.0, 10.0], tau_ref=[15.0, 15.0], K_ref=[15.0, 15.0])


def _single_m2(w) -> RDMNetwork:
    return RDMNetwork(M=2, w=torch.tensor(w), **_M2, dt=DT, hazard=HAZARD, Qm=list(QM2))


def _batch_m2(w) -> RDMNetwork:
    # element 0 is the single system; element 1 has its weights scaled by 0.9
    w0 = torch.tensor(w)
    W = torch.stack([w0, 0.9 * w0])
    col = {k: torch.tensor(v).expand(2, 2).clone() for k, v in _M2.items() if k != "N"}
    return RDMNetwork(M=2, w=W, N=list(_M2["N"]), **col, dt=DT, hazard=HAZARD,
                      Qm=list(QM2))


def _build(cfg: str, kind: str):
    """ A fresh system for configuration `cfg`, in `kind` = "single" or "batch", already
    placed where the configuration says: on the fixed point, or N_EQUI steps in. """
    m1 = cfg.startswith("M1")
    fp = cfg.endswith("fp")
    if m1:
        J = 10.0 if fp else 17.0
        net = _single_m1(J) if kind == "single" else _batch_m1(J)
    else:
        w = W2_FP if fp else W2_CYC
        net = _single_m2(w) if kind == "single" else _batch_m2(w)
    if fp:
        root = torch.tensor(LIT["fp_root"][cfg[:2]])
        net.set_state(root if kind == "single" else root.expand(net.B, net.M).clone())
    else:
        net.forward(N_EQUI, pb=False)
    return net


CONFIGS = ("M1_fp", "M1_cyc", "M2_fp", "M2_cyc")


# Fingerprints and comparison
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

def _fp(x):
    """ (shape, fsum, weighted fsum, four entries) of a tensor; a float stays a float. """
    if isinstance(x, (bool, int, float)):
        return x
    x = torch.as_tensor(x).detach().cpu()
    if x.dtype == torch.bool:
        return (tuple(x.shape), tuple(x.reshape(-1).tolist()))
    v = [float(a) for a in x.reshape(-1).tolist()]
    n = len(v)
    picks = sorted({0, n // 3, (2 * n) // 3, n - 1}) if n else []
    return (tuple(x.shape), math.fsum(v),
            math.fsum(a * ((i % 7) + 1) for i, a in enumerate(v)),
            tuple(v[i] for i in picks))


def _fps(out: dict) -> dict:
    """ Fingerprint every entry of an output dict, splitting per-population lists. """
    res = {}
    for k, v in out.items():
        if isinstance(v, (list, tuple)):
            for i, t in enumerate(v):
                res[f"{k}[{i}]"] = _fp(t)
        else:
            res[k] = _fp(v)
    return res


def _close(a, b, rtol, atol) -> bool:
    if isinstance(a, tuple):
        return (isinstance(b, tuple) and len(a) == len(b)
                and all(_close(x, y, rtol, atol) for x, y in zip(a, b)))
    if isinstance(a, float) and isinstance(b, float):
        if rtol == 0.0 and atol == 0.0:
            return a == b
        return abs(a - b) <= atol + rtol * abs(b)
    return a == b


def _check(got: dict, want: dict, name: str = "") -> None:
    assert set(got) == set(want), (sorted(set(got) ^ set(want)))
    for k in want:
        base = k.split("[")[0]
        rtol, atol = (EPR_RTOL, EPR_ATOL) if base in EPR_KEYS else (0.0, 0.0)
        if _moved(name, k):
            rtol, atol = max(rtol, SINGLE_RTOL), max(atol, SINGLE_ATOL)
        assert _close(got[k], want[k], rtol, atol), (k, got[k], want[k])


def _state(net) -> dict:
    return {"p": list(net.p), "S": list(net.S), "m": net.m}


# The cases
# ~~~~~~~~~
#
# Each returns a dict of named outputs; LIT[case] holds their fingerprints.

def case_trajectory(cfg: str, kind: str) -> dict:
    net = _build(cfg, kind)
    out = net.trajectory(N_REC, r_tot=True, q=True, S=True, pb=False)
    return _fps(out) | {f"after.{k}": v for k, v in _fps(_state(net)).items()}


def _entropy(net, method: str, inplace: bool = True) -> dict:
    """ The sliding path is the driver's entropy_trajectory; the joint path, which left the
    library in Step 3, is the same driver loop fed the oracle in tests/epr_reference.py. Both
    default to inplace=True, as entropy_trajectory did for both methods when these literals
    were recorded. """
    if method == "joint":
        return joint_entropy_trajectory(net, N_REC, inplace=inplace)
    return net.entropy_trajectory(N_REC, pb=False, inplace=inplace)


def case_entropy(cfg: str, kind: str, method: str, **opts) -> dict:
    net = _build(cfg, kind)
    out = _entropy(net, method, **opts)
    out = dict(out)
    ref = epr.poisson_reference_entropy(out["m"], net.nu0, net.dt)
    out["H_fwd_ref"] = out["H_fwd"] - ref
    out["H_rev_ref"] = out["H_rev"] - ref
    return _fps(out) | {f"after.{k}": v for k, v in _fps(_state(net)).items()}


def case_methods(cfg: str, kind: str) -> dict:
    """ The per-method equations, called directly on the configuration's state, then one
    update and a short forward. """
    net = _build(cfg, kind)
    M = net.M
    probe = torch.tensor([0.007] if M == 1 else [0.006, 0.004])
    if kind == "batch":
        probe = torch.stack([probe, 1.1 * probe])
    out = {
        "rates": net.rates(), "overlaps": net.overlaps(), "activity": net.activity(),
        "age_density": net.age_density(), "total_input": net.total_input(),
        "field": net.field(), "firing_prob": net.firing_prob(),
        "d_firing_prob": net.d_firing_prob(), "hazard_rate": net.hazard_rate(),
        "max_step_prob": net.max_step_prob(), "truncation_error": net.truncation_error(),
        "fp_residual": net.fp_residual(probe), "fp_state.p": net.fp_state(probe)[0],
        "fp_state.S": net.fp_state(probe)[1], "R": net.R, "R_abs": net.R_abs,
        "alpha_int": net.alpha_int,
    }
    if kind == "single":
        x = net.flatten_state(net.p, net.S)
        out["stationary_rate"] = net.stationary_rate()
        out["state_map"] = net.state_map(x)
    res = _fps(out)
    net.update()
    res |= {f"update.{k}": v for k, v in _fps(_state(net)).items()}
    net.forward(20, pb=False)
    res |= {f"forward.{k}": v for k, v in _fps(_state(net)).items()}
    return res


def case_tangent(cfg: str, kind: str) -> dict:
    net = _build(cfg, kind)
    blocks = net.init_tangent(3, generator=torch.Generator().manual_seed(0))
    res = {f"init[{j}]": _fp(b) for j, b in enumerate(blocks)}
    for _ in range(40):
        net.step_tangent(blocks)
    res |= {f"step[{j}]": _fp(b) for j, b in enumerate(blocks)}
    return res | {f"after.{k}": v for k, v in _fps(_state(net)).items()}


def case_fixed_points_single(cfg: str) -> dict:
    """ RDMNetwork.fixed_points with its default method (bracket for M = 1, Newton for
    M = 2), plus M = 1 by Newton too, and each root's stability and leading eigenvalues. """
    m1 = cfg.startswith("M1")
    fp = cfg.endswith("fp")
    net = _single_m1(10.0 if fp else 17.0) if m1 else _single_m2(W2_FP if fp else W2_CYC)
    roots = net.fixed_points()
    res = {"roots": _fp(roots)}
    if m1:
        res["roots_newton"] = _fp(net.fixed_points(method="newton"))
    for r_i, r in enumerate(roots):
        ev = net.fp_eigenvalues(r)
        lead = ev.abs().sort(descending=True).values[:4]
        res[f"root{r_i}.lead_abs"] = _fp(lead)
        res[f"root{r_i}.stable"] = net.is_stable(r)
    return res


def case_fp_sweep(cfg: str) -> dict:
    m1 = cfg.startswith("M1")
    fp = cfg.endswith("fp")
    net = _batch_m1(10.0 if fp else 17.0) if m1 else _batch_m2(W2_FP if fp else W2_CYC)
    roots, conv = net.fp_sweep()
    # rows not converged hold junk (possibly NaN), so they are zeroed rather than pinned
    return {"roots": _fp(torch.where(conv.unsqueeze(-1), roots, 0.0)), "converged": _fp(conv)}


def case_rdm_free_functions() -> dict:
    """ rdm.py's free functions on fixed, small inputs, both inplace branches. """
    Q = 7
    g = torch.arange(1, Q + 1, dtype=torch.float64)
    P = (g / g.sum()).sqrt(); P = P / P.sum()
    phi = 0.05 + 0.9 * torch.sin(g) ** 2
    PB = torch.stack([P, P.flip(0)])
    phiB = torch.stack([phi, phi.flip(0)])
    S0 = torch.cos(g) * 0.3
    SB = torch.stack([S0, -S0])
    w = torch.tensor([[1.5, -0.7], [0.4, -2.1]])
    out = {
        "total_input.single": rdm.compute_total_input(torch.tensor([0.013, 0.021]), w,
                                                      torch.tensor([0.1, -0.2])),
        "total_input.batch": rdm.compute_total_input(
            torch.tensor([[0.013, 0.021], [0.05, 0.002]]), torch.stack([w, w.T]),
            torch.tensor([[0.1, -0.2], [0.0, 0.3]])),
        "syn": rdm.update_synaptic_term(S0, 0.0327, 0.71),
        "syn.inplace": rdm.update_synaptic_term(S0.clone(), 0.0327, 0.71, inplace=True),
        "syn.batch": rdm.update_synaptic_term(SB, torch.tensor([[0.0327], [0.1]]),
                                              torch.tensor([[0.71], [-0.2]])),
        "syn.batch.inplace": rdm.update_synaptic_term(
            SB.clone(), torch.tensor([[0.0327], [0.1]]), torch.tensor([[0.71], [-0.2]]),
            inplace=True),
        "rate": rdm.compute_firing_rate(P, phi),
        "rate.batch": rdm.compute_firing_rate(PB, phiB),
        "age": rdm.update_age_distribution(P, phi),
        "age.inplace": rdm.update_age_distribution(P.clone(), phi, inplace=True),
        "age.batch": rdm.update_age_distribution(PB, phiB),
        "age.batch.inplace": rdm.update_age_distribution(PB.clone(), phiB, inplace=True),
        "hazard_clamp.f64": rdm.hazard_clamp(torch.float64),
        "hazard_clamp.f32": rdm.hazard_clamp(torch.float32),
    }
    return _fps(out)


# The parametrisation of every case: (name, callable). LIT is keyed by name.
CASES = {}
for _c in CONFIGS:
    for _k in ("single", "batch"):
        CASES[f"trajectory/{_c}/{_k}"] = (case_trajectory, (_c, _k), {})
        CASES[f"methods/{_c}/{_k}"] = (case_methods, (_c, _k), {})
        for _m in ("sliding", "joint"):
            CASES[f"entropy/{_c}/{_k}/{_m}"] = (case_entropy, (_c, _k, _m), {})
    CASES[f"fixed_points/{_c}"] = (case_fixed_points_single, (_c,), {})
    CASES[f"fp_sweep/{_c}"] = (case_fp_sweep, (_c,), {})
# non-default EPR code paths, on the configuration that exercises the most (M = 2, unequal
# Qm, moving input): inplace, one system and batched. (The batch's `chunk` cases went with
# `chunk` itself in Step 3.)
for _m in ("sliding", "joint"):
    CASES[f"entropy/M2_cyc/single/{_m}/inplace"] = (
        case_entropy, ("M2_cyc", "single", _m), {"inplace": True})
    CASES[f"entropy/M2_cyc/batch/{_m}/inplace"] = (
        case_entropy, ("M2_cyc", "batch", _m), {"inplace": True})
for _c in ("M1_cyc", "M2_cyc"):
    for _k in ("single", "batch"):
        CASES[f"tangent/{_c}/{_k}"] = (case_tangent, (_c, _k), {})
CASES["rdm_free_functions"] = (case_rdm_free_functions, (), {})

# At Step 0 every non-default EPR path above reproduces its default bit for bit, so it is
# checked against the default's literal rather than carrying a copy. A later step that makes
# one of them differ gives it its own LIT entry and drops it from here.
SAME_AS = {name: name.rsplit("/", 1)[0] for name in CASES
           if name.startswith("entropy/") and name.count("/") == 4}


def run_case(name: str) -> dict:
    fn, args, kw = CASES[name]
    return fn(*args, **kw)


@pytest.mark.parametrize("name", list(CASES))
def test_characterization(name):
    _check(run_case(name), LIT[SAME_AS.get(name, name)], name)


def _equal_to_row0(single, batch, tag: str) -> None:
    """ single == batch[0] bit for bit; lists are per-population. """
    if isinstance(single, (list, tuple)):
        for i, (s, b) in enumerate(zip(single, batch, strict=True)):
            _equal_to_row0(s, b, f"{tag}[{i}]")
        return
    s, b = torch.as_tensor(single), torch.as_tensor(batch)[0]
    assert torch.equal(s.reshape(b.shape), b), (tag, (s.reshape(b.shape) - b).abs().max())


def _state_equal(s, b, tag: str) -> None:
    for k, v in _state(s).items():
        _equal_to_row0(v, _state(b)[k], f"{tag}.{k}")


@pytest.mark.parametrize("cfg", CONFIGS)
def test_single_equals_batch_element0(cfg):
    """ Since Step 1 both classes call the same equations with the same arithmetic, so the
    single system reproduces batch element 0 (its own parameters) BIT FOR BIT: observables,
    trajectory, both EPR paths (and inplace on M2_cyc, as in CASES), and the tangent step fed
    the same blocks. With the batch literals pinned exactly, this pins the single system. """
    s, b = _build(cfg, "single"), _build(cfg, "batch")
    _state_equal(s, b, "build")
    for k in ("rates", "overlaps", "activity", "age_density", "total_input", "field",
              "firing_prob", "d_firing_prob", "hazard_rate", "max_step_prob",
              "truncation_error"):
        _equal_to_row0(getattr(s, k)(), getattr(b, k)(), k)

    kw = dict(r_tot=True, q=True, S=True, pb=False)
    ts, tb = s.trajectory(N_REC, **kw), b.trajectory(N_REC, **kw)
    for k in ts:
        _equal_to_row0(ts[k], tb[k], f"trajectory.{k}")
    _state_equal(s, b, "trajectory.after")

    for method in ("sliding", "joint"):
        for inplace in ((False, True) if cfg == "M2_cyc" else (False,)):
            s, b = _build(cfg, "single"), _build(cfg, "batch")
            es, eb = _entropy(s, method, inplace), _entropy(b, method, inplace)
            for k in es:
                _equal_to_row0(es[k], eb[k], f"entropy.{method}.{inplace}.{k}")
            _state_equal(s, b, f"entropy.{method}.{inplace}.after")

    s, b = _build(cfg, "single"), _build(cfg, "batch")
    blocks_b = b.init_tangent(3, generator=torch.Generator().manual_seed(0))
    blocks_s = [t[0].clone() for t in blocks_b]
    for _ in range(40):
        s.step_tangent(blocks_s)
        b.step_tangent(blocks_b)
    _equal_to_row0(blocks_s, blocks_b, "tangent")
    _state_equal(s, b, "tangent.after")


# Checks on the configurations themselves, and the round trip
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.mark.parametrize("cfg", ["M1_cyc", "M2_cyc"])
def test_cycle_configurations_oscillate(cfg):
    """ The *_cyc configurations are on an oscillation over the recorded window, and the *_fp
    ones are not: otherwise the cycle literals would be pinning a transient to a fixed point. """
    r = _build(cfg, "single").trajectory(N_REC, pb=False)["r"]
    assert (r.max(0).values - r.min(0).values).min() > 0.05
    r = _build(cfg.replace("cyc", "fp"), "single").trajectory(N_REC, pb=False)["r"]
    assert (r.max(0).values - r.min(0).values).max() < 1e-12


@pytest.mark.parametrize("kind", ["single", "batch"])
def test_save_load_round_trip_is_exact(kind):
    """ RDMNetwork.save/load through an in-memory buffer (no file), for one system and for a
    batch: every saved field is equal after load, and the loaded network continues bit for
    bit, through update and through the EPR. (Before Step 3 only the batch class had
    save/load.) """
    a = _build("M2_cyc", kind)
    buf = io.BytesIO()
    a.save(buf)
    buf.seek(0)
    b = RDMNetwork.load(buf, device="cpu")
    for k in RDMNetwork._SAVE_SCALARS:
        assert getattr(a, k) == getattr(b, k), k
    assert (a.Qm, a.N, a.grid_shape) == (b.Qm, b.N, b.grid_shape)
    for k in RDMNetwork._SAVE_TENSORS + ("R_abs",):
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    for k in RDMNetwork._SAVE_LISTS:
        assert all(torch.equal(x, y) for x, y in zip(getattr(a, k), getattr(b, k))), k
    ta, tb = a.trajectory(40, q=True, pb=False), b.trajectory(40, q=True, pb=False)
    assert torch.equal(ta["m"], tb["m"]) and all(torch.equal(x, y) for x, y in zip(ta["q"], tb["q"]))
    ea, eb = (n.entropy_trajectory(30, pb=False) for n in (a, b))
    assert all(torch.equal(ea[k], eb[k]) for k in ea)


# The recorded literals
# ~~~~~~~~~~~~~~~~~~~~~

# Recorded from the code at 8e8a724 (branch rebuild, before Step 1), float64 reprs. Regenerate
# only when a step is MEANT to move a number, and say which in that step's report.
LIT: dict = {
    'fp_root': {'M1': [0.0065338668097958425], 'M2': [0.006115006911590174, 0.005698000283481283]},
    'trajectory/M1_fp/single': {
        'r': ((140, 1), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591685)),
        'm': ((140, 1), 0.9147413533714183, 3.658965413485673, (0.0065338668097958425, 0.0065338668097958495, 0.0065338668097958425, 0.0065338668097958425)),
        'r_tot': ((140,), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591685)),
        'q[0]': ((140, 200), 280.0, 1120.0, (0.013067733619591685, 0.009621783963384107, 0.0130676647140843, 0.018646672323727742)),
        'S[0]': ((140, 200), 3101.6325543459593, 12406.530217383837, (0.0, 0.1291255636291112, 0.11619787461903268, 0.13050539489445465)),
        'after.p[0]': ((200,), 1.0, 3.972986585493066, (0.006533866809795848, 0.00653383235704214, 0.004810891981692056, 0.00932333616186376)),
        'after.S[0]': ((200,), 22.154518245328276, 88.3653303755948, (0.0, 0.11619787461903267, 0.12912556362911118, 0.13050539489445465)),
        'after.m': ((1,), 0.006533866809795848, 0.006533866809795848, (0.006533866809795848,)),
    },
    'methods/M1_fp/single': {
        'rates': ((1,), 0.013067733619591685, 0.013067733619591685, (0.013067733619591685,)),
        'overlaps': ((1,), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
        'activity': ((), 0.013067733619591685, 0.013067733619591685, (0.013067733619591685,)),
        'age_density[0]': ((200,), 1.9999999999999998, 7.945973170986131, (0.013067733619591685, 0.013067664714084293, 0.009621783963384109, 0.01864667232372748)),
        'total_input': ((1,), 0.13067733619591684, 0.13067733619591684, (0.13067733619591684,)),
        'field[0]': ((200,), -535.9715313268812, -2076.6503185299807, (-16.666666666666664, -2.0458495008159754, -0.5489970487509586, -0.38923115313300677)),
        'firing_prob[0]': ((200,), 2.697385352618303, 10.69505606200569, (1.8600379880104444e-44, 2.3332571407243514e-06, 0.01838184882427377, 0.047234543182813664)),
        'd_firing_prob[0]': ((200,), 15.899416286694231, 63.04435965127735, (1.1160227928062666e-43, 1.3999526512066751e-05, 0.1092711470376055, 0.2766060060561697)),
        'hazard_rate[0]': ((200,), 5.490989459645388, 21.770377859107985, (3.720075976020889e-44, 4.6665197255460555e-06, 0.03710578867717104, 0.09677303197655145)),
        'max_step_prob': 0.047234543182813664,
        'truncation_error': 0.008320801512137972,
        'fp_residual': ((1,), 0.0003996398607001234, 0.0003996398607001234, (0.0003996398607001234,)),
        'fp_state.p[0]': ((200,), 1.0, 3.972838729587903, (0.006600360139299877, 0.006600323585931611, 0.00477671392465138, 0.0076441518269347196)),
        'fp_state.S[0]': ((200,), 23.73504575954827, 94.66940949910352, (0.0, 0.12448755782927326, 0.1383375222844527, 0.13981579221841037)),
        'R[0]': ((200,), -458.1260495722094, -1768.0156489055755, (-16.166666666666664, -1.662047375435008, -0.17812261238006974, -0.01973654802746148)),
        'R_abs': ((1,), 16.166666666666664, 16.166666666666664, (16.166666666666664,)),
        'alpha_int': ((1,), 0.0327838995179941, 0.0327838995179941, (0.0327838995179941,)),
        'stationary_rate': ((1,), 0.013067733619591687, 0.013067733619591687, (0.013067733619591687,)),
        'state_map': ((400,), 23.15451824532828, 92.71314514274601, (0.0065338668097958425, 0.004810891981692054, 0.11619787461903272, 0.13050539489445476)),
        'update.p[0]': ((200,), 0.9999999999999999, 3.9729865854930657, (0.0065338668097958425, 0.006533832357042146, 0.004810891981692054, 0.00932333616186376)),
        'update.S[0]': ((200,), 22.15451824532828, 88.36533037559481, (0.0, 0.11619787461903272, 0.12912556362911115, 0.13050539489445476)),
        'update.m': ((1,), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
        'forward.p[0]': ((200,), 1.0, 3.972986585493066, (0.006533866809795848, 0.006533832357042145, 0.004810891981692053, 0.009323336161863871)),
        'forward.S[0]': ((200,), 22.154518245328283, 88.36533037559481, (0.0, 0.11619787461903273, 0.12912556362911118, 0.1305053948944547)),
        'forward.m': ((1,), 0.006533866809795848, 0.006533866809795848, (0.006533866809795848,)),
    },
    'entropy/M1_fp/single/sliding': {
        'r': ((140, 1), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591675)),
        'm': ((140, 1), 0.9147413533714183, 3.658965413485673, (0.0065338668097958425, 0.0065338668097958495, 0.0065338668097958425, 0.006533866809795837)),
        'sigma': ((140, 1), 3.7686263831826494e-11, 1.5074469797426993e-10, (2.69187450108177e-13, 2.6920826678988874e-13, 2.692117362368407e-13, 2.6916316397951334e-13)),
        'H_fwd': ((140, 1), 4.354987482789685, 17.41994993115874, (0.03110705344849774, 0.03110705344849775, 0.03110705344849772, 0.031107053448497745)),
        'H_rev': ((140, 1), 4.354987482827371, 17.419949931309485, (0.03110705344876693, 0.031107053448766957, 0.031107053448766932, 0.031107053448766908)),
        'r_tot': ((140,), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591675)),
        'm_tot': ((140,), 0.9147413533714183, 3.658965413485673, (0.0065338668097958425, 0.0065338668097958495, 0.0065338668097958425, 0.006533866809795837)),
        'sigma_tot': ((140,), 3.7686263831826494e-11, 1.5074469797426993e-10, (2.69187450108177e-13, 2.6920826678988874e-13, 2.692117362368407e-13, 2.6916316397951334e-13)),
        'H_fwd_tot': ((140,), 4.354987482789685, 17.41994993115874, (0.03110705344849774, 0.03110705344849775, 0.03110705344849772, 0.031107053448497745)),
        'H_rev_tot': ((140,), 4.354987482827371, 17.419949931309485, (0.03110705344876693, 0.031107053448766957, 0.031107053448766932, 0.031107053448766908)),
        'H_fwd_ref': ((140, 1), -92.68561779560265, -370.7424711824106, (-0.6620401271114476, -0.6620401271114476, -0.6620401271114476, -0.6620401271114476)),
        'H_rev_ref': ((140, 1), -92.68561779556497, -370.74247118225986, (-0.6620401271111783, -0.6620401271111783, -0.6620401271111783, -0.6620401271111784)),
        'after.p[0]': ((200,), 1.0, 3.972986585493066, (0.0065338668097958425, 0.0065338323570421405, 0.004810891981692056, 0.00932333616186376)),
        'after.S[0]': ((200,), 22.154518245328276, 88.3653303755948, (0.0, 0.11619787461903267, 0.12912556362911118, 0.13050539489445465)),
        'after.m': ((1,), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
    },
    'entropy/M1_fp/single/joint': {
        'r': ((140, 1), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591675)),
        'm': ((140, 1), 0.9147413533714183, 3.658965413485673, (0.0065338668097958425, 0.0065338668097958495, 0.0065338668097958425, 0.006533866809795837)),
        'sigma': ((140, 1), 3.76859238260252e-11, 1.5074340387055685e-10, (2.69187450108177e-13, 2.692047973429368e-13, 2.6920826678988874e-13, 2.691596945325614e-13)),
        'H_fwd': ((140, 1), 4.354987482789685, 17.419949931158744, (0.03110705344849774, 0.031107053448497752, 0.031107053448497724, 0.03110705344849775)),
        'H_rev': ((140, 1), 4.354987482827371, 17.419949931309485, (0.03110705344876693, 0.031107053448766957, 0.031107053448766932, 0.031107053448766908)),
        'r_tot': ((140,), 1.8294827067428365, 7.317930826971346, (0.013067733619591685, 0.013067733619591699, 0.013067733619591685, 0.013067733619591675)),
        'm_tot': ((140,), 0.9147413533714183, 3.658965413485673, (0.0065338668097958425, 0.0065338668097958495, 0.0065338668097958425, 0.006533866809795837)),
        'sigma_tot': ((140,), 3.76859238260252e-11, 1.5074340387055685e-10, (2.69187450108177e-13, 2.692047973429368e-13, 2.6920826678988874e-13, 2.691596945325614e-13)),
        'H_fwd_tot': ((140,), 4.354987482789685, 17.419949931158744, (0.03110705344849774, 0.031107053448497752, 0.031107053448497724, 0.03110705344849775)),
        'H_rev_tot': ((140,), 4.354987482827371, 17.419949931309485, (0.03110705344876693, 0.031107053448766957, 0.031107053448766932, 0.031107053448766908)),
        'H_fwd_ref': ((140, 1), -92.68561779560265, -370.7424711824106, (-0.6620401271114476, -0.6620401271114476, -0.6620401271114476, -0.6620401271114476)),
        'H_rev_ref': ((140, 1), -92.68561779556497, -370.74247118225986, (-0.6620401271111783, -0.6620401271111783, -0.6620401271111783, -0.6620401271111784)),
        'after.p[0]': ((200,), 1.0, 3.972986585493066, (0.0065338668097958425, 0.0065338323570421405, 0.004810891981692056, 0.00932333616186376)),
        'after.S[0]': ((200,), 22.154518245328276, 88.3653303755948, (0.0, 0.11619787461903267, 0.12912556362911118, 0.13050539489445465)),
        'after.m': ((1,), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
    },
    'trajectory/M1_fp/batch': {
        'r': ((2, 140, 1), 3.6897623813520264, 14.759647711512304, (0.013067733619591685, 0.013067733619591687, 0.01333294540703631, 0.013260435353456775)),
        'm': ((2, 140, 1), 1.8448811906760132, 7.379823855756152, (0.0065338668097958425, 0.006533866809795843, 0.006666472703518155, 0.006630217676728388)),
        'r_tot': ((2, 140), 3.6897623813520264, 14.759647711512304, (0.013067733619591685, 0.013067733619591687, 0.01333294540703631, 0.013260435353456775)),
        'q[0]': ((2, 140, 200), 560.0, 2239.998579586871, (0.013067733619591685, 0.0130676647140843, 0.00949059516350201, 0.013359147556350592)),
        'S[0]': ((2, 140, 200), 6561.861844208572, 26247.45147960468, (0.0, 0.11619787461903269, 0.1442187012073692, 0.14567584012824478)),
        'after.p[0]': ((2, 200), 1.9999999999999998, 7.967626983667033, (0.0065338668097958425, 0.004810891981692056, 0.006646748429308432, 0.006679569355888537)),
        'after.S[0]': ((2, 200), 46.881644591743566, 187.41011390871387, (0.0, 0.12912556362911115, 0.1296614349584798, 0.14567583400841427)),
        'after.m': ((2, 1), 0.01316469911131022, 0.0197955314128246, (0.0065338668097958425, 0.006630832301514378)),
    },
    'methods/M1_fp/batch': {
        'rates': ((2, 1), 0.02613546723918337, 0.03920320085877506, (0.013067733619591685, 0.013067733619591685)),
        'overlaps': ((2, 1), 0.013067733619591685, 0.01960160042938753, (0.0065338668097958425, 0.0065338668097958425)),
        'activity': ((2,), 0.02613546723918337, 0.03920320085877506, (0.013067733619591685, 0.013067733619591685)),
        'age_density[0]': ((2, 200), 4.0, 15.933178890510758, (0.013067733619591685, 0.009621783963384109, 0.01325351988973665, 0.014080796222900753)),
        'total_input': ((2, 1), 0.27442240601142537, 0.4181674758269339, (0.13067733619591684, 0.14374506981550852)),
        'field[0]': ((2, 200), -1070.7276108292294, -4250.19322755226, (-16.666666666666664, -0.5489970487509586, -2.034229713354072, -0.37618061364356126)),
        'firing_prob[0]': ((2, 200), 5.609267991320543, 22.393190706496142, (1.8600379880104444e-44, 0.01838184882427377, 2.5017333869928292e-06, 0.05098210520225966)),
        'd_firing_prob[0]': ((2, 200), 33.03836322277231, 131.89595787045735, (1.1160227928062666e-43, 0.1092711470376055, 1.5010381545931499e-05, 0.29795910981869694)),
        'hazard_rate[0]': ((2, 200), 11.427119717194353, 45.61875721980086, (3.720075976020889e-44, 0.03710578867717104, 5.0034730326660364e-06, 0.10465524814373127)),
        'max_step_prob': ((2,), 0.09821664838507332, 0.14919875358733298, (0.047234543182813664, 0.05098210520225966)),
        'truncation_error': ((2,), 0.015093692426324839, 0.021866583340511708, (0.008320801512137972, 0.006772890914186868)),
        'fp_residual': ((2, 1), 0.0012957547650909408, 0.002191869669481758, (0.0003996398607001234, 0.0008961149043908174)),
        'fp_state.p[0]': ((2, 200), 2.0, 7.975850117315328, (0.006600360139299877, 0.00477671392465138, 0.006803841110195146, 0.003843967678539284)),
        'fp_state.S[0]': ((2, 200), 52.454451128601676, 209.7052932812858, (0.0, 0.1383375222844527, 0.15062994497342067, 0.16917710858427656)),
        'R[0]': ((2, 200), -917.2520991444188, -3637.6727323408336, (-16.166666666666664, -0.17812261238006974, -1.662047375435008, -0.01973654802746148)),
        'R_abs': ((2, 1), 33.33333333333333, 50.49999999999999, (16.166666666666664, 17.166666666666664)),
        'alpha_int': ((2, 1), 0.0655677990359882, 0.09835169855398229, (0.0327838995179941, 0.0327838995179941)),
        'update.p[0]': ((2, 200), 2.0, 7.96658944525538, (0.0065338668097958425, 0.004810891981692054, 0.006626759944868326, 0.007040398111450408)),
        'update.S[0]': ((2, 200), 46.524488315189394, 185.97950478857305, (0.0, 0.12912556362911115, 0.127817662080936, 0.14355593438390024)),
        'update.m': ((2, 1), 0.013160664184896118, 0.019787461559996396, (0.0065338668097958425, 0.006626797375100276)),
        'forward.p[0]': ((2, 200), 2.0, 7.966738133492843, (0.006533866809795848, 0.004810891981692053, 0.006626759710304244, 0.007014261465596139)),
        'forward.S[0]': ((2, 200), 46.75515636975048, 186.90320170833968, (0.0, 0.12912556362911118, 0.12902456999614495, 0.14476284229910918)),
        'forward.m': ((2, 1), 0.013194889402530025, 0.019855911995264205, (0.006533866809795846, 0.006661022592734179)),
    },
    'entropy/M1_fp/batch/sliding': {
        'r': ((2, 140, 1), 3.6899482424826355, 14.759833572642913, (0.013067733619591685, 0.013067733619591685, 0.013332945407036306, 0.013260435353456775)),
        'm': ((2, 140, 1), 1.8449741212413178, 7.3799167863214565, (0.0065338668097958425, 0.0065338668097958425, 0.006666472703518153, 0.006630217676728388)),
        'sigma': ((2, 140, 1), -3.2460660571997924e-07, 7.749416017113417e-05, (2.69187450108177e-13, 2.6921520568379265e-13, 1.2602225011343338e-05, -2.35088034376274e-05)),
        'H_fwd': ((2, 140, 1), 8.739976508597497, 34.95989633406582, (0.03110705344849774, 0.03110705344849772, 0.031403775895740046, 0.031274383770150974)),
        'H_rev': ((2, 140, 1), 8.739976183990892, 34.95997382822599, (0.03110705344876693, 0.031107053448766936, 0.03141637812075139, 0.031250874966713346)),
        'r_tot': ((2, 140), 3.6899482424826355, 14.759833572642913, (0.013067733619591685, 0.013067733619591685, 0.013332945407036306, 0.013260435353456775)),
        'm_tot': ((2, 140), 1.8449741212413178, 7.3799167863214565, (0.0065338668097958425, 0.0065338668097958425, 0.006666472703518153, 0.006630217676728388)),
        'sigma_tot': ((2, 140), -3.2460660571997924e-07, 7.749416017113417e-05, (2.69187450108177e-13, 2.6921520568379265e-13, 1.2602225011343338e-05, -2.35088034376274e-05)),
        'H_fwd_tot': ((2, 140), 8.739976508597497, 34.95989633406582, (0.03110705344849774, 0.03110705344849772, 0.031403775895740046, 0.031274383770150974)),
        'H_rev_tot': ((2, 140), 8.739976183990892, 34.95997382822599, (0.03110705344876693, 0.031107053448766936, 0.03141637812075139, 0.031250874966713346)),
        'H_fwd_ref': ((2, 140, 1), -185.34123404818718, -741.3649458930729, (-0.6620401271114476, -0.6620401271114476, -0.6617434046642052, -0.6618727967897943)),
        'H_rev_ref': ((2, 140, 1), -185.3412343727938, -741.3648683989127, (-0.6620401271111783, -0.6620401271111783, -0.6617308024391939, -0.661896305593232)),
        'after.p[0]': ((2, 200), 1.9999999999999998, 7.967626983667033, (0.0065338668097958425, 0.004810891981692056, 0.006646748429308432, 0.006679569355888537)),
        'after.S[0]': ((2, 200), 46.881644591743566, 187.41011390871387, (0.0, 0.12912556362911115, 0.1296614349584798, 0.14567583400841427)),
        'after.m': ((2, 1), 0.013164699111310223, 0.019795531412824603, (0.0065338668097958425, 0.00663083230151438)),
    },
    'entropy/M1_fp/batch/joint': {
        'r': ((2, 140, 1), 3.6899482424826355, 14.759833572642913, (0.013067733619591685, 0.013067733619591685, 0.013332945407036306, 0.013260435353456775)),
        'm': ((2, 140, 1), 1.8449741212413178, 7.3799167863214565, (0.0065338668097958425, 0.0065338668097958425, 0.006666472703518153, 0.006630217676728388)),
        'sigma': ((2, 140, 1), -3.2460660611202674e-07, 7.74941601695868e-05, (2.69187450108177e-13, 2.692047973429368e-13, 1.2602225011350277e-05, -2.350880343763434e-05)),
        'H_fwd': ((2, 140, 1), 8.739976508597499, 34.959896334065824, (0.03110705344849774, 0.031107053448497724, 0.031403775895740046, 0.03127438377015098)),
        'H_rev': ((2, 140, 1), 8.739976183990892, 34.95997382822599, (0.03110705344876693, 0.03110705344876693, 0.0314163781207514, 0.031250874966713346)),
        'r_tot': ((2, 140), 3.6899482424826355, 14.759833572642913, (0.013067733619591685, 0.013067733619591685, 0.013332945407036306, 0.013260435353456775)),
        'm_tot': ((2, 140), 1.8449741212413178, 7.3799167863214565, (0.0065338668097958425, 0.0065338668097958425, 0.006666472703518153, 0.006630217676728388)),
        'sigma_tot': ((2, 140), -3.2460660611202674e-07, 7.74941601695868e-05, (2.69187450108177e-13, 2.692047973429368e-13, 1.2602225011350277e-05, -2.350880343763434e-05)),
        'H_fwd_tot': ((2, 140), 8.739976508597499, 34.959896334065824, (0.03110705344849774, 0.031107053448497724, 0.031403775895740046, 0.03127438377015098)),
        'H_rev_tot': ((2, 140), 8.739976183990892, 34.95997382822599, (0.03110705344876693, 0.03110705344876693, 0.0314163781207514, 0.031250874966713346)),
        'H_fwd_ref': ((2, 140, 1), -185.34123404818718, -741.3649458930729, (-0.6620401271114476, -0.6620401271114476, -0.6617434046642052, -0.6618727967897943)),
        'H_rev_ref': ((2, 140, 1), -185.3412343727938, -741.3648683989127, (-0.6620401271111783, -0.6620401271111783, -0.6617308024391939, -0.661896305593232)),
        'after.p[0]': ((2, 200), 1.9999999999999998, 7.967626983667033, (0.0065338668097958425, 0.004810891981692056, 0.006646748429308432, 0.006679569355888537)),
        'after.S[0]': ((2, 200), 46.881644591743566, 187.41011390871387, (0.0, 0.12912556362911115, 0.1296614349584798, 0.14567583400841427)),
        'after.m': ((2, 1), 0.013164699111310223, 0.019795531412824603, (0.0065338668097958425, 0.00663083230151438)),
    },
    'fixed_points/M1_fp': {
        'roots': ((1, 1), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
        'roots_newton': ((1, 1), 0.0065338668097958425, 0.0065338668097958425, (0.0065338668097958425,)),
        'root0.lead_abs': ((4,), 3.966959525496672, 9.888151548431752, (0.9990516977016497, 0.9990516977016497, 0.9844280650466862, 0.9844280650466862)),
        'root0.stable': True,
    },
    'fp_sweep/M1_fp': {
        'roots': ((2, 5, 1), 0.06588826573275137, 0.22423994254527185, (0.0065338668097958425, 0.0065338668097958425, 0.006643786336754431, 0.006643786336754433)),
        'converged': ((2, 5), (True, True, True, True, True, True, True, True, True, True)),
    },
    'trajectory/M1_cyc/single': {
        'r': ((140, 1), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908446, 0.0003207332692202176, 0.002750446790299433)),
        'm': ((140, 1), 1.0070636230368752, 4.01366141664644, (0.0013612166664155431, 0.02908130123954223, 0.0001603666346101088, 0.0013752233951497164)),
        'r_tot': ((140,), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908446, 0.0003207332692202176, 0.002750446790299433)),
        'q[0]': ((140, 200), 280.0, 1120.03624614513, (0.0027224333328310863, 0.07183758578322792, 0.014395430378377474, 1.1847211522919565e-07)),
        'S[0]': ((140, 200), 5805.7007026841, 23222.78986300689, (0.0, 0.2958831004643424, 0.24049155919894757, 0.07076817840824495)),
        'after.p[0]': ((200,), 1.0, 3.9575080733460397, (0.0014602369691376705, 0.00013495634467502175, 0.001610666955264851, 7.697475967738399e-08)),
        'after.S[0]': ((200,), 9.0576157069342, 36.08808990025952, (0.0, 0.018308528019460445, 0.06968156735392202, 0.0699808347721472)),
        'after.m': ((1,), 0.0014602369691376705, 0.0014602369691376705, (0.0014602369691376705,)),
    },
    'methods/M1_cyc/single': {
        'rates': ((1,), 0.0027224333328310863, 0.0027224333328310863, (0.0027224333328310863,)),
        'overlaps': ((1,), 0.0013612166664155431, 0.0013612166664155431, (0.0013612166664155431,)),
        'activity': ((), 0.0027224333328310863, 0.0027224333328310863, (0.0027224333328310863,)),
        'age_density[0]': ((200,), 2.0, 8.02781794632677, (0.0027224333328310863, 0.00026823014477565853, 0.002978230583406843, 1.054416263190916e-07)),
        'total_input': ((1,), 0.046281366658128466, 0.046281366658128466, (0.046281366658128466,)),
        'field[0]': ((200,), -556.0212076783256, -2135.73860363272, (-23.666666666666664, -2.1449108171611506, -0.6076646587283293, -0.449001056013442)),
        'firing_prob[0]': ((200,), 1.8985581699142955, 7.52773964945323, (1.0694329824497696e-62, 1.2877517012446502e-06, 0.01296305431515994, 0.03323976813850776)),
        'd_firing_prob[0]': ((200,), 11.250692628224483, 44.610518331518996, (6.416597894698618e-62, 7.726505232552433e-06, 0.0772720110080812, 0.19608661341363007)),
        'hazard_rate[0]': ((200,), 3.844433699916279, 15.242488073355913, (2.138865964899539e-62, 2.575505060795168e-06, 0.026095615888848428, 0.06760952955766024)),
        'max_step_prob': 0.03323976813850776,
        'truncation_error': 3.5342294625931084e-08,
        'fp_residual': ((1,), -0.0002471357567855187, -0.0002471357567855187, (-0.0002471357567855187,)),
        'fp_state.p[0]': ((200,), 1.0, 3.9709255096657836, (0.007247135756785519, 0.007247068535847931, 0.00407593420831166, 0.0004946144139281866)),
        'fp_state.S[0]': ((200,), 40.34957779123206, 160.937996148476, (0.0, 0.21162884830976456, 0.23517378788356957, 0.23768684677129762)),
        'R[0]': ((200,), -465.1260495722094, -1775.0156489055755, (-23.166666666666664, -1.662047375435008, -0.17812261238006974, -0.01973654802746148)),
        'R_abs': ((1,), 23.166666666666664, 23.166666666666664, (23.166666666666664,)),
        'alpha_int': ((1,), 0.0327838995179941, 0.0327838995179941, (0.0327838995179941,)),
        'stationary_rate': ((1,), 0.01216122876590269, 0.01216122876590269, (0.01216122876590269,)),
        'state_map': ((400,), 10.039872616497101, 40.19216042608101, (0.0014456011207092628, 0.0015911308954082012, 0.018075474541604648, 0.0699336196317719)),
        'update.p[0]': ((200,), 1.0, 3.9608959967288895, (0.0014456011207092626, 0.00013096403171160864, 0.0015911308954082012, 6.852381062572022e-08)),
        'update.S[0]': ((200,), 9.039872616497101, 36.01708211594664, (0.0, 0.018075474541604648, 0.06963971746838596, 0.0699336196317719)),
        'update.m': ((1,), 0.0014456011207092626, 0.0014456011207092626, (0.0014456011207092626,)),
        'forward.p[0]': ((200,), 0.9999999999999999, 3.9992839458409875, (0.0049525649280042034, 0.00016132805463557666, 0.005728725841612417, 1.4025841746256873e-05)),
        'forward.S[0]': ((200,), 13.289723382956815, 52.99733612369024, (0.0, 0.057693255347868705, 0.08358555892581664, 0.0845635002069356)),
        'forward.m': ((1,), 0.0049525649280042034, 0.0049525649280042034, (0.0049525649280042034,)),
    },
    'entropy/M1_cyc/single/sliding': {
        'r': ((140, 1), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908448, 0.0003207332692201953, 0.0027504467902994406)),
        'm': ((140, 1), 1.0070636230368752, 4.01366141664644, (0.0013612166664155431, 0.02908130123954224, 0.00016036663461009766, 0.0013752233951497203)),
        'sigma': ((140, 1), 0.9222501435718918, 3.4734049670369664, (-0.004336213356325636, -0.0013453473335626603, 0.0007190866522317836, -0.004368912079605458)),
        'H_fwd': ((140, 1), 3.642064172147644, 14.667479528015665, (0.009333489504726502, 0.12270025925646391, 0.0009763515780326928, 0.009418527393230014)),
        'H_rev': ((140, 1), 4.564314315719535, 18.14088449505263, (0.004997276148400866, 0.12135491192290125, 0.0016954382302644764, 0.005049615313624556)),
        'r_tot': ((140,), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908448, 0.0003207332692201953, 0.0027504467902994406)),
        'm_tot': ((140,), 1.0070636230368752, 4.01366141664644, (0.0013612166664155431, 0.02908130123954224, 0.00016036663461009766, 0.0013752233951497203)),
        'sigma_tot': ((140,), 0.9222501435718918, 3.4734049670369664, (-0.004336213356325636, -0.0013453473335626603, 0.0007190866522317836, -0.004368912079605458)),
        'H_fwd_tot': ((140,), 3.642064172147644, 14.667479528015665, (0.009333489504726502, 0.12270025925646391, 0.0009763515780326928, 0.009418527393230014)),
        'H_rev_tot': ((140,), 4.564314315719535, 18.14088449505263, (0.004997276148400866, 0.12135491192290125, 0.0016954382302644764, 0.005049615313624556)),
        'H_fwd_ref': ((140, 1), -93.39854110624469, -373.4949415855537, (-0.6838136910552188, -0.5704469213034814, -0.6921708289819126, -0.6837286531667153)),
        'H_rev_ref': ((140, 1), -92.4762909626728, -370.02153661851673, (-0.6881499044115444, -0.571792268637044, -0.6914517423296808, -0.6880975652463207)),
        'after.p[0]': ((200,), 1.0, 3.95750807334604, (0.001460236969137667, 0.00013495634467505907, 0.0016106669552648483, 7.697475967738399e-08)),
        'after.S[0]': ((200,), 9.0576157069342, 36.08808990025952, (0.0, 0.018308528019460445, 0.06968156735392202, 0.0699808347721472)),
        'after.m': ((1,), 0.001460236969137667, 0.001460236969137667, (0.001460236969137667,)),
    },
    'entropy/M1_cyc/single/joint': {
        'r': ((140, 1), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908448, 0.0003207332692201953, 0.0027504467902994406)),
        'm': ((140, 1), 1.0070636230368752, 4.01366141664644, (0.0013612166664155431, 0.02908130123954224, 0.00016036663461009766, 0.0013752233951497203)),
        'sigma': ((140, 1), 0.922250143571892, 3.473404967036967, (-0.004336213356325635, -0.0013453473335626465, 0.0007190866522317836, -0.004368912079605458)),
        'H_fwd': ((140, 1), 3.6420641721476437, 14.667479528015665, (0.009333489504726502, 0.12270025925646391, 0.0009763515780326928, 0.009418527393230014)),
        'H_rev': ((140, 1), 4.564314315719535, 18.14088449505263, (0.004997276148400867, 0.12135491192290127, 0.0016954382302644764, 0.005049615313624556)),
        'r_tot': ((140,), 2.0141272460737505, 8.02732283329288, (0.0027224333328310863, 0.05816260247908448, 0.0003207332692201953, 0.0027504467902994406)),
        'm_tot': ((140,), 1.0070636230368752, 4.01366141664644, (0.0013612166664155431, 0.02908130123954224, 0.00016036663461009766, 0.0013752233951497203)),
        'sigma_tot': ((140,), 0.922250143571892, 3.473404967036967, (-0.004336213356325635, -0.0013453473335626465, 0.0007190866522317836, -0.004368912079605458)),
        'H_fwd_tot': ((140,), 3.6420641721476437, 14.667479528015665, (0.009333489504726502, 0.12270025925646391, 0.0009763515780326928, 0.009418527393230014)),
        'H_rev_tot': ((140,), 4.564314315719535, 18.14088449505263, (0.004997276148400867, 0.12135491192290127, 0.0016954382302644764, 0.005049615313624556)),
        'H_fwd_ref': ((140, 1), -93.39854110624469, -373.4949415855537, (-0.6838136910552188, -0.5704469213034814, -0.6921708289819126, -0.6837286531667153)),
        'H_rev_ref': ((140, 1), -92.4762909626728, -370.02153661851673, (-0.6881499044115444, -0.571792268637044, -0.6914517423296808, -0.6880975652463207)),
        'after.p[0]': ((200,), 1.0, 3.95750807334604, (0.001460236969137667, 0.00013495634467505907, 0.0016106669552648483, 7.697475967738399e-08)),
        'after.S[0]': ((200,), 9.0576157069342, 36.08808990025952, (0.0, 0.018308528019460445, 0.06968156735392202, 0.0699808347721472)),
        'after.m': ((1,), 0.001460236969137667, 0.001460236969137667, (0.001460236969137667,)),
    },
    'trajectory/M1_cyc/batch': {
        'r': ((2, 140, 1), 4.0336630445236885, 16.33242048178541, (0.002722433332831078, 0.00032073326922021797, 0.14560049032482614, 0.00424987927923005)),
        'm': ((2, 140, 1), 2.0168315222618443, 8.166210240892704, (0.001361216666415539, 0.00016036663461010899, 0.07280024516241307, 0.002124939639615025)),
        'r_tot': ((2, 140), 4.0336630445236885, 16.33242048178541, (0.002722433332831078, 0.00032073326922021797, 0.14560049032482614, 0.00424987927923005)),
        'q[0]': ((2, 140, 200), 560.0, 2240.3126296752466, (0.0027224333328310785, 0.014395430378377478, 0.00491063846454442, 1.5660540686468494e-07)),
        'S[0]': ((2, 140, 200), 11960.940564962344, 47843.76450046755, (0.0, 0.24049155919894755, 0.6242684814330153, 0.07007156426590291)),
        'after.p[0]': ((2, 200), 2.0, 8.018849136036575, (0.0014602369691376679, 0.0016106669552648529, 0.0001090317299939727, 1.1489512974360849e-07)),
        'after.S[0]': ((2, 200), 18.597413474612715, 74.309199712369, (0.0, 0.06968156735392209, 0.027965378112350368, 0.07028206736885968)),
        'after.m': ((2, 1), 0.003726460400867844, 0.0059926838325980196, (0.001460236969137668, 0.002266223431730176)),
    },
    'methods/M1_cyc/batch': {
        'rates': ((2, 1), 0.006614651580942276, 0.010506869829053475, (0.002722433332831078, 0.0038922182481111985)),
        'overlaps': ((2, 1), 0.003307325790471138, 0.005253434914526738, (0.001361216666415539, 0.0019461091240555992)),
        'activity': ((2,), 0.006614651580942276, 0.010506869829053475, (0.002722433332831078, 0.0038922182481111985)),
        'age_density[0]': ((2, 200), 4.0, 16.162183011877627, (0.0027224333328310785, 0.0029782305834068407, 0.00020967375433593238, 8.674522145035723e-08)),
        'total_input': ((2, 1), 0.11634129512412991, 0.18640122359013148, (0.04628136665812833, 0.07005992846600158)),
        'field[0]': ((2, 200), -1112.8692682934357, -4404.70944145994, (-23.666666666666664, -0.6076646587283293, -2.1378298303660355, -0.44980854451783137)),
        'firing_prob[0]': ((2, 200), 3.786761945136327, 15.113401053545434, (1.0694329824497696e-62, 0.01296305431515994, 1.343641846577609e-06, 0.03308180049672204)),
        'd_firing_prob[0]': ((2, 200), 22.44063511635752, 89.5640959099195, (6.416597894698618e-62, 0.0772720110080812, 8.061845663342993e-06, 0.19517077048571485)),
        'hazard_rate[0]': ((2, 200), 7.667693611574168, 30.602322586196586, (2.138865964899539e-62, 0.026095615888848428, 2.687285498530247e-06, 0.06728275828157865)),
        'max_step_prob': ((2,), 0.0663215686352298, 0.09940336913195183, (0.03323976813850776, 0.03308180049672204)),
        'truncation_error': ((2,), 1.2693197027539602e-07, 2.1852164599928544e-07, (3.5342294551506574e-08, 9.158967572388943e-08)),
        'fp_residual': ((2, 1), -3.089508382341476e-05, 0.00018534558913868918, (-0.0002471357567855187, 0.00021624067296210393)),
        'fp_state.p[0]': ((2, 200), 1.9999999999999998, 7.985623311602429, (0.007247135756785519, 0.00407593420831166, 0.007483674006517006, 0.00010621438462478307)),
        'fp_state.S[0]': ((2, 200), 87.34496839513764, 349.17853324659245, (0.0, 0.23517378788356957, 0.2464853645019611, 0.27683526859245255)),
        'R[0]': ((2, 200), -931.2520991444188, -3679.6727323408336, (-23.166666666666664, -0.17812261238006974, -1.662047375435008, -0.01973654802746148)),
        'R_abs': ((2, 1), 47.33333333333333, 71.5, (23.166666666666664, 24.166666666666664)),
        'alpha_int': ((2, 1), 0.0655677990359882, 0.09835169855398229, (0.0327838995179941, 0.0327838995179941)),
        'update.p[0]': ((2, 200), 2.0, 8.075637453239379, (0.0014456011207092585, 0.0015911308954082036, 0.00010545368513813009, 6.441185163019725e-08)),
        'update.S[0]': ((2, 200), 18.40312811838717, 73.54057139543238, (0.0, 0.06963971746838597, 0.025706727375438457, 0.06993216628204663)),
        'update.m': ((2, 1), 0.0035211026126111883, 0.0055966041045131185, (0.0014456011207092587, 0.0020755014919019298)),
        'forward.p[0]': ((2, 200), 2.0, 8.075556848027155, (0.0049525649280042034, 0.005728725841612412, 0.00017519826019457332, 2.543539234300951e-05)),
        'forward.S[0]': ((2, 200), 32.20400753555423, 128.75587679755387, (0.0, 0.08358555892581666, 0.08992637904039587, 0.11288674559050592)),
        'forward.m': ((2, 1), 0.012546774505049072, 0.02014098408209394, (0.0049525649280042034, 0.007594209577044868)),
    },
    'entropy/M1_cyc/batch/sliding': {
        'r': ((2, 140, 1), 4.033663044523688, 16.332420481785405, (0.0027224333328310785, 0.000320733269220218, 0.14560049032482622, 0.00424987927923005)),
        'm': ((2, 140, 1), 2.016831522261844, 8.166210240892703, (0.0013612166664155392, 0.000160366634610109, 0.07280024516241311, 0.002124939639615025)),
        'sigma': ((2, 140, 1), 2.1976094817273766, 9.174259114506228, (-0.00433621335632563, 0.000719086652231847, 0.17424076727339238, -0.006643658187751977)),
        'H_fwd': ((2, 140, 1), 7.191388167404863, 28.88108944266632, (0.009333489504726487, 0.0009763515780326941, 0.1256567089219799, 0.014506965616909891)),
        'H_rev': ((2, 140, 1), 9.38899764913224, 38.05534855717254, (0.004997276148400857, 0.001695438230264541, 0.29989747619537227, 0.007863307429157914)),
        'r_tot': ((2, 140), 4.033663044523688, 16.332420481785405, (0.0027224333328310785, 0.000320733269220218, 0.14560049032482622, 0.00424987927923005)),
        'm_tot': ((2, 140), 2.016831522261844, 8.166210240892703, (0.0013612166664155392, 0.000160366634610109, 0.07280024516241311, 0.002124939639615025)),
        'sigma_tot': ((2, 140), 2.1976094817273766, 9.174259114506228, (-0.00433621335632563, 0.000719086652231847, 0.17424076727339238, -0.006643658187751977)),
        'H_fwd_tot': ((2, 140), 7.191388167404863, 28.88108944266632, (0.009333489504726487, 0.0009763515780326941, 0.1256567089219799, 0.014506965616909891)),
        'H_rev_tot': ((2, 140), 9.38899764913224, 38.05534855717254, (0.004997276148400857, 0.001695438230264541, 0.29989747619537227, 0.007863307429157914)),
        'H_fwd_ref': ((2, 140, 1), -186.88982238937982, -747.4437527844724, (-0.6838136910552188, -0.6921708289819126, -0.5674904716379653, -0.6786402149430354)),
        'H_rev_ref': ((2, 140, 1), -184.69221290765245, -738.2694936699662, (-0.6881499044115444, -0.6914517423296808, -0.3932497043645729, -0.6852838731307874)),
        'after.p[0]': ((2, 200), 2.0, 8.018849136036575, (0.0014602369691376679, 0.0016106669552648529, 0.0001090317299939727, 1.1489512974360849e-07)),
        'after.S[0]': ((2, 200), 18.597413474612715, 74.309199712369, (0.0, 0.06968156735392209, 0.027965378112350368, 0.07028206736885968)),
        'after.m': ((2, 1), 0.003726460400867845, 0.005992683832598022, (0.0014602369691376679, 0.002266223431730177)),
    },
    'entropy/M1_cyc/batch/joint': {
        'r': ((2, 140, 1), 4.033663044523688, 16.332420481785405, (0.0027224333328310785, 0.000320733269220218, 0.14560049032482622, 0.00424987927923005)),
        'm': ((2, 140, 1), 2.016831522261844, 8.166210240892703, (0.0013612166664155392, 0.000160366634610109, 0.07280024516241311, 0.002124939639615025)),
        'sigma': ((2, 140, 1), 2.1976094817273766, 9.174259114506228, (-0.00433621335632563, 0.0007190866522318471, 0.17424076727339233, -0.006643658187751977)),
        'H_fwd': ((2, 140, 1), 7.191388167404863, 28.88108944266632, (0.009333489504726487, 0.000976351578032694, 0.1256567089219799, 0.014506965616909891)),
        'H_rev': ((2, 140, 1), 9.38899764913224, 38.05534855717254, (0.004997276148400857, 0.001695438230264541, 0.2998974761953722, 0.007863307429157914)),
        'r_tot': ((2, 140), 4.033663044523688, 16.332420481785405, (0.0027224333328310785, 0.000320733269220218, 0.14560049032482622, 0.00424987927923005)),
        'm_tot': ((2, 140), 2.016831522261844, 8.166210240892703, (0.0013612166664155392, 0.000160366634610109, 0.07280024516241311, 0.002124939639615025)),
        'sigma_tot': ((2, 140), 2.1976094817273766, 9.174259114506228, (-0.00433621335632563, 0.0007190866522318471, 0.17424076727339233, -0.006643658187751977)),
        'H_fwd_tot': ((2, 140), 7.191388167404863, 28.88108944266632, (0.009333489504726487, 0.000976351578032694, 0.1256567089219799, 0.014506965616909891)),
        'H_rev_tot': ((2, 140), 9.38899764913224, 38.05534855717254, (0.004997276148400857, 0.001695438230264541, 0.2998974761953722, 0.007863307429157914)),
        'H_fwd_ref': ((2, 140, 1), -186.88982238937982, -747.4437527844724, (-0.6838136910552188, -0.6921708289819126, -0.5674904716379653, -0.6786402149430354)),
        'H_rev_ref': ((2, 140, 1), -184.69221290765245, -738.2694936699662, (-0.6881499044115444, -0.6914517423296808, -0.39324970436457296, -0.6852838731307874)),
        'after.p[0]': ((2, 200), 2.0, 8.018849136036575, (0.0014602369691376679, 0.0016106669552648529, 0.0001090317299939727, 1.1489512974360849e-07)),
        'after.S[0]': ((2, 200), 18.597413474612715, 74.309199712369, (0.0, 0.06968156735392209, 0.027965378112350368, 0.07028206736885968)),
        'after.m': ((2, 1), 0.003726460400867845, 0.005992683832598022, (0.0014602369691376679, 0.002266223431730177)),
    },
    'fixed_points/M1_cyc': {
        'roots': ((1, 1), 0.007312316325683587, 0.007312316325683587, (0.007312316325683587,)),
        'roots_newton': ((1, 1), 0.007312316325683587, 0.007312316325683587, (0.007312316325683587,)),
        'root0.lead_abs': ((4,), 3.9864278357685086, 9.939828578624436, (1.0031672116413362, 1.0031672116413362, 0.9900467062429182, 0.9900467062429182)),
        'root0.stable': False,
    },
    'fp_sweep/M1_cyc': {
        'roots': ((2, 5, 1), 0.07368665903657613, 0.250760039036255, (0.007312316325683588, 0.007312316325683588, 0.007425015481631641, 0.00742501548163164)),
        'converged': ((2, 5), (True, True, True, True, True, True, True, True, True, True)),
    },
    'trajectory/M2_fp/single': {
        'r': ((140, 2), 3.3076420146200083, 13.230568058480035, (0.012230013823180348, 0.011396000566962564, 0.012230013823180345, 0.011396000566962567)),
        'm': ((140, 2), 1.6538210073100041, 6.6152840292400175, (0.006115006911590174, 0.005698000283481282, 0.006115006911590172, 0.0056980002834812835)),
        'r_tot': ((140,), 1.6888495640711514, 6.7553982562846056, (0.012063211171936792, 0.012063211171936797, 0.012063211171936788, 0.012063211171936799)),
        'q[0]': ((140, 200), 280.0, 1120.0, (0.012230013823180348, 0.00980804684281458, 0.01222996570071141, 0.05180659310374658)),
        'q[1]': ((140, 150), 280.0, 1120.0, (0.011396000566962565, 0.011281724810027318, 0.011396000546875902, 0.3779747651981089)),
        'S[0]': ((140, 200), 1781.270311999358, 7125.081247997432, (0.0, 0.07415692509753831, 0.06673254189516734, 0.07494936341041354)),
        'S[1]': ((140, 150), 1123.833668094804, 4495.334672379216, (0.0, 0.061566436910309254, 0.056896119055900446, 0.06194804224733474)),
        'after.p[0]': ((200,), 1.0, 3.9737774006732485, (0.0061150069115901776, 0.0061149828503557005, 0.004904023421407297, 0.02590329655187329)),
        'after.p[1]': ((150,), 1.0, 3.784748592136602, (0.0056980002834812835, 0.005698000273437953, 0.005640862405013657, 0.18898738259905445)),
        'after.S[0]': ((200,), 12.72335937142398, 50.74828718428044, (0.0, 0.06673254189516739, 0.07415692509753827, 0.07494936341041354)),
        'after.S[1]': ((150,), 8.027383343534316, 31.991355562918944, (0.0, 0.05689611905590048, 0.06156643691030926, 0.061948042247334745)),
        'after.m': ((2,), 0.011813007195071461, 0.017511007478552745, (0.0061150069115901776, 0.0056980002834812835)),
    },
    'methods/M2_fp/single': {
        'rates': ((2,), 0.023626014390142912, 0.03502201495710548, (0.012230013823180348, 0.011396000566962565)),
        'overlaps': ((2,), 0.011813007195071456, 0.01751100747855274, (0.006115006911590174, 0.005698000283481283)),
        'activity': ((), 0.012063211171936792, 0.012063211171936792, (0.012063211171936792,)),
        'age_density[0]': ((200,), 1.9999999999999996, 7.947554801346495, (0.012230013823180348, 0.012229965700711408, 0.009808046842814578, 0.05180659310374637)),
        'age_density[1]': ((150,), 2.0, 7.569497184273205, (0.011396000566962565, 0.011396000546875904, 0.01128172481002732, 0.37797476519810874)),
        'total_input': ((2,), 0.13703219182363718, 0.1990162741957567, (0.07504810945151766, 0.061984082372119526)),
        'field[0]': ((200,), -543.4026902007854, -2112.267361721295, (-14.666666666666666, -2.095314833539841, -0.6039656872825314, -0.4447871846170478)),
        'field[1]': ((150,), -518.5980577235401, -2018.859464726716, (-12.666666666666666, -3.276237923507527, -0.9735434632984767, -0.542546914914684)),
        'firing_prob[0]': ((200,), 1.945365660141432, 7.7132984388317825, (3.027300947700593e-39, 1.7340739233297383e-06, 0.013252034358933627, 0.034076223996286924)),
        'firing_prob[1]': ((150,), 0.4728529294115426, 1.8524094450362174, (4.927077343055629e-34, 1.451635881467801e-09, 0.0014515342669583163, 0.019100215441192912)),
        'd_firing_prob[0]': ((200,), 11.52457374289115, 45.69635952835991, (1.8163805686203557e-38, 1.0404434518936102e-05, 0.0789830140906213, 0.2009335196292315)),
        'd_firing_prob[1]': ((150,), 2.8196654949562534, 11.04694228578642, (2.9562464058333774e-33, 8.709815282485067e-09, 0.008702881686031245, 0.1134998025320261)),
        'hazard_rate[0]': ((200,), 3.9404027898150646, 15.622905516314981, (6.054601895401186e-39, 3.4681508536753246e-06, 0.02668125223489666, 0.06934070942276198)),
        'hazard_rate[1]': ((150,), 0.9515513542568191, 3.7274312009629242, (9.854154686111257e-34, 2.9032717650428487e-09, 0.0029051775267420784, 0.03856996209627282)),
        'max_step_prob': 0.034076223996286924,
        'truncation_error': 0.5427367776189442,
        'fp_residual': ((2,), -0.0018730234713142209, -0.003592110558798289, (-0.0001539363838301529, -0.001719087087484068)),
        'fp_state.p[0]': ((200,), 1.0, 3.9737106199595633, (0.006153936383830153, 0.006153911530156074, 0.004903158555014375, 0.02389201219422917)),
        'fp_state.p[1]': ((150,), 1.0000000000000002, 3.7872666928661043, (0.005719087087484068, 0.005719087077292184, 0.005661050296458911, 0.18639306019082996)),
        'fp_state.S[0]': ((200,), 13.562883291170438, 54.096805428059156, (0.0, 0.07113574733101329, 0.07905001273397297, 0.07989473841052021)),
        'fp_state.S[1]': ((150,), 8.288459138620437, 33.031815228545746, (0.0, 0.05874656008807048, 0.06356877139205853, 0.0639627877368196)),
        'R[0]': ((200,), -456.1260495722094, -1766.0156489055755, (-14.166666666666666, -1.662047375435008, -0.17812261238006974, -0.01973654802746148)),
        'R[1]': ((150,), -451.62544106707446, -1753.8508202896348, (-12.166666666666666, -2.8331340425634273, -0.5351099002087859, -0.1044949571620187)),
        'R_abs': ((2,), 26.333333333333332, 38.5, (14.166666666666666, 12.166666666666666)),
        'alpha_int': ((2,), 0.08155447501728008, 0.13032505051656607, (0.0327838995179941, 0.048770575499285984)),
        'stationary_rate': ((2,), 0.02362601439014292, 0.03502201495710549, (0.012230013823180352, 0.011396000566962569)),
        'state_map': ((700,), 22.750742714958303, 91.27587145669965, (0.006115006911590174, 0.05006676392889889, 0.005697978722470764, 0.06194804224733473)),
        'update.p[0]': ((200,), 1.0, 3.9737774006732485, (0.006115006911590174, 0.006114982850355704, 0.004904023421407289, 0.0259032965518734)),
        'update.p[1]': ((150,), 1.0, 3.784748592136602, (0.0056980002834812835, 0.005698000273437953, 0.005640862405013661, 0.18898738259905434)),
        'update.S[0]': ((200,), 12.723359371423989, 50.74828718428047, (0.0, 0.06673254189516738, 0.07415692509753834, 0.07494936341041365)),
        'update.S[1]': ((150,), 8.027383343534312, 31.991355562918933, (0.0, 0.05689611905590046, 0.06156643691030923, 0.06194804224733473)),
        'update.m': ((2,), 0.011813007195071458, 0.01751100747855274, (0.006115006911590174, 0.0056980002834812835)),
        'forward.p[0]': ((200,), 1.0, 3.9737774006732485, (0.0061150069115901845, 0.0061149828503557005, 0.004904023421407288, 0.02590329655187329)),
        'forward.p[1]': ((150,), 1.0, 3.784748592136602, (0.005698000283481284, 0.005698000273437952, 0.005640862405013659, 0.18898738259905434)),
        'forward.S[0]': ((200,), 12.723359371424007, 50.74828718428054, (0.0, 0.06673254189516747, 0.07415692509753843, 0.07494936341041378)),
        'forward.S[1]': ((150,), 8.027383343534321, 31.99135556291897, (0.0, 0.05689611905590053, 0.0615664369103093, 0.06194804224733478)),
        'forward.m': ((2,), 0.01181300719507147, 0.017511007478552755, (0.0061150069115901845, 0.005698000283481284)),
    },
    'entropy/M2_fp/single/sliding': {
        'r': ((140, 2), 3.3076420146200087, 13.230568058480035, (0.012230013823180348, 0.011396000566962564, 0.012230013823180352, 0.011396000566962567)),
        'm': ((140, 2), 1.6538210073100044, 6.6152840292400175, (0.006115006911590174, 0.005698000283481282, 0.006115006911590176, 0.0056980002834812835)),
        'sigma': ((140, 2), 5.597533200840665e-11, 2.239012933391571e-10, (1.7797222029436455e-13, 2.2184684644876995e-13, 1.779756897413165e-13, 2.2185725478962581e-13)),
        'H_fwd': ((140, 2), 8.385071235051626, 33.540284940206504, (0.03022723714518021, 0.02966612881947425, 0.030227237145180212, 0.02966612881947425)),
        'H_rev': ((140, 2), 8.3850712351076, 33.5402849404304, (0.03022723714535818, 0.029666128819696096, 0.030227237145358188, 0.029666128819696106)),
        'r_tot': ((140,), 1.6888495640711514, 6.7553982562846056, (0.012063211171936792, 0.012063211171936792, 0.012063211171936795, 0.012063211171936795)),
        'm_tot': ((140,), 0.8444247820355757, 3.3776991281423028, (0.006031605585968396, 0.006031605585968396, 0.006031605585968398, 0.006031605585968398)),
        'sigma_tot': ((140,), 2.6144826581475656e-11, 1.045792362430742e-10, (1.8674714552524564e-13, 1.8673326773743783e-13, 1.867506149721976e-13, 1.8674367607829366e-13)),
        'H_fwd_tot': ((140,), 4.216102167205463, 16.864408668821852, (0.030115015480039017, 0.03011501548003903, 0.03011501548003902, 0.03011501548003903)),
        'H_rev_tot': ((140,), 4.216102167231608, 16.864408668926433, (0.030115015480225767, 0.030115015480225764, 0.030115015480225774, 0.030115015480225774)),
        'H_fwd_ref': ((140, 2), -185.69613932173306, -742.7845572869322, (-0.6629199434147651, -0.663481051740471, -0.6629199434147651, -0.663481051740471)),
        'H_rev_ref': ((140, 2), -185.6961393216771, -742.7845572867084, (-0.6629199434145872, -0.6634810517402492, -0.6629199434145872, -0.6634810517402492)),
        'after.p[0]': ((200,), 1.0, 3.9737774006732485, (0.006115006911590177, 0.0061149828503557005, 0.00490402342140729, 0.02590329655187329)),
        'after.p[1]': ((150,), 1.0, 3.784748592136602, (0.005698000283481282, 0.005698000273437953, 0.005640862405013657, 0.18898738259905445)),
        'after.S[0]': ((200,), 12.72335937142398, 50.74828718428044, (0.0, 0.06673254189516739, 0.07415692509753827, 0.07494936341041354)),
        'after.S[1]': ((150,), 8.027383343534316, 31.991355562918944, (0.0, 0.05689611905590048, 0.06156643691030926, 0.061948042247334745)),
        'after.m': ((2,), 0.01181300719507146, 0.01751100747855274, (0.006115006911590177, 0.005698000283481282)),
    },
    'entropy/M2_fp/single/joint': {
        'r': ((140, 2), 3.3076420146200087, 13.230568058480035, (0.012230013823180348, 0.011396000566962564, 0.012230013823180352, 0.011396000566962567)),
        'm': ((140, 2), 1.6538210073100044, 6.6152840292400175, (0.006115006911590174, 0.005698000283481282, 0.006115006911590176, 0.0056980002834812835)),
        'sigma': ((140, 2), 5.597484975528033e-11, 2.2389944759337865e-10, (1.779687508474126e-13, 2.2184684644876995e-13, 1.7797222029436455e-13, 2.2185378534267386e-13)),
        'H_fwd': ((140, 2), 8.385071235051626, 33.540284940206504, (0.030227237145180212, 0.029666128819474252, 0.030227237145180216, 0.029666128819474252)),
        'H_rev': ((140, 2), 8.3850712351076, 33.5402849404304, (0.03022723714535818, 0.0296661288196961, 0.030227237145358188, 0.029666128819696106)),
        'r_tot': ((140,), 1.6888495640711514, 6.7553982562846056, (0.012063211171936792, 0.012063211171936792, 0.012063211171936795, 0.012063211171936795)),
        'm_tot': ((140,), 0.8444247820355757, 3.3776991281423028, (0.006031605585968396, 0.006031605585968396, 0.006031605585968398, 0.006031605585968398)),
        'sigma_tot': ((140,), 2.61446718441416e-11, 1.045786374165303e-10, (1.8674298218890328e-13, 1.8673604329499938e-13, 1.8674714552524564e-13, 1.8674298218890328e-13)),
        'H_fwd_tot': ((140,), 4.216102167205464, 16.864408668821856, (0.030115015480039024, 0.03011501548003903, 0.030115015480039028, 0.03011501548003903)),
        'H_rev_tot': ((140,), 4.216102167231608, 16.864408668926433, (0.030115015480225767, 0.03011501548022577, 0.030115015480225774, 0.030115015480225774)),
        'H_fwd_ref': ((140, 2), -185.69613932173306, -742.7845572869322, (-0.6629199434147651, -0.663481051740471, -0.6629199434147651, -0.663481051740471)),
        'H_rev_ref': ((140, 2), -185.6961393216771, -742.7845572867084, (-0.6629199434145872, -0.6634810517402492, -0.6629199434145872, -0.6634810517402492)),
        'after.p[0]': ((200,), 1.0, 3.9737774006732485, (0.006115006911590177, 0.0061149828503557005, 0.00490402342140729, 0.02590329655187329)),
        'after.p[1]': ((150,), 1.0, 3.784748592136602, (0.005698000283481282, 0.005698000273437953, 0.005640862405013657, 0.18898738259905445)),
        'after.S[0]': ((200,), 12.72335937142398, 50.74828718428044, (0.0, 0.06673254189516739, 0.07415692509753827, 0.07494936341041354)),
        'after.S[1]': ((150,), 8.027383343534316, 31.991355562918944, (0.0, 0.05689611905590048, 0.06156643691030926, 0.061948042247334745)),
        'after.m': ((2,), 0.01181300719507146, 0.01751100747855274, (0.006115006911590177, 0.005698000283481282)),
    },
    'trajectory/M2_fp/batch': {
        'r': ((2, 140, 2), 6.576843701845393, 26.30673642891653, (0.012230013823180348, 0.012230013823180348, 0.01124218346318946, 0.011259211739426605)),
        'm': ((2, 140, 2), 3.2884218509226963, 13.153368214458265, (0.006115006911590174, 0.006115006911590174, 0.00562109173159473, 0.0056296058697133024)),
        'r_tot': ((2, 140), 3.3590153774834253, 13.435675351159798, (0.012063211171936792, 0.012063211171936792, 0.01191937218955619, 0.01193713037920309)),
        'q[0]': ((2, 140, 200), 560.0, 2240.0011969518146, (0.012230013823180348, 0.012229965700711411, 0.00981090636063148, 0.05896353457376624)),
        'q[1]': ((2, 140, 150), 560.0, 2240.0016991967655, (0.011396000566962565, 0.011396000546875906, 0.011156746461956359, 0.3958509030313211)),
        'S[0]': ((2, 140, 200), 3370.9711951406175, 13483.883521173624, (0.0, 0.06673254189516738, 0.0661799555071635, 0.06680353029916328)),
        'S[1]': ((2, 140, 150), 2126.045718523551, 8504.181856023004, (0.0, 0.056896119055900446, 0.05486053146859394, 0.055204637034062386)),
        'after.p[0]': ((2, 200), 2.0, 7.901328583616155, (0.006115006911590176, 0.00490402342140729, 0.006047848304535051, 0.029482593543045166)),
        'after.p[1]': ((2, 150), 2.0, 8.188026384992414, (0.005698000283481282, 0.005640862405013659, 0.005626787950546866, 0.1979270620986211)),
        'after.S[0]': ((2, 200), 24.063861037485673, 96.17278702673141, (0.0, 0.07415692509753825, 0.05948164259786133, 0.06680381254935086)),
        'after.S[1]': ((2, 150), 15.18122002549157, 60.66232503617198, (0.0, 0.0615664369103092, 0.050706302686292455, 0.05520489700559276)),
        'after.m': ((2, 2), 0.023495809714791427, 0.05818896700245501, (0.006115006911590174, 0.005698000283481282, 0.006053250554977616, 0.0056295519647423554)),
    },
    'methods/M2_fp/batch': {
        'rates': ((2, 2), 0.047252028780285824, 0.11729605869449679, (0.012230013823180348, 0.011396000566962565, 0.012230013823180348, 0.011396000566962565)),
        'overlaps': ((2, 2), 0.023626014390142912, 0.058648029347248394, (0.006115006911590174, 0.005698000283481283, 0.006115006911590174, 0.005698000283481283)),
        'activity': ((2,), 0.024126422343873584, 0.03618963351581038, (0.012063211171936792, 0.012063211171936792)),
        'age_density[0]': ((2, 200), 3.999999999999999, 15.804423459445898, (0.012230013823180348, 0.009808046842814578, 0.012110662344672583, 0.05836132993442022)),
        'age_density[1]': ((2, 150), 4.000000000000001, 16.3726832302656, (0.011396000566962565, 0.01128172481002732, 0.011265313441247917, 0.3941561622567495)),
        'total_input': ((2, 2), 0.26036116446491064, 0.6247888662544847, (0.07504810945151766, 0.061984082372119526, 0.0675432985063659, 0.05578567413490757)),
        'field[0]': ((2, 200), -1087.2777163387132, -4318.557248950865, (-14.666666666666666, -0.6039656872825314, -2.1019880877293575, -0.4522821209580892)),
        'field[1]': ((2, 150), -1037.3988537814337, -4125.550546819453, (-12.666666666666666, -0.9735434632984767, -3.281927535413117, -0.5487417191394175)),
        'firing_prob[0]': ((2, 200), 3.806600636273706, 15.190597942713808, (3.027300947700593e-39, 0.013252034358933627, 1.6660141182034624e-06, 0.0326024767568574)),
        'firing_prob[1]': ((2, 150), 0.9285753116215991, 3.6948559848279623, (4.927077343055629e-34, 0.0014515342669583163, 1.4029167233182243e-09, 0.01840976996882578)),
        'd_firing_prob[0]': ((2, 200), 22.556916893959414, 90.01671434985305, (1.8163805686203557e-38, 0.0789830140906213, 9.996076382407024e-06, 0.19239086599864252)),
        'd_firing_prob[1]': ((2, 150), 5.5377915931556725, 22.035638110278736, (2.9562464058333774e-33, 0.008702881686031245, 8.41750033400482e-09, 0.10943556341405992)),
        'hazard_rate[0]': ((2, 200), 7.708301427514058, 30.760289391395972, (6.054601895401186e-39, 0.02668125223489666, 3.3320310120130497e-06, 0.06629155763312397)),
        'hazard_rate[1]': ((2, 150), 1.8684240249145734, 7.43442196502385, (9.854154686111257e-34, 0.0029051775267420784, 2.805833448604624e-09, 0.03716267748188718)),
        'max_step_prob': ((2,), 0.06667870075314433, 0.09928117751000173, (0.034076223996286924, 0.0326024767568574)),
        'truncation_error': ((2,), 1.0885839504014792, 1.6344311231840143, (0.5427367776189442, 0.545847172782535)),
        'fp_residual': ((2, 2), -0.0027330950260860496, -0.007484726657498787, (-0.0001539363838301529, -0.001719087087484068, 0.0004523298796131831, -0.0013124014343850118)),
        'fp_state.p[0]': ((2, 200), 2.0, 7.916817573763064, (0.006153936383830153, 0.004903158555014375, 0.006147645396320771, 0.024208985394752653)),
        'fp_state.p[1]': ((2, 150), 2.0, 8.169307417902079, (0.005719087087484068, 0.005661050296458911, 0.005712401424240496, 0.18721470315849414)),
        'fp_state.S[0]': ((2, 200), 26.990137749429177, 107.879816027521, (0.0, 0.07905001273397297, 0.07042438985770318, 0.07909579102641502)),
        'fp_state.S[1]': ((2, 150), 16.49403368585467, 65.9179157964121, (0.0, 0.06356877139205853, 0.05815909448718978, 0.06332315985945142)),
        'R[0]': ((2, 200), -911.4520991444188, -3616.6727323408336, (-14.166666666666666, -0.17812261238006974, -1.662047375435008, -0.01973654802746148)),
        'R[1]': ((2, 150), -902.6508821341489, -3587.9966575882872, (-12.166666666666666, -0.5351099002087859, -2.8331340425634273, -0.1044949571620187)),
        'R_abs': ((2, 2), 51.266666666666666, 124.86666666666666, (14.166666666666666, 12.166666666666666, 13.366666666666667, 11.566666666666666)),
        'alpha_int': ((2, 2), 0.16310895003456016, 0.4237590510676923, (0.0327838995179941, 0.048770575499285984, 0.0327838995179941, 0.048770575499285984)),
        'update.p[0]': ((2, 200), 2.0, 7.90221172972295, (0.006115006911590174, 0.004904023421407289, 0.0060553311723362925, 0.02918066496721028)),
        'update.p[1]': ((2, 150), 2.0, 8.186341615132799, (0.005698000283481283, 0.005640862405013661, 0.005632656720623959, 0.19707808112837455)),
        'update.S[0]': ((2, 200), 24.17438280570558, 96.6154833899685, (0.0, 0.07415692509753834, 0.060059287705650645, 0.06745442706937228)),
        'update.S[1]': ((2, 150), 15.252028352715193, 60.946110768834195, (0.0, 0.06156643691030923, 0.05120650715031041, 0.05575323802260126)),
        'update.m': ((2, 2), 0.023501018001543406, 0.05820769662819086, (0.006115006911590172, 0.0056980002834812835, 0.006055354076249686, 0.005632656730222264)),
        'forward.p[0]': ((2, 200), 2.0, 7.9021352076770075, (0.0061150069115901776, 0.004904023421407288, 0.006055331212019103, 0.029200244383986718)),
        'forward.p[1]': ((2, 150), 2.0, 8.186486862340228, (0.005698000283481283, 0.005640862405013659, 0.005632656720643348, 0.1971627132242466)),
        'forward.S[0]': ((2, 200), 24.111268406431787, 96.36274229671508, (0.0, 0.07415692509753831, 0.059728573896349176, 0.06712371326007083)),
        'forward.S[1]': ((2, 150), 15.200533764310705, 60.73978325639487, (0.0, 0.06156643691030923, 0.05084224847843788, 0.05538897935072872)),
        'forward.m': ((2, 2), 0.02348184539249718, 0.05814017189761331, (0.006115006911590176, 0.005698000283481283, 0.006046188370642308, 0.005622649826783411)),
    },
    'entropy/M2_fp/batch/sliding': {
        'r': ((2, 140, 2), 6.576593709068193, 26.306355749032814, (0.012230013823180348, 0.012230013823180352, 0.011242183463189462, 0.011259211739426605)),
        'm': ((2, 140, 2), 3.2882968545340967, 13.153177874516407, (0.006115006911590174, 0.006115006911590176, 0.005621091731594731, 0.0056296058697133024)),
        'sigma': ((2, 140, 2), 0.00031928436701916807, 0.0012355597063723259, (1.7797222029436455e-13, 1.7799303697607627e-13, 2.361079327495297e-06, 2.061991534220431e-07)),
        'H_fwd': ((2, 140, 2), 16.720633089110333, 66.88252588889901, (0.03022723714518021, 0.030227237145180195, 0.02944435328990372, 0.02948414053275032)),
        'H_rev': ((2, 140, 2), 16.720952373477353, 66.88376144860538, (0.03022723714535818, 0.030227237145358188, 0.029446714369231214, 0.02948434673190374)),
        'r_tot': ((2, 140), 3.358893795525577, 13.43555376920195, (0.012063211171936792, 0.012063211171936795, 0.011919372189556193, 0.01193713037920309)),
        'm_tot': ((2, 140), 1.6794468977627885, 6.717776884600975, (0.006031605585968396, 0.006031605585968398, 0.0059596860947780965, 0.005968565189601545)),
        'sigma_tot': ((2, 140), 0.00012922001117273986, 0.0004718048768598299, (1.8674714552524564e-13, 1.86763798870615e-13, 1.3632609576236245e-06, 7.111990184251183e-07)),
        'H_fwd_tot': ((2, 140), 8.409471321191972, 33.6378787083782, (0.030115015480039017, 0.030115015480039007, 0.02993205885628121, 0.029971278595026907)),
        'H_rev_tot': ((2, 140), 8.409600541203146, 33.63835051325506, (0.030115015480225767, 0.030115015480225774, 0.029933422117238838, 0.029971989794045334)),
        'H_fwd_ref': ((2, 140, 2), -371.441788024459, -1485.7671585653784, (-0.6629199434147651, -0.6629199434147651, -0.6637028272700416, -0.663663040027195)),
        'H_rev_ref': ((2, 140, 2), -371.44146874009203, -1485.765923005672, (-0.6629199434145872, -0.6629199434145872, -0.6637004661907141, -0.6636628338280416)),
        'after.p[0]': ((2, 200), 2.0, 7.901328583616155, (0.006115006911590176, 0.00490402342140729, 0.006047848304535051, 0.029482593543045166)),
        'after.p[1]': ((2, 150), 2.0, 8.188026384992414, (0.005698000283481282, 0.005640862405013659, 0.005626787950546866, 0.1979270620986211)),
        'after.S[0]': ((2, 200), 24.063861037485673, 96.17278702673141, (0.0, 0.07415692509753825, 0.05948164259786133, 0.06680381254935086)),
        'after.S[1]': ((2, 150), 15.18122002549157, 60.66232503617198, (0.0, 0.0615664369103092, 0.050706302686292455, 0.05520489700559276)),
        'after.m': ((2, 2), 0.02349580971479143, 0.05818896700245501, (0.006115006911590176, 0.005698000283481282, 0.006053250554977617, 0.005629551964742355)),
    },
    'entropy/M2_fp/batch/joint': {
        'r': ((2, 140, 2), 6.576593709068193, 26.306355749032814, (0.012230013823180348, 0.012230013823180352, 0.011242183463189462, 0.011259211739426605)),
        'm': ((2, 140, 2), 3.2882968545340967, 13.153177874516407, (0.006115006911590174, 0.006115006911590176, 0.005621091731594731, 0.0056296058697133024)),
        'sigma': ((2, 140, 2), 0.00031928436701894256, 0.001235559706371573, (1.779687508474126e-13, 1.7798956752912432e-13, 2.361079327495297e-06, 2.0619915342898198e-07)),
        'H_fwd': ((2, 140, 2), 16.720633089110336, 66.88252588889902, (0.030227237145180212, 0.0302272371451802, 0.02944435328990372, 0.029484140532750315)),
        'H_rev': ((2, 140, 2), 16.720952373477353, 66.8837614486054, (0.03022723714535818, 0.030227237145358188, 0.029446714369231214, 0.029484346731903744)),
        'r_tot': ((2, 140), 3.358893795525577, 13.43555376920195, (0.012063211171936792, 0.012063211171936795, 0.011919372189556193, 0.01193713037920309)),
        'm_tot': ((2, 140), 1.6794468977627885, 6.717776884600975, (0.006031605585968396, 0.006031605585968398, 0.0059596860947780965, 0.005968565189601545)),
        'sigma_tot': ((2, 140), 0.00012922001117269267, 0.00047180487685967795, (1.8674298218890328e-13, 1.8676032942366308e-13, 1.3632609576236245e-06, 7.111990184292816e-07)),
        'H_fwd_tot': ((2, 140), 8.409471321191972, 33.6378787083782, (0.030115015480039024, 0.030115015480039014, 0.02993205885628121, 0.029971278595026907)),
        'H_rev_tot': ((2, 140), 8.409600541203146, 33.63835051325506, (0.030115015480225767, 0.030115015480225774, 0.029933422117238838, 0.029971989794045337)),
        'H_fwd_ref': ((2, 140, 2), -371.441788024459, -1485.7671585653784, (-0.6629199434147651, -0.6629199434147651, -0.6637028272700416, -0.663663040027195)),
        'H_rev_ref': ((2, 140, 2), -371.44146874009203, -1485.765923005672, (-0.6629199434145872, -0.6629199434145872, -0.6637004661907141, -0.6636628338280416)),
        'after.p[0]': ((2, 200), 2.0, 7.901328583616155, (0.006115006911590176, 0.00490402342140729, 0.006047848304535051, 0.029482593543045166)),
        'after.p[1]': ((2, 150), 2.0, 8.188026384992414, (0.005698000283481282, 0.005640862405013659, 0.005626787950546866, 0.1979270620986211)),
        'after.S[0]': ((2, 200), 24.063861037485673, 96.17278702673141, (0.0, 0.07415692509753825, 0.05948164259786133, 0.06680381254935086)),
        'after.S[1]': ((2, 150), 15.18122002549157, 60.66232503617198, (0.0, 0.0615664369103092, 0.050706302686292455, 0.05520489700559276)),
        'after.m': ((2, 2), 0.02349580971479143, 0.05818896700245501, (0.006115006911590176, 0.005698000283481282, 0.006053250554977617, 0.005629551964742355)),
    },
    'fixed_points/M2_fp': {
        'roots': ((1, 2), 0.011813007195071456, 0.01751100747855274, (0.006115006911590174, 0.005698000283481283)),
        'root0.lead_abs': ((4,), 3.978575646731681, 9.93725613810654, (0.9969396563635856, 0.9969396563635856, 0.9923481670022547, 0.9923481670022547)),
        'root0.stable': True,
    },
    'fp_sweep/M2_fp': {
        'roots': ((2, 25, 2), 0.5872369241793163, 2.319330535127534, (0.006115006911590174, 0.005698000283481283, 0.006049907627847028, 0.005626562144254164)),
        'converged': ((2, 25), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)),
    },
    'trajectory/M2_cyc/single': {
        'r': ((140, 2), 3.9742454439314483, 15.879432705992608, (0.001610146950983874, 0.015385975908856755, 0.0003561286092405903, 0.0013820331221009756)),
        'm': ((140, 2), 1.9871227219657241, 7.939716352996304, (0.000805073475491937, 0.0076929879544283776, 0.00017806430462029516, 0.0006910165610504878)),
        'r_tot': ((140,), 2.001010995102807, 7.98300937525382, (0.0015733268017489952, 0.02076000501486739, 0.0009219054681348076, 0.0014487982599047616)),
        'q[0]': ((140, 200), 280.0, 1119.975324987014, (0.001610146950983874, 0.02910858736310836, 0.00730193963276609, 1.2738283930335825e-05)),
        'q[1]': ((140, 150), 280.0, 1120.0029307524558, (0.0014260462048094808, 0.005821311042358439, 0.01286584949203801, 0.03767989534269667)),
        'S[0]': ((140, 200), 5121.087898191706, 20484.35094217826, (0.0, 0.13400108960600812, 0.33189376114980834, 0.08069784927038062)),
        'S[1]': ((140, 150), 2348.472740800214, 9393.88602930924, (0.0, 0.09195708008913464, 0.179541435380633, 0.024923235339269253)),
        'after.p[0]': ((200,), 1.0, 3.997422201013824, (0.0007708775448681813, 0.015373675397728595, 0.0008585387698484452, 6.2465566122993366e-06)),
        'after.p[1]': ((150,), 1.0000000000000002, 3.987199004906244, (0.0007099407778928114, 0.002118519073587349, 0.0053449164845772315, 0.018911636296204515)),
        'after.S[0]': ((200,), 10.189840048989288, 40.601420065853404, (0.0, 0.011175787647679899, 0.07862439195394125, 0.0787787172771766)),
        'after.S[1]': ((150,), 2.237482021070018, 8.901906631125827, (0.0, 0.006629072832885101, 0.023919668179180557, 0.024354793779689743)),
        'after.m': ((2,), 0.0014808183227609927, 0.0021907591006538043, (0.0007708775448681813, 0.0007099407778928114)),
    },
    'methods/M2_cyc/single': {
        'rates': ((2,), 0.0030361931557933547, 0.004462239360602836, (0.001610146950983874, 0.0014260462048094808)),
        'overlaps': ((2,), 0.0015180965778966774, 0.002231119680301418, (0.000805073475491937, 0.0007130231024047404)),
        'activity': ((), 0.0015733268017489952, 0.0015733268017489952, (0.0015733268017489952,)),
        'age_density[0]': ((200,), 2.0, 7.970165598720337, (0.001610146950983874, 0.018705946446534342, 0.0018070882023969064, 1.171428337332081e-05)),
        'age_density[1]': ((150,), 2.0, 7.969002602554542, (0.0014260462048094808, 0.0037469051505991213, 0.011369146930067768, 0.03668640426795178)),
        'total_input': ((2,), 0.03919582906213616, 0.05387125236716542, (0.024520405757106897, 0.01467542330502926)),
        'field[0]': ((200,), -555.2166457909364, -2132.5322436972847, (-23.666666666666664, -2.1520415691833756, -0.601445171716939, -0.44289851539360353)),
        'field[1]': ((150,), -528.4298355636156, -2046.115287389497, (-16.666666666666664, -3.326213974514039, -1.0118067743960553, -0.5807318379074455)),
        'firing_prob[0]': ((200,), 1.9698074767107423, 7.810302472426807, (1.0694329824497696e-62, 1.2338178996084557e-06, 0.013452604842398727, 0.034457803938837836)),
        'firing_prob[1]': ((150,), 0.3765402052740817, 1.4750825457289192, (1.8600379880104444e-44, 1.0755528503347343e-09, 0.0011539492617208165, 0.015219214830368756)),
        'd_firing_prob[0]': ((200,), 11.66755711577824, 46.263873327531634, (6.416597894698618e-62, 7.402902830729027e-06, 0.08017026026177895, 0.20314316993352544)),
        'd_firing_prob[1]': ((150,), 2.248179655005314, 8.80770354216889, (1.1160227928062666e-43, 6.453317098537963e-09, 0.006919699236144263, 0.09061686326660748)),
        'hazard_rate[0]': ((200,), 3.9905261051330223, 15.821818138569524, (2.138865964899539e-62, 2.467637321524773e-06, 0.027087821850657844, 0.07013094844956154)),
        'hazard_rate[1]': ((150,), 0.7567817792774095, 2.964483176770054, (3.720075976020889e-44, 2.1511057018262823e-09, 0.0023092311476260335, 0.03067243141189657)),
        'max_step_prob': 0.034457803938837836,
        'truncation_error': 0.2984075807173827,
        'fp_residual': ((2,), -0.00312434651934807, -0.005320527848663023, (-0.0009281651900331167, -0.002196181329314953)),
        'fp_state.p[0]': ((200,), 1.0000000000000002, 3.9719783067048176, (0.006928165190033117, 0.006928115795391601, 0.004511955277738354, 0.0023585472419507527)),
        'fp_state.p[1]': ((150,), 1.0, 3.841785513184494, (0.006196181329314953, 0.0061961813149703825, 0.006112694385652188, 0.1302564198003373)),
        'fp_state.S[0]': ((200,), 31.87277573425053, 127.12749275593902, (0.0, 0.16716900622788122, 0.18576752992483647, 0.18775263526472247)),
        'fp_state.S[1]': ((150,), 14.504803492585763, 57.80567664995505, (0.0, 0.10280648015412332, 0.11124534993610241, 0.11193487853943429)),
        'R[0]': ((200,), -465.1260495722094, -1775.0156489055755, (-23.166666666666664, -1.662047375435008, -0.17812261238006974, -0.01973654802746148)),
        'R[1]': ((150,), -455.62544106707446, -1757.8508202896348, (-16.166666666666664, -2.8331340425634273, -0.5351099002087859, -0.1044949571620187)),
        'R_abs': ((2,), 39.33333333333333, 55.49999999999999, (23.166666666666664, 16.166666666666664)),
        'alpha_int': ((2,), 0.08155447501728008, 0.13032505051656607, (0.0327838995179941, 0.048770575499285984)),
        'stationary_rate': ((2,), 0.02283100160132022, 0.033397968451225696, (0.012264034751414743, 0.010566966849905476)),
        'state_map': ((700,), 13.84275090874442, 55.508049438823384, (0.0008480521317512039, 0.009158357056312315, 0.02351914994400571, 0.023319639708701864)),
        'update.p[0]': ((200,), 1.0, 3.9819394004432835, (0.0008480521317512041, 0.005620124272834138, 0.0009623703802770446, 5.742658370389009e-06)),
        'update.p[1]': ((150,), 1.0, 3.9761870327587054, (0.0007354208917619777, 0.0017028389760362559, 0.006032066984692892, 0.01842263752528306)),
        'update.S[0]': ((200,), 9.670186930309526, 38.53131465242135, (0.0, 0.009753247105920184, 0.07495280964626139, 0.0751231951311842)),
        'update.S[1]': ((150,), 2.172563978434894, 8.644648665492149, (0.0, 0.007298828893501953, 0.022832175565672783, 0.023319639708701864)),
        'update.m': ((2,), 0.0015834730235131818, 0.0023188939152751593, (0.0008480521317512041, 0.0007354208917619777)),
        'forward.p[0]': ((200,), 1.0000000000000002, 3.9778007235776354, (0.0025813376806685267, 0.00016647686929693384, 0.0032862380375108403, 1.206444148416086e-05)),
        'forward.p[1]': ((150,), 0.9999999999999999, 3.9748144931731266, (0.0018035060493096526, 0.0006098300720492262, 0.023031104129724055, 0.023163176377658634)),
        'forward.S[0]': ((200,), 8.819339323017905, 35.155643615059354, (0.0, 0.02890847764343638, 0.06208081159723712, 0.06261915306425532)),
        'forward.S[1]': ((150,), 3.366687249805168, 13.414923146967999, (0.0, 0.02176403899320377, 0.026070114168500375, 0.02775118223662969)),
        'forward.m': ((2,), 0.00438484372997818, 0.006188349779287831, (0.0025813376806685267, 0.0018035060493096526)),
    },
    'entropy/M2_cyc/single/sliding': {
        'r': ((140, 2), 3.9742454439314483, 15.879432705992608, (0.001610146950983874, 0.01538597590885676, 0.00035612860924059037, 0.0013820331221009758)),
        'm': ((140, 2), 1.9871227219657241, 7.939716352996304, (0.000805073475491937, 0.00769298795442838, 0.00017806430462029518, 0.0006910165610504879)),
        'sigma': ((140, 2), 0.9231160996214133, 3.6503719492131883, (-0.002132892921462303, -0.014308069015866287, 0.0028031175140490004, -0.0007376772811022512)),
        'H_fwd': ((140, 2), 7.778998991659343, 31.11085227533991, (0.005393120787750272, 0.044729507409419986, 0.0009099182017988164, 0.004282740088019738)),
        'H_rev': ((140, 2), 8.702115091280756, 34.7612242245531, (0.0032602278662879685, 0.0304214383935537, 0.0037130357158478167, 0.003545062806917487)),
        'r_tot': ((140,), 2.001010995102807, 7.98300937525382, (0.0015733268017489952, 0.020760005014867394, 0.0009219054681348058, 0.001448798259904762)),
        'm_tot': ((140,), 1.0005054975514036, 3.99150468762691, (0.0007866634008744976, 0.010380002507433697, 0.0004609527340674029, 0.000724399129952381)),
        'sigma_tot': ((140,), 0.49858319538839146, 1.9216784383440697, (-0.001882386786637218, -0.00962625373367309, 0.003284935987620667, -0.0016360927716109613)),
        'H_fwd_tot': ((140,), 3.8411004486085742, 15.37364891735965, (0.005212316010077223, 0.05681589529261557, 0.00201395524932281, 0.00476621515087923)),
        'H_rev_tot': ((140,), 4.339683643996966, 17.29532735570372, (0.003329929223440004, 0.04718964155894248, 0.005298891236943476, 0.0031301223792682686)),
        'H_fwd_ref': ((140, 2), -186.30221156512533, -745.2139899517988, (-0.6877540597721951, -0.6484176731505253, -0.6922372623581464, -0.6888644404719254)),
        'H_rev_ref': ((140, 2), -185.37909546550392, -741.5636180025856, (-0.6898869526936573, -0.6627257421663916, -0.6894341448440975, -0.6896021177530277)),
        'after.p[0]': ((200,), 1.0, 3.9974222010138245, (0.0007708775448681813, 0.015373675397728593, 0.0008585387698484507, 6.246556612410359e-06)),
        'after.p[1]': ((150,), 1.0, 3.9871990049062433, (0.0007099407778928149, 0.002118519073587344, 0.00534491648457723, 0.018911636296204404)),
        'after.S[0]': ((200,), 10.189840048989288, 40.601420065853404, (0.0, 0.011175787647679899, 0.07862439195394125, 0.0787787172771766)),
        'after.S[1]': ((150,), 2.237482021070018, 8.901906631125827, (0.0, 0.006629072832885101, 0.023919668179180557, 0.024354793779689743)),
        'after.m': ((2,), 0.0014808183227609962, 0.0021907591006538113, (0.0007708775448681813, 0.0007099407778928149)),
    },
    'entropy/M2_cyc/single/joint': {
        'r': ((140, 2), 3.9742454439314483, 15.879432705992608, (0.001610146950983874, 0.01538597590885676, 0.00035612860924059037, 0.0013820331221009758)),
        'm': ((140, 2), 1.9871227219657241, 7.939716352996304, (0.000805073475491937, 0.00769298795442838, 0.00017806430462029518, 0.0006910165610504879)),
        'sigma': ((140, 2), 0.9231160996214135, 3.650371949213189, (-0.002132892921462303, -0.01430806901586628, 0.0028031175140489995, -0.0007376772811022512)),
        'H_fwd': ((140, 2), 7.778998991659343, 31.11085227533991, (0.005393120787750272, 0.044729507409419986, 0.0009099182017988164, 0.004282740088019738)),
        'H_rev': ((140, 2), 8.702115091280756, 34.7612242245531, (0.0032602278662879685, 0.030421438393553706, 0.003713035715847816, 0.003545062806917487)),
        'r_tot': ((140,), 2.001010995102807, 7.98300937525382, (0.0015733268017489952, 0.020760005014867394, 0.0009219054681348058, 0.001448798259904762)),
        'm_tot': ((140,), 1.0005054975514036, 3.99150468762691, (0.0007866634008744976, 0.010380002507433697, 0.0004609527340674029, 0.000724399129952381)),
        'sigma_tot': ((140,), 0.4985831953883916, 1.9216784383440702, (-0.001882386786637218, -0.009626253733673096, 0.003284935987620666, -0.0016360927716109613)),
        'H_fwd_tot': ((140,), 3.8411004486085742, 15.37364891735965, (0.005212316010077223, 0.05681589529261557, 0.00201395524932281, 0.00476621515087923)),
        'H_rev_tot': ((140,), 4.339683643996966, 17.295327355703723, (0.003329929223440004, 0.04718964155894247, 0.005298891236943475, 0.0031301223792682686)),
        'H_fwd_ref': ((140, 2), -186.30221156512533, -745.2139899517988, (-0.6877540597721951, -0.6484176731505253, -0.6922372623581464, -0.6888644404719254)),
        'H_rev_ref': ((140, 2), -185.37909546550392, -741.5636180025856, (-0.6898869526936573, -0.6627257421663916, -0.6894341448440975, -0.6896021177530277)),
        'after.p[0]': ((200,), 1.0, 3.9974222010138245, (0.0007708775448681813, 0.015373675397728593, 0.0008585387698484507, 6.246556612410359e-06)),
        'after.p[1]': ((150,), 1.0, 3.9871990049062433, (0.0007099407778928149, 0.002118519073587344, 0.00534491648457723, 0.018911636296204404)),
        'after.S[0]': ((200,), 10.189840048989288, 40.601420065853404, (0.0, 0.011175787647679899, 0.07862439195394125, 0.0787787172771766)),
        'after.S[1]': ((150,), 2.237482021070018, 8.901906631125827, (0.0, 0.006629072832885101, 0.023919668179180557, 0.024354793779689743)),
        'after.m': ((2,), 0.0014808183227609962, 0.0021907591006538113, (0.0007708775448681813, 0.0007099407778928149)),
    },
    'trajectory/M2_cyc/batch': {
        'r': ((2, 140, 2), 7.893206900115615, 31.555389353752734, (0.0016101469509838889, 0.0003561286092405531, 0.007371814013253911, 0.0024669990964253487)),
        'm': ((2, 140, 2), 3.9466034500578075, 15.777694676876367, (0.0008050734754919444, 0.00017806430462027656, 0.0036859070066269555, 0.0012334995482126744)),
        'r_tot': ((2, 140), 3.9914285245229317, 15.9455400208625, (0.0015733268017490074, 0.0009219054681347783, 0.009407873040613775, 0.0012662361617463425)),
        'q[0]': ((2, 140, 200), 560.0, 2239.977339959696, (0.0016101469509838887, 0.007301939632766095, 0.014008519022483882, 0.00012409078277686447)),
        'q[1]': ((2, 140, 150), 560.0, 2240.006406585521, (0.0014260462048094808, 0.012865849492038013, 0.05549388878023368, 0.0921800247778286)),
        'S[0]': ((2, 140, 200), 9701.261911074045, 38805.0732979692, (0.0, 0.33189376114980823, 0.07276893199027296, 0.1208845303350973)),
        'S[1]': ((2, 140, 150), 4458.552795610214, 17834.219244376374, (0.0, 0.17954143538063302, 0.04025085460179458, 0.04398198491033538)),
        'after.p[0]': ((2, 200), 2.0, 7.9988356834068135, (0.0007708775448681813, 0.0008585387698484476, 0.02153792685770775, 7.062798385326641e-05)),
        'after.p[1]': ((2, 150), 2.0, 8.080816655981167, (0.0007099407778928115, 0.005344916484577228, 0.02740003679516146, 0.045916270999287856)),
        'after.S[0]': ((2, 200), 26.773591070627987, 107.05289224373755, (0.0, 0.07862439195394123, 0.09165444279438459, 0.11721973395892642)),
        'after.S[1]': ((2, 150), 6.308301797500569, 25.227493384193775, (0.0, 0.02391966817918056, 0.01349074756375897, 0.0421525649981249)),
        'after.m': ((2, 2), 0.0032049811292257763, 0.008588303320633496, (0.0007708775448681814, 0.0007099407778928114, 0.0004991070058794418, 0.0012250558005853416)),
    },
    'methods/M2_cyc/batch': {
        'rates': ((2, 2), 0.006479028439448798, 0.01712526907691254, (0.0016101469509838889, 0.0014260462048094804, 0.0011083114183120253, 0.002334523865343404)),
        'overlaps': ((2, 2), 0.003239514219724399, 0.00856263453845627, (0.0008050734754919444, 0.0007130231024047402, 0.0005541557091560126, 0.001167261932671702)),
        'activity': ((2,), 0.0029268807094673084, 0.0042804346171856095, (0.0015733268017490074, 0.001353553907718301)),
        'age_density[0]': ((2, 200), 4.0, 15.971236638167856, (0.0016101469509838887, 0.0018070882023969064, 0.05317171354240667, 0.0002133519937257411)),
        'age_density[1]': ((2, 150), 4.0, 16.14937914604325, (0.0014260462048094808, 0.011369146930067764, 0.04252887808106451, 0.08865139131171307)),
        'total_input': ((2, 2), 0.05982458209069159, 0.12363124273883021, (0.02452040575710715, 0.01467542330502941, 0.012755021742555862, 0.007873731285999164)),
        'field[0]': ((2, 200), -1104.2099266803864, -4367.354791042912, (-23.666666666666664, -0.601445171716939, -2.088919426518861, -0.4155292910881614)),
        'field[1]': ((2, 150), -1054.6697838841828, -4182.583210663327, (-16.666666666666664, -1.0118067743960553, -3.3256759587771336, -0.5686263506681914)),
        'firing_prob[0]': ((2, 200), 4.284052244955478, 17.107443648195346, (1.0694329824497696e-62, 0.013452604842398727, 1.8019076645529564e-06, 0.04048136446486177)),
        'firing_prob[1]': ((2, 150), 0.7813675088245807, 3.1117680127640233, (1.8600379880104444e-44, 0.0011539492617208165, 1.079030446473444e-09, 0.016356339123515258)),
        'd_firing_prob[0]': ((2, 200), 25.34406314517401, 101.20596582235359, (6.416597894698618e-62, 0.08017026026177895, 1.0811436246698193e-05, 0.23790424946028996)),
        'd_firing_prob[1]': ((2, 150), 4.6643588296940415, 18.575770093873608, (1.1160227928062666e-43, 0.006919699236144263, 6.474182675347743e-09, 0.09733103330272641)),
        'hazard_rate[0]': ((2, 200), 8.68945394881794, 34.69951094685114, (2.138865964899539e-62, 0.027087821850657844, 3.603818575981045e-06, 0.08264708318305429)),
        'hazard_rate[1]': ((2, 150), 1.5707155559871453, 6.255275103975109, (3.720075976020889e-44, 0.0023092311476260335, 2.1580608941111947e-09, 0.03298316154314077)),
        'max_step_prob': ((2,), 0.0749391684036996, 0.11542053286856138, (0.034457803938837836, 0.04048136446486177)),
        'truncation_error': ((2,), 0.6555098487585366, 1.0126121167996915, (0.2984075807173816, 0.35710226804115497)),
        'fp_residual': ((2, 2), -0.005225706862074593, -0.013410217533117342, (-0.0009281651900331167, -0.002196181329314953, -0.0003157516864517743, -0.0017856086562747491)),
        'fp_state.p[0]': ((2, 200), 2.0000000000000004, 7.97902829785861, (0.006928165190033117, 0.004511955277738354, 0.0069157028657214435, 0.0024830160210419206)),
        'fp_state.p[1]': ((2, 150), 2.0, 8.113422132629717, (0.006196181329314953, 0.006112694385652188, 0.006185608642041818, 0.1314391800701902)),
        'fp_state.S[0]': ((2, 200), 63.42682371115856, 253.51756766467435, (0.0, 0.18576752992483647, 0.16549731616560245, 0.1858751089120753)),
        'fp_state.S[1]': ((2, 150), 28.864558950245673, 115.35635264372118, (0.0, 0.11124534993610241, 0.10177841535258211, 0.11081552975403998)),
        'R[0]': ((2, 200), -928.5520991444188, -3666.1727323408336, (-23.166666666666664, -0.17812261238006974, -1.662047375435008, -0.01973654802746148)),
        'R[1]': ((2, 150), -910.2508821341489, -3606.396657588287, (-16.166666666666664, -0.5351099002087859, -2.8331340425634273, -0.1044949571620187)),
        'R_abs': ((2, 2), 75.96666666666667, 180.56666666666666, (23.166666666666664, 16.166666666666664, 21.46666666666667, 15.166666666666666)),
        'alpha_int': ((2, 2), 0.16310895003456016, 0.4237590510676923, (0.0327838995179941, 0.048770575499285984, 0.0327838995179941, 0.048770575499285984)),
        'update.p[0]': ((2, 200), 1.9999999999999998, 7.9832856677386035, (0.0008480521317512037, 0.0009623703802770446, 0.027976218240772507, 0.00011691261210855508)),
        'update.p[1]': ((2, 150), 2.0, 8.065850464009655, (0.000735420891761976, 0.006032066984692893, 0.018959066852073967, 0.044205329446787234)),
        'update.S[0]': ((2, 200), 23.612215948988496, 94.39991389595575, (0.0, 0.0749528096462614, 0.06842942424292005, 0.10117376911491621)),
        'update.S[1]': ((2, 150), 5.416041975949131, 21.65320407815865, (0.0, 0.022832175565672786, 0.006747759456540799, 0.0345031165803494)),
        'update.m': ((2, 2), 0.0033255399703023235, 0.008711517920583273, (0.0008480521317512038, 0.000735420891761976, 0.0005756437818484573, 0.0011664231649406863)),
        'forward.p[0]': ((2, 200), 2.0, 7.979968478618294, (0.0025813376806685297, 0.0032862380375108377, 0.0033661109190006922, 0.00013361217579532791)),
        'forward.p[1]': ((2, 150), 2.0, 8.066085530366921, (0.0018035060493096534, 0.02303110412972404, 0.0024255158078646455, 0.044690756940457765)),
        'forward.S[0]': ((2, 200), 17.179037885751892, 68.65580284342934, (0.0, 0.062080811597237134, 0.01518938554054323, 0.0635105865942239)),
        'forward.S[1]': ((2, 150), 5.645991657514912, 22.5545400872015, (0.0, 0.02607011416850038, 0.010855948220639611, 0.022083966363763346)),
        'forward.m': ((2, 2), 0.00726699097407956, 0.016324616261490222, (0.00258133768066853, 0.0018035060493096534, 0.0013923224942031208, 0.0014898247498982557)),
    },
    'entropy/M2_cyc/batch/sliding': {
        'r': ((2, 140, 2), 7.893206900115615, 31.555389353752734, (0.0016101469509838887, 0.0003561286092405531, 0.007371814013253911, 0.0024669990964253487)),
        'm': ((2, 140, 2), 3.9466034500578075, 15.777694676876367, (0.0008050734754919443, 0.00017806430462027656, 0.0036859070066269555, 0.0012334995482126744)),
        'sigma': ((2, 140, 2), 1.362396477312174, 5.4080470467204576, (-0.0021328929214622745, 0.0028031175140489518, -0.006336750499654958, 0.0007251806292003157)),
        'H_fwd': ((2, 140, 2), 16.033917681989433, 64.13098614179616, (0.00539312078775027, 0.0009099182017987676, 0.022755339202024303, 0.006631095052860492)),
        'H_rev': ((2, 140, 2), 17.396314159301607, 69.53903318851663, (0.0032602278662879954, 0.0037130357158477196, 0.016418588702369345, 0.007356275682060808)),
        'r_tot': ((2, 140), 3.991428524522932, 15.9455400208625, (0.0015733268017490074, 0.0009219054681347783, 0.009407873040613775, 0.0012662361617463427)),
        'm_tot': ((2, 140), 1.995714262261466, 7.97277001043125, (0.0007866634008745037, 0.00046095273406738916, 0.0047039365203068876, 0.0006331180808731713)),
        'sigma_tot': ((2, 140), 0.7412348004427723, 2.8962342309003914, (-0.001882386786637193, 0.003284935987620633, -0.009981332204139989, -8.821035749725613e-05)),
        'H_fwd_tot': ((2, 140), 7.918495499047634, 31.682676650524485, (0.005212316010077218, 0.002013955249322771, 0.028619503671252933, 0.0037141760926348005)),
        'H_rev_tot': ((2, 140), 8.659730299490407, 34.578910881424875, (0.0033299292234400257, 0.005298891236943405, 0.01863817146711294, 0.003625965735137544)),
        'H_fwd_ref': ((2, 140, 2), -372.12850343157993, -1488.5186983124813, (-0.6877540597721951, -0.6922372623581465, -0.670391841357921, -0.6865160855070847)),
        'H_rev_ref': ((2, 140, 2), -370.76610695426774, -1483.110651265761, (-0.6898869526936573, -0.6894341448440976, -0.676728591857576, -0.6857909048778844)),
        'after.p[0]': ((2, 200), 2.0, 7.9988356834068135, (0.0007708775448681813, 0.0008585387698484476, 0.02153792685770775, 7.062798385326641e-05)),
        'after.p[1]': ((2, 150), 2.0, 8.080816655981167, (0.0007099407778928115, 0.005344916484577228, 0.02740003679516146, 0.045916270999287856)),
        'after.S[0]': ((2, 200), 26.773591070627987, 107.05289224373755, (0.0, 0.07862439195394123, 0.09165444279438459, 0.11721973395892642)),
        'after.S[1]': ((2, 150), 6.308301797500569, 25.227493384193775, (0.0, 0.02391966817918056, 0.01349074756375897, 0.0421525649981249)),
        'after.m': ((2, 2), 0.0032049811292257763, 0.008588303320633497, (0.0007708775448681813, 0.0007099407778928115, 0.000499107005879442, 0.0012250558005853416)),
    },
    'entropy/M2_cyc/batch/joint': {
        'r': ((2, 140, 2), 7.893206900115615, 31.555389353752734, (0.0016101469509838887, 0.0003561286092405531, 0.007371814013253911, 0.0024669990964253487)),
        'm': ((2, 140, 2), 3.9466034500578075, 15.777694676876367, (0.0008050734754919443, 0.00017806430462027656, 0.0036859070066269555, 0.0012334995482126744)),
        'sigma': ((2, 140, 2), 1.362396477312174, 5.4080470467204576, (-0.0021328929214622745, 0.0028031175140489518, -0.006336750499654958, 0.0007251806292003157)),
        'H_fwd': ((2, 140, 2), 16.033917681989433, 64.13098614179616, (0.00539312078775027, 0.0009099182017987675, 0.022755339202024303, 0.006631095052860492)),
        'H_rev': ((2, 140, 2), 17.396314159301607, 69.53903318851663, (0.0032602278662879954, 0.003713035715847719, 0.016418588702369345, 0.007356275682060808)),
        'r_tot': ((2, 140), 3.991428524522932, 15.9455400208625, (0.0015733268017490074, 0.0009219054681347783, 0.009407873040613775, 0.0012662361617463427)),
        'm_tot': ((2, 140), 1.995714262261466, 7.97277001043125, (0.0007866634008745037, 0.00046095273406738916, 0.0047039365203068876, 0.0006331180808731713)),
        'sigma_tot': ((2, 140), 0.7412348004427723, 2.896234230900392, (-0.001882386786637193, 0.003284935987620633, -0.009981332204139989, -8.821035749725613e-05)),
        'H_fwd_tot': ((2, 140), 7.918495499047634, 31.682676650524485, (0.005212316010077218, 0.002013955249322771, 0.028619503671252933, 0.0037141760926348005)),
        'H_rev_tot': ((2, 140), 8.659730299490407, 34.578910881424875, (0.0033299292234400257, 0.005298891236943404, 0.01863817146711294, 0.003625965735137544)),
        'H_fwd_ref': ((2, 140, 2), -372.12850343157993, -1488.5186983124813, (-0.6877540597721951, -0.6922372623581465, -0.670391841357921, -0.6865160855070847)),
        'H_rev_ref': ((2, 140, 2), -370.76610695426774, -1483.110651265761, (-0.6898869526936573, -0.6894341448440976, -0.676728591857576, -0.6857909048778844)),
        'after.p[0]': ((2, 200), 2.0, 7.9988356834068135, (0.0007708775448681813, 0.0008585387698484476, 0.02153792685770775, 7.062798385326641e-05)),
        'after.p[1]': ((2, 150), 2.0, 8.080816655981167, (0.0007099407778928115, 0.005344916484577228, 0.02740003679516146, 0.045916270999287856)),
        'after.S[0]': ((2, 200), 26.773591070627987, 107.05289224373755, (0.0, 0.07862439195394123, 0.09165444279438459, 0.11721973395892642)),
        'after.S[1]': ((2, 150), 6.308301797500569, 25.227493384193775, (0.0, 0.02391966817918056, 0.01349074756375897, 0.0421525649981249)),
        'after.m': ((2, 2), 0.0032049811292257763, 0.008588303320633497, (0.0007708775448681813, 0.0007099407778928115, 0.000499107005879442, 0.0012250558005853416)),
    },
    'fixed_points/M2_cyc': {
        'roots': ((1, 2), 0.013469678795397462, 0.019827704858661782, (0.0071116527321331445, 0.006358026063264318)),
        'root0.lead_abs': ((4,), 3.9931711488307817, 9.96640130289341, (1.0024244295035816, 1.0024244295035816, 0.9941611449118092, 0.9941611449118092)),
        'root0.stable': False,
    },
    'fp_sweep/M2_cyc': {
        'roots': ((2, 25, 2), 0.665603440161832, 2.6287275226797946, (0.0071116527321331445, 0.006358026063264318, 0.0069416546472040475, 0.006212804163871768)),
        'converged': ((2, 25), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True)),
    },
    'tangent/M1_cyc/single': {
        'init[0]': ((3, 200), 2.2731653798874207e-16, 4.953606008867406, (-0.110802145740968, -0.013262105588164229, -0.06928356528919498, 0.03476280858224397)),
        'init[1]': ((3, 200), 0.8216780235233764, 6.340288158896245, (-0.019280429635101442, -0.02307954163902813, 0.04449075440751303, 0.0008674061595549874)),
        'step[0]': ((3, 200), 6.171820866873734e-17, -1.7498987644454156, (-0.03366950220129068, -0.006866255657463467, 0.0069625615699679885, 0.03527101904595073)),
        'step[1]': ((3, 200), -77.6461985786119, -310.7007977193913, (0.0, 0.0, 0.0, 0.1344685879896785)),
        'after.p[0]': ((200,), 1.0, 3.973607504387919, (0.01744778349840925, 0.00033767010448464683, 0.021554322901949786, 0.0002225220027275432)),
        'after.S[0]': ((200,), 36.902331630411915, 147.23391788818998, (0.0, 0.18942581399100394, 0.2006945813735592, 0.20410187001848137)),
        'after.m': ((1,), 0.01744778349840925, 0.01744778349840925, (0.01744778349840925,)),
    },
    'tangent/M1_cyc/batch': {
        'init[0]': ((2, 3, 200), 3.145948128738252e-16, 8.011608796830574, (-0.11317118135000799, -0.07042724988163138, 0.05247507313357884, -0.04560107079726168)),
        'init[1]': ((2, 3, 200), -0.8732242902854306, -1.6888188549030372, (0.0976464653562162, 0.045612551285289286, -0.13363860864642288, 0.017454619865267935)),
        'step[0]': ((2, 3, 200), -1.3586408473958977e-16, -4.0208020502745905, (-0.0342491780687153, 0.0067115362929523375, -0.16392800468663168, 0.006108294401996339)),
        'step[1]': ((2, 3, 200), -412.065965435864, -1648.3301178305355, (0.0, 0.0, 0.0, 0.2887586794030045)),
        'after.p[0]': ((2, 200), 2.0000000000000004, 7.971979580069154, (0.017447783498409263, 0.021554322901949797, 0.00041659559717927984, 0.00018700144029970556)),
        'after.S[0]': ((2, 200), 103.77576242960042, 415.01803696187045, (0.0, 0.2006945813735592, 0.34719439708466265, 0.35989616743739206)),
        'after.m': ((2, 1), 0.05445282461346862, 0.09145786572852799, (0.01744778349840926, 0.03700504111505936)),
    },
    'tangent/M2_cyc/single': {
        'init[0]': ((3, 200), 9.736813135277633e-17, 3.84413247003286, (-0.0836121142595116, -0.008333263856338551, -0.056107001307787396, 0.02843690005001411)),
        'init[1]': ((3, 200), 0.6486101540598092, 4.933557323234768, (-0.014549154033454114, -0.017533689715973338, 0.03381449058308663, 0.0007418902738753398)),
        'init[2]': ((3, 150), 5.849270720559296e-17, 1.5347370647934644, (-0.053663191974777896, -0.02859938293939513, -0.007301488911289353, -0.04146874509967057)),
        'init[3]': ((3, 150), -0.3362601767987113, -1.1494095934420256, (0.037277840492773065, 0.04549411346806189, -0.04680777527835724, -0.013889209143188554)),
        'step[0]': ((3, 200), 2.1927988938519327e-17, -1.4603010538956291, (-0.007399218675734483, -0.0017177528841497754, 0.001029746792720435, 0.03511235285289041)),
        'step[1]': ((3, 200), -32.47649763692044, -130.22692022162212, (0.0, 0.0, 0.0, 0.05680524724877966)),
        'step[2]': ((3, 150), -8.039359108980015e-17, -0.8171204696250629, (-0.002538567917420744, -0.0015810024814853114, 0.0020454642386148573, 0.16137836436958536)),
        'step[3]': ((3, 150), -15.703372034864637, -62.84336892974799, (0.0, 0.0, 0.0, 0.03291705942968892)),
        'after.p[0]': ((200,), 0.9999999999999998, 3.979692490118073, (0.0077399735047773595, 0.00025137748783108055, 0.010030762410535961, 0.00015617406052303995)),
        'after.p[1]': ((150,), 1.0, 3.9409233044557688, (0.005389786578958746, 0.0005719915684359874, 0.007576995267612543, 0.04078289362036758)),
        'after.S[0]': ((200,), 17.6519859118392, 70.41356975081085, (0.0, 0.08577973019602475, 0.10223916540372516, 0.10385751539678902)),
        'after.S[1]': ((150,), 9.178080675461139, 36.589518360170516, (0.0, 0.06461489876481255, 0.06494151200547389, 0.06720397679996698)),
        'after.m': ((2,), 0.013129760083736106, 0.01851954666269485, (0.0077399735047773595, 0.005389786578958746)),
    },
    'tangent/M2_cyc/batch': {
        'init[0]': ((2, 3, 200), 2.9548574958376816e-16, 6.029046341384413, (-0.08354139155898867, -0.05603787342996456, 0.0373619658366908, -0.03464594262566843)),
        'init[1]': ((2, 3, 200), -0.6689793988403289, -1.2198100031307448, (0.07208126220266148, 0.03685805337972646, -0.09687593612688866, 0.013805231649142248)),
        'init[2]': ((2, 3, 150), 1.87512765731368e-16, 0.02392971113012337, (-0.06950915887554109, -0.05500699865129959, -0.02975582174564593, 0.04732098840691997)),
        'init[3]': ((2, 3, 150), -0.616341683917165, -2.1524443414893613, (-0.005441981640483311, -0.07129181994356486, 0.01632023953724171, -0.028259962732888503)),
        'step[0]': ((2, 3, 200), 8.405954968551677e-17, -3.3304180499617266, (-0.006684649475246044, 0.001451171951594167, -0.006102581937298342, 0.013289967455358043)),
        'step[1]': ((2, 3, 200), -72.02038007774371, -287.70047927054515, (0.0, 0.0, 0.0, 0.033307275024466475)),
        'step[2]': ((2, 3, 150), -1.6759986817436665e-16, 0.2789171835220552, (-0.00423392466899347, 0.0009023893928763977, -0.0019692406087625327, 0.1967071325425726)),
        'step[3]': ((2, 3, 150), -34.23942211925327, -136.76087895773347, (0.0, 0.0, 0.0, 0.04038387390752336)),
        'after.p[0]': ((2, 200), 2.0, 7.986132217266932, (0.007739973504777361, 0.010030762410535961, 0.00035595866583910604, 0.0001669104616459638)),
        'after.p[1]': ((2, 150), 2.0, 8.050315431674381, (0.005389786578958743, 0.007576995267612545, 0.0012662352114988955, 0.052545231372635115)),
        'after.S[0]': ((2, 200), 27.3633328536158, 109.31740242462877, (0.0, 0.10223916540372519, 0.0381769996552902, 0.06443912525126537)),
        'after.S[1]': ((2, 150), 13.41460693202065, 53.56965671019876, (0.0, 0.06494151200547389, 0.028634735961245068, 0.03314451532382741)),
        'after.m': ((2, 2), 0.019610367428925562, 0.040801674551950905, (0.0077399735047773595, 0.005389786578958743, 0.003640301491501773, 0.002840305853687685)),
    },
    'rdm_free_functions': {
        'total_input.single': ((2,), -0.1341, -0.373, (0.1048, -0.2389)),
        'total_input.batch': ((2, 2), 0.20249999999999999, 0.8976, (0.1048, -0.2389, 0.0758, 0.2608)),
        'syn': ((7,), 0.05931165579800869, 0.9249250847198439, (0.0, -0.09754465049761527, -0.16646384233841158, 0.3018488154830697)),
        'syn.inplace': ((7,), 0.05931165579800869, 0.9249250847198439, (0.0, -0.09754465049761527, -0.16646384233841158, 0.3018488154830697)),
        'syn.batch': ((2, 7), 0.013736663325964987, 0.10759697942897273, (0.0, -0.16646384233841158, 0.09235964586772845, -0.2792459773955988)),
        'syn.batch.inplace': ((2, 7), 0.013736663325964987, 0.10759697942897273, (0.0, -0.16646384233841158, 0.09235964586772845, -0.2792459773955988)),
        'rate': ((), 0.48049798747029987, 0.48049798747029987, (0.48049798747029987,)),
        'rate.batch': ((2,), 0.9609959749405996, 1.4414939624108993, (0.4804979874702998, 0.4804979874702998)),
        'age': ((7,), 1.0, 3.4059613417219796, (0.4804979874702998, 0.02160110870150426, 0.06448118987718496, 0.270120840697901)),
        'age.inplace': ((7,), 0.9999999999999999, 3.405961341721979, (0.48049798747029987, 0.02160110870150426, 0.06448118987718496, 0.2701208406979009)),
        'age.batch': ((2, 7), 2.0, 6.022579076450432, (0.4804979874702998, 0.06448118987718496, 0.1598878399378932, 0.04480513172866607)),
        'age.batch.inplace': ((2, 7), 2.0, 6.022579076450431, (0.4804979874702998, 0.06448118987718496, 0.1598878399378932, 0.04480513172866596)),
        'hazard_clamp.f64': 0.999999999999,
        'hazard_clamp.f32': 0.9999998807907104,
    },
}
