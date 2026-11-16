"""
Benchmark: cost of the EPR on the S1 cycle, old (Q, Q) ring against the new O(Q) observer.

From `agents/briefs/P1 2026-11-13 code rebuild.md`, Step 2 (#run benchmark). Measures cost,
not a physical quantity: peak RSS, wall time, and the split between equilibration (`forward`)
and EPR measurement (`entropy_trajectory`), single run, per dt.

THE CODE VERSION IS PYTHONPATH. This script imports `rdme` and never says which one; the
driver runs each (version, dt, mode) in its own process under `/usr/bin/time -f %M`, so the
peak RSS of one measurement is not contaminated by another.

    python scripts/S1-epr_bench.py run <dt> <mode>          # one measurement, prints one JSON line
    python scripts/S1-epr_bench.py all [old=<src>] [new=<src>] [step1=<src>]   # the whole table

Modes: `control` (equilibration only, no EPR), `epr` (equilibration + entropy_trajectory on
the version's DEFAULT path: inplace=False at 8e8a724, inplace=True in the current tree),
`epr_inplace` / `epr_outofplace` (the same with inplace forced True / False). The driver runs
`epr_outofplace` for the new code only, and only the versions it is given. Inside
`entropy_trajectory` the time is split by wrapping `epr.init_epr_state` and the per-step
observer: `pre_init` is the fill phase (plus, in the old code, the ring's index setup), `init`
the observer's initial state, `obs` the summed observer steps, and the rest of the loop is the
model step and bookkeeping.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

N_THREADS = 8                       # fixed: one per physical core of the benchmark machine
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))

import torch

torch.set_num_threads(N_THREADS)
torch.set_default_dtype(torch.float64)

PRESET = "S1"
HAZARD = "poisson_exact"
J      = 17.0                       # the cycle point
T_EQUI, T_MEAS = 8000.0, 4000.0     # ms, as the goldens
U_MAX  = 138.5                      # ms, age window held in ms (Q = 554 at dt = 1/4)

DTS       = [1 / 2, 1 / 8, 1 / 32, 1 / 64]
DT_STEP1  = 1 / 8                   # the Step-1 snapshot is run at this dt only


def run(dt: float, mode: str) -> dict:
    import rdme.epr as epr
    import rdme.helpers as hlp
    import rdme.presets as pre
    try:
        from rdme.mean_field import RDMIsingModel          # since Step 3 of the rebuild
    except ImportError:
        from rdme.rdm_single import RDMIsingModel       # 8e8a724 and the Step-1/2 trees

    Q = round(U_MAX / dt)
    out = {"dt": dt, "mode": mode, "Q": Q, "threads": torch.get_num_threads(),
           "rdme": str(Path(epr.__file__).parent)}
    net = RDMIsingModel(**pre.params(PRESET, hazard=HAZARD, J=J, dt=dt), Q=Q)

    t0 = time.perf_counter()
    net.forward(hlp.n_steps(T_EQUI, dt))
    out["t_equi"] = time.perf_counter() - t0
    if mode == "control":
        return out

    # time split: wrap the init and the observer step, whichever names this version uses
    marks = {"init_start": None, "init": 0.0, "obs": 0.0}
    init0 = epr.init_epr_state

    def init_timed(*a, **k):
        marks["init_start"] = time.perf_counter()
        s = init0(*a, **k)
        marks["init"] = time.perf_counter() - marks["init_start"]
        return s
    epr.init_epr_state = init_timed

    step_name = "epr_step" if hasattr(epr, "epr_step") else "step_epr"
    step0 = getattr(epr, step_name)

    def step_timed(*a, **k):
        s = time.perf_counter()
        r = step0(*a, **k)
        marks["obs"] += time.perf_counter() - s
        return r
    setattr(epr, step_name, step_timed)

    t0 = time.perf_counter()
    kw = {"epr_inplace": {"inplace": True}, "epr_outofplace": {"inplace": False}}.get(mode, {})
    traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, dt), pb=False, **kw)
    t1 = time.perf_counter()
    out["t_epr"] = t1 - t0
    out["t_pre_init"] = marks["init_start"] - t0
    out["t_init"] = marks["init"]
    out["t_obs"] = marks["obs"]
    out["t_loop_rest"] = out["t_epr"] - out["t_pre_init"] - out["t_init"] - out["t_obs"]

    r = traj["r_tot"]
    a, z, n_cyc = hlp.cycle_window(r)
    out["n_cycles"] = n_cyc
    out["sigma"] = float(traj["sigma_tot"][a:z].mean()) / dt      # nats/ms, whole cycles
    out["r"] = float(r[a:z].mean())                               # 1/ms
    return out


def drive(versions: dict[str, str]) -> None:
    jobs = []
    for dt in DTS:
        if "old" in versions:
            jobs += [("old", dt, "control"), ("old", dt, "epr")]
        if "new" in versions:
            jobs += [("new", dt, m) for m in ("control", "epr", "epr_outofplace")]
    if "step1" in versions:
        jobs += [("step1", DT_STEP1, "control"), ("step1", DT_STEP1, "epr")]

    for v, dt, mode in jobs:                 # sequential: wall times contend for the cores
        env = dict(os.environ, PYTHONPATH=versions[v])
        p = subprocess.run(["/usr/bin/time", "-f", "MAXRSS_KB %M", sys.executable, __file__,
                            "run", repr(dt), mode], env=env, capture_output=True, text=True)
        rss = [l for l in p.stderr.splitlines() if l.startswith("MAXRSS_KB")]
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        rec = json.loads(line[-1]) if line else {"dt": dt, "mode": mode, "error": p.stderr[-2000:]}
        rec["version"] = v
        rec["rss_MiB"] = int(rss[-1].split()[1]) / 1024 if rss else None
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        print(json.dumps(run(float(sys.argv[2]), sys.argv[3])), flush=True)
    elif sys.argv[1] == "all":
        drive(dict(a.split("=", 1) for a in sys.argv[2:]))
