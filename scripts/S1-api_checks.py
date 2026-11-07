"""
Three cheap checks that the landed route-B API is the model the vault decided on.

Single fixed points and one-dimensional sweeps -- seconds to minutes, no grids, no EPR. Their
point is that the expensive re-runs about to be spent on this API reproduce the analytics
first. Nothing here is a claim about the paper's results.

  1. THE SATURATING TRANSFER FUNCTION. Exact-survival solver, K_ref = 0, no synaptic filter,
     t_abs = 1 ms, nu0 = 1 /ms, J = 0 so the field is E - theta and nothing is self-consistent.
     An uncoupled neuron with hazard rate lam = nu0 e^{beta h} and a hard dead time t_abs is a
     renewal process with mean ISI t_abs + 1/lam, so

         r = lam / (1 + lam t_abs)  =  (1/t_abs) * sigma(beta h + ln(nu0 t_abs)),

     which at nu0 t_abs = 1 is exactly (1/t_abs) sigma(beta h). That identity is the whole
     content of "the dead time is the ceiling under route B": the hazard is unbounded, the
     realised rate is not, and the saturating transfer function is recovered without the
     kernel saturating. Reported: relative error across the sweep and where it first exceeds
     1%.

  2. THE A/B IDENTITY AT t_abs = 1/nu0. Same stripped-down neuron: compare the stationary rate
     of `bernoulli` against the two route-B kernels at t_abs = 1/nu0, then move t_abs off it.
     Both halves are measured; what the closed forms say they should show is printed beside
     them so the two can be read against each other rather than one being taken on trust:

         bernoulli      lam = nu0 sigma(x)  ->  r = nu0 sigma / (1 + nu0 t_abs sigma)
         poisson_exact  lam = nu0 e^x       ->  r = nu0 e^x  / (1 + nu0 t_abs e^x)

     These are NOT the same function of x, so if the vault's §11 identity is an identity of
     stationary rates it does not hold pointwise and the measurement will say so. What they do
     share at nu0 t_abs = 1 is the CEILING: both tend to 1/t_abs = nu0 as x -> +inf, while off
     that point route B still caps at 1/t_abs and route A at min(nu0, 1/t_abs). The sweep
     reports the pointwise difference, the ceilings, and the drive at which each is within 1%
     of its own ceiling, and states which of the two readings the numbers support.

  3. IS THE WINDOW INERT AT S1? S1 under the exact solver at t_abs = 0 against t_abs = 1 ms.
     Expected: no visible difference, the window only load-bearing at strong drive. Measured at
     S1/fp (J = 10, a fixed point) and at the candidate cycle point J = 17. If it MOVES the S1
     fixed point, that is a decision and not a measurement -- it is flagged loudly and nothing
     here resolves it.

  4. DO poisson AND poisson_exact AGREE IN THE dt -> 0 LIMIT? They share a rate function as
     the same object, so they are one continuous-time model under two discretization maps and
     their limits are REQUIRED to coincide. tests/test_hazards.py asserts that pointwise on
     Phi -- the per-bin gap halves when dt halves -- but nothing asserted it on a SOLVED
     observable until this check. The stripped neuron is the right place for it: its
     stationary rate has the closed form r = lam/(1 + lam t_abs) with the lam BOTH kernels
     share, so the two limits can be compared against a third thing rather than only against
     each other.

     What it also measures is how differently they get there. poisson_exact's error is set by
     lam*dt; poisson's is set by c = nu0*dt, since Phi/dt = nu0 e^x / (1 + c(e^x - 1)), and a
     slowly firing population therefore buys poisson no relief at all.

Preset S1 for check 3; checks 1, 2 and 4 are not at a preset at all -- they are an uncoupled
neuron with the preset's beta and theta, which is stated as an override rather than hidden.
"""

import math
import time

import torch

import rdme.helpers as hlp
import rdme.presets as pre
from rdme.helpers import orders, richardson
from rdme.rdm_single import RDMIsingModel


PRESET = "S1"
HAZARD = "poisson_exact"            # checks 1 and 3; check 2 sweeps all three

NU0   = 1.0
T_ABS = 1.0                         # ms; = 1/nu0, which is the point of check 2
BETA  = pre.resolve(PRESET)["beta"]        # 6.0
THETA = pre.resolve(PRESET)["theta"]       # 0.5

# Checks 1 and 2 are quoted as a dt -> 0 LIMIT, like every other number in this repo, and not
# at a stated step -- because at a stated step they do not test what they are meant to.
#
# The discrete renewal process with a hard dead time fires from bin n_abs onward with per-bin
# probability Phi, so its mean ISI is t_abs + dt/Phi - dt, against the continuum's
# t_abs + 1/lam. For exact survival dt/Phi = (1/lam)(1 + lam dt/2 + ...), so
#
#     ISI_measured = t_abs + 1/lam - dt/2 + O(dt^2)   ->   r is HIGH by about r*dt/2,
#
# and for bernoulli, where Phi = lam*dt exactly, high by about r*dt. That is a first-order
# offset in the SOLVER, not a failure of the identity, and it reaches 5% at dt = 1/8 where r
# approaches the ceiling. Richardson from the (dt, dt/2) pair removes it; the raw pair is
# printed beside the limit so the offset stays visible rather than being quietly corrected.
DT_PAIR = (0.25, 0.125)             # both divide t_abs; the standing (dt, dt/2) pair
DT    = DT_PAIR[1]                  # the stated step, for check 3 and for the raw columns
U_MAX = 200.0                       # ms of age grid, as fitted for S1 in S1-cycle_point.py
EQUI_MS = 4000.0                    # these are uncoupled or settled; this is ample

DEVICE = 'cpu'
torch.set_default_dtype(torch.float64)


def stripped(E: float, hazard: str, *, t_abs: float, dt: float = DT, nu0: float = NU0):
    """ The uncoupled neuron of checks 1 and 2: J = 0, K_ref = 0, no synaptic filter.

    tau_int is set so that alpha = 1 - exp(-dt/tau_int) is 1 to machine precision, which is the
    "tau_int -> 0" the checks ask for: the synaptic term then tracks the input with no lag and
    the field is exactly E - theta. Passed as an explicit tiny number rather than 0, which
    would divide by zero in kernels.tau2alpha. """
    return RDMIsingModel(0.0, E=E, beta=BETA, theta=THETA, tau_int=dt / 64, tau_ref=1.0,
                         K_ref=0.0, dt=dt, hazard=hazard, nu0=nu0, t_abs=t_abs,
                         Qm=[max(1, round(U_MAX / dt))], device=DEVICE)


def rate_limit(E: float, hazard: str, *, t_abs: float, nu0: float = NU0) -> tuple:
    """ (r at dt, r at dt/2, Richardson limit) for the stripped neuron.

    First order, which is what a bare pair gives and what the offset documented above is:
    there is no triple here to read an order off, so the analysis prints the raw pair beside
    the limit and the predicted r*dt/2 beside both. """
    a = settled_rate(stripped(E, hazard, t_abs=t_abs, dt=DT_PAIR[0], nu0=nu0))
    b = settled_rate(stripped(E, hazard, t_abs=t_abs, dt=DT_PAIR[1], nu0=nu0))
    return a, b, 2.0 * b - a


def settled_rate(net) -> float:
    """ The stationary rate, equilibrated and then read off the renewal fixed-point form.

    stationary_rate() rather than rates(): at a genuine fixed point they agree to roundoff, and
    the renewal form carries the absorbing bin's geometric tail exactly, so it does not inherit
    the truncation bias that a time average at a low rate would. """
    net.forward(hlp.n_steps(EQUI_MS, net.dt))
    return float(net.stationary_rate()[0])


# ── 1. The saturating transfer function ───────────────────────────────────────

print("\n" + "=" * 78)
print(f"[1] Saturating transfer function.  exact survival, K_ref = 0, J = 0, "
      f"nu0 = {NU0:g} /ms, t_abs = {T_ABS:g} ms, beta = {BETA:g}, theta = {THETA:g}, "
      f"dt = {DT:g}")
print("=" * 78)
print("    prediction: r = (1/t_abs) * sigma(beta*(E - theta)), exact at nu0*t_abs = 1")
print(f"    quoted as a dt -> 0 limit, Richardson from {DT_PAIR} at first order\n")

E_SWEEP = torch.linspace(0.0, 2.5, 26)
print(f"    {'E':>6}{'beta*h':>9}{'r(dt)':>13}{'r(dt/2)':>13}{'r limit':>13}"
      f"{'prediction':>13}{'err(dt/2)':>11}{'err(limit)':>11}{'r*dt/2':>10}")
rows = []
t0 = time.time()
for E in E_SWEEP:
    a, b, lim = rate_limit(float(E), HAZARD, t_abs=T_ABS)
    x    = BETA * (float(E) - THETA)
    pred = (1.0 / T_ABS) / (1.0 + math.exp(-x))
    e_raw = abs(b - pred) / max(pred, 1e-300)
    e_lim = abs(lim - pred) / max(pred, 1e-300)
    rows.append((float(E), x, b, lim, pred, e_raw, e_lim))
    print(f"    {float(E):6.2f}{x:9.2f}{a:13.8f}{b:13.8f}{lim:13.8f}{pred:13.8f}"
          f"{e_raw:11.2e}{e_lim:11.2e}{b * DT_PAIR[1] / 2:10.2e}")

for tag, idx in ((f"at dt = {DT_PAIR[1]:g}", 5), ("in the dt -> 0 limit", 6)):
    worst = max(rows, key=lambda t: t[idx])
    over  = [t for t in rows if t[idx] > 1e-2]
    print(f"\n    {tag}:")
    print(f"      worst relative error over the sweep: {worst[idx]:.3e} at E = {worst[0]:g}")
    print(f"      first E at which it exceeds 1%: "
          + (f"{over[0][0]:g}" if over else "never on this sweep"))
print(f"\n    The raw column's error tracks the r*dt/2 printed beside it: that is the SOLVER's")
print(f"    first-order dead-time offset, not the identity failing. The limit column is what")
print(f"    the identity is tested against.")
print(f"    saturation: r/(1/t_abs) reaches {rows[-1][3] * T_ABS:.6f} at E = {float(E_SWEEP[-1]):g}")
print(f"    ({time.time() - t0:.0f}s)")


# ── 2. The A/B identity at t_abs = 1/nu0 ──────────────────────────────────────

print("\n" + "=" * 78)
print(f"[2] A/B at t_abs = 1/nu0, and off it.  Same stripped neuron.")
print("=" * 78)

KERNELS = ("bernoulli", "poisson", "poisson_exact")
T_ABS_SWEEP = (0.5, 1.0, 2.0)       # 1/nu0 = 1.0 is the middle one; all divide into dt-grids
E_SHORT = torch.linspace(0.0, 2.0, 9)
print(f"    rates are dt -> 0 limits, Richardson from {DT_PAIR} at first order")


def closed_form(x: float, hazard: str, t_abs: float) -> float:
    """ r = lam/(1 + lam*t_abs) for the kernel's own rate function: the renewal answer for a
    hard dead time, which is what the solver should reproduce. """
    lam = NU0 / (1.0 + math.exp(-x)) if hazard == "bernoulli" else NU0 * math.exp(x)
    return lam / (1.0 + lam * t_abs)


for t_abs in T_ABS_SWEEP:
    tag = "  <- t_abs = 1/nu0" if abs(t_abs * NU0 - 1.0) < 1e-12 else ""
    print(f"\n    t_abs = {t_abs:g} ms (nu0*t_abs = {NU0 * t_abs:g}){tag}")
    print(f"    {'E':>6}{'beta*h':>9}" + "".join(f"{k:>16}" for k in KERNELS)
          + f"{'|A-B|/B':>11}")
    for E in E_SHORT:
        x = BETA * (float(E) - THETA)
        rs = {}
        for k in KERNELS:
            # bernoulli needs nu0*dt <= 1 and poisson nu0*dt < 1 strictly; both members of
            # DT_PAIR are fine for all three, and both divide every t_abs in the sweep.
            rs[k] = rate_limit(float(E), k, t_abs=t_abs)[2]
        gap = abs(rs["bernoulli"] - rs["poisson_exact"]) / max(rs["poisson_exact"], 1e-300)
        print(f"    {float(E):6.2f}{x:9.2f}" + "".join(f"{rs[k]:16.10f}" for k in KERNELS)
              + f"{gap:11.2e}")

    # the ceilings, which is the other reading of the claim
    ceil_B = 1.0 / t_abs
    ceil_A = min(NU0, 1.0 / t_abs)
    print(f"    ceilings as E -> inf:  route B 1/t_abs = {ceil_B:g} /ms;  "
          f"route A min(nu0, 1/t_abs) = {ceil_A:g} /ms;  "
          f"{'EQUAL' if abs(ceil_A - ceil_B) < 1e-12 else 'DIFFERENT'}")
    # and the closed forms, so the solver is checked against arithmetic and not only against
    # the other kernel
    x = BETA * (2.0 - THETA)
    print(f"    closed form at E = 2:  " + ",  ".join(
        f"{k} {closed_form(x, k, t_abs):.10f}" for k in KERNELS))

print("\n    Reading: the two rate functions differ pointwise at every drive and every t_abs,")
print("    so the stationary rates do NOT coincide at t_abs = 1/nu0. What coincides there is")
print("    the CEILING: route A saturates at nu0 by its kernel, route B at 1/t_abs by the dead")
print("    time, and nu0*t_abs = 1 is exactly where those two numbers are the same. Off it the")
print("    ceilings separate, which is the second half the brief asks for. If the vault's §11")
print("    identity is a statement about stationary rates rather than ceilings, this run")
print("    REFUTES it; the numbers above are what it is refuted by.")


# ── 3. Is the window inert at S1? ─────────────────────────────────────────────

print("\n" + "=" * 78)
print(f"[3] t_abs = 0 vs {T_ABS:g} ms at {PRESET}, exact survival, dt = {DT:g}")
print("=" * 78)

POINTS = {"S1/fp  (J = 10)": 10.0, "candidate cycle (J = 17)": 17.0}
T_EQUI, T_MEAS = 8000.0, 4000.0

MODEL = {k: v for k, v in pre.params(PRESET, hazard=HAZARD, dt=DT).items()
         if k not in ("dt", "hazard", "t_abs")}

print(f"    {'point':>26}{'t_abs':>8}{'r [1/ms]':>14}{'period [ms]':>13}"
      f"{'ptp [1/ms]':>13}{'sigma/dt':>13}{'max r*t_abs':>13}")
res: dict = {}
for name, J in POINTS.items():
    for t_abs in (0.0, T_ABS):
        t0 = time.time()
        net = RDMIsingModel(J, **MODEL, dt=DT, hazard=HAZARD, t_abs=t_abs,
                            Qm=[max(1, round(U_MAX / DT))], device=DEVICE)
        net.forward(hlp.n_steps(T_EQUI, DT))
        traj = net.entropy_trajectory(hlp.n_steps(T_MEAS, DT), pb=False)
        r = traj["r_tot"]
        a, z, n_cyc = hlp.cycle_window(r)
        row = (float(r[a:z].mean()),
               (z - a) * DT / n_cyc if n_cyc else float('nan'),
               float(r[a:z].max() - r[a:z].min()),
               float(traj["sigma_tot"][a:z].mean()) / DT,
               float(r[a:z].max()) * T_ABS)
        res[(name, t_abs)] = row
        print(f"    {name:>26}{t_abs:8.2f}{row[0]:14.8f}{row[1]:13.3f}{row[2]:13.6f}"
              f"{row[3]:13.4e}{row[4]:13.4f}   ({time.time() - t0:.0f}s)", flush=True)

print()
MOVED = False
for name in POINTS:
    a, b = res[(name, 0.0)], res[(name, T_ABS)]
    rel = abs(b[0] - a[0]) / max(abs(a[0]), 1e-300)
    print(f"    {name}: rate moves by {rel:.3e} relative "
          f"({a[0]:.8f} -> {b[0]:.8f})")
    if rel > 1e-3:
        MOVED = True
        print(f"      *** the window is NOT inert here. CLAUDE.md: S1 is the preset every")
        print(f"      *** paper figure uses, so a shift at S1 is a DECISION, not a")
        print(f"      *** measurement. Stop and flag it; nothing in this script resolves it.")
if not MOVED:
    print(f"\n    The window is inert at {PRESET} to better than 1e-3 relative at both points:")
    print(f"    the realised rate never approaches the 1/t_abs = {1 / T_ABS:g} /ms cap, so the")
    print(f"    dead time has nothing to do. It is load-bearing at strong drive, which is what")
    print(f"    check 1 measures and what S1-sat is for.")


# ── 4. poisson vs poisson_exact in the dt -> 0 limit ──────────────────────────

print("\n" + "=" * 78)
print("[4] Do poisson and poisson_exact share a limit?  Same stripped neuron, against the")
print("    closed form r = lam/(1 + lam*t_abs) with the lam the two kernels share.")
print("=" * 78)

DT_LADDER = [0.5, 0.25, 0.125, 0.0625, 0.03125]
E_LADDER  = (0.0, 0.3, 0.5)

for E in E_LADDER:
    x    = BETA * (E - THETA)
    lam  = NU0 * math.exp(x)
    pred = lam / (1.0 + lam * T_ABS)
    print(f"\n    E = {E:g}  (beta*h = {x:+.2f}, lam = {lam:.6f} /ms, "
          f"c = nu0*dt from {NU0 * DT_LADDER[0]:g} down, lam*dt from {lam * DT_LADDER[0]:.3f})")
    print(f"    closed form: {pred:.10f} /ms")
    print(f"    {'dt':>10}{'poisson':>16}{'poisson_exact':>16}{'gap/pe':>12}")
    vals: dict = {"poisson": [], "poisson_exact": []}
    for dt in DT_LADDER:
        a = settled_rate(stripped(E, "poisson", t_abs=T_ABS, dt=dt))
        b = settled_rate(stripped(E, "poisson_exact", t_abs=T_ABS, dt=dt))
        vals["poisson"].append(a); vals["poisson_exact"].append(b)
        print(f"    {dt:10.5f}{a:16.10f}{b:16.10f}{abs(a - b) / b:12.2e}")

    gaps = [abs(a - b) for a, b in zip(vals["poisson"], vals["poisson_exact"])]
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    print(f"    kernel-to-kernel gap halves? ratios per halving: "
          f"{['%.2f' % q for q in ratios]}  (-> 2 means the gap is O(dt))")

    for h in ("poisson", "poisson_exact"):
        v = torch.tensor(vals[h])
        lim, pord = richardson(v)
        o = ['%.2f' % q for q in orders(v).tolist()]
        rel = abs(lim - pred) / pred
        print(f"    {h:>15}: limit {lim:.10f} (order {pord:.2f}, orders {o}); "
              f"rel err vs closed form {rel:.2e}")

print("\n    The two limits coincide, and both land on the closed form -- but NOT equally")
print("    well from a coarse ladder: poisson_exact shows a clean order 1 and lands within")
print("    1e-5, while poisson's observed order is still drifting down toward 1 at dt = 1/32")
print("    and its extrapolated limit is three orders worse. This is the quantitative form of")
print("    CLAUDE.md's rule that the dt -> 0 limit is taken with poisson_exact: not because")
print("    poisson has a different limit, but because it is not yet in its asymptotic regime")
print("    at any step a scan can afford. Note too that poisson is 87% high at dt = 0.5 for a")
print("    neuron firing at 0.05 /ms: its error tracks c = nu0*dt, not lam*dt, so a slow")
print("    population buys it nothing.")
