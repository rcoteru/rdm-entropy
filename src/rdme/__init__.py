""" Refractory density method for SRM_0 populations, with the hazard as a parameter.

THE MODEL (route B, decided 2026-11-07). A population of SRM_0 neurons whose firing is a
maximum-caliber tilt of a Poisson reference process of rate nu0. Writing x = beta*h with
h(u, t) = S(u, t) + R(u) - theta, c = nu0*dt and ell = logit(c):

    poisson        Phi = sigma(x + ell)            -> rate nu0 e^x      (MaxCal at every dt)
    poisson_exact  Phi = 1 - exp(-nu0 e^x dt)      -> rate nu0 e^x      (exact survival)
    bernoulli      Phi = nu0*dt * sigma(x)         -> rate nu0 sigma(x) (route A, legacy)

One model, two kernels for it. `poisson` and `poisson_exact` share a rate function exactly and
so are the same continuous-time model under two discretization maps; `bernoulli` is route A,
kept for comparison and for equivalence claims about the time-discrete construction this
project started from, and is not the hazard of a measurement.

Which to use. `poisson` is the model of BINNED DATA: maximum caliber against the reference, at
every bin width, so refining dt refines one variational problem instead of walking across a
family of them. `poisson_exact` is the SOLVER for the same continuum: its per-bin error is
single-signed, so it approaches the limit monotonically and can be Richardson-extrapolated,
while `poisson`'s error changes sign at threshold and the extrapolation fails on an attractor
that crosses it. Measurements are quoted as a dt -> 0 limit, by default Richardson from a
coarse pair (dt, dt/2) with `poisson_exact`.

Parameters, and what is NOT one. nu0 is a rate UNIT, 1 ms^-1: the reference's rate, against
which the tilt is measured. It is not identifiable on its own -- (nu0, theta) -> (nu0
e^{beta delta}, theta + delta) leaves Phi unchanged for poisson_exact exactly -- which is why
it is fixed rather than fitted. t_abs is the absolute refractory period in ms, a PHYSICAL dead
time and an independent parameter: the route-B rate is unbounded and exponential, so t_abs is
the only thing that caps the realised firing rate, at 1/t_abs. dt is the solver step and
nothing else; there is no time quantum, no dt0, and no model parameter that moves when the
step does.

The continuum equations are the transport pair

    d_t q^a + d_u q^a = -lambda(h^a(u,t)) q^a(u,t),      q^a(0,t) = r^a(t),
    d_t S^a + d_u S^a = (I^a(t) - S^a(u,t)) / tau_int^a, S^a(0,t) = 0,

with R^a(u) = -K^a exp(-u/tau_ref^a) floored to -R_abs over the dead time, the input
I^a(t) = sum_b w^{ab} r^b(t) + E^a, and the closure r^a(t) = int lambda(h^a(u,t)) q^a(u,t) du.

Scheme. Age and time are advanced together on a single grid, du = dt, so cohorts move along
the characteristics u - t = const exactly and no interpolation or numerical diffusion enters.
Over one bin a cohort's mass is multiplied by 1 - Phi, and the synaptic recursion with
alpha = 1 - exp(-dt/tau_int) is the exact solution of the second equation for an input frozen
across the bin. The only errors are the variation of rate and input within a bin, O(dt) over a
fixed physical time -- bounded locally by lambda*dt/2 for `poisson_exact`
(hazards.step_error).

Call-site conventions:

  Weights are NOT pre-divided by dt. The input is built from the rate r = m/dt directly, so w
  is passed in physical units. Passing w/dt, out of habit, silently rescales the whole network.

  The natural observable is the rate r (per unit time), not the per-bin overlap m. Both are
  exposed, since m is what the discrete implementation reports and the comparison needs it,
  but only r is meaningful across different dt.

  t_abs must be an integer number of bins: kernels.n_abs_bins refuses a dt that does not
  divide it, because a window that rounds moves with the step.

Entropy production is a functional of Phi alone, so it is computed once for every hazard:
epr_from_joint writes the plain Bernoulli entropy rather than the softplus/logit form, which
is an identity of the logistic kernel only. sigma is the reported quantity; the forward and
reverse terms H and H_rev each diverge as rate*ln(1/dt) against a counting measure, and are
reported RELATIVE TO THE POISSON REFERENCE, which is finite as dt -> 0 and is the model's own
reference measure under route B. The raw values are kept as a control (increment per halving
-> r ln 2). """
