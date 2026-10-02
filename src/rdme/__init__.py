""" The mean field with the hazard as a parameter, spanning the synchronous model of
rdme.mean_field and the two continuous-time models of App. "Time Discretization and the
Continuous-Time Limit".

Three per-bin kernels, all passing through the synchronous model at lam0*dt = 1:

    synchronous   Phi = sigma(beta*h)                   no dt -> 0 limit
    async         Phi = lam0*dt * sigma(beta*h)         -> rate lam0*sigma(beta*h)
    escape        Phi = 1 - (1 + e^{beta*h})^(-lam0*dt) -> rate lam0*softplus(beta*h)

The two families differ in the discretization map (linear thinning against exact survival),
which only changes the speed of convergence, and in the rate function, which changes the
limit -- so they converge to different continuous-time models. The gap is set by the time
quantum relative to the kernels, not by the step: measured here at strong drive against
tau_ref = 3, the two rates differ by 3% at dt0 = 0.5, 27% at dt0 = 3 and 104% at dt0 = 12.

rdme.mean_field stays the reference implementation and is deliberately untouched. Everything
here is checked against it at hazard="sigmoid", lam0*dt = 1, where the two models coincide --
so the discrete model is the oracle for the dynamics, the fixed points, the Jacobian spectrum
and the entropy production alike.

The model of rdme.mean_field fixes the firing probability *per bin*, so its escape rate
softplus(beta*h)/dt diverges as dt -> 0: it is a discrete-time model tied to the step at which
it is defined, its time quantum dt0. This module implements the continuous-time model that the
appendix builds instead, in which the *rate* is held fixed,

    lambda(h) = lam0 * softplus(beta*h),          lam0 = 1/dt0,

so the one-bin survival is exp(-lambda*dt) and the hazard

    Phi(beta*h) = 1 - (1 + exp(beta*h))**(-lam0*dt)

reduces to the sigmoid exactly at lam0*dt = 1. Refining dt here refines the *numerics* of one
model, instead of walking across a family of different ones, which is what refining dt does in
rdme.mean_field.

The continuum equations are the transport pair

    d_t q^a + d_u q^a = -lambda(h^a(u,t)) q^a(u,t),      q^a(0,t) = r^a(t),
    d_t S^a + d_u S^a = (I^a(t) - S^a(u,t)) / tau_m^a,   S^a(0,t) = 0,

with h^a(u,t) = S^a(u,t) + R^a(u) - theta^a, R^a(u) = -K^a exp(-u/tau_ref^a), the input
I^a(t) = sum_b w^{ab} r^b(t) + E^a, and the closure r^a(t) = int lambda(h^a(u,t)) q^a(u,t) du.

Scheme. Age and time are advanced together on a single grid, du = dt, so cohorts move along
the characteristics u - t = const exactly and no interpolation or numerical diffusion enters.
Over one bin a cohort's mass is multiplied by 1 - Phi, the exact survival for a rate frozen at
its value at the start of the bin, and the synaptic recursion with alpha = 1 - exp(-dt/tau_m)
is the exact solution of the second equation for an input frozen across the bin. The only
errors are the variation of rate and input within a bin, O(dt) over a fixed physical time.

That scheme is, deliberately, the update of rdme.mean_field with a different hazard -- which is
the appendix's point, that the discrete model is an exponential integrator for these equations,
consistent with them only at dt = dt0. The step functions are imported from there rather than
re-derived so the two cannot drift apart, and `hazard="sigmoid"` reproduces RDMNetwork exactly,
which is how the equivalence is tested.

API mirrors rdme.mean_field.RDMNetwork, with two differences that matter at the call site:

  Weights are NOT pre-divided by dt. RDMNetwork is given w/dt and multiplies by the per-bin
  overlap m; here the input is built from the rate r = m/dt directly, so w is passed in its
  physical units. Passing w/dt here, out of habit, silently rescales the whole network.

  The natural observable is the rate r (per unit time), not the per-bin overlap m. Both are
  exposed, since m is what RDMNetwork reports and the comparison needs it, but only r is
  meaningful across different dt.

Single-system only for now; the batched counterpart comes later.

Entropy production is a functional of Phi alone, so it is computed once for all three
hazards: epr_from_joint writes the plain Bernoulli entropy rather than mean_field's
softplus/logit form, which is an identity of the logistic kernel only. It is computed by the
O(Q^2) reference path (joint distribution -> EPR), not
the O(Q) sliding recursions of rdme.mean_field. The sliding version is ~200 lines of
hand-optimised state updates with the sigmoid baked in, including one bare torch.sigmoid that
bypasses compute_firing_prob; threading a hazard through it is a much larger change with much
more to go wrong, and the reference path is what it was validated against in the first place.
At a few hundred age bins the quadratic cost is seconds per run, not minutes.

A caveat on reading those outputs across resolutions: by the appendix only sigma/dt has a
continuum limit, while H_fwd and H_rev each diverge as -r ln dt, being entropies of a binned
process against a counting measure. They are comparable at fixed dt and not across it. """

