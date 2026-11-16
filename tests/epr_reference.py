""" The joint-distribution EPR: the reference oracle the library's O(Q) observer is checked
against. Test support, not a test module.

Moved here from rdme.epr in Step 3 of `P1 2026-11-13 code rebuild` (Target section 4): the
library computes the EPR only by the sliding observer (rdme.epr.epr_step), and the explicit
(Q, K) joint of forward and reverse age -- the definition, O(Q^2) per step -- lives only in
tests. It shares epr_from_parts with the observer, so the two can only disagree through the
four marginals they hand it.

`joint_entropy_trajectory(net, T)` is what `net.entropy_trajectory(T, method="joint")` was:
the driver's own observer loop, fed this module's (init, step) pair instead of the sliding one.
"""

from typing import NamedTuple

import torch

import rdme.rdm as rdm
from rdme.epr import (EPRParams, _cumlog1m, anchor_reverse_synaptic_term, epr_from_parts,
                      hazard_field)
from rdme.hazards import Hazard


def joint_distribution(p: torch.Tensor,        # (Q,)   p_t(n)
                       buff_S: torch.Tensor,   # (K, Q) buff_S[j, n] = S_{t+j}(n), j=0 is t
                       R: torch.Tensor,        # (Q,)
                       beta: float | torch.Tensor,
                       theta: float | torch.Tensor,
                       hazard: Hazard,
                       nu0: float,
                       dt: float,
                       inplace: bool = False,
                       ) -> torch.Tensor:      # (Q, K) p_t(n, n_dagger=k)
    """ Joint distribution of forward and reverse age, built from hazards.

    The hazard-parametrised port of mean_field.compute_joint_distribution; the structure is
    unchanged. A cohort's survival prod_{j<k} (1 - Phi_{t+j}(n+j)) is built by multiplying
    hazards rather than reading density ratios, which is what lets it track a cohort through
    the lumped final bin and means only the marginal p_t is ever needed. Rows sum to 1 by
    telescoping, so no normalization.

    inplace folds the chain of (Q, K) temporaries into three live buffers, in the same order so
    the two branches agree bit for bit; buff_S, R and p are read-only either way, only the
    intermediates this function allocates itself are reused. It is the expensive flag in the
    package: this is the one (B, Q, K) object in an EPR trajectory, so each temporary dropped
    is the whole joint again. """
    Q, K = buff_S.shape[-1], buff_S.shape[-2]
    # buff_S carries an extra lag axis that R, beta and theta do not: they are (..., Q) and
    # (..., 1) as everywhere else in the package, so each is lifted one axis here rather than
    # making every caller remember a different convention for these two functions. A 0-dim
    # scalar becomes (1,), which broadcasts just as well.
    lift = lambda x: x.unsqueeze(-1) if torch.is_tensor(x) else x
    R, beta, theta = R.unsqueeze(-2), lift(beta), lift(theta)
    n_idx = torch.arange(Q, device=buff_S.device)
    j_idx = torch.arange(K, device=buff_S.device)
    future_age  = torch.clamp(n_idx.unsqueeze(1) + j_idx.unsqueeze(0), max=Q - 1)   # (Q, K)
    future_time = j_idx.unsqueeze(0)                                                # (1, K)

    phi_all = hazard.phi(hazard_field(buff_S, R, theta, inplace), beta, nu0, dt, inplace=inplace)
    # the two index tensors consume the last two axes, so the ellipsis carries any batch
    phi = phi_all[..., future_time, future_age]                     # (..., Q, K)
    del phi_all             # the gather already copied; holding the source doubles the peak

    if inplace:
        log_surv = _cumlog1m(phi, True)
        surv = torch.empty_like(phi)
        surv[..., 0] = 1.0
        torch.exp(log_surv[..., :-1], out=surv[..., 1:])
        del log_surv
        surv[..., :-1].mul_(phi[..., :-1])   # first passage; last column keeps bare S
        return surv.mul_(p.unsqueeze(-1))

    log_surv = _cumlog1m(phi, False)
    surv = torch.ones_like(phi)
    surv[..., 1:] = torch.exp(log_surv[..., :-1])
    surv[..., :-1] = surv[..., :-1] * phi[..., :-1]  # first passage; last column keeps bare S
    return surv * p.unsqueeze(-1)



def epr_from_joint(p_joint: torch.Tensor,   # (Q, K)
                   phi_fwd: torch.Tensor,   # (Q,)
                   phi_rev: torch.Tensor,   # (K,)
                   ) -> torch.Tensor:       # (3,) [sigma, H_fwd, H_rev]
    """ The EPR read off the explicit joint: the reference path, and the oracle the sliding
    recursions are checked against. """
    return epr_from_parts(p_joint.sum(dim=-1), p_joint.sum(dim=-2),
                          p_joint[..., :, 0], p_joint[..., 0, :], phi_fwd, phi_rev)



def synaptic_window(S: torch.Tensor,      # (..., Q) S_t
                    I: torch.Tensor,      # (..., K-1) inputs I_t .. I_{t+K-2}
                    alpha: float | torch.Tensor,
                    ) -> torch.Tensor:    # (..., K, Q) row j = S_{t+j}
    """ The look-ahead block the joint path reads, replayed from S_t and the input window by
    rdm.update_synaptic_term -- the model's own update on the model's own inputs, so every row
    is bit for bit the S the model held at that time. O(K*Q): the oracle's cost, not the
    observer's. """
    out = S.new_empty(S.shape[:-1] + (I.shape[-1] + 1, S.shape[-1]))
    out[..., 0, :] = S
    for j in range(I.shape[-1]):
        out[..., j + 1, :] = rdm.update_synaptic_term(out[..., j, :], alpha, I[..., j:j + 1])
    return out


class JointState(NamedTuple):
    """ What the joint path carries between timesteps: the reported marginal, the model's S
    lagged to the reported time, and the inputs that rebuild the look-ahead from it. """
    p: torch.Tensor   # (..., Q)   p_t
    S: torch.Tensor   # (..., Q)   S_t
    I: torch.Tensor   # (..., K-1) I_t .. I_{t+K-2}


def epr_step_joint(state: JointState, S_live, I_new, prm: EPRParams, inplace: bool = False):
    """ The reference path as an observer: same feed and same return as epr_step, but the EPR
    is read off the explicit (Q, K) joint, rebuilt every step from the lagged S and the input
    window. O(Q^2) per step; it is the oracle the sliding observer is checked against. It
    replays the window's last row itself, so S_live is taken for the shared interface only. """
    R, alpha, beta, theta, hazard, nu0, dt = prm
    I_full = torch.cat([state.I, I_new.unsqueeze(-1)], dim=-1)          # I_t .. I_{t+K-1}
    phi = lambda h: hazard.phi(h, beta, nu0, dt, inplace=inplace)
    J = joint_distribution(state.p, synaptic_window(state.S, state.I, alpha), R, beta, theta,
                           hazard, nu0, dt, inplace=inplace)
    phi_f = phi(hazard_field(state.S, R, theta, inplace))
    phi_r = phi(hazard_field(anchor_reverse_synaptic_term(I_full, alpha), R, theta, inplace))
    e = epr_from_joint(J, phi_f, phi_r)
    nxt = JointState(rdm.update_age_distribution(state.p, phi_f),
                     rdm.update_synaptic_term(state.S, alpha, state.I[..., :1]),
                     I_full[..., 1:])
    return nxt, e[..., 0], e[..., 1], e[..., 2]


def init_joint_state(p, S, I, prm: EPRParams, inplace: bool = False) -> JointState:
    """ The joint path's state at the first reported step: it carries the inputs and rebuilds
    everything else every step. Same signature as rdme.epr.init_epr_state. """
    return JointState(p, S, I)


def joint_entropy_trajectory(net, T: int, pb: bool = False, inplace: bool = False) -> dict:
    """ net.entropy_trajectory with the joint observer in place of the sliding one: the same
    loop, feed, rewind and output assembly (RDMNetwork._observe), so a difference from the
    sliding path is a difference between the two observers and nothing else. """
    return net._observe(T, init_joint_state, epr_step_joint, pb=pb, inplace=inplace)
