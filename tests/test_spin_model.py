""" Tests for rdme.spin_model's entropy production: the transition pairing, the chunked
estimator, and the time origin it shares with the mean field.

These were in the mean-field entropy test file because they were written alongside it; they
test the spin model, so they live here now. """

import pytest
import torch

from rdme.single import RDMIsingModel, RDMWilsonCowan


torch.manual_seed(0)


def test_entropy_trajectories_share_a_time_origin():
    """ All three entropy_trajectory variants must report index 0 for the state they are
    called on, so a mean-field run and a spin run of the same model overlay step for step.
    The spin ones used to discard a lead-in first (chunked: a burn_in that defaulted to
    overlap = 512), which offset the curves by that many steps -- invisible in the
    equilibration plots, which use trajectory() and have no lead-in, and easy to misread as
    noise in an oscillatory regime where the offset aliases against the period. That lead-in
    is gone: equilibrating is forward()'s job, so there is nothing left to get out of sync. """
    from rdme.spin_model import SpinIsingModel

    torch.manual_seed(3)
    dt = 0.2
    args = dict(J=4 / dt, E=1, beta=30, theta=1, tau_int=12, tau_ref=3, K_ref=0, dt=dt)
    T = 60

    def spin():
        torch.manual_seed(3)
        m = SpinIsingModel(N=2000, ic="silent", device="cpu",
                           **{k: v for k, v in args.items()})
        m.forward(50)
        return m

    # index 0 of each spin variant is the activity of the state it was called on
    for run in (lambda m: m.entropy_trajectory(T, buffer=20),
                lambda m: m.entropy_trajectory_chunked(T, chunk=64, overlap=16,
                                                       check_overlap=False)):
        m = spin()
        m0 = m.activity().item()
        assert run(m)["m_tot"][0].item() == pytest.approx(m0, abs=1e-6)

    # and the mean field agrees: its index 0 is the overlap it was sitting at
    net = RDMIsingModel(**{**args, "J": args["J"] * dt},
                            hazard="synchronous", eps=0.01, device="cpu")
    net.forward(50)
    p0 = net.m[0].item()
    assert net.entropy_trajectory(T, pb=False)["m_tot"][0].item() == pytest.approx(p0, abs=1e-12)

    # and no lead-in knob survives anywhere to reintroduce a silent offset
    import inspect
    for fn in (SpinIsingModel(N=1, ic="silent", device="cpu", **args).entropy_trajectory,
               SpinIsingModel(N=1, ic="silent", device="cpu", **args).entropy_trajectory_chunked,
               net.entropy_trajectory):
        assert "burn_in" not in inspect.signature(fn).parameters


def test_spin_epr_pairs_both_halves_of_the_same_transition():
    """ sigma[t] is a property of the transition t -> t+1, per eq. (total-epr): the forward
    half emits s[t+1] from hf[t], and the reverse half emits s[t] from hr[t+1]. The spin model
    used to pair hr[t+2] with s[t+1] -- matching on the realized spike rather than on the bond
    -- which sums to the same total EPR but shifts H_rev one step against H_fwd. """
    from rdme.spin_model import SpinIsingModel

    torch.manual_seed(7)
    dt = 0.2
    T = 80
    m = SpinIsingModel(N=3000, J=4 / dt, E=1, beta=30, theta=1, tau_int=12, tau_ref=3,
                       K_ref=0, dt=dt, device="cpu", ic="silent")
    m.forward(400)
    out = m.entropy_trajectory(T, buffer=60, s=True, fields=True)

    softplus = torch.nn.functional.softplus
    s_t = out["s"].float()                      # s[t], index-aligned with the report

    # the reverse half is emitted against s[t] -- this is the pairing under test
    H_rev = (-s_t * out["hr"] + softplus(out["hr"])).mean(dim=1)
    assert torch.allclose(H_rev, out["H_rev_tot"].float(), atol=1e-5)

    # the forward half against s[t+1], which is s_t shifted by one
    H_fwd = (-s_t[1:] * out["hf"][:-1] + softplus(out["hf"][:-1])).mean(dim=1)
    assert torch.allclose(H_fwd, out["H_fwd_tot"][:-1].float(), atol=1e-5)

    # pairing the reverse half against s[t+1] instead is the old convention, and it is
    # a genuinely different per-step series -- so the assertion above has teeth
    old = (-s_t[1:] * out["hr"][:-1] + softplus(out["hr"][:-1])).mean(dim=1)
    assert not torch.allclose(old, out["H_rev_tot"][:-1].float(), atol=1e-3)

    # identical up to float32 rounding: both come from the same ent_f/ent_r reductions
    assert torch.allclose(out["sigma_tot"], out["H_rev_tot"] - out["H_fwd_tot"], atol=1e-6)


def test_spin_epr_chunked_matches_the_windowed_reference():
    """ The chunked estimator is the same quantity as the plain one, so both had to take the
    transition pairing together; a slip in either index would show up right here. """
    from rdme.spin_model import SpinIsingModel

    dt = 0.2
    T = 60
    kw = dict(N=4000, J=4 / dt, E=1, beta=30, theta=1, tau_int=12, tau_ref=3,
              K_ref=0, dt=dt, device="cpu", ic="silent")

    # both runs have to sample the SAME trajectory to be compared step by step, and the
    # draws depend on the default dtype -- so pin it rather than inherit it from whichever
    # test ran last
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(5)
        a = SpinIsingModel(**kw); a.forward(400)
        torch.manual_seed(99)
        ref = a.entropy_trajectory(T, buffer=80)

        torch.manual_seed(5)
        b = SpinIsingModel(**kw); b.forward(400)
        torch.manual_seed(99)
        got = b.entropy_trajectory_chunked(T, chunk=80, overlap=80, check_overlap=False)
    finally:
        torch.set_default_dtype(old_dtype)

    # the trajectories are identical, so any gap is an index slip, not sampling noise
    assert torch.allclose(ref["m_tot"].double(), got["m_tot"].double(), atol=1e-10)
    for k in ("sigma", "H_fwd", "H_rev"):
        assert torch.allclose(ref[k].double(), got[k].double(), atol=1e-4), k


def test_entropy_trajectory_leaves_the_model_at_index_T():
    """ Both spin estimators run past the reported window -- the plain one by `buffer`, the
    chunked one by up to chunk+overlap -- because the reverse recursion needs the lookahead.
    That overshoot is not simulation the caller asked for, so the model is rewound onto the
    end of the window, exactly as RDMNetwork.entropy_trajectory rewinds p and S. Without it,
    a post-hoc fdist() is sampled at the wrong phase and will not line up with a mean-field
    p(n) run for the same T. """
    from rdme.spin_model import SpinIsingModel

    dt = 0.2
    T = 300
    kw = dict(N=5000, J=4 / dt, E=1, beta=30, theta=1, tau_int=12, tau_ref=3,
              K_ref=0, dt=dt, device="cpu", ic="silent")

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        def fresh():
            torch.manual_seed(11)
            m = SpinIsingModel(**kw)
            m.forward(200)
            return m

        torch.manual_seed(42); ref = fresh(); ref.forward(T)         # ground truth
        torch.manual_seed(42); a = fresh(); a.entropy_trajectory(T, buffer=80)
        torch.manual_seed(42); b = fresh()
        b.entropy_trajectory_chunked(T, chunk=256, overlap=128, check_overlap=False)

        # the evolving state is exactly (s, n, S, R) -- all four must land on forward(T)
        for name, m in (("entropy_trajectory", a), ("entropy_trajectory_chunked", b)):
            for k in ("s", "n", "S", "R"):
                assert torch.equal(getattr(ref, k), getattr(m, k)), f"{name}: {k}"

        # and the age distribution that the mean field is compared against follows
        assert torch.equal(ref.fdist(100), a.fdist(100))
        assert torch.equal(ref.fdist(100), b.fdist(100))
    finally:
        torch.set_default_dtype(old_dtype)


def test_spin_and_mean_field_entropy_trajectories_share_a_dict_contract():
    """ Both entropy_trajectory families return per-population (T, M) arrays plus _tot
    network aggregates under the same key names, so a script can plot either without
    knowing which model produced it. The spin side used to return network-wide scalars
    under the bare names, with no per-population breakdown at all -- which left the
    Wilson-Cowan phase maps unplottable from a spin run. """
    from rdme.spin_model import SpinWilsonCowan

    torch.manual_seed(13)
    dt = 0.5
    T = 40
    kw = dict(E_ratio=0.8, w_EE=5.0 / dt, w_EI=3.0 / dt, w_IE=-6.0 / dt, w_II=-0.5 / dt,
              E_exc=1.0, E_inh=0.5, beta_E=10.0, beta_I=10.0, theta_E=1.0, theta_I=1.0,
              tau_int_E=10.0, tau_int_I=5.0, tau_ref_E=3.0, tau_ref_I=3.0)

    # the spin model takes w pre-divided by dt; srm takes it in physical units, so the two
    # need different weight kwargs for the same network
    srm_kw = {**kw, **{k: kw[k] * dt for k in ("w_EE", "w_EI", "w_IE", "w_II")}}
    net = RDMWilsonCowan(K_ref_E=0.0, K_ref_I=0.0, dt=dt, hazard="synchronous", **srm_kw)
    net.forward(200)
    mf_out = net.entropy_trajectory(T, pb=False)

    spin = SpinWilsonCowan(N=4000, K_ref1=0.0, K_ref2=0.0, dt=dt, device="cpu",
                           ic="silent", **kw)
    spin.forward(200)
    sm_out = spin.entropy_trajectory(T, buffer=60)
    sm_chunk = SpinWilsonCowan(N=4000, K_ref1=0.0, K_ref2=0.0, dt=dt, device="cpu",
                               ic="silent", **kw)
    sm_chunk.forward(200)
    ck_out = sm_chunk.entropy_trajectory_chunked(T, chunk=64, overlap=32,
                                                 check_overlap=False)

    per_pop = ("m", "sigma", "H_fwd", "H_rev")
    totals  = ("m_tot", "sigma_tot", "H_fwd_tot", "H_rev_tot")

    for out in (mf_out, sm_out, ck_out):
        for k in per_pop:
            assert k in out, k
            assert out[k].shape == (T, 2), (k, out[k].shape)
        for k in totals:
            assert k in out, k
            assert out[k].shape == (T,), (k, out[k].shape)

    # both constructors must split N the same way for the same E_ratio
    mf_ratios = net.N_ratios.double()
    sm_ratios = (spin.Nm / spin.N).double()
    assert torch.allclose(mf_ratios, sm_ratios, atol=1e-12)
    assert torch.allclose(mf_ratios, torch.tensor([0.8, 0.2], dtype=torch.float64), atol=1e-12)

    # the aggregates are the N-weighted combination of the per-population entries
    for out, ratios in ((mf_out, mf_ratios), (sm_out, sm_ratios), (ck_out, sm_ratios)):
        for k in per_pop:
            got = (out[k].double() * ratios).sum(dim=1)
            assert torch.allclose(got, out[f"{k}_tot"].double(), atol=1e-6), k
        assert torch.allclose(out["sigma"].double(),
                              (out["H_rev"] - out["H_fwd"]).double(), atol=1e-6)