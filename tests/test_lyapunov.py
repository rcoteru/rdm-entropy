""" Tests for the model-free half of rdme.lyapunov: the driver, not the tangent map.

Nothing here builds a network: the driver takes an `advance` callable and a basis, so it can be
checked against maps whose spectra are known in closed form, which is a far sharper reference
than any model supplies. The model-specific half -- the tangent map itself -- is tested in
test_srm.py against autodiff on the map it linearizes.
"""

import pytest
import torch

import rdme.lyapunov as lyap


@pytest.fixture(autouse=True)
def float64():
    """ The linear-map spectra below are asserted to 1e-9; float32 would not reach it. """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def test_orthonormalize_recovers_the_norms_it_removes():
    """ log|R_ii| of a matrix built with known column norms from an orthonormal start. """
    torch.manual_seed(0)
    B, k, Q = 3, 4, 7
    blocks = [torch.randn(B, k, Q), torch.randn(B, k, Q)]
    lyap.orthonormalize(blocks)                       # start from an orthonormal basis

    scale = torch.tensor([2.0, 0.5, 3.0, 0.25]).expand(B, k).contiguous()
    for b in blocks:
        b *= scale.unsqueeze(-1)                      # columns are still orthogonal, now scaled

    logs = lyap.orthonormalize(blocks)
    assert torch.allclose(logs, scale.log(), rtol=0, atol=1e-12)

    gram = sum(torch.einsum('bia,bja->bij', b, b) for b in blocks)
    assert torch.allclose(gram, torch.eye(k).expand_as(gram), rtol=0, atol=1e-12)


def test_benettin_recovers_a_known_linear_spectrum():
    """ For a linear map x -> A x the exponents are log|eigenvalues of A| / dt exactly. """
    dt = 0.5
    diag = torch.tensor([0.9, 0.6, 0.3, 0.05])
    A = torch.diag(diag)
    # rotate so the answer is not trivially aligned with the coordinate axes
    torch.manual_seed(0)
    Qr, _ = torch.linalg.qr(torch.randn(4, 4))
    A = Qr @ A @ Qr.T

    blocks = [torch.randn(2, 4, 4)]
    lyap.orthonormalize(blocks)

    def advance(bl):
        bl[0] = torch.einsum('ab,ib->ia', A, bl[0].reshape(-1, 4)).reshape(bl[0].shape)

    out = lyap.benettin(advance, blocks, n_steps=400, dt=dt,
                        cadence=5, n_blocks=8, warmup=300)
    want = torch.log(diag) / dt
    assert torch.allclose(out.lam[0], want, rtol=0, atol=1e-9), out.lam[0] - want
    assert torch.allclose(out.lam[1], want, rtol=0, atol=1e-9)
    assert out.T == pytest.approx(400 * dt)


def test_benettin_returns_exponents_in_order():
    """ QR orders the exponents only asymptotically, and classify_spectrum reads column 0 --
    a positive exponent landing in column 1 would be silently missed. """
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([0.5, 0.95, 0.7]))     # deliberately NOT in descending order
    blocks = [torch.randn(4, 3, 3)]
    lyap.orthonormalize(blocks)

    def advance(bl):
        bl[0] = torch.einsum('ab,ib->ia', A, bl[0].reshape(-1, 3)).reshape(bl[0].shape)

    out = lyap.benettin(advance, blocks, n_steps=300, dt=1.0, cadence=5, n_blocks=6, warmup=300)
    assert bool((out.lam[:, :-1] >= out.lam[:, 1:]).all()), out.lam
    want = torch.tensor([0.95, 0.7, 0.5]).log()
    assert torch.allclose(out.lam[0], want, rtol=0, atol=1e-9), out.lam[0] - want


def test_classification_separates_a_locked_cycle_from_a_dead_cell():
    """ The map-vs-flow trap: both have lam_1 < 0, and only the `moving` bit tells them apart. """
    lam = torch.tensor([[-0.01, -0.02],      # not moving -> fixed point
                        [-0.01, -0.02],      # moving     -> locked cycle
                        [ 0.0,  -0.02],      # marginal   -> invariant circle
                        [ 0.0,   0.0],       # two zeros  -> 2-torus
                        [ 0.01, -0.02]])     # positive   -> chaos
    moving = torch.tensor([False, True, True, True, True])
    got = lyap.classify_spectrum(lam, tol=1e-3, moving=moving)
    assert got.tolist() == [lyap.FIXED, lyap.LOCKED, lyap.CIRCLE, lyap.TORUS, lyap.CHAOS]


def test_classification_uses_per_cell_error_bars():
    """ A scalar threshold calls a noisy cell chaotic; its own error bar does not. This is the
    distinction that turned 138 'chaotic' cells into 3 on the real sweep. """
    lam = torch.tensor([[2e-3, -1e-2],      # well resolved, genuinely positive
                        [8e-4, -1e-2]])     # positive but within its own noise
    se  = torch.tensor([[2e-4, 1e-4],
                        [9e-4, 1e-4]])
    moving = torch.tensor([True, True])

    loose = lyap.classify_spectrum(lam, tol=5e-4, moving=moving)
    assert loose.tolist() == [lyap.CHAOS, lyap.CHAOS], "scalar tol cannot tell these apart"

    tight = lyap.classify_spectrum(lam, tol=5e-4, moving=moving, se=se, n_sigma=3.0)
    assert tight.tolist() == [lyap.CHAOS, lyap.CIRCLE]


def test_kaplan_yorke_known_cases():
    # all negative -> dimension 0
    assert lyap.kaplan_yorke(torch.tensor([[-0.1, -0.2]]))[0] == 0.0
    # one zero exponent then a strongly negative one -> an invariant circle, dimension 1
    assert lyap.kaplan_yorke(torch.tensor([[0.0, -0.2]]))[0] == pytest.approx(1.0)
    # lam1 + lam2 still positive within the k supplied -> not determined, and says so
    assert torch.isnan(lyap.kaplan_yorke(torch.tensor([[0.3, 0.2]]))[0])
    # textbook: 1 + lam1/|lam2|
    assert lyap.kaplan_yorke(torch.tensor([[0.1, -0.4]]))[0] == pytest.approx(1.25)


# Map vs flow semantics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_continuous_mode_defaults_off():
    """ continuous=False is the historical behaviour, so an un-updated caller keeps the map
    reading rather than silently acquiring the flow one. """
    lam = torch.tensor([0.0, -0.2])
    se = torch.tensor([1e-4, 1e-4])
    mv = torch.tensor(True)
    assert lyap.classify_spectrum(lam, tol=5e-4, moving=mv, se=se) == lyap.CIRCLE


def test_a_zero_leading_exponent_is_a_cycle_for_a_flow_and_a_circle_for_a_map():
    """ The one cell where the two tables disagree, and the one that dominates these sweeps.

    lam_1 ~ 0 > lam_2 is an invariant circle of a map -- quasiperiodicity -- but for a flow it
    is the phase direction of an ordinary limit cycle. Reading a refined model with the map
    table turns every oscillation into a claim of quasiperiodicity. """
    lam = torch.tensor([0.0, -0.2])
    se = torch.tensor([1e-4, 1e-4])
    mv = torch.tensor(True)
    assert lyap.classify_spectrum(lam, tol=5e-4, moving=mv, se=se, continuous=False) == lyap.CIRCLE
    assert lyap.classify_spectrum(lam, tol=5e-4, moving=mv, se=se, continuous=True) == lyap.LOCKED


def test_the_other_cells_agree_between_the_two_tables():
    """ Only the near-zero lam_1 cell moves; fixed points, tori and chaos read the same either
    way, which is what makes the codes mean the attractor rather than the signature. """
    se = torch.tensor([1e-4, 1e-4])
    cases = [([-0.05, -0.20], False, lyap.FIXED),
             ([ 0.00,  0.00], True,  lyap.TORUS),
             ([ 0.02, -0.20], True,  lyap.CHAOS),
             ([ 0.02,  0.01], True,  lyap.CHAOS)]
    for vals, mv, want in cases:
        lam, m = torch.tensor(vals), torch.tensor(mv)
        for cont in (False, True):
            got = lyap.classify_spectrum(lam, tol=5e-4, moving=m, se=se, continuous=cont)
            assert got == want, (vals, cont, got)


def test_a_moving_cell_with_negative_lam1_is_a_cycle_in_both_modes():
    """ In a map this is a locked cycle. In a flow it is a limit cycle whose O(dt) phase bias
    escaped the zero band -- the failure mode of a tol that has not been scaled with dt. Both
    are LOCKED, so an under-resolved cycle degrades to 'cycle' rather than to 'dead'. """
    lam = torch.tensor([-0.05, -0.20])
    se = torch.tensor([1e-4, 1e-4])
    for cont in (False, True):
        got = lyap.classify_spectrum(lam, tol=5e-4, moving=torch.tensor(True),
                                     se=se, continuous=cont)
        assert got == lyap.LOCKED, cont


def test_class_names_track_the_regime():
    """ CIRCLE is unreachable in continuous mode, and the name for LOCKED changes: printing
    'invariant circle' for a flow's limit cycle is the same error as classifying it that way. """
    assert lyap.class_names(False)[lyap.CIRCLE] == "invariant circle"
    assert lyap.class_names(True)[lyap.LOCKED] == "limit cycle"
    lam = torch.randn(500, 2).sort(dim=-1, descending=True).values * 0.01
    se = torch.full((500, 2), 1e-4)
    cls = lyap.classify_spectrum(lam, tol=5e-4, moving=torch.ones(500, dtype=torch.bool),
                                 se=se, continuous=True)
    assert not (cls == lyap.CIRCLE).any(), "CIRCLE must never be emitted for a flow"
