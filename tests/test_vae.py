import numpy as np
import pytest

from aie.data import digits
from aie.gradcheck import numerical_gradient, relative_error
from scripts.vae import VAE, kl_term


@pytest.mark.parametrize(
    ("activation", "n_z", "sigma_x2"), [("tanh", 1, 1.0), ("sigmoid", 3, 1.0), ("relu", 2, 0.1)]
)
def test_gradient_check_full_loss(activation, n_z, sigma_x2):
    rng = np.random.default_rng(0)
    X = rng.random((5, 6))
    vae = VAE(6, n_latent=2, hidden=(4,), activation=activation, seed=3, sigma_x2=sigma_x2)
    eta = rng.standard_normal((n_z, 5, 2))
    vae.backward(vae.forward(X, eta))
    params = vae.parameters()
    analytic = [g.copy() for _, g, *_ in params]
    for (p, *_), g in zip(params, analytic, strict=True):
        num = numerical_gradient(lambda: vae.loss(X, eta), p)
        assert relative_error(g, num) < 1e-6


def test_kl_term_zero_at_prior_and_negative_elsewhere():
    assert np.allclose(kl_term(np.zeros((3, 4)), np.zeros((3, 4))), 0.0)
    rng = np.random.default_rng(1)
    assert np.all(kl_term(rng.standard_normal((10, 2)), rng.standard_normal((10, 2))) < 0)


def test_kl_matches_closed_form_gaussian_kl():
    mu, logvar = np.array([[0.5, -1.0]]), np.array([[0.2, -0.3]])
    var = np.exp(logvar)
    kl = 0.5 * np.sum(var + mu**2 - 1.0 - logvar)
    assert np.isclose(kl_term(mu, logvar)[0], -kl)


def test_decoder_term_matches_eq_8_35():
    rng = np.random.default_rng(2)
    X = rng.random((3, 6))
    vae = VAE(6, n_latent=2, hidden=(4,), seed=0)
    eta = rng.standard_normal((4, 3, 2))
    out = vae.forward(X, eta)
    expected = -np.mean(np.sum((X[None] - out["Xr"]) ** 2, axis=(0, 2)) / (2 * 4))
    assert np.isclose(out["recon"], expected)
    assert np.isclose(out["L"], out["recon"] + out["kl"])


def test_training_on_digits_reduces_loss_with_2d_latent():
    X, _ = digits(n_per_class=30, seed=0)
    X = X.reshape(len(X), -1)
    vae = VAE(144, n_latent=2, hidden=(64,), seed=0)
    eta = np.random.default_rng(5).standard_normal((1, len(X), 2))
    before = vae.loss(X, eta)
    vae.fit(X, n_epochs=30, lr=3e-3, batch_size=32)
    assert vae.loss(X, eta) < 0.7 * before
    assert vae.history["loss"][-1] < vae.history["loss"][0]
    mu, logvar = vae.encode(X)
    assert mu.shape == (len(X), 2) and logvar.shape == (len(X), 2)
    assert vae.sample(7).shape == (7, 144)
    assert vae.reconstruct(X[:4]).shape == (4, 144)
