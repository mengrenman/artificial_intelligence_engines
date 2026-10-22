import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aie.data import digits  # noqa: E402
from aie.losses import softmax_cross_entropy  # noqa: E402
from scripts.deep_backprop import as_images, lenet  # noqa: E402
from scripts.torch_models import (  # noqa: E402
    TorchLeNet,
    TorchVAE,
    load_numpy_weights,
    paired_layers,
    torch_grads_as_numpy,
)
from scripts.vae import VAE  # noqa: E402


def _assert_grads_match(module, np_model, atol=1e-6):
    for (_, lay), tg in zip(paired_layers(module, np_model), torch_grads_as_numpy(module),
                            strict=True):
        for k in ("W", "b"):
            np.testing.assert_allclose(tg[k], lay.grads[k], rtol=1e-6, atol=atol)


@pytest.mark.parametrize(("activation", "n_z", "sigma_x2"),
                         [("tanh", 1, 0.1), ("relu", 3, 1.0), ("sigmoid", 2, 0.5)])
def test_vae_forward_and_gradients_match_numpy(activation, n_z, sigma_x2):
    rng = np.random.default_rng(0)
    X = rng.random((7, 12))
    eta = rng.standard_normal((n_z, 7, 2))
    vae = VAE(12, n_latent=2, hidden=(8, 5), activation=activation, seed=1, sigma_x2=sigma_x2)
    tv = TorchVAE.from_numpy(vae)
    assert tv.dtype == torch.float64

    cache = vae.forward(X, eta)
    vae.backward(cache)
    t = tv.terms(torch.as_tensor(X), torch.as_tensor(eta))
    for k in ("L", "recon", "kl"):
        assert abs(t[k].item() - cache[k]) < 1e-10
    loss = tv.loss(X, eta)
    assert abs(loss.item() - vae.loss(X, eta)) < 1e-10
    loss.backward()
    _assert_grads_match(tv.encoder, vae.encoder)
    _assert_grads_match(tv.decoder, vae.decoder)


@pytest.mark.parametrize(("activation", "pool"), [("relu", "max"), ("tanh", "avg")])
def test_lenet_forward_and_gradients_match_numpy(activation, pool):
    rng = np.random.default_rng(0)
    X = rng.random((6, 1, 12, 12))
    y = rng.integers(0, 10, 6)
    model = lenet(np.random.default_rng(1), activation=activation, pool=pool)
    tl = TorchLeNet.from_numpy(model)

    logits = model.forward(X)
    _, g = softmax_cross_entropy(logits, y)
    model.backward(g)
    t_logits = tl(torch.as_tensor(X))
    np.testing.assert_allclose(t_logits.detach().numpy(), logits, rtol=1e-10, atol=1e-12)
    torch.nn.functional.cross_entropy(t_logits, torch.as_tensor(y)).backward()
    _assert_grads_match(tl.net, model)


def test_load_numpy_weights_rejects_mismatched_architecture():
    with pytest.raises(ValueError):
        load_numpy_weights(TorchLeNet().net, VAE(144).encoder)


def test_torch_vae_training_reduces_loss():
    X, _ = digits(n_per_class=30, max_shift=0, seed=0)
    tv = TorchVAE(144, n_latent=2, hidden=(64,), seed=0, sigma_x2=0.1)
    eta = np.random.default_rng(5).standard_normal((1, len(X), 2))
    before = tv.loss(X, eta).item()
    hist = tv.fit(X, n_epochs=30, lr=3e-3, batch_size=32, seed=0)
    assert tv.loss(X, eta).item() < 0.7 * before
    assert hist["loss"][-1] < hist["loss"][0]
    assert len(hist["kl"]) == 30 and hist["kl"][-1] > 0
    mu, logvar = tv.encode_numpy(X)
    assert mu.shape == logvar.shape == (len(X), 2)


def test_torch_lenet_learns_digits():
    X, y = digits(n_per_class=60, max_shift=1, seed=0)
    X_test, y_test = digits(n_per_class=30, max_shift=1, seed=1)
    tl = TorchLeNet(seed=0)
    hist = tl.fit(as_images(X), y, n_epochs=8, lr=3e-3, X_val=X_test, y_val=y_test)
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["acc"][-1] > 0.9
    assert tl.accuracy(X_test, y_test) > 0.8
    assert tl.predict(X_test[:5]).shape == (5,)
