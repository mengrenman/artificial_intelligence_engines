import numpy as np
import pytest

from aie.data import digits
from aie.gradcheck import numerical_gradient, relative_error
from aie.layers import Dense, ReLU, Sequential, Tanh
from aie.losses import mean_sse, softmax_cross_entropy
from scripts.deep_backprop import (
    GAN,
    Residual,
    as_images,
    autoencoder,
    bce_with_logits,
    corrupt,
    deep_mlp,
    fgsm,
    input_gradient,
    layer_gradient_norms,
    lenet,
    ring_of_gaussians,
    train_classifier,
    train_denoising_autoencoder,
)


def _check_model(model, x, loss_fn, tol=1e-6):
    """Gradcheck all parameters and the input of ``model`` under ``loss_fn``."""

    def loss():
        return loss_fn(model.forward(x))[0]

    _, g = loss_fn(model.forward(x))
    dx = model.backward(g)
    for p, _, layer, name in model.parameters():
        analytic = layer.grads[name].copy()
        assert relative_error(analytic, numerical_gradient(loss, p)) < tol, name
    assert relative_error(dx, numerical_gradient(loss, x)) < tol


def test_residual_block_gradients():
    rng = np.random.default_rng(0)
    block = Residual(Dense(4, 4, rng), Tanh(), Dense(4, 4, rng))
    model = Sequential(block, ReLU(), Dense(4, 3, rng))
    assert len(model.parameters()) == 6  # inner params are visible to optimisers
    x = rng.standard_normal((5, 4))
    labels = np.array([0, 1, 2, 1, 0])
    _check_model(model, x, lambda out: softmax_cross_entropy(out, labels))


def test_residual_is_identity_plus_branch():
    rng = np.random.default_rng(1)
    d = Dense(3, 3, rng)
    x = rng.standard_normal((2, 3))
    assert np.allclose(Residual(d).forward(x), x + d.forward(x))


def test_lenet_gradients_tiny_batch():
    rng = np.random.default_rng(0)
    model = lenet(rng, n_maps=(2, 3), n_hidden=5, activation="tanh", pool="avg", size=8)
    x = rng.random((2, 1, 8, 8))
    labels = np.array([3, 7])
    _check_model(model, x, lambda out: softmax_cross_entropy(out, labels))


def test_lenet_maxpool_relu_gradients():
    rng = np.random.default_rng(3)
    model = lenet(rng, n_maps=(2, 2), n_hidden=4, size=8)
    x = rng.random((2, 1, 8, 8))
    labels = np.array([1, 4])
    _check_model(model, x, lambda out: softmax_cross_entropy(out, labels), tol=1e-5)


def test_cnn_learns_digits_subset():
    X, y = digits(n_per_class=30, seed=0)
    model = lenet(np.random.default_rng(0))
    hist = train_classifier(model, as_images(X), y, n_epochs=10, batch_size=16, lr=5e-3)
    assert hist["acc"][-1] > 0.9


def test_vanishing_gradients_sigmoid_vs_relu_vs_residual():
    X, y = digits(n_per_class=10, seed=0)
    Xf = X.reshape(len(X), -1)
    ratio = {}
    for kind in ["sigmoid", "relu", "residual"]:
        norms = layer_gradient_norms(deep_mlp(kind, depth=15, rng=np.random.default_rng(0)), Xf, y)
        ratio[kind] = norms[0] / norms[-1]
    assert ratio["sigmoid"] < 1e-4
    assert ratio["relu"] > 1e-2 and ratio["residual"] > 1e-2


@pytest.fixture(scope="module")
def trained_cnn():
    X, y = digits(n_per_class=30, seed=0)
    model = lenet(np.random.default_rng(0))
    train_classifier(model, as_images(X), y, n_epochs=10, batch_size=16, lr=5e-3)
    return model, as_images(X), y


def test_fgsm_increases_loss_and_flips_predictions(trained_cnn):
    model, X, y = trained_cnn
    loss0, g = input_gradient(model, X, y)
    assert g.shape == X.shape
    Xa = fgsm(model, X, y, eps=0.1)
    assert np.max(np.abs(Xa - X)) <= 0.1 + 1e-12
    loss1, _ = input_gradient(model, Xa, y)
    assert loss1 > loss0
    acc0 = np.mean(model.forward(X).argmax(1) == y)
    acc1 = np.mean(model.forward(Xa).argmax(1) == y)
    assert acc1 < acc0 - 0.2


def test_input_gradient_matches_numerical():
    rng = np.random.default_rng(0)
    model = lenet(rng, n_maps=(2, 2), n_hidden=4, activation="tanh", size=8)
    x, y = rng.random((2, 1, 8, 8)), np.array([0, 5])
    _, g = input_gradient(model, x, y)
    num = numerical_gradient(lambda: softmax_cross_entropy(model.forward(x), y)[0], x)
    assert relative_error(g, num) < 1e-6


def test_denoising_autoencoder_reduces_error():
    X, _ = digits(n_per_class=30, seed=0)
    Xf = X.reshape(len(X), -1)
    ae = autoencoder(rng=np.random.default_rng(0))
    Xc = corrupt(Xf, np.random.default_rng(9))
    before = mean_sse(ae.forward(Xc), Xf)
    hist = train_denoising_autoencoder(ae, Xf, n_epochs=30, lr=1e-2)
    after = mean_sse(ae.forward(Xc), Xf)
    assert hist[-1] < hist[0]
    assert after < 0.5 * before
    assert after < mean_sse(Xc, Xf)  # better than leaving the noise in


def test_bce_with_logits_values_and_gradient():
    a = np.array([[-30.0], [0.0], [2.0], [40.0]])
    t = np.array([[0.0], [1.0], [1.0], [0.0]])
    loss, g = bce_with_logits(a, t)
    p = 1 / (1 + np.exp(-a[1:3]))
    expected = (0.0 + -np.log(p[0, 0]) - np.log(p[1, 0]) + 40.0) / 4
    assert np.isclose(loss, expected, atol=1e-10)
    num = numerical_gradient(lambda: bce_with_logits(a, t)[0], a)
    assert relative_error(g, num) < 1e-6


def _tiny_gan():
    gan = GAN(z_dim=2, x_dim=2, n_hidden=4, seed=0)
    rng = np.random.default_rng(1)
    # Random biases keep ReLU inputs away from the kink at exactly 0, which
    # zero biases plus dead units would otherwise hit (finite differences
    # are invalid there).
    for p, _, _, name in gan.G.parameters() + gan.D.parameters():
        if name == "b":
            p[:] = 0.1 * rng.standard_normal(p.shape)
    return gan, ring_of_gaussians(6, rng), rng.standard_normal((6, 2))


def test_gan_discriminator_gradients():
    gan, x, z = _tiny_gan()
    gan.discriminator_gradients(x, z)
    for p, _, layer, name in gan.D.parameters():
        analytic = layer.grads[name].copy()
        num = numerical_gradient(lambda: gan.discriminator_loss(x, z), p)
        assert relative_error(analytic, num) < 1e-6, name


def test_gan_generator_gradients():
    gan, _, z = _tiny_gan()
    gan.generator_gradients(z)
    for p, _, layer, name in gan.G.parameters():
        analytic = layer.grads[name].copy()
        num = numerical_gradient(lambda: gan.generator_loss(z), p)
        assert relative_error(analytic, num) < 1e-6, name


def test_gan_learns_ring_radius():
    gan = GAN(seed=0)
    gan.train(ring_of_gaussians, n_steps=600, batch_size=64)
    r = np.linalg.norm(gan.generate(gan.sample_z(500)), axis=1)
    assert abs(np.median(r) - 2.0) < 0.4
