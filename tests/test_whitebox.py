import numpy as np
import pytest

from src.attacks.budget import quantise
from src.attacks.whitebox import (ATTACK_REGISTRY, PAPER_ATTACK_REGISTRY, bim, cw, fgsm, mim,
                                  paper_fgsm, paper_pgd, pgd)
from src.defences.inference import bit_depth_reduction
from src.models.arcface_inference import predict_logits

EPS = 4 / 255
LINF_ATTACKS = [fgsm, bim, pgd, mim]


def margin(model, x, labels):
    """True-class logit minus the best other logit.

    Measured instead of cross-entropy: with logit gaps of ~30, float32
    cross-entropy rounds to exactly 0 before and after a partial attack,
    although its gradient (softmax - one_hot) is still non-zero.
    """
    logits = predict_logits(model, x)
    true = logits[np.arange(len(labels)), labels]
    logits[np.arange(len(labels)), labels] = -np.inf
    return true - logits.max(axis=1)


@pytest.mark.parametrize("attack", LINF_ATTACKS, ids=lambda f: f.__name__)
def test_linf_attacks_respect_budget_and_contract(model, images, labels, attack):
    adv = attack(images, model, epsilon=EPS, label=labels)
    assert adv.shape == images.shape and adv.dtype == np.float32
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    assert np.abs(adv - images).max() <= EPS + 1e-6
    assert np.abs(adv - images).max() > 0


@pytest.mark.parametrize("attack", LINF_ATTACKS, ids=lambda f: f.__name__)
def test_linf_attacks_shrink_the_margin(model, images, labels, attack):
    adv = attack(images, model, epsilon=EPS, label=labels)
    assert (margin(model, adv, labels) < margin(model, images, labels)).all()


@pytest.mark.parametrize("attack", LINF_ATTACKS, ids=lambda f: f.__name__)
def test_single_image_input(model, images, attack):
    adv = attack(images[0], model, epsilon=EPS)
    assert adv.shape == (256, 256, 3)


def test_epsilon_is_required(model, images):
    with pytest.raises(TypeError):
        fgsm(images, model)


def test_pgd_is_deterministic_per_seed(model, images):
    a = pgd(images[:2], model, epsilon=EPS, seed=1)
    b = pgd(images[:2], model, epsilon=EPS, seed=1)
    c = pgd(images[:2], model, epsilon=EPS, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cw_finds_small_adversarial_examples(model, images, labels):
    adv = cw(images[:3], model, label=labels[:3])
    assert (predict_logits(model, adv).argmax(1) != labels[:3]).all()
    # CW minimises L2: it should need less than an L-inf attack that also succeeds.
    strong = pgd(images[:3], model, epsilon=16 / 255, label=labels[:3])
    l2 = lambda a: np.sqrt(((a - images[:3]) ** 2).sum(axis=(1, 2, 3)))  # noqa: E731
    assert (l2(adv) < l2(strong)).all()


def test_cw_returns_input_when_it_fails(model, images, labels):
    adv = cw(images[:2], model, label=labels[:2], steps=1, binary_steps=1, initial_const=1e-6)
    np.testing.assert_array_equal(adv, images[:2])


def fooled(model, x, labels):
    return predict_logits(model, x).argmax(1) != labels


def test_cw_with_quantise_fn_returns_8bit_adversarial_images(model, images, labels):
    clean = quantise(images[:2])                       # loaded PNGs are on the grid
    plain = cw(clean, model, label=labels[:2], steps=20, binary_steps=3)
    rounded = cw(clean, model, label=labels[:2], steps=20, binary_steps=3, quantise_fn=quantise)
    assert rounded.shape == clean.shape and rounded.dtype == np.float32
    assert np.allclose(rounded * 255, np.round(rounded * 255), atol=1e-4)
    assert fooled(model, plain, labels[:2]).any()
    assert (fooled(model, rounded, labels[:2]) | ~fooled(model, plain, labels[:2])).all()


def test_cw_judges_success_after_quantise_fn(model, images, labels):
    # A 6-bit grid is coarse enough that rounding the float result afterwards undoes
    # most successes, which is what 8-bit rounding does to some kappa = 0 examples.
    six_bit = lambda x: bit_depth_reduction(x, bits=6)  # noqa: E731
    clean = six_bit(images)
    plain = cw(clean, model, label=labels, steps=20, binary_steps=3)
    aware = cw(clean, model, label=labels, steps=20, binary_steps=3, quantise_fn=six_bit)
    assert fooled(model, plain, labels).all()
    assert not fooled(model, six_bit(plain), labels).all()          # rounding afterwards loses some
    assert fooled(model, aware, labels).all()                       # judged after rounding: none lost
    assert np.allclose(aware * 63, np.round(aware * 63), atol=1e-4)


def test_cw_successes_clear_the_kappa_margin(model, images, labels):
    # Before, any misclassified iterate counted as a success, so kappa = 5 still returned
    # examples on the decision boundary (margins 0.4-4 on this model).
    adv = cw(images, model, label=labels, kappa=5.0, steps=20, binary_steps=3)
    assert (-margin(model, adv, labels) > 5.0).all()


def test_cw_quantise_fn_failures_and_bad_shapes(model, images, labels):
    clean = quantise(images[:2])
    adv = cw(clean, model, label=labels[:2], steps=1, binary_steps=1, initial_const=1e-6,
             quantise_fn=quantise)
    np.testing.assert_array_equal(adv, clean)
    with pytest.raises(ValueError, match="shape"):
        cw(clean, model, steps=1, binary_steps=1, quantise_fn=lambda x: x[:, :128])


def test_paper_wrappers_load_without_torch(model, images):
    adv = paper_fgsm(images[0], model)
    assert adv.shape == (256, 256, 3)
    assert np.abs(adv - images[0]).max() <= 0.001 + 1e-6
    assert set(PAPER_ATTACK_REGISTRY) == set(ATTACK_REGISTRY)


def test_paper_pgd_ignores_epsilon(model, images):
    # Vendored behaviour documented in src/attacks/whitebox.py: no projection,
    # so the budget is num_steps * step_size, whatever epsilon says.
    adv = paper_pgd(images[0], model, epsilon=0.001, num_steps=10, step_size=0.002)
    assert np.abs(adv - images[0]).max() > 0.001 * 5
