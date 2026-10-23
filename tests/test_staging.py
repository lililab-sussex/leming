"""Regression tests for hard staging with fixed event orders and Gaussian fits.

Run from the repository root: PYTHONPATH=src python -m unittest discover -s tests -v
"""

from itertools import permutations
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import norm
import torch

from leming import LEMING


def fixed_model(X, theta, sequence):
    """Isolate prediction from mixture fitting and event-order optimisation."""
    sequence = np.asarray(sequence, dtype=int)
    with patch('torch.cuda.is_available', return_value=False):
        model = LEMING(X=X, labels=np.zeros(len(X), dtype=int))
    model.thetas = np.asarray(theta).tolist()
    matrix = np.zeros((len(sequence), len(sequence)))
    matrix[sequence, np.arange(len(sequence))] = 1.
    model.params = [torch.tensor(100. * matrix - 50., dtype=model.dtype)]
    return model


def enumerated_probabilities(X, theta, sequence):
    """Enumerate stages by direct density products, without cumulative sums.

    At stage k, sequence[:k] is abnormal and every other feature is normal.
    Inputs deliberately avoid extreme tails to test indexing independently
    of density underflow. Missing values contribute 0.5 in either state.
    """
    scores = np.ones((len(X), X.shape[1] + 1))
    for i, row in enumerate(X):
        for stage in range(X.shape[1] + 1):
            abnormal = set(sequence[:stage])
            for feature, value in enumerate(row):
                state = 2 if feature in abnormal else 0
                density = .5 if np.isnan(value) else norm.pdf(
                    value, loc=theta[feature, state],
                    scale=theta[feature, state + 1])
                scores[i, stage] *= density
    return scores / scores.sum(axis=1, keepdims=True)


class HardStagingTests(unittest.TestCase):
    def setUp(self):
        # LEMING sets the global default dtype during construction.
        self.addCleanup(torch.set_default_dtype, torch.get_default_dtype())

    def assert_matches_enumeration(self, X, theta, sequence):
        model = fixed_model(X, theta, sequence)
        with torch.no_grad():
            stages, probabilities = model.predict_stage(X, hard_perm=True)
        expected = enumerated_probabilities(X, theta, sequence)
        self.assertEqual(probabilities.shape, expected.shape)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        # The production calculation intentionally remains float32.
        np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-7)
        np.testing.assert_array_equal(stages, probabilities.argmax(axis=1))
        return stages, probabilities

    def test_every_stage_for_multiple_permutations(self):
        orders = [(0,), (1, 0), *permutations(range(3)), (3, 0, 4, 1, 2)]
        for order in orders:
            with self.subTest(sequence=order):
                sequence = np.asarray(order)
                p = len(sequence)
                theta = np.tile([0., .2, 1., .2, .5], (p, 1))
                X = np.zeros((p + 1, p))
                for stage in range(p + 1):
                    X[stage, sequence[:stage]] = 1.
                stages, _ = self.assert_matches_enumeration(X, theta, sequence)
                np.testing.assert_array_equal(stages, np.arange(p + 1))

    def test_unequal_gaussians_and_both_event_directions(self):
        theta = np.array([
            [-1., .7, 1.5, .9, .2],
            [2., 1.1, -.5, .6, .7],
            [.2, .8, 1.1, 1.2, .4],
            [-.3, .5, -.9, .7, .6],
            [1., 1.3, 2.5, .8, .8],
        ])
        X = np.random.default_rng(716).normal(.5, .7, (13, 5))
        self.assert_matches_enumeration(X, theta, np.array([2, 4, 0, 3, 1]))

    def test_missing_measurements(self):
        theta = np.array([[0., .7, 1.2, .8, .5],
                          [.1, .6, 1.6, .9, .5],
                          [-.2, .8, 2., .7, .5]])
        X = np.array([[np.nan, .1, 2.], [0., np.nan, np.nan],
                      [np.nan, np.nan, np.nan], [.3, .7, np.nan]])
        _, probabilities = self.assert_matches_enumeration(
            X, theta, np.array([2, 0, 1]))
        # Check the tied probabilities, not which tied stage argmax chooses.
        np.testing.assert_allclose(probabilities[2], np.full(4, .25))

    def test_feature_relabeling_preserves_predictions(self):
        sequence = np.array([2, 0, 3, 1])
        theta = np.array([[j * .1, .7, 1.5 + j * .3, .9, .5]
                          for j in range(4)])
        X = np.random.default_rng(189).normal(.8, .9, (9, 4))
        stages, probabilities = self.assert_matches_enumeration(X, theta, sequence)
        relabel = np.array([3, 1, 0, 2])
        other_stages, other_probabilities = self.assert_matches_enumeration(
            X[:, relabel], theta[relabel], np.argsort(relabel)[sequence])
        np.testing.assert_array_equal(stages, other_stages)
        np.testing.assert_allclose(probabilities, other_probabilities,
                                   rtol=1e-5, atol=1e-7)

    def test_hard_and_soft_staging_agree_for_near_exact_permutation(self):
        sequence = np.array([2, 0, 1])
        theta = np.tile([0., .2, 1., .2, .5], (3, 1))
        X = np.zeros((4, 3))
        for stage in range(4):
            X[stage, sequence[:stage]] = 1.
        model = fixed_model(X, theta, sequence)
        with torch.no_grad():
            hard_stages, hard_probs = model.predict_stage(X, hard_perm=True)
            soft_stages, soft_probs = model.predict_stage(X, hard_perm=False)
        np.testing.assert_array_equal(hard_stages, soft_stages)
        np.testing.assert_allclose(hard_probs, soft_probs, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
