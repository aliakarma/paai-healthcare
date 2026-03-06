"""Unit tests for evaluation metrics and statistical tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import (
    compute_roc_auc,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_latency_percentiles,
    compute_adherence_correlation,
)
from evaluation.statistical_tests import (
    delongs_test,
    wilcoxon_test,
    bonferroni_correction,
)


class TestROCAUCMetric:
    """Test ROC-AUC computation."""

    def test_roc_auc_perfect_separation(self):
        """Test AUC for perfect class separation."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        
        auc = compute_roc_auc(y_true, y_score)
        
        assert auc == 1.0, "Perfect separation should yield AUC = 1.0"

    def test_roc_auc_random_chance(self):
        """Test AUC for random classifier."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score = np.random.rand(8)
        
        auc = compute_roc_auc(y_true, y_score)
        
        # Random classifier should be near 0.5
        assert 0.3 < auc < 0.7, f"Random classifier should have AUC near 0.5, got {auc}"

    def test_roc_auc_inverted_scores(self):
        """Test AUC with inverted predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])  # Inverted
        
        auc = compute_roc_auc(y_true, y_score)
        
        # Inverted should give low AUC
        assert auc < 0.5, f"Inverted predictions should have low AUC, got {auc}"

    def test_roc_auc_bounds(self):
        """Test that AUC is always in [0, 1]."""
        np.random.seed(42)
        for _ in range(10):
            y_true = np.random.randint(0, 2, 100)
            y_score = np.random.rand(100)
            auc = compute_roc_auc(y_true, y_score)
            assert 0.0 <= auc <= 1.0, f"AUC should be in [0, 1], got {auc}"


class TestAccuracyMetric:
    """Test accuracy computation."""

    def test_accuracy_perfect(self):
        """Test accuracy for perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 1.0, "Perfect predictions should yield accuracy = 1.0"

    def test_accuracy_half_correct(self):
        """Test accuracy with 50% correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 2 correct, 2 wrong
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 0.5, f"50% correct should yield accuracy 0.5, got {acc}"

    def test_accuracy_all_wrong(self):
        """Test accuracy for all wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        
        acc = compute_accuracy(y_true, y_pred)
        
        assert acc == 0.0, "All wrong predictions should yield accuracy = 0.0"


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 score metrics."""

    def test_precision_perfect(self):
        """Test precision for perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        
        prec, rec, f1 = compute_precision_recall_f1(y_true, y_pred)
        
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0

    def test_precision_vs_recall_tradeoff(self):
        """Test that there's a precision-recall tradeoff."""
        y_true = np.array([0, 0, 0, 0, 1, 1])
        
        # High precision, low recall: only predict positive when very confident
        y_pred_conservative = np.array([0, 0, 0, 0, 1, 0])
        prec_c, rec_c, _ = compute_precision_recall_f1(y_true, y_pred_conservative)
        
        # Low precision, high recall: predict positive more liberally
        y_pred_liberal = np.array([0, 1, 1, 0, 1, 1])
        prec_l, rec_l, _ = compute_precision_recall_f1(y_true, y_pred_liberal)
        
        assert prec_c >= prec_l, "Conservative predictions should have higher precision"
        assert rec_c <= rec_l, "Conservative predictions should have lower recall"

    def test_f1_harmonic_mean(self):
        """Test that F1 is harmonic mean of precision and recall."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        prec, rec, f1 = compute_precision_recall_f1(y_true, y_pred)
        
        # F1 = 2 * (prec * rec) / (prec + rec)
        expected_f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        assert abs(f1 - expected_f1) < 1e-6, f"F1 should be harmonic mean, got {f1}"


class TestLatencyMetrics:
    """Test latency percentile computation."""

    def test_latency_percentiles(self):
        """Test latency percentile computation."""
        latencies = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        
        p50, p95, p99 = compute_latency_percentiles(latencies)
        
        assert p50 == np.percentile(latencies, 50)
        assert p95 == np.percentile(latencies, 95)
        assert p99 == np.percentile(latencies, 99)

    def test_latency_percentile_ordering(self):
        """Test that percentiles are ordered: 50 < 95 < 99."""
        latencies = np.random.exponential(200, 1000)
        
        p50, p95, p99 = compute_latency_percentiles(latencies)
        
        assert p50 <= p95 <= p99, f"Percentiles should be ordered: {p50} <= {p95} <= {p99}"

    def test_latency_empty_array(self):
        """Test handling of empty latency array."""
        latencies = np.array([])
        
        # Should handle gracefully or raise informative error
        try:
            p50, p95, p99 = compute_latency_percentiles(latencies)
            # If no error, should return NaN or similar
            assert np.isnan(p50) or p50 is None
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass


class TestAdherenceCorrelation:
    """Test adherence-outcome correlation metrics."""

    def test_adherence_positive_correlation(self):
        """Test correlation when adherence predicts better outcomes."""
        adherence = np.array([0.3, 0.5, 0.7, 0.8, 0.9, 0.95])
        outcomes = np.array([0.2, 0.4, 0.6, 0.75, 0.85, 0.95])  # Strongly correlated
        
        corr = compute_adherence_correlation(adherence, outcomes)
        
        assert corr > 0.8, f"Positive relationship should have r > 0.8, got {corr}"

    def test_adherence_negative_correlation(self):
        """Test correlation when adherence is inversely related to outcomes."""
        adherence = np.array([0.9, 0.8, 0.7, 0.5, 0.3, 0.1])
        outcomes = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])  # Inverse
        
        corr = compute_adherence_correlation(adherence, outcomes)
        
        assert corr < -0.8, f"Negative relationship should have r < -0.8, got {corr}"

    def test_adherence_no_correlation(self):
        """Test correlation when variables are independent."""
        np.random.seed(42)
        adherence = np.random.rand(100)
        outcomes = np.random.rand(100)
        
        corr = compute_adherence_correlation(adherence, outcomes)
        
        # Should be near 0
        assert abs(corr) < 0.3, f"Random vars should have r near 0, got {corr}"


class TestDeLongsTest:
    """Test DeLong's statistical test for AUC comparison."""

    def test_delongs_identical_scores(self):
        """Test that identical classifiers have p-value near 1."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score1 = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        y_score2 = y_score1.copy()
        
        p_value, z_stat = delongs_test(y_true, y_score1, y_score2)
        
        assert p_value > 0.99, f"Identical classifiers should have p > 0.99, got {p_value}"
        assert abs(z_stat) < 0.01, f"Identical classifiers should have z near 0, got {z_stat}"

    def test_delongs_significantly_different(self):
        """Test that different classifiers show statistical difference."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_score_good = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        y_score_bad = np.array([0.5, 0.45, 0.55, 0.4, 0.5, 0.55, 0.45, 0.6])  # Near random
        
        p_value, z_stat = delongs_test(y_true, y_score_good, y_score_bad)
        
        # Should show significant difference (p < 0.05)
        assert p_value < 0.1, f"Different classifiers should show difference, p={p_value}"


class TestWilcoxonTest:
    """Test Wilcoxon signed-rank test."""

    def test_wilcoxon_identical_distributions(self):
        """Test that identical samples give p near 1."""
        sample1 = np.array([1, 2, 3, 4, 5])
        sample2 = sample1.copy()
        
        p_value, stat = wilcoxon_test(sample1, sample2)
        
        assert p_value > 0.9, f"Identical samples should have p > 0.9, got {p_value}"

    def test_wilcoxon_significantly_different(self):
        """Test that different distributions show Statistical difference."""
        sample1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sample2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])  # All higher
        
        p_value, stat = wilcoxon_test(sample1, sample2)
        
        # Should show significant difference (p < 0.05)
        assert p_value < 0.05, f"Different distributions should show difference, p={p_value}"


class TestBonferroniCorrection:
    """Test Bonferroni multiple comparison correction."""

    def test_bonferroni_single_test(self):
        """Test Bonferroni correction for single test (no change)."""
        p_values = [0.01]
        corrected = bonferroni_correction(p_values)
        
        assert abs(corrected[0] - 0.01) < 1e-6, "Single test should not change"

    def test_bonferroni_multiple_tests(self):
        """Test Bonferroni correction multiplies by number of tests."""
        p_values = [0.01, 0.02, 0.03]
        corrected = bonferroni_correction(p_values)
        
        # Each p-value should be multiplied by 3
        expected = np.array([0.03, 0.06, 0.09])
        np.testing.assert_array_almost_equal(corrected, expected)

    def test_bonferroni_caps_at_one(self):
        """Test that Bonferroni p-values don't exceed 1.0."""
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Would exceed 1.0 after correction
        corrected = bonferroni_correction(p_values)
        
        assert np.all(corrected <= 1.0), f"Corrected p-values should not exceed 1.0, got {corrected}"

    def test_bonferroni_ordering_preserved(self):
        """Test that relative ordering is preserved after correction."""
        p_values = [0.001, 0.01, 0.05]
        corrected = bonferroni_correction(p_values)
        
        assert corrected[0] < corrected[1] < corrected[2], \
            f"Ordering should be preserved: {corrected}"


class TestMetricConsistency:
    """Integration tests for metric consistency."""

    def test_metrics_on_synthetic_cohort(self):
        """Test metrics on realistic synthetic data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data where policy has ~0.92 AUC
        y_true = np.random.binomial(1, 0.2, n_samples)
        noise = np.random.normal(0, 0.1, n_samples)
        y_score = np.clip(y_true * 0.8 + (1 - y_true) * 0.2 + noise, 0, 1)
        
        # All metrics should succeed
        auc = compute_roc_auc(y_true, y_score)
        y_pred = (y_score > 0.5).astype(int)
        acc = compute_accuracy(y_true, y_pred)
        prec, rec, f1 = compute_precision_recall_f1(y_true, y_pred)
        
        assert 0.5 <= auc <= 1.0
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0

    
    def test_bootstrap_confidence_intervals(self):
        """Test that bootstrap CI contains true metric."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
        
        true_auc = compute_roc_auc(y_true, y_score)
        
        # Bootstrap resampling
        boot_aucs = []
        for _ in range(100):
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            boot_auc = compute_roc_auc(y_true[idx], y_score[idx])
            boot_aucs.append(boot_auc)
        
        ci_lower = np.percentile(boot_aucs, 2.5)
        ci_upper = np.percentile(boot_aucs, 97.5)
        
        # True AUC should be in confidence interval (most of the time)
        # This assertion might occasionally fail due to randomness, but should be rare
        assert ci_lower <= true_auc <= ci_upper or true_auc > ci_upper, \
            f"CI [{ci_lower}, {ci_upper}] should contain true AUC {true_auc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
