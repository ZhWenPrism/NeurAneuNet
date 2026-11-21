import random
import tracemalloc
import unittest

from neuraneunet.evaluation import binary_metrics, dice_score


class BinaryMetricAllocationsTest(unittest.TestCase):
    def test_metrics_preserve_counts_and_empty_foreground_conventions(self):
        rng = random.Random(42)
        cases = [([0] * 8, [0] * 8), ([1] * 8, [1] * 8)]
        cases += [([rng.randrange(2) for _ in range(64)], [rng.randrange(2) for _ in range(64)]) for _ in range(12)]
        for left, right in cases:
            with self.subTest(left=left, right=right):
                tp = sum(a == b == 1 for a, b in zip(left, right))
                tn = sum(a == b == 0 for a, b in zip(left, right))
                fp = sum(a == 0 and b == 1 for a, b in zip(left, right))
                fn = sum(a == 1 and b == 0 for a, b in zip(left, right))
                expected_dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
                self.assertEqual(dice_score(iter(left), iter(right)), expected_dice)
                metrics = binary_metrics(iter(left), iter(right))
                self.assertEqual(metrics.accuracy, (tp + tn) / len(left))
                self.assertEqual(metrics.sensitivity, tp / (tp + fn) if tp + fn else 0.0)
                self.assertEqual(metrics.specificity, tn / (tn + fp) if tn + fp else 0.0)
                self.assertEqual(metrics.precision, tp / (tp + fp) if tp + fp else 0.0)

    def test_dense_masks_do_not_allocate_per_positive_index_or_pair(self):
        mask = [1] * 20000
        for metric in (dice_score, binary_metrics):
            with self.subTest(metric=metric.__name__):
                tracemalloc.start()
                try:
                    metric(iter(mask), iter(mask))
                    _, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, len(mask) * 64)
