import unittest
from src.pipelines.metrics.cale import CALE
import torch

class TestCALE(unittest.TestCase):
    def test_calc(self):
        mean = torch.tensor([1, 2, 3, 4, 5])
        stddev = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        y = torch.tensor([1.05, 2.1, 3.5, 4.5, 6.0])
        result = CALE.calc(mean, stddev, y)
        self.assertAlmostEqual(result.item(), 0.0345989353954792)
        