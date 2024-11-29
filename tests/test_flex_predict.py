import unittest
from src.util.flex_predict import flex_predict, prob_flex_predict

class TestFlexPredict(unittest.TestCase):
    def test_flex_predict_within_bounds(self):
        forecasts = [20, 21, 22, 23, 24]
        lower_bound = 19
        upper_bound = 25
        result = flex_predict(forecasts, lower_bound, upper_bound)
        self.assertEqual(result, 5)

    def test_flex_predict_out_of_bounds(self):
        forecasts = [20, 21, 22, 26, 24]
        lower_bound = 19
        upper_bound = 25
        result = flex_predict(forecasts, lower_bound, upper_bound)
        self.assertEqual(result, 3)

    def test_flex_predict_with_error(self):
        forecasts = [20, 21, 22, 23, 24]
        lower_bound = 19
        upper_bound = 25
        result = flex_predict(forecasts, lower_bound, upper_bound, error=1)
        self.assertEqual(result, 5)

    
    def test_prob_flex_predict_within_bounds(self):
        forecasts = ([21, 22, 23, 24], [1, 1.25, 1.1, 0.6])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=0.95)
        self.assertEqual(result, 2)
        
    def test_prob_flex_predict_out_of_upper_bound(self):
        forecasts = ([21, 22, 23, 24], [1, 1.1, 1.2, 1.3])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=0.95)
        self.assertEqual(result, 2)
        
    def test_prob_flex_predict_out_of_lower_bound(self):
        forecasts = ([24, 23, 22, 21], [0.5, 1.1, 1.2, 1.3])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=0.95)
        self.assertEqual(result, 1)
        
    def test_prob_flex_predict_with_error(self):
        forecasts = ([21, 22, 23, 24], [0.6, 0.8, 1, 1.1])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, error=1, confidence=0.95)
        self.assertEqual(result, 1)
        
    def test_prob_flex_predict_with_low_confidence(self):
        forecasts = ([21, 22, 23, 24], [0.6, 0.8, 1, 1.1])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=0.85)
        self.assertEqual(result, 3)
        
    def test_prob_flex_predict_with_high_confidence(self):
        forecasts = ([21, 22, 23, 24], [0.6, 0.8, 1, 1.1])
        lower_bound = 19
        upper_bound = 25
        result, _ = prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=0.99)
        self.assertEqual(result, 2)
        
    def test_prob_flex_predict_impossible_confidence(self):
        forecasts = ([21, 22, 23, 24], [0.6, 0.8, 1, 1.1])
        lower_bound = 19
        upper_bound = 25
        with self.assertRaises(AssertionError):
            prob_flex_predict(forecasts, lower_bound, upper_bound, confidence=1)
            
    if __name__ == '__main__':
        unittest.main()
            