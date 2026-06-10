"""Unit tests for the EmotionDetection package."""

import unittest

from EmotionDetection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test expected dominant emotions."""

    def test_anger(self):
        result = emotion_detector("I hate working long hours.")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        result = emotion_detector("This food is disgusting and gross.")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_fear(self):
        result = emotion_detector("I am afraid of walking alone at night.")
        self.assertEqual(result["dominant_emotion"], "fear")

    def test_joy(self):
        result = emotion_detector("I am so happy I am doing this.")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_sadness(self):
        result = emotion_detector("I am sad about the bad news.")
        self.assertEqual(result["dominant_emotion"], "sadness")


if __name__ == "__main__":
    unittest.main()
