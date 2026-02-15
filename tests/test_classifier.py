"""Tests for nadirclaw.classifier — binary complexity classification."""

import pytest


class TestBinaryClassifier:
    @pytest.fixture(autouse=True)
    def classifier(self):
        from nadirclaw.classifier import BinaryComplexityClassifier
        self.clf = BinaryComplexityClassifier()

    def test_simple_prompt(self):
        is_complex, confidence = self.clf.classify("What is 2+2?")
        assert is_complex is False
        assert 0.0 <= confidence <= 1.0

    def test_complex_prompt(self):
        is_complex, confidence = self.clf.classify(
            "Design a distributed database with sharding, replication, "
            "and consensus protocol for high availability"
        )
        assert is_complex is True
        assert 0.0 <= confidence <= 1.0

    def test_confidence_score_range(self):
        """Confidence-to-score should map to [0, 1]."""
        score_simple = self.clf._confidence_to_score(False, 0.5)
        score_complex = self.clf._confidence_to_score(True, 0.5)
        assert 0.0 <= score_simple <= 0.5
        assert 0.5 <= score_complex <= 1.0

    def test_analyze_sync_returns_expected_keys(self):
        result = self.clf._analyze_sync("Hello world")
        expected_keys = {
            "recommended_model", "confidence", "complexity_score",
            "tier_name", "reasoning", "analyzer_type",
        }
        assert expected_keys.issubset(result.keys())
        assert result["analyzer_type"] == "binary"

    @pytest.mark.asyncio
    async def test_analyze_async(self):
        result = await self.clf.analyze(text="What is Python?")
        assert result["tier_name"] in ("simple", "complex")
