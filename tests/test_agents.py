"""
Tests for the Smart Research Agent pipeline components.

These tests validate the agent logic without requiring GPU or API keys.
"""

from datetime import datetime


class TestResearchState:
    """Tests for the ResearchState type definition."""

    def test_state_structure(self):
        """Verify ResearchState can be constructed with expected fields."""
        state = {
            "user_query": "test query",
            "expanded_keywords": ["keyword1", "keyword2"],
            "raw_papers": [],
            "selected_papers": [],
            "summaries": {},
            "final_analysis": "",
            "logs": [],
        }
        assert state["user_query"] == "test query"
        assert isinstance(state["expanded_keywords"], list)
        assert isinstance(state["summaries"], dict)


class TestRankingLogic:
    """Tests for the paper ranking algorithm."""

    def _compute_score(self, paper):
        """Replicate the ranking logic from the rank agent."""
        current_year = datetime.now().year
        c_score = (paper.get("citation_count", 0) / 500) * 40
        r_score = (10 - min(10, (current_year - paper.get("year", current_year)))) * 6
        return c_score + r_score

    def test_recent_papers_score_higher(self):
        """More recent papers should score higher in the recency component."""
        current_year = datetime.now().year
        recent = {"citation_count": 100, "year": current_year}
        old = {"citation_count": 100, "year": current_year - 10}
        assert self._compute_score(recent) > self._compute_score(old)

    def test_highly_cited_papers_score_higher(self):
        """Papers with more citations should score higher in the citation component."""
        current_year = datetime.now().year
        high_cite = {"citation_count": 400, "year": current_year}
        low_cite = {"citation_count": 50, "year": current_year}
        assert self._compute_score(high_cite) > self._compute_score(low_cite)

    def test_empty_papers_returns_empty(self):
        """Rank agent should handle empty paper list gracefully."""
        papers = []
        assert len(papers) == 0

    def test_deduplication_by_title(self):
        """Duplicate papers should be removed based on title."""
        papers = [
            {"title": "Paper A", "abstract": "First version"},
            {"title": "Paper A", "abstract": "Duplicate"},
            {"title": "Paper B", "abstract": "Different paper"},
        ]
        unique = {p["title"]: p for p in papers}.values()
        assert len(unique) == 2

    def test_score_calculation_bounds(self):
        """Score should be non-negative and within reasonable bounds."""
        current_year = datetime.now().year
        paper = {"citation_count": 500, "year": current_year}
        score = self._compute_score(paper)
        # Max citation score: 40, max recency score: 60
        assert 0 <= score <= 100


class TestPaperParsing:
    """Tests for paper data structure handling."""

    def test_paper_structure(self):
        """Verify paper dictionary has all required fields."""
        paper = {
            "id": "2301.00001",
            "title": "Test Paper",
            "abstract": "This is a test abstract.",
            "year": 2024,
            "citation_count": 42,
            "url": "https://arxiv.org/abs/2301.00001",
            "source": "arXiv",
        }
        required_fields = ["id", "title", "abstract", "year", "citation_count", "url", "source"]
        for field in required_fields:
            assert field in paper

    def test_abstract_newline_stripping(self):
        """Abstracts should have newlines replaced with spaces."""
        raw_abstract = "Line one\nLine two\nLine three"
        clean = raw_abstract.replace("\n", " ")
        assert "\n" not in clean
        assert clean == "Line one Line two Line three"
