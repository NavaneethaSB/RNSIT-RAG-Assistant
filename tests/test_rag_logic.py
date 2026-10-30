import unittest
from unittest.mock import patch

import numpy as np

from ask import _is_abstention, build_answer_sources, build_sources, retrieve_chunks, select_relevant_chunks


class FakeModel:
    def encode(self, *args, **kwargs):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeIndex:
    ntotal = 2
    d = 2

    def search(self, _vectors, _limit):
        return np.asarray([[0.9, 0.8]], dtype=np.float32), np.asarray([[0, 1]], dtype=np.int64)


class ReverseSemanticIndex(FakeIndex):
    def search(self, _vectors, limit):
        self.last_limit = limit
        return np.asarray([[0.9, 0.8]], dtype=np.float32), np.asarray([[1, 0]], dtype=np.int64)


class EmptyIndex:
    ntotal = 0
    d = 2


class FakeBM25:
    def get_scores(self, _tokens):
        return np.asarray([2.0, 1.0])


class NoMatchBM25:
    def get_scores(self, _tokens):
        return np.asarray([0.0, 0.0])


class RagLogicTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {"chunk_id": 0, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "CSE has an intake of 720."},
            {"chunk_id": 1, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "CSE has laboratories and faculty."},
        ]

    def test_abstention_policy_handles_wording_variations(self):
        for answer in (
            "I'm sorry, I cannot disclose that information.",
            "The provided context does not contain enough information.",
            "Insufficient evidence is available to answer.",
            "There is not enough information in the sources.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(_is_abstention(answer))
        self.assertFalse(_is_abstention("The CSE intake is 720 students."))

    def test_empty_query_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            retrieve_chunks("  ", self.chunks, FakeIndex(), FakeModel(), FakeBM25())

    def test_chunk_index_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "count"):
            retrieve_chunks("CSE intake", self.chunks[:1], FakeIndex(), FakeModel(), FakeBM25())

    def test_empty_index_returns_no_retrievals(self):
        self.assertEqual(retrieve_chunks("CSE intake", [], EmptyIndex(), FakeModel(), FakeBM25()), [])

    def test_query_embedding_dimension_mismatch_is_rejected(self):
        index = type("WrongDimensionIndex", (), {"ntotal": 2, "d": 3})()
        with self.assertRaisesRegex(ValueError, "dimension"):
            retrieve_chunks("unmatched words", self.chunks, index, FakeModel(), FakeBM25())

    def test_retrieved_records_keep_metadata_and_rank(self):
        results = retrieve_chunks("CSE department overview", self.chunks, FakeIndex(), FakeModel(), FakeBM25())
        self.assertEqual([item["chunk_id"] for item in results], [0, 1])
        self.assertEqual(results[0]["url"], self.chunks[0]["url"])
        self.assertEqual(results[0]["retrieval_rank"], 1)
        self.assertGreater(results[0]["retrieval_score"], results[1]["retrieval_score"])
        self.assertNotIn("retrieval_rank", self.chunks[0])

    def test_zero_bm25_scores_do_not_add_keyword_votes(self):
        results = retrieve_chunks("unmatched words", self.chunks, FakeIndex(), FakeModel(), NoMatchBM25())
        expected = 1.5 / (60 + 1)
        self.assertAlmostEqual(results[0]["retrieval_score"], expected)

    def test_numeric_intake_retrieval_keeps_matching_fact_and_original_rrf_rank(self):
        chunks = [
            {"chunk_id": 0, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "The CSE department has a current intake of 720 students."},
            {"chunk_id": 1, "title": "ISE - RNSIT", "url": "https://www.rnsit.ac.in/ise/", "text": "The ISE department has a current intake of 240 students."},
        ]
        index = ReverseSemanticIndex()
        results = retrieve_chunks(
            "Which department has an intake of 720 students?",
            chunks,
            index,
            FakeModel(),
            FakeBM25(),
        )
        self.assertEqual(index.last_limit, index.ntotal)
        self.assertEqual([result["url"] for result in results], [chunks[0]["url"]])
        self.assertEqual(results[0]["retrieval_rank"], 2)
        self.assertTrue(np.isfinite(results[0]["retrieval_score"]))

    def test_unsupported_numeric_intake_does_not_return_wrong_value_evidence(self):
        chunks = [
            {"chunk_id": 0, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "The CSE department has a current intake of 720 students."},
            {"chunk_id": 1, "title": "ISE - RNSIT", "url": "https://www.rnsit.ac.in/ise/", "text": "The ISE department has a current intake of 240 students."},
        ]
        results = retrieve_chunks(
            "Which department has an intake of 9999 students?",
            chunks,
            ReverseSemanticIndex(),
            FakeModel(),
            FakeBM25(),
        )
        self.assertEqual(results, [])

    def test_source_grouping_deduplicates_url_and_aggregates_excerpts(self):
        results = retrieve_chunks("CSE department overview", self.chunks, FakeIndex(), FakeModel(), FakeBM25())
        sources = build_sources(results)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["id"], "source-1")
        self.assertEqual(sources[0]["title"], "CSE - RNSIT")
        self.assertIn("720", sources[0]["excerpt"])
        self.assertIn("laboratories", sources[0]["excerpt"])

    def test_abstained_answer_has_no_sources_even_when_retrieval_found_pages(self):
        retrieved = retrieve_chunks("CSE intake", self.chunks, FakeIndex(), FakeModel(), FakeBM25())
        self.assertEqual(
            build_answer_sources(
                "principal private phone password",
                "I couldn't find relevant information on the RNSIT website.",
                retrieved,
                self.chunks,
            ),
            [],
        )

    def test_supported_answer_keeps_validated_sources(self):
        retrieved = retrieve_chunks("CSE intake", self.chunks, FakeIndex(), FakeModel(), FakeBM25())
        sources = build_answer_sources("CSE intake", "The CSE intake is 720.", retrieved, self.chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], self.chunks[0]["url"])
        self.assertIn("720", sources[0]["excerpt"])

    def test_answer_source_selection_uses_one_matching_official_page(self):
        corpus = [
            {"chunk_id": 0, "title": "RNSIT", "url": "https://www.rnsit.ac.in/", "text": "RNSIT is an institute of technology in Bengaluru, Karnataka."},
            {"chunk_id": 1, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "The Department of Computer Science and Engineering (CSE) offers undergraduate engineering education."},
            {"chunk_id": 2, "title": "MCA - RNSIT", "url": "https://www.rnsit.ac.in/mca/", "text": "The MCA programme is a two-year course."},
            {"chunk_id": 3, "title": "CIVIL - RNSIT", "url": "https://www.rnsit.ac.in/civil/", "text": "The Department of Civil Engineering offers a Bachelor of Engineering course."},
            {"chunk_id": 4, "title": "Admissions - RNSIT", "url": "https://www.rnsit.ac.in/admissions/", "text": "Admissions are open for 2026. Applicants can find admission information here."},
            {"chunk_id": 5, "title": "Placement Statistics - RNSIT", "url": "https://www.rnsit.ac.in/cse-placement-statistics", "text": "Placement statistics include 1060 placements in 2025."},
        ]
        cases = [
            ("Tell me about CSE", "CSE offers undergraduate engineering education.", 1),
            ("What is the MCA programme duration?", "The MCA programme is a two-year course.", 2),
            ("Tell me about Civil Engineering", "Civil Engineering offers a Bachelor of Engineering course.", 3),
            ("What is RNSIT?", "RNSIT is an institute of technology in Bengaluru, Karnataka.", 0),
            ("What are RNSIT admissions?", "Admissions are open for 2026.", 4),
            ("What are RNSIT placement statistics?", "There were 1060 placements in 2025.", 5),
            # Resolved follow-up query retains the MCA topic from conversation history.
            ("MCA programme duration", "The MCA programme is a two-year course.", 2),
        ]
        for question, answer, chunk_id in cases:
            with self.subTest(question=question):
                retrieved = [
                    {**chunk, "retrieval_rank": rank, "retrieval_score": 0.03 / rank}
                    for rank, chunk in enumerate(corpus, start=1)
                ]
                sources = build_answer_sources(question, answer, retrieved, corpus)
                self.assertEqual(len(sources), 1)
                source = sources[0]
                expected = corpus[chunk_id]
                self.assertEqual(source["url"], expected["url"])
                self.assertEqual(source["title"], expected["title"])
                self.assertEqual(source["excerpt"], expected["text"])

    def test_unsupported_answer_has_no_source_and_source_count_is_capped(self):
        retrieved = [
            {**chunk, "retrieval_rank": rank, "retrieval_score": 0.03 / rank}
            for rank, chunk in enumerate(self.chunks, start=1)
        ]
        self.assertEqual(
            build_answer_sources("private password", "I cannot provide that information.", retrieved, self.chunks),
            [],
        )
        sources = build_answer_sources("CSE intake", "The CSE intake is 720.", retrieved, self.chunks)
        self.assertLessEqual(len(sources), 1)

    def test_query_anchor_filters_unrelated_department_chunks(self):
        corpus = [
            {"chunk_id": 0, "title": "MCA - RNSIT", "url": "https://www.rnsit.ac.in/mca/", "text": "MCA is a two-year programme."},
            {"chunk_id": 1, "title": "ISE - RNSIT", "url": "https://www.rnsit.ac.in/ise/", "text": "ISE has an intake of 240 and links to the MCA programme."},
            {"chunk_id": 2, "title": "Notifications - RNSIT", "url": "https://www.rnsit.ac.in/notifications/", "text": "Latest notifications about the MCA programme."},
        ]
        selected = select_relevant_chunks("mca programme duration", corpus, corpus)
        self.assertEqual([chunk["chunk_id"] for chunk in selected], [0])

    def test_single_entity_numeric_answer_excludes_conflicting_page_source(self):
        corpus = [
            {"chunk_id": 0, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "The CSE department has an intake of 720 students."},
            {"chunk_id": 1, "title": "CSE (Data Science) - RNSIT", "url": "https://www.rnsit.ac.in/cse-ds/", "text": "The CSE Data Science programme has an intake of 180 students."},
            {"chunk_id": 2, "title": "ISE - RNSIT", "url": "https://www.rnsit.ac.in/ise/", "text": "ISE has an intake of 240 students."},
        ]
        retrieved = [
            {**chunk, "retrieval_rank": rank, "retrieval_score": 0.03 / rank}
            for rank, chunk in enumerate(corpus, start=1)
        ]
        sources = build_answer_sources("CSE intake", "The CSE intake is 720 students.", retrieved, corpus)
        self.assertEqual([source["url"] for source in sources], [corpus[0]["url"]])

    def test_sources_require_matching_original_chunk_metadata(self):
        retrieved = [
            {**self.chunks[0], "retrieval_rank": 1, "retrieval_score": 0.02},
            {**self.chunks[1], "retrieval_rank": 2, "url": "https://example.com/fake/"},
            {**self.chunks[1], "retrieval_rank": 3, "chunk_id": 99},
        ]
        sources = build_sources(retrieved, self.chunks)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], self.chunks[0]["url"])
        self.assertEqual(sources[0]["title"], self.chunks[0]["title"])
        self.assertEqual(sources[0]["retrieval_rank"], 1)
        self.assertEqual(sources[0]["retrieval_score"], 0.02)

    def test_source_validation_preserves_retrieval_rank_and_score(self):
        for score in (0.0, None):
            with self.subTest(score=score):
                retrieved = [{
                    **self.chunks[0],
                    "retrieval_rank": 4,
                    "retrieval_score": score,
                }]
                sources = build_sources(retrieved, self.chunks)
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0]["retrieval_rank"], 4)
                self.assertIs(sources[0]["retrieval_score"], score)
                self.assertEqual(sources[0]["title"], self.chunks[0]["title"])
                self.assertEqual(sources[0]["url"], self.chunks[0]["url"])
                self.assertEqual(sources[0]["excerpt"], self.chunks[0]["text"])


if __name__ == "__main__":
    unittest.main()
