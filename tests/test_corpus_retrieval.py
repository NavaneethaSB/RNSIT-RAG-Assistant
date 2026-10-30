"""Regression tests against the checked-in RNSIT corpus."""

import json
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest

from api import AskRequest, chat
from ask import generate_answer_with_source, get_query_type, get_targeted_chunks, retrieve_chunks


class UnusedIndex:
    d = 2

    def __init__(self, count):
        self.ntotal = count

    def search(self, *_args):
        raise AssertionError("targeted queries must not fall through to hybrid search")


class UnusedModel:
    def encode(self, *_args, **_kwargs):
        raise AssertionError("targeted queries must not fall through to semantic search")


class UnusedBM25:
    def get_scores(self, *_args):
        raise AssertionError("targeted queries must not fall through to keyword search")


class SelectedChunkClient:
    def __init__(self, chunk_id, answer="RNSIT offers the programmes listed in the admissions branch selector."):
        self.chunk_ids = chunk_id if isinstance(chunk_id, list) else [chunk_id]
        self.answer = answer
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_kwargs):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "chunk_ids": self.chunk_ids,
                "answer": self.answer,
            })))],
        )
        return response


class CorpusRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = json.loads(Path("chunks.json").read_text(encoding="utf-8"))

    def selected(self, question):
        return get_targeted_chunks(question, self.chunks)

    def test_general_admission_documents_select_route_evidence_and_cite_all_used_chunks(self):
        question = "What documents are required for admission?"
        retrieved = retrieve_chunks(
            question, self.chunks, UnusedIndex(len(self.chunks)), UnusedModel(), UnusedBM25()
        )
        ids = {chunk["chunk_id"] for chunk in retrieved}
        self.assertEqual(ids, {32, 33, 34, 35, 36, 37})
        self.assertTrue(ids.isdisjoint({26, 27, 40}))
        self.assertIn("i year b.e. management quota", self.chunks[39]["text"].casefold())
        selected_ids = [33, 34, 36, 37]
        answer_text = (
            "Requirements vary by admission route; the corpus does not establish one universal list. "
            "For KCET and Diploma CET, the listed requirements include the KEA admission order, SSLC and II PUC marks cards, and a seven-year study certificate. "
            "Reserved category and SNQ applicants have additional certificate requirements, and special-category applicants must submit relevant originals. "
            "For COMEDK, the list includes the seat allotment letter with fee receipt, marks cards, study certificate, and applicable certificates."
        )
        answer, sources = generate_answer_with_source(
            question, retrieved, SelectedChunkClient(selected_ids, answer_text), self.chunks
        )
        self.assertEqual(answer, answer_text)
        self.assertEqual({source["url"] for source in sources}, {"https://www.rnsit.ac.in/admissions/"})
        self.assertEqual([source["id"] for source in sources], [f"source-{i}" for i in range(1, 5)])
        for source, chunk_id in zip(sources, selected_ids):
            self.assertEqual(source["title"], self.chunks[chunk_id]["title"])
            self.assertEqual(source["url"], self.chunks[chunk_id]["url"])
            self.assertIn(source["excerpt"], self.chunks[chunk_id]["text"])
            self.assertNotIn("management quota", source["excerpt"].casefold())

    def test_api_chat_uses_real_retrieval_generation_and_source_contract(self):
        question = "What documents are required for admission?"
        selected_ids = [33, 34, 36, 37]
        answer_text = "Requirements vary by route; KCET and COMEDK lists are separate."
        client = SelectedChunkClient(selected_ids, answer_text)
        resources = (
            self.chunks,
            UnusedIndex(len(self.chunks)),
            UnusedModel(),
            UnusedBM25(),
            client,
        )
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag_resources=resources)))
        response = asyncio.run(chat(AskRequest(question=question), request))
        self.assertEqual(response.answer, answer_text)
        self.assertEqual([source.url for source in response.sources], [self.chunks[i]["url"] for i in selected_ids])
        for source, chunk_id in zip(response.sources, selected_ids):
            self.assertIn(source.excerpt, self.chunks[chunk_id]["text"])
        self.assertEqual([source.id for source in response.sources], [f"source-{i}" for i in range(1, 5)])
        self.assertTrue(all("management quota" not in source.excerpt.casefold() for source in response.sources))
        self.assertEqual(get_query_type("What documents are required for admission?"), "admission_documents")
        self.assertEqual(get_query_type("WHAT DOCUMENTS ARE REQUIRED FOR ADMISSION?!"), "admission_documents")
        self.assertEqual(get_query_type("What is the vision and mission of RNSIT?"), "vision_mission")
        by_id = {chunk["chunk_id"]: chunk["text"].casefold() for chunk in self.chunks}
        self.assertIn("be management quota admissions", by_id[26])
        self.assertIn("be management quota admissions", by_id[27])
        self.assertIn("documents required for admission through kcet", by_id[33])
        self.assertIn("special category quota", by_id[35])
        self.assertIn("documents required for admission through comedk", by_id[36])

    def test_management_query_selects_its_real_admissions_section(self):
        self.assertEqual(self.selected("What documents are required for management quota?"), [])
        selected = self.selected("What documents are required for BE management-quota admission?")
        self.assertTrue(selected)
        self.assertEqual({chunk["chunk_id"] for chunk in selected}, {26, 27})
        self.assertTrue(all("management quota admissions" in c["text"].casefold() for c in selected))
        self.assertTrue(all(c["url"].rstrip("/") == "https://www.rnsit.ac.in/admissions" for c in selected))
        answer, sources = generate_answer_with_source(
            "What documents are required for BE management-quota admission?",
            selected,
            SelectedChunkClient([26, 27], "BE management-quota documents are listed in the supplied evidence."),
            self.chunks,
        )
        self.assertIn("BE management-quota documents", answer)
        self.assertTrue(all(source["excerpt"] in self.chunks[i]["text"] for source, i in zip(sources, (26, 27))))

    def test_route_specific_sections_on_same_url_are_not_interchangeable(self):
        cases = {
            "KCET": ("KCET", "documents required for admission through kcet"),
            "COMEDK": ("COMEDK", "documents required for admission through comedk"),
        }
        for route, (query_term, evidence) in cases.items():
            with self.subTest(route=route):
                selected = self.selected(f"What documents are required through {query_term}?")
                self.assertTrue(selected)
                if route == "KCET":
                    self.assertEqual({chunk["chunk_id"] for chunk in selected}, {32, 33, 34, 35})
                else:
                    self.assertEqual({chunk["chunk_id"] for chunk in selected}, {36, 37})
                self.assertTrue(any(evidence in c["text"].casefold() for c in selected))
                selected_ids = sorted(c["chunk_id"] for c in selected)
                expected_answer = f"{route} applicants have route-specific document requirements."
                answer, sources = generate_answer_with_source(
                    f"What documents are required through {query_term}?",
                    selected,
                    SelectedChunkClient(selected_ids, expected_answer),
                    self.chunks,
                )
                self.assertEqual(answer, expected_answer)
                self.assertEqual({source["url"] for source in sources}, {selected[0]["url"]})
                self.assertEqual(len(sources), len(selected_ids))
                for source, chunk_id in zip(sources, selected_ids):
                    self.assertIn(source["excerpt"], self.chunks[chunk_id]["text"])

    def test_vision_and_mission_resolve_to_institutional_homepage_chunk(self):
        for question, evidence in (
            ("What is the vision of RNSIT?", "building rnsit into a world class institution"),
            ("What is the mission of RNSIT?", "to impart high quality education"),
            ("What are the vision and mission of RNSIT?", "building rnsit into a world class institution"),
            ("Tell me about RNSIT's vision.", "building rnsit into a world class institution"),
            ("Tell me about RNSIT's mission.", "to impart high quality education"),
        ):
            with self.subTest(question=question):
                selected = self.selected(question)
                self.assertTrue(selected)
                self.assertEqual({c["chunk_id"] for c in selected}, {6})
                self.assertIn(evidence, selected[0]["text"].casefold())
                self.assertEqual(selected[0]["url"], "https://www.rnsit.ac.in/")

    def test_department_vision_does_not_select_institutional_statement(self):
        self.assertIsNone(self.selected("What is the vision of the CSE department?"))

    def test_mission_substring_inside_another_word_is_not_vision_intent(self):
        self.assertIsNone(get_query_type("What is the admissions commission process?"))
        self.assertIsNone(self.selected("What is the admissions commission process?"))

    def test_unrelated_and_intake_questions_do_not_trigger_targeted_selection(self):
        cse = self.selected("What is the CSE intake?")
        self.assertTrue(cse)
        self.assertEqual({chunk["chunk_id"] for chunk in cse}, {55})
        self.assertIn("current intake of 720", cse[0]["text"].casefold())
        self.assertIsNone(self.selected("When is the next cultural event?"))
        self.assertIsNone(self.selected("What is the library's opening time?"))

    def test_programme_specific_document_questions_abstain_without_matching_evidence(self):
        for question in (
            "What documents are required for the MCA programme?",
            "What documents are required for MBA admission?",
        ):
            with self.subTest(question=question):
                retrieved = retrieve_chunks(
                    question, self.chunks, UnusedIndex(len(self.chunks)), UnusedModel(), UnusedBM25()
                )
                self.assertEqual(retrieved, [])
                answer, source = generate_answer_with_source(
                    question, retrieved, SelectedChunkClient(27), self.chunks
                )
                self.assertIn("couldn't find relevant information", answer)
                self.assertIsNone(source)

    def test_cse_intake_retrieval_returns_the_core_cse_fact_only(self):
        for question in (
            "What is the CSE intake?",
            "WHAT IS THE CSE INTAKE!!!",
            "How many students can the CSE intake accommodate?",
            "What is the intake of Computer Science and Engineering?",
        ):
            with self.subTest(question=question):
                retrieved = retrieve_chunks(
                    question,
                    self.chunks,
                    UnusedIndex(len(self.chunks)),
                    UnusedModel(),
                    UnusedBM25(),
                )
                self.assertEqual({chunk["chunk_id"] for chunk in retrieved}, {55})
                self.assertEqual(retrieved[0]["url"], "https://www.rnsit.ac.in/cse/")

    def test_programme_variations_select_official_institutional_listing(self):
        for question in (
            "What programs does RNSIT offer?",
            "Which programmes are offered at RNSIT?",
            "What courses are available at RNSIT?",
        ):
            with self.subTest(question=question):
                selected = self.selected(question)
                self.assertTrue(selected)
                self.assertEqual({c["chunk_id"] for c in selected}, {17})
                text = selected[0]["text"].casefold()
                self.assertTrue(all(name in text for name in ("cse", "ece", "mba", "mca")))
                self.assertEqual(selected[0]["url"].rstrip("/"), "https://www.rnsit.ac.in/admissions")

    def test_source_is_exactly_the_chunk_used_for_answer_generation(self):
        for question, answer_text, chunk_id in (
            ("What programs does RNSIT offer?", "RNSIT offers the listed programmes.", 17),
            ("What is the vision and mission of RNSIT?", "The institutional vision and mission are stated here.", 6),
            ("What is the CSE intake?", "CSE has a current intake of 720.", 55),
        ):
            with self.subTest(question=question):
                selected = self.selected(question)
                selected = [
                    {**chunk, "retrieval_rank": rank, "retrieval_score": 1.0 / rank}
                    for rank, chunk in enumerate(selected, start=1)
                ]
                self.assertEqual({chunk["chunk_id"] for chunk in selected}, {chunk_id})
                answer, source = generate_answer_with_source(
                    question,
                    selected,
                    SelectedChunkClient(chunk_id, answer_text),
                    self.chunks,
                )
                self.assertIn(answer_text.split()[0].casefold(), answer.casefold())
                self.assertEqual(source["url"], selected[0]["url"])
                self.assertEqual(source["title"], selected[0]["title"])
                self.assertEqual(source["excerpt"], selected[0]["text"])

    def test_empty_retrieval_returns_fallback_without_source(self):
        answer, source = generate_answer_with_source(
            "What is the principal's private phone password?",
            [],
            SelectedChunkClient(27),
            self.chunks,
        )
        self.assertIn("couldn't find relevant information", answer)
        self.assertIsNone(source)


if __name__ == "__main__":
    unittest.main()
