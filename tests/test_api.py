import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pydantic import ValidationError
from fastapi import HTTPException

from api import AskRequest, AskResponse, ConversationTurn, _needs_clarification, _retrieval_query, chat, health
from ask import _is_abstention


class ApiTests(unittest.TestCase):
    def setUp(self):
        chunks = [
            {"chunk_id": 0, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "CSE intake is 720.", "retrieval_rank": 1, "retrieval_score": 0.04},
            {"chunk_id": 1, "title": "CSE - RNSIT", "url": "https://www.rnsit.ac.in/cse/", "text": "CSE department overview.", "retrieval_rank": 2, "retrieval_score": 0.03},
        ]
        resources = (chunks, object(), object(), object(), object())
        self.request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag_resources=resources)))
        self.chunks = chunks

    def source(self, chunk=None):
        chunk = chunk or self.chunks[0]
        return {
            "id": "source-1",
            "title": chunk["title"],
            "url": chunk["url"],
            "excerpt": chunk["text"],
            "retrieval_rank": chunk["retrieval_rank"],
            "retrieval_score": chunk["retrieval_score"],
        }

    def test_empty_and_malformed_questions_fail_validation(self):
        for body in ({"question": ""}, {"question": "hello", "conversation": [{"role": "system", "content": "bad"}]}, {"question": "hello", "extra": True}):
            with self.assertRaises(ValidationError):
                AskRequest.model_validate(body)

    def test_whitespace_only_question_is_rejected_by_route(self):
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(chat(AskRequest(question="   "), self.request))
        self.assertEqual(raised.exception.status_code, 422)

    def test_health_requires_loaded_pipeline_resources(self):
        self.assertEqual(asyncio.run(health(self.request)), {"status": "ok"})
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(health(request))
        self.assertEqual(raised.exception.status_code, 503)

    def test_follow_up_uses_prior_user_topic_for_retrieval(self):
        request = AskRequest(question="Tell me more.", conversation=[{"role": "user", "content": "Explain MCA."}, {"role": "assistant", "content": "Prior answer text."}])
        query = _retrieval_query(request.question, request.conversation)
        self.assertEqual(query, "mca")
        self.assertNotIn("Prior answer text", query)
        self.assertFalse(_needs_clarification(request.question, request.conversation))
        self.assertTrue(_needs_clarification("Tell me more.", []))

    def test_duration_follow_up_keeps_topic_and_current_request_terms(self):
        conversation = [
            ConversationTurn(role="user", content="What is the MCA programme?"),
            ConversationTurn(role="assistant", content="The MCA programme is two years."),
        ]
        query = _retrieval_query("What is its duration?", conversation)
        self.assertEqual(query, "mca programme duration")
        self.assertNotIn("two years", query)

    def test_department_intake_follow_up_keeps_department_topic(self):
        conversation = [
            ConversationTurn(role="user", content="Tell me about the CSE department."),
            ConversationTurn(role="assistant", content="Prior answer text."),
        ]
        self.assertEqual(_retrieval_query("What is its intake?", conversation), "cse department intake")

    def test_other_website_topic_follow_up_keeps_topic_and_request(self):
        conversation = [
            ConversationTurn(role="user", content="Tell me about placements at RNSIT."),
            ConversationTurn(role="assistant", content="Prior answer text."),
        ]
        self.assertEqual(_retrieval_query("What are their statistics?", conversation), "placements rnsit statistics")

    def test_chat_retrieves_with_resolved_topic_but_generates_from_current_question(self):
        payload = AskRequest(
            question="What is its duration?",
            conversation=[
                {"role": "user", "content": "What is the MCA programme?"},
                {"role": "assistant", "content": "The MCA programme is two years."},
            ],
        )
        with patch("api.retrieve_chunks", return_value=self.chunks) as retrieve, patch(
            "api.generate_answer_with_source",
            return_value=("The MCA programme lasts two years.", self.source()),
        ) as generate:
            asyncio.run(chat(payload, self.request))
        retrieve.assert_called_once_with(
            "mca programme duration", *self.request.app.state.rag_resources[:4]
        )
        generate.assert_called_once_with(
            "What is its duration?",
            self.chunks,
            self.request.app.state.rag_resources[4],
            self.chunks,
        )

    def test_abstention_response_does_not_return_unrelated_sources(self):
        payload = AskRequest(question="What is the principal's private phone password?")
        with patch("api.retrieve_chunks", return_value=self.chunks), patch(
            "api.generate_answer_with_source",
            return_value=("I'm sorry, but I cannot provide that private password.", None),
        ):
            response = asyncio.run(chat(payload, self.request))
        self.assertTrue(_is_abstention(response.answer))
        self.assertEqual(response.sources, [])

    def test_supported_answer_keeps_sources(self):
        payload = AskRequest(question="What is CSE intake?")
        with patch("api.retrieve_chunks", return_value=self.chunks), patch(
            "api.generate_answer_with_source",
            return_value=("The CSE intake is 720.", self.source()),
        ):
            response = asyncio.run(chat(payload, self.request))
        self.assertFalse(_is_abstention(response.answer))
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].url, self.chunks[0]["url"])

    def test_abstention_detector_covers_retrieval_fallback_and_preserves_answers(self):
        self.assertTrue(_is_abstention("I couldn't find relevant information on the RNSIT website."))
        self.assertTrue(_is_abstention("The provided information does not specify the password."))
        self.assertTrue(_is_abstention("I'm sorry, I cannot assist with that request."))
        self.assertTrue(_is_abstention("I cannot disclose the principal's private password."))
        self.assertFalse(_is_abstention("The CSE intake is 720 students."))

    def test_explicit_new_subject_does_not_inherit_old_topic(self):
        conversation = [ConversationTurn(role="user", content="Explain CSE.")]
        self.assertEqual(_retrieval_query("What about MCA?", conversation), "What about MCA?")

    def test_api_returns_deduplicated_answer_level_sources(self):
        payload = AskRequest(question="What is CSE intake?")
        with patch("api.retrieve_chunks", return_value=self.chunks), patch(
            "api.generate_answer_with_source",
            return_value=("The CSE intake is 720.", self.source()),
        ) as generated:
            response = asyncio.run(chat(payload, self.request))
        generated.assert_called_once()
        self.assertEqual(generated.call_args.args[3], self.chunks)
        validated = AskResponse.model_validate(response.model_dump())
        self.assertEqual(validated.answer, "The CSE intake is 720.")
        self.assertEqual(len(validated.sources), 1)
        self.assertEqual(validated.sources[0].id, "source-1")
        self.assertEqual(validated.sources[0].url, "https://www.rnsit.ac.in/cse/")
        self.assertIn("720", validated.sources[0].excerpt)

    def test_upstream_failures_are_sanitized(self):
        payload = AskRequest(question="What is CSE intake?")
        with patch("api.retrieve_chunks", return_value=self.chunks), patch("api.generate_answer_with_source", side_effect=TimeoutError("secret detail")):
            with self.assertRaises(Exception) as raised:
                asyncio.run(chat(payload, self.request))
        self.assertEqual(raised.exception.status_code, 504)
        self.assertNotIn("secret", raised.exception.detail)

        with patch("api.retrieve_chunks", return_value=self.chunks), patch("api.generate_answer_with_source", side_effect=RuntimeError("secret detail")):
            with self.assertRaises(Exception) as raised:
                asyncio.run(chat(payload, self.request))
        self.assertEqual(raised.exception.status_code, 502)
        self.assertNotIn("secret", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
