"""Citation and abstention checks without live API requests."""

import json
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import chatbot


SOURCE = "https://docs.aws.amazon.com/bedrock/latest/userguide/kb-how-it-works.html"


class ChatbotTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(page_content="Knowledge Bases use RAG to draw on your data.",
                                 metadata={"source": SOURCE, "title": "Knowledge Bases"})

    def answer(self, text="Knowledge Bases use RAG.", ids=None, insufficient=False):
        return chatbot.GroundedAnswer(
            parts=[chatbot.AnswerPart(text=text, source_ids=[1] if ids is None else ids)],
            insufficient_context=insufficient,
        )

    def test_citations_use_retrieved_source(self):
        result = chatbot.render_answer(self.answer(), [self.document])
        self.assertEqual(result, f"Knowledge Bases use RAG. [1]({SOURCE})")

    def test_invalid_or_missing_citations_rejected(self):
        for ids in ([], [0], [-1], [2]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                chatbot.render_answer(self.answer(ids=ids), [self.document])

    def test_model_cannot_invent_citation_url(self):
        with self.assertRaises(ValueError):
            chatbot.render_answer(self.answer(text="Read https://invented.example for facts."), [self.document])

    def test_partial_answer_includes_notice_and_citation(self):
        result = chatbot.render_answer(self.answer(insufficient=True), [self.document])
        self.assertIn(f"[1]({SOURCE})", result)
        self.assertIn(chatbot.INSUFFICIENT, result)

    def test_no_evidence_skips_answer_model(self):
        with patch.object(chatbot, "retrieve_reranked", return_value=[]), patch.object(chatbot, "ChatOpenAI") as model:
            result = chatbot.answer_question("What is a knowledge base?")
            self.assertIn("couldn't find", result)
            model.assert_not_called()

    def test_generation_receives_question_passages_and_preserves_sources(self):
        client = MagicMock()
        structured = client.with_structured_output.return_value
        structured.invoke.return_value = self.answer()
        with patch.object(chatbot, "retrieve_reranked", return_value=[(self.document, 0.7)]) as retrieve, patch.object(chatbot, "ChatOpenAI", return_value=client):
            result = chatbot.answer_question("What is a knowledge base?")
        payload = json.loads(structured.invoke.call_args.args[0][1][1])
        self.assertEqual(payload["question"], "What is a knowledge base?")
        self.assertEqual(payload["passages"][0]["text"], self.document.page_content)
        self.assertEqual(payload["passages"][0]["source"], SOURCE)
        self.assertIn(SOURCE, result)
        self.assertEqual(retrieve.call_args.kwargs["namespace"], "bedrock-sections-v2")

    def test_full_abstention_lists_checked_sources(self):
        answer = chatbot.GroundedAnswer(parts=[], insufficient_context=True)
        result = chatbot.render_answer(answer, [self.document])
        self.assertIn(chatbot.INSUFFICIENT, result)
        self.assertIn(f"Sources checked: [1]({SOURCE})", result)

    def test_blank_question_rejected_before_api_calls(self):
        with patch.object(chatbot, "retrieve_reranked") as retrieve, self.assertRaises(ValueError):
            chatbot.answer_question("  ")
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
