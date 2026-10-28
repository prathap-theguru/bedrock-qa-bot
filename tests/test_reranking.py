"""Local checks for candidate merging and safe passage selection."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import rerank_retrieval as module


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        local_file = str(Path(self.temp.name) / "rerank_retrieval.py")
        patched_file = patch.object(module, "__file__", local_file)
        patched_file.start()
        self.addCleanup(patched_file.stop)
        self.documents = [
            (Document(page_content="API object", metadata={"source": "api"}), 0.9),
            (Document(page_content="RAG explanation", metadata={"source": "guide"}), 0.6),
        ]

    def test_merge_removes_duplicates_and_keeps_best_score(self):
        result = module.merge_candidates(self.documents, [(self.documents[1][0], 0.7)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1][1], 0.7)

    def test_selection_preserves_original_documents_and_is_cached(self):
        client = MagicMock()
        client.with_structured_output.return_value.invoke.return_value = module.Selection(ranked_ids=[1, 0])
        with patch.object(module, "ChatOpenAI", return_value=client) as constructor:
            result = module.rerank("What is RAG?", self.documents, k=2)
            self.assertIs(result[0][0], self.documents[1][0])
            self.assertEqual(result[0][1], 0.6)
            again = module.rerank("What is RAG?", self.documents, k=2)
            self.assertEqual(result, again)
            self.assertEqual(constructor.call_count, 1)

    def test_invalid_ids_are_rejected(self):
        for ids in ([2], [-1], [0, 0], [0, 1]):
            client = MagicMock()
            client.with_structured_output.return_value.invoke.return_value = module.Selection(ranked_ids=ids)
            with patch.object(module, "ChatOpenAI", return_value=client), self.assertRaises(ValueError):
                module.rerank("What is RAG?", self.documents, k=1)

    def test_no_candidates_does_not_call_model(self):
        with patch.object(module, "ChatOpenAI") as client:
            self.assertEqual(module.rerank("What is RAG?", [], k=3), [])
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
