"""URL ingestion checks with mocked network, embeddings and Pinecone."""

from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import chatbot
import ingest
import url_ingestion as module


URL = "https://example.com/guide"
HTML = '<html><body><main><h1>Example guide</h1><h2>Overview</h2><p>Use <a href="x">example data</a> for retrieval.</p></main></body></html>'


class URLTests(unittest.TestCase):
    def test_public_url_validation_blocks_private_hosts(self):
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1"):
            with patch.object(module.socket, "getaddrinfo", return_value=[(0, 0, 0, "", (address, 80))]), self.assertRaises(ValueError):
                module.validate_public_url("http://internal.example/page")
        for url in ("file:///etc/passwd", "https://user:password@example.com", ""):
            with self.assertRaises(ValueError):
                module.validate_public_url(url)

    def test_redirects_are_validated_before_following(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 302
        response.headers = {"Location": "http://127.0.0.1/private"}
        with patch.object(module, "validate_public_url", side_effect=[None, ValueError("private")]) as validate, patch.object(module.requests, "get", return_value=response) as fetch:
            with self.assertRaises(ValueError):
                module.fetch_page(URL)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(validate.call_count, 2)

    def test_non_html_is_rejected_before_embedding(self):
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.headers = {"Content-Type": "application/pdf"}
        with patch.object(module, "validate_public_url"), patch.object(module.requests, "get", return_value=response), self.assertRaisesRegex(ValueError, "HTML"):
            module.fetch_page(URL)

    def test_generic_sections_keep_link_text_without_bedrock_label(self):
        chunks = module.page_chunks(HTML, URL)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Use example data for retrieval.", chunks[0].page_content)
        self.assertNotIn("Amazon Bedrock", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["source"], URL)

    def test_external_source_can_be_cited(self):
        document = Document(page_content="Example", metadata={"source": URL + "(one)"})
        answer = chatbot.GroundedAnswer(parts=[chatbot.AnswerPart(text="Example.", source_ids=[1])], insufficient_context=False)
        rendered = chatbot.render_answer(answer, [document])
        self.assertIn("https://example.com/guide%28one%29", rendered)


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.index = MagicMock()
        self.index.upsert.side_effect = lambda vectors, namespace: SimpleNamespace(upserted_count=len(vectors))
        self.model = MagicMock()
        self.model.embed_documents.side_effect = lambda texts: [[0.1] * module.DIMENSIONS for _ in texts]
        patches = [
            patch.object(module, "DATA_DIR", root),
            patch.object(ingest, "ROOT", root / "sections-v2"),
            patch.object(module, "fetch_page", return_value=(URL, HTML)),
            patch.object(module, "destination", return_value=(self.index, "test-host")),
            patch.object(module, "OpenAIEmbeddings", return_value=self.model),
            patch.dict(module.os.environ, {"OPENAI_API_KEY": "test", "PINECONE_API_KEY": "test"}),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def test_upload_and_unchanged_page_skip_paid_request(self):
        first = module.ingest_url(URL)
        self.assertEqual(first.chunks, 1)
        self.assertFalse(first.unchanged)
        call = self.index.upsert.call_args.kwargs
        record = call["vectors"][0]
        self.assertEqual(call["namespace"], "bedrock-sections-v2")
        self.assertEqual(record["metadata"]["source"], URL)
        self.assertEqual(record["metadata"]["guide"], "url-upload")
        self.index.fetch.return_value = SimpleNamespace(vectors={record["id"]: SimpleNamespace(metadata=record["metadata"])})
        again = module.ingest_url(URL)
        self.assertTrue(again.unchanged)
        self.assertEqual(self.model.embed_documents.call_count, 1)
        self.assertEqual(self.index.upsert.call_count, 1)

    def test_failed_upsert_reuses_paid_embedding_on_retry(self):
        self.index.upsert.side_effect = RuntimeError("network")
        with self.assertRaises(RuntimeError):
            module.ingest_url(URL)
        self.index.upsert.side_effect = lambda vectors, namespace: SimpleNamespace(upserted_count=len(vectors))
        module.ingest_url(URL)
        self.assertEqual(self.model.embed_documents.call_count, 1)

    def test_changed_page_deletes_only_previous_upload_ids(self):
        module.ingest_url(URL)
        old_id = self.index.upsert.call_args.kwargs["vectors"][0]["id"]
        changed = HTML.replace("example data", "updated data")
        with patch.object(module, "fetch_page", return_value=(URL, changed)):
            module.ingest_url(URL)
        self.index.delete.assert_called_once_with(ids=[old_id], namespace="bedrock-sections-v2")
        self.assertTrue(old_id.startswith("url-upload-"))


if __name__ == "__main__":
    unittest.main()
