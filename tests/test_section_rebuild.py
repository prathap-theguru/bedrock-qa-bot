"""Local regression checks; these tests make no API calls."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import ingest as rebuild
from chunking import section_chunks


URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/example.html"
HTML = """<div id="main-col-body"><h1>Example</h1>
<p>Access to <a href="x">foundation models</a> from providers.</p>
<h2 id="code">Code</h2><pre>if True:\n    call()</pre>
<h2>Limits</h2><table><tr><th>Model</th><th>Limit</th></tr>
<tr><td>Example</td><td>10</td></tr></table><!-- private comment -->
<script>ignore()</script></div>"""


class SectionTests(unittest.TestCase):
    def test_inline_order_sections_code_tables_metadata(self):
        chunks = section_chunks(HTML, URL, "Example")
        self.assertEqual(len(chunks), 3)
        self.assertIn("Access to foundation models from providers.", chunks[0].page_content)
        self.assertIn("if True:\n    call()", chunks[1].page_content)
        self.assertIn("Model\tLimit", chunks[2].page_content)
        self.assertIn("Example\t10", chunks[2].page_content)
        self.assertTrue(all(c.metadata["source"] == URL for c in chunks))
        self.assertFalse(any("private comment" in c.page_content or "ignore()" in c.page_content for c in chunks))

    def test_large_section_keeps_content_and_size(self):
        words = [f"word{i:04}" for i in range(500)]
        chunks = section_chunks("<h1>Example</h1><p>" + " ".join(words) + "</p>", URL, "Example")
        self.assertTrue(all(len(c.page_content) <= 1000 for c in chunks))
        combined = " ".join(c.page_content for c in chunks)
        self.assertTrue(all(word in combined for word in words))

    def test_heading_hierarchy_does_not_leak_siblings(self):
        chunks = section_chunks("<h1>X</h1><h2>A</h2><h3>Child</h3><p>one</p><h2>B</h2><p>two</p>", URL, "X")
        self.assertEqual(chunks[0].metadata["section"], "X > A > Child")
        self.assertEqual(chunks[1].metadata["section"], "X > B")
        self.assertEqual(len(chunks), 2)


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.root_patch = patch.object(rebuild, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        rebuild.write_json(self.root / "html" / f"{rebuild.digest(URL)}.json", {"source": URL, "html": HTML})
        rebuild.write_json(self.root / "inventory.json", {"pages": {URL: {"guide": "user-guide"}}})
        self.page = rebuild.prepare_page(URL, "user-guide")

    def test_prepare_uses_cache_and_stable_ids(self):
        with patch.object(rebuild.requests, "Session", side_effect=AssertionError("Network forbidden")):
            again = rebuild.prepare_page(URL, "user-guide")
        self.assertEqual(self.page, again)
        self.assertTrue(all(c["metadata"]["embedding_model"] == rebuild.EMBEDDING_MODEL for c in self.page["chunks"]))

    def test_redirect_deduplicates_pages(self):
        alias = URL.replace("example", "old-example")
        rebuild.write_json(rebuild.page_path(alias), {**self.page, "url": alias})
        rebuild.write_json(self.root / "inventory.json", {"pages": {URL: {}, alias: {}}})
        self.assertEqual(len(list(rebuild.prepared_pages())), 1)

    def test_chunker_change_blocks_upload_input(self):
        rebuild.write_json(rebuild.page_path(URL), {**self.page, "chunker_hash": "outdated"})
        with self.assertRaisesRegex(ValueError, "Chunker changed"):
            list(rebuild.prepared_pages())

    def test_upload_failure_reuses_paid_embeddings_and_resume(self):
        index = MagicMock()
        index.upsert.side_effect = RuntimeError("Upload unavailable")
        model = MagicMock()
        model.embed_documents.side_effect = lambda texts: [[0.1] * rebuild.DIMENSIONS for _ in texts]
        with patch.object(rebuild, "estimate"), patch.object(rebuild, "destination", return_value=(index, "test-host")), patch.object(rebuild, "OpenAIEmbeddings", return_value=model), patch.object(rebuild.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                rebuild.upload()
            self.assertEqual(model.embed_documents.call_count, 1)
            index.upsert.side_effect = None
            index.upsert.return_value = SimpleNamespace(upserted_count=len(self.page["chunks"]))
            rebuild.upload()
            self.assertEqual(model.embed_documents.call_count, 1, "Upload retry must not re-embed")
            calls = index.upsert.call_count
            rebuild.upload()
            self.assertEqual(index.upsert.call_count, calls, "Checkpoint must skip completed pages")
            records = index.upsert.call_args.kwargs["vectors"]
            self.assertEqual(index.upsert.call_args.kwargs["namespace"], "bedrock-sections-v2")
            self.assertEqual(records[0]["metadata"]["source"], URL)

    def test_verification_detects_missing_and_schedules_repair(self):
        host = "test-host"
        state_path = self.root / f"upload-{rebuild.digest(host + rebuild.NAMESPACE)[:16]}.json"
        rebuild.write_json(state_path, {"pages": {URL: ["old-id"]}})
        index = MagicMock()
        index.fetch.return_value = SimpleNamespace(vectors={})
        with patch.object(rebuild, "destination", return_value=(index, host)), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rebuild.verify(), 1)
        self.assertNotIn(URL, json.loads(state_path.read_text())["pages"])
        report = json.loads((self.root / "verification.json").read_text())
        self.assertEqual(report["missing"], len(self.page["chunks"]))


if __name__ == "__main__":
    unittest.main()
