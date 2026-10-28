"""Ingest Bedrock documentation using section-based chunks and cached embeddings.

Preparation is free of OpenAI/Pinecone calls. Upload caches paid embeddings
locally before sending them to Pinecone, so upload retries can reuse them.
"""

import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import requests
from requests.adapters import HTTPAdapter
import tiktoken
from urllib3.util.retry import Retry

from config import DATA_DIR, EMBEDDING_MODEL, GUIDES
from documentation import discover, write_json
from chunking import CHUNKER_VERSION, section_chunks


ROOT = DATA_DIR / CHUNKER_VERSION
NAMESPACE = "bedrock-sections-v2"
DIMENSIONS = 1536
BATCH_SIZE = 32  # Keep 1536-dimensional JSON upserts comfortably below 2 MB.
PRICE_PER_MILLION = 0.02  # Standard input price checked 2026-10-28; estimate only.
# Invalidate prepared artifacts if the implementation changes.
CHUNKER_HASH = sha256(Path(__file__).with_name("chunking.py").read_bytes()).hexdigest()
# Adding an optional context name preserves the previous Bedrock chunk output.
COMPATIBLE_CHUNKER_HASHES = {
    CHUNKER_HASH,
    "12c38dc7485a460a741a66080fab578bc2ef223dbd7e90cc1c172207db4e8610",
}


def digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def page_path(url: str) -> Path:
    return ROOT / "pages" / f"{digest(url)}.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_page(url: str, guide_name: str) -> dict:
    path = page_path(url)
    if path.exists():
        saved = read_json(path)
        if saved.get("chunker_hash") in COMPATIBLE_CHUNKER_HASHES:
            return saved
    cache = ROOT / "html" / f"{digest(url)}.json"
    if cache.exists():
        fetched = read_json(cache)
    else:
        retry = Retry(total=3, backoff_factor=1,
                      status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            response = session.get(url, timeout=(10, 45))
            response.raise_for_status()
            fetched = {"source": response.url, "html": response.text}
        write_json(cache, fetched)
    source, guide = urlparse(fetched["source"]), urlparse(GUIDES[guide_name])
    if not (source.scheme == guide.scheme and source.netloc == guide.netloc
            and source.path.startswith(guide.path) and source.path.endswith(".html")
            and not source.query):
        raise ValueError(f"Unexpected redirect: {fetched['source']}")
    article = BeautifulSoup(fetched["html"], "html.parser").select_one("#main-col-body")
    if article is None or article.find("h1") is None:
        raise ValueError("Missing article or page heading")
    title = article.find("h1").get_text(" ", strip=True)
    documents = section_chunks(str(article), fetched["source"], title)
    chunks = []
    for number, document in enumerate(documents):
        text = document.page_content
        if not text.strip() or len(text) > 1000:
            raise ValueError("Invalid chunk length")
        metadata = {**document.metadata, "guide": guide_name, "chunk_index": number,
                    "text": text, "content_hash": digest(text), "embedding_model": EMBEDDING_MODEL}
        chunks.append({"id": digest(f"{fetched['source']}:{CHUNKER_VERSION}:{number}:{text}"),
                       "metadata": metadata})
    page = {"url": url, "source": fetched["source"], "chunker_hash": CHUNKER_HASH, "chunks": chunks}
    write_json(path, page)
    return page


def prepare(workers: int) -> int:
    inventory_path = DATA_DIR / "inventory.json"
    inventory = read_json(inventory_path) if inventory_path.exists() else discover()
    pages = inventory["pages"]
    report = {"chunker_hash": CHUNKER_HASH, "namespace": NAMESPACE, "pages": {}}
    # Pin the inventory for upload; failed preparation never silently disappears.
    write_json(ROOT / "inventory.json", inventory)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(prepare_page, url, entry["guide"]): url for url, entry in pages.items()}
        for number, job in enumerate(as_completed(jobs), 1):
            url = jobs[job]
            try:
                page = job.result()
                report["pages"][url] = {"status": "prepared", "chunks": len(page["chunks"])}
            except Exception as error:
                report["pages"][url] = {"status": "failed", "error": str(error)}
                print(f"Failed: {url}: {error}", flush=True)
            if number % 25 == 0 or number == len(pages):
                write_json(ROOT / "preparation.json", report)
                print(f"Prepared/checked {number}/{len(pages)} pages", flush=True)
    write_json(ROOT / "preparation.json", report)
    failures = sum(p["status"] == "failed" for p in report["pages"].values())
    print(f"Preparation failures: {failures}")
    if failures:
        return 1
    estimate()
    return 0


def prepared_pages():
    inventory = read_json(ROOT / "inventory.json")
    seen = set()
    for url in inventory["pages"]:
        page = read_json(page_path(url))
        if page["chunker_hash"] not in COMPATIBLE_CHUNKER_HASHES:
            raise ValueError("Chunker changed: rerun --prepare before upload")
        # Multiple old URLs can redirect to the same canonical page.
        if page["source"] not in seen:
            seen.add(page["source"])
            yield page


def cache_key(text: str) -> str:
    return digest(f"{EMBEDDING_MODEL}:{DIMENSIONS}:{text}")


def open_cache():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "embeddings.sqlite3")
    db.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
    return db


def cached_vector(db, key):
    row = db.execute("SELECT vector FROM embeddings WHERE key=?", (key,)).fetchone()
    if row is None:
        return None
    vector = array("f")
    vector.frombytes(row[0])
    if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
        raise ValueError("Corrupt local embedding cache")
    return list(vector)


def estimate() -> dict:
    encoding = tiktoken.get_encoding("cl100k_base")
    unique = {}
    pages = chunks = 0
    for page in prepared_pages():
        pages += 1
        for chunk in page["chunks"]:
            chunks += 1
            text = chunk["metadata"]["text"]
            tokens = len(encoding.encode(text, disallowed_special=()))
            if tokens > 8191:
                raise ValueError("Chunk exceeds embedding input limit")
            unique[cache_key(text)] = tokens
    with open_cache() as db:
        needed_tokens = sum(tokens for key, tokens in unique.items() if cached_vector(db, key) is None)
    result = {"canonical_pages": pages, "chunks": chunks, "unique_embedding_inputs": len(unique),
              "uncached_tokens": needed_tokens,
              "estimated_embedding_usd": needed_tokens / 1_000_000 * PRICE_PER_MILLION,
              "price_per_million_tokens": PRICE_PER_MILLION}
    write_json(ROOT / "estimate.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("Estimate excludes Pinecone costs and API retries; pricing can change.")
    return result


def destination():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    name = os.getenv("PINECONE_INDEX_NAME") or "bedrock-docs"
    description = pc.describe_index(name)
    if description.dimension != DIMENSIONS or description.metric != "cosine" or not description.status["ready"]:
        raise ValueError("Index must be ready with 1536 dimensions and cosine metric")
    return pc.Index(host=description.host), description.host


def upload() -> None:
    # Complete local validation before any paid embedding request.
    estimate()
    index, host = destination()
    state_path = ROOT / f"upload-{digest(host + NAMESPACE)[:16]}.json"
    state = read_json(state_path) if state_path.exists() else {
        "host": host, "namespace": NAMESPACE, "chunker_hash": CHUNKER_HASH, "pages": {},
    }
    if state["chunker_hash"] not in COMPATIBLE_CHUNKER_HASHES:
        raise ValueError("Existing namespace checkpoint uses a different chunker")
    model = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=DIMENSIONS,
                             check_embedding_ctx_length=False, max_retries=2, request_timeout=60)
    db = open_cache()
    try:
        for number, page in enumerate(prepared_pages(), 1):
            ids = [c["id"] for c in page["chunks"]]
            if state["pages"].get(page["source"]) == ids:
                continue
            for offset in range(0, len(page["chunks"]), BATCH_SIZE):
                batch = page["chunks"][offset:offset + BATCH_SIZE]
                texts = {cache_key(c["metadata"]["text"]): c["metadata"]["text"] for c in batch}
                vectors = {key: cached_vector(db, key) for key in texts}
                missing = [key for key in texts if vectors[key] is None]
                if missing:
                    generated = model.embed_documents([texts[key] for key in missing])
                    if len(generated) != len(missing):
                        raise ValueError("Embedding response count mismatch")
                    for key, vector in zip(missing, generated):
                        if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
                            raise ValueError("Invalid embedding dimensions or values")
                        db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                   (key, array("f", vector).tobytes()))
                        vectors[key] = vector
                    db.commit()  # Persist paid results BEFORE any upload can fail.
                records = [{"id": c["id"], "values": vectors[cache_key(c["metadata"]["text"])],
                            "metadata": c["metadata"]} for c in batch]
                # Retrying an upsert is safe: IDs are deterministic and embeddings are cached.
                for attempt in range(3):
                    try:
                        response = index.upsert(vectors=records, namespace=NAMESPACE)
                        if response.upserted_count != len(records):
                            raise ValueError("Pinecone upsert count mismatch")
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
            state["pages"][page["source"]] = ids
            write_json(state_path, state)
            print(f"[{number}] Stored {len(ids)} chunks: {page['source']}", flush=True)
    finally:
        db.close()
    print(f"Upload complete in {NAMESPACE}. Run --verify to audit stored IDs and metadata.")


def verify() -> int:
    index, host = destination()
    missing = mismatched = checked = 0
    failed_pages = []
    for page in prepared_pages():
        failures = 0
        for offset in range(0, len(page["chunks"]), 100):
            batch = page["chunks"][offset:offset + 100]
            fetched = index.fetch(ids=[c["id"] for c in batch], namespace=NAMESPACE).vectors
            for chunk in batch:
                checked += 1
                stored = fetched.get(chunk["id"])
                if stored is None:
                    missing += 1
                    failures += 1
                elif (len(stored.values) != DIMENSIONS
                      or stored.metadata != chunk["metadata"]):
                    mismatched += 1
                    failures += 1
        if failures:
            failed_pages.append(page["source"])
    report = {"checked": checked, "missing": missing, "mismatched": mismatched,
              "failed_pages": failed_pages, "namespace": NAMESPACE}
    write_json(ROOT / "verification.json", report)
    # Let a subsequent --upload repair these pages using cached embeddings.
    state_path = ROOT / f"upload-{digest(host + NAMESPACE)[:16]}.json"
    if failed_pages and state_path.exists():
        state = read_json(state_path)
        for source in failed_pages:
            state["pages"].pop(source, None)
        write_json(state_path, state)
    print(json.dumps(report, indent=2))
    return int(bool(missing or mismatched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true", help="Fetch HTML, chunk and estimate; no embedding calls")
    modes.add_argument("--estimate", action="store_true", help="Validate prepared pages and estimate remaining embedding cost")
    modes.add_argument("--upload", action="store_true", help="Generate cached embeddings and upload; uses API credits")
    modes.add_argument("--verify", action="store_true", help="Read back every expected vector and metadata")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    load_dotenv(Path(__file__).with_name(".env"))
    if args.prepare:
        raise SystemExit(prepare(args.workers))
    elif args.estimate:
        estimate()
    elif args.upload:
        upload()
    else:
        raise SystemExit(verify())
