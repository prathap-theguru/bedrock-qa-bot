"""Fetch one public HTML page and add its section embeddings to Pinecone."""

from array import array
from dataclasses import dataclass
from hashlib import sha256
import ipaddress
import json
import math
import os
import socket
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
import requests

from chunking import section_chunks
from config import DATA_DIR, EMBEDDING_MODEL, GUIDES
from documentation import write_json
from ingest import BATCH_SIZE, DIMENSIONS, NAMESPACE, cache_key, cached_vector, destination, open_cache


MAX_PAGE_BYTES = 5 * 1024 * 1024


@dataclass
class IngestionResult:
    source: str
    chunks: int
    unchanged: bool = False


def validate_public_url(url: str) -> None:
    """Keep the URL fetcher restricted to public web servers, including redirects."""
    parsed = urlparse(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or any(c.isspace() for c in url)):
        raise ValueError("Enter a public HTTP or HTTPS URL without login credentials.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except (OSError, ValueError) as error:
        raise ValueError("The URL's host could not be resolved.") from error
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Only public URLs are supported; local and private network addresses are not allowed.")


def fetch_page(url: str) -> tuple[str, str]:
    url = urldefrag(url.strip())[0]
    for _ in range(6):
        validate_public_url(url)
        with requests.get(url, timeout=(10, 45), allow_redirects=False, stream=True) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("The page returned a redirect without a destination.")
                url = urldefrag(urljoin(url, location))[0]
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if not any(kind in content_type for kind in ("text/html", "application/xhtml+xml")):
                raise ValueError("This version supports HTML pages only, not PDFs or downloads.")
            content = bytearray()
            for block in response.iter_content(chunk_size=65536):
                content.extend(block)
                if len(content) > MAX_PAGE_BYTES:
                    raise ValueError("The page is too large (maximum 5 MB).")
            # Let the HTML parser read the page's declared encoding.
            return url, str(BeautifulSoup(bytes(content), "html.parser"))
    raise ValueError("The URL redirected too many times.")


def page_chunks(html: str, source: str):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("#main-col-body") or soup.find("article") or soup.find("main") or soup.body
    if article is None:
        raise ValueError("No readable page content found.")
    heading = article.find("h1") or soup.title
    title = heading.get_text(" ", strip=True) if heading else urlparse(source).hostname
    for element in article.select("script, style, nav, noscript, header, footer, aside, form"):
        element.decompose()
    context = "Amazon Bedrock" if any(source.startswith(base) for base in GUIDES.values()) else ""
    return section_chunks(str(article), source, title, context_name=context)


def ingest_url(url: str) -> IngestionResult:
    for key in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
        if not os.getenv(key):
            raise ValueError(f"Set {key} in your .env file first.")
    source, html = fetch_page(url)
    chunks = page_chunks(html, source)
    records = []
    for number, chunk in enumerate(chunks):
        text = chunk.page_content
        content_hash = sha256(text.encode()).hexdigest()
        records.append({
            "id": "url-upload-" + sha256(f"{source}:{number}:{text}".encode()).hexdigest(),
            "metadata": {**chunk.metadata, "text": text, "guide": "url-upload",
                         "embedding_model": EMBEDDING_MODEL, "content_hash": content_hash},
        })
    page_hash = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    index, host = destination()
    state_path = DATA_DIR / "url-uploads" / f"{sha256((host + NAMESPACE + source).encode()).hexdigest()}.json"
    prior = json.loads(state_path.read_text()) if state_path.exists() else {}
    if prior.get("page_hash") == page_hash:
        present = True
        for offset in range(0, len(records), 100):
            batch = records[offset:offset + 100]
            fetched = index.fetch(ids=[r["id"] for r in batch], namespace=NAMESPACE).vectors
            if any(r["id"] not in fetched or fetched[r["id"]].metadata != r["metadata"] for r in batch):
                present = False
                break
        if present:
            return IngestionResult(source=source, chunks=len(records), unchanged=True)

    model = OpenAIEmbeddings(model=EMBEDDING_MODEL, dimensions=DIMENSIONS,
                             check_embedding_ctx_length=False, max_retries=2, request_timeout=60)
    db = open_cache()
    try:
        for offset in range(0, len(records), BATCH_SIZE):
            batch = records[offset:offset + BATCH_SIZE]
            texts = {cache_key(r["metadata"]["text"]): r["metadata"]["text"] for r in batch}
            vectors = {key: cached_vector(db, key) for key in texts}
            missing = [key for key in texts if vectors[key] is None]
            if missing:
                generated = model.embed_documents([texts[key] for key in missing])
                if len(generated) != len(missing):
                    raise ValueError("Embedding response count mismatch.")
                for key, vector in zip(missing, generated):
                    if len(vector) != DIMENSIONS or not all(math.isfinite(x) for x in vector):
                        raise ValueError("Invalid embedding returned by the model.")
                    db.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)", (key, array("f", vector).tobytes()))
                    vectors[key] = vector
                db.commit()
            payload = [{**r, "values": vectors[cache_key(r["metadata"]["text"])]} for r in batch]
            response = index.upsert(vectors=payload, namespace=NAMESPACE)
            if response.upserted_count != len(batch):
                raise ValueError("Pinecone upload count mismatch; retry the URL.")
    finally:
        db.close()
    # Delete only this uploader's obsolete records after replacements are uploaded.
    ids = [r["id"] for r in records]
    obsolete = sorted(set(prior.get("ids", [])) - set(ids))
    for offset in range(0, len(obsolete), 100):
        index.delete(ids=obsolete[offset:offset + 100], namespace=NAMESPACE)
    write_json(state_path, {"source": source, "page_hash": page_hash, "ids": ids})
    return IngestionResult(source=source, chunks=len(records))
