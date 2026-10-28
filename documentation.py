"""Discover documentation pages and persist JSON artifacts."""

from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests

from config import DATA_DIR, GUIDES


def write_json(path: Path, value: dict) -> None:
    """Replace a checkpoint atomically so interruption won't truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def toc_urls(tree: dict, base: str) -> tuple[list[str], list[str]]:
    """Read every nested contents entry, deduplicating page URLs."""
    urls, excluded = set(), set()

    def visit(node):
        if isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, dict):
            href = node.get("href")
            if href:
                url = urldefrag(urljoin(base, href))[0]
                parsed = urlparse(url)
                if url.startswith(base) and parsed.path.endswith(".html") and not parsed.query:
                    urls.add(url)
                else:
                    excluded.add(url)
            visit(node.get("contents", []))

    visit(tree)
    return sorted(urls), sorted(excluded)


def discover() -> dict:
    """Build the inventory from official AWS tables of contents."""
    inventory = {"created_at": datetime.now(timezone.utc).isoformat(), "guides": {}, "pages": {}}
    for guide, base in GUIDES.items():
        response = requests.get(urljoin(base, "toc-contents.json"), timeout=30)
        response.raise_for_status()
        urls, excluded = toc_urls(response.json(), base)
        if not urls:
            raise ValueError(f"No pages discovered for {guide}; check the AWS TOC format.")
        inventory["guides"][guide] = {"toc": response.url, "pages": len(urls), "excluded": excluded}
        for url in urls:
            inventory["pages"][url] = {"guide": guide}
        print(f"{guide}: {len(urls)} pages, {len(excluded)} out-of-scope links", flush=True)
    write_json(DATA_DIR / "inventory.json", inventory)
    return inventory


if __name__ == "__main__":
    discover()
