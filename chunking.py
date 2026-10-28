"""Section-aware HTML chunking with inline text and source context preserved."""

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKER_VERSION = "sections-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def section_chunks(
    html: str, url: str, title: str, context_name: str = "Amazon Bedrock",
) -> list[Document]:
    """Keep headings together; prefix each section with service/page context."""
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    headings = {}
    fragments = []
    consumed = []

    def flush():
        # Inline text stays in DOM order, including links inside sentences.
        text = "".join(fragments).strip()
        if text:
            sections.append((list(headings.values()), text))
        fragments.clear()

    def visit(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            fragments.append(str(node))
            consumed.append(str(node))
        elif isinstance(node, Tag):
            if node.name in {"script", "style", "nav", "noscript"}:
                return
            if node.name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                flush()
                level = int(node.name[1])
                for key in list(headings):
                    if key >= level:
                        del headings[key]
                headings[level] = node.get_text(" ", strip=True)
                consumed.append(node.get_text())
                return
            block = node.name in {"p", "div", "section", "li", "pre", "table", "tr", "br"}
            if block:
                fragments.append("\n")
            if node.name == "pre":
                fragments.append(node.get_text())  # Preserve code indentation.
                consumed.append(node.get_text())
            else:
                for child in node.children:
                    visit(child)
            if node.name in {"td", "th"}:
                fragments.append("\t")
            if block:
                fragments.append("\n")

    visit(soup)
    flush()
    # Fail instead of silently dropping body text when markup changes.
    for unwanted in soup.select("script, style, nav, noscript"):
        unwanted.decompose()
    expected = "".join(soup.strings)
    if "".join("".join(consumed).split()) != "".join(expected.split()):
        raise ValueError(f"HTML text coverage mismatch: {url}")
    result = []
    for section_headings, section_text in sections:
        content = " ".join(section_text.split())
        if not content or content in {" ".join(h.split()) for h in [title, *section_headings]}:
            continue  # A matching heading alone does not provide an answer.
        context = " > ".join(dict.fromkeys(value for value in [context_name, title, *section_headings] if value))
        # Include context within the same 1,000-character budget as the baseline.
        budget = CHUNK_SIZE - len(context) - 2
        if budget <= CHUNK_OVERLAP:
            raise ValueError("Heading context leaves too little room for section text")
        splitter = RecursiveCharacterTextSplitter(chunk_size=budget, chunk_overlap=CHUNK_OVERLAP)
        for text in splitter.split_text(section_text):
            result.append(Document(
                page_content=f"{context}\n\n{text}",
                metadata={"source": url, "title": title,
                          "section": " > ".join(section_headings), "chunker": CHUNKER_VERSION},
            ))
    if not result:
        raise ValueError(f"No sections extracted: {url}")
    return result

