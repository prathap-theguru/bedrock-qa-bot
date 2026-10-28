"""Search the stored Bedrock documentation without generating an answer."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import EMBEDDING_MODEL


def retrieve(
    question: str, k: int = 20, namespace: str = "bedrock-user-guide",
    metadata_filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """Embed the question and return the top matching chunks and scores."""
    for key in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
        if not os.getenv(key):
            raise ValueError(f"Set {key} in your .env file.")
    if not question.strip():
        raise ValueError("Question must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.getenv("PINECONE_INDEX_NAME") or "bedrock-docs"
    description = pc.describe_index(index_name)
    if description.dimension != 1536 or description.metric != "cosine":
        raise ValueError("The Pinecone index must use 1536 dimensions and cosine similarity.")

    vector_store = PineconeVectorStore(
        index=pc.Index(host=description.host),
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        namespace=namespace,
    )
    return vector_store.similarity_search_with_score(question, k=k, filter=metadata_filter)


if __name__ == "__main__":
    load_dotenv(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="bedrock-user-guide")
    parser.add_argument("--question", default="What is Amazon Bedrock?")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    question = args.question
    print(f"Question: {question}")

    print(f"Namespace: {args.namespace}")
    results = retrieve(question, k=args.k, namespace=args.namespace)
    if not results:
        print("No chunks found. Confirm ingestion completed in this index and namespace.")
    for number, (document, score) in enumerate(results, start=1):
        print(f"\nMatch {number} | Similarity: {score:.4f}")
        print(f"Title: {document.metadata.get('title', 'Untitled')}")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(document.page_content)
