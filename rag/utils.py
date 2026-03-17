def format_docs(docs):
    """RAG 검색 결과 문서를 XML 형식으로 포맷."""
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content>"
            f"<source>{doc.metadata.get('source', 'unknown')}</source>"
            f"<page>{int(doc.metadata.get('page', 0)) + 1}</page></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    """웹 검색 결과를 XML 형식으로 포맷."""
    return "\n".join(
        [
            f"<document><content>{doc.get('content', doc.get('snippet', ''))}</content>"
            f"<source>{doc.get('url', doc.get('link', 'unknown'))}</source></document>"
            for doc in docs
        ]
    )


def extract_sources_from_docs(docs) -> list[str]:
    """검색된 문서에서 출처 목록 추출."""
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if source and source not in sources:
            sources.append(source)
    return sources


def extract_sources_from_web(docs) -> list[str]:
    """웹 검색 결과에서 URL 목록 추출."""
    sources = []
    for doc in docs:
        url = doc.get("url", doc.get("link", ""))
        if url and url not in sources:
            sources.append(url)
    return sources
