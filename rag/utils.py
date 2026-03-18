from pathlib import Path
from urllib.parse import urlparse

SOURCE_LABELS = {
    "IEA_GlobalEVOutlook2025_summary.pdf": "IEA Global EV Outlook 2025",
    "[LG에너지솔루션]사업보고서_사업개요.pdf": "LGES 사업보고서",
    "CATL 사업보고서_summary.pdf": "CATL 사업보고서",
    "배터리 산업 - ESS 배터리의 빛과 그림자.pdf": "SK증권 ESS 보고서",
}


def _get_source_label(source: str) -> str:
    filename = Path(source).name
    return SOURCE_LABELS.get(filename, filename)


def _get_site_label(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    hostname = hostname.removeprefix("www.").removeprefix("m.")
    if not hostname:
        return "웹페이지"
    return hostname


def _build_web_reference(doc: dict) -> str:
    url = doc.get("url", doc.get("link", ""))
    if not url:
        return ""

    title = (doc.get("title") or doc.get("headline") or "").strip()
    site = (doc.get("site_name") or doc.get("source") or _get_site_label(url)).strip()
    published_at = (
        doc.get("published_date")
        or doc.get("published_at")
        or doc.get("date")
        or ""
    )

    if not title:
        title = site

    return f"[웹] {site} | {published_at} | {title} | {url}"


def format_docs(docs):
    """RAG 검색 결과 문서를 XML 형식으로 포맷."""
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content>"
            f"<source>{doc.metadata.get('source', 'unknown')}</source>"
            f"<page>{int(doc.metadata.get('page', 0)) + 1}</page>"
            f"<citation>{_get_source_label(doc.metadata.get('source', 'unknown'))} (p.{int(doc.metadata.get('page', 0)) + 1})</citation></document>"
            for doc in docs
        ]
    )


def format_searched_docs(docs):
    """웹 검색 결과를 XML 형식으로 포맷."""
    return "\n".join(
        [
            f"<document><content>{doc.get('content', doc.get('snippet', ''))}</content>"
            f"<title>{doc.get('title', doc.get('headline', 'unknown'))}</title>"
            f"<site>{doc.get('site_name', doc.get('source', _get_site_label(doc.get('url', doc.get('link', '')))))}</site>"
            f"<source>{doc.get('url', doc.get('link', 'unknown'))}</source></document>"
            for doc in docs
        ]
    )


def extract_sources_from_docs(docs) -> list[str]:
    """검색된 문서에서 출처 목록 추출. REFERENCE용으로 파일명만 보존한다."""
    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        if not source:
            continue

        source_label = _get_source_label(source)
        if source_label not in sources:
            sources.append(source_label)
    return sources


def extract_sources_from_web(docs) -> list[str]:
    """웹 검색 결과에서 REFERENCE용 웹 출처 목록 추출."""
    sources = []
    for doc in docs:
        reference = _build_web_reference(doc)
        if reference and reference not in sources:
            sources.append(reference)
    return sources
