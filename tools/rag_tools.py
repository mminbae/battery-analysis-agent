"""
Agent별 RAG 검색 @tool 래퍼.

각 Tool은 해당 Agent가 접근 가능한 문서 카테고리만 검색한다.
- market_rag_search  : Market Research Agent (T1) — 시장·ESS 문서
- lges_rag_search    : LGES Strategy Agent (T2) — LGES 사업보고서
- catl_rag_search    : CATL Strategy Agent (T3) — CATL 사업보고서
- critic_rag_search  : Critic Agent (T4) — 전체 문서 (반론 근거용)
"""
from langchain_core.tools import tool

from rag.retriever import get_retriever
from rag.utils import format_docs, extract_sources_from_docs


@tool("market_rag_search")
def market_rag_search(query: str) -> str:
    """
    시장 환경 분석을 위한 RAG 검색.
    IEA 글로벌 EV 전망 보고서 및 ESS 배터리 보고서에서 관련 정보를 검색한다.
    전기차 캐즘, HEV 트렌드, ESS 시장, 배터리 기술 경쟁 지형 등을 조회할 때 사용.
    """
    retriever = get_retriever()
    docs = retriever.search("market", query)
    if not docs:
        return "검색 결과 없음: 관련 문서를 찾지 못했습니다."
    return format_docs(docs)


@tool("lges_rag_search")
def lges_rag_search(query: str) -> str:
    """
    LG에너지솔루션(LGES) 전략 분석을 위한 RAG 검색.
    LGES 사업보고서에서 포트폴리오, 지역 전략, 기술 경쟁력 정보를 검색한다.
    """
    retriever = get_retriever()
    docs = retriever.search("lges", query)
    if not docs:
        return "검색 결과 없음: LGES 관련 문서를 찾지 못했습니다."
    return format_docs(docs)


@tool("catl_rag_search")
def catl_rag_search(query: str) -> str:
    """
    CATL 전략 분석을 위한 RAG 검색.
    CATL 사업보고서에서 포트폴리오, 지역 전략, 기술 경쟁력 정보를 검색한다.
    """
    retriever = get_retriever()
    docs = retriever.search("catl", query)
    if not docs:
        return "검색 결과 없음: CATL 관련 문서를 찾지 못했습니다."
    return format_docs(docs)


@tool("critic_rag_search")
def critic_rag_search(query: str) -> str:
    """
    반론 및 리스크 탐색을 위한 RAG 검색.
    LGES·CATL 사업보고서 및 시장 보고서 전체에서 약점·위협 근거를 검색한다.
    """
    retriever = get_retriever()
    docs = retriever.search("critic", query)
    if not docs:
        return "검색 결과 없음: 반론 근거 문서를 찾지 못했습니다."
    return format_docs(docs)


def get_rag_sources(category: str, query: str) -> list[str]:
    """검색 결과에서 출처 목록만 추출 (quantitative_check용)."""
    retriever = get_retriever()
    docs = retriever.search(category, query)
    return extract_sources_from_docs(docs)
