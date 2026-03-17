"""
Tavily 웹 검색 Tool 래퍼.

긍정·부정 양방향 쿼리를 지원하여 확증 편향 방지 1단계를 구현한다.
"""
import os
from langchain_tavily import TavilySearch


def get_web_search_tool(max_results: int = 5) -> TavilySearch:
    """TavilySearch 도구 인스턴스 반환."""
    return TavilySearch(max_results=max_results)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    단일 쿼리 웹 검색 실행.
    결과: [{"content": ..., "url": ...}, ...]
    """
    tool = get_web_search_tool(max_results=max_results)
    results = tool.invoke({"query": query})
    if isinstance(results, list):
        return results
    # TavilySearch가 dict로 반환하는 경우
    return results.get("results", []) if isinstance(results, dict) else []


def web_search_bidirectional(topic: str, entity: str) -> tuple[list[dict], list[dict]]:
    """
    확증 편향 방지를 위한 양방향 검색.
    긍정·부정 쿼리를 각각 실행하여 결과를 반환한다.

    Args:
        topic: 분석 주제 (예: "포트폴리오 전략 ESS 수주")
        entity: 분석 대상 기업 (예: "LGES" or "CATL")

    Returns:
        (positive_results, negative_results)
    """
    positive_query = f"{entity} 강점 경쟁우위 {topic}"
    negative_query = f"{entity} 리스크 약점 실적부진 위협 {topic}"

    positive_results = web_search(positive_query)
    negative_results = web_search(negative_query)

    return positive_results, negative_results
