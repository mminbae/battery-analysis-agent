"""
T3: CATL Strategy Agent

CATL 전략 분석 — 포트폴리오·지역·기술 전략 정리.
확증 편향 방지: 긍·부정 양방향 쿼리 사용.
RAG (CATL 문서) + Web Search 활용.

[P1 수정]
- MIN_CONTENT_CHARS: 출력 내용 최소 길이 체크 추가
- Agentic RAG: RAG 결과 부족 시 대안 쿼리로 보완 검색
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from rag.retriever import get_retriever
from rag.utils import format_docs, format_searched_docs, extract_sources_from_docs, extract_sources_from_web
from tools.web_search import web_search, web_search_bidirectional
from prompts.prompts import CATL_STRATEGY_SYSTEM, CATL_STRATEGY_HUMAN

MODEL_NAME = "gpt-4o-mini"
MIN_SOURCES = 3
MIN_CONTENT_CHARS = 1200  # [P1] 5개 항목 × 최소 200자 + 여유


def _build_catl_web_queries(user_query: str) -> list[str]:
    """최종 보고서 Section 4 소제목에 대응되는 웹 검색 쿼리."""
    return [
        f"CATL 포트폴리오 다각화 전략 개요 2025 {user_query}",
        "CATL LFP NCM 나트륨이온 Shenxing 제품 화학 포트폴리오 2025",
        "CATL 글로벌 OEM ESS 신흥시장 고객 시장 다각화 수주 2025",
        "CATL 중국 유럽 동남아 생산 전략 공장 2025",
        "CATL 핵심 경쟁력 R&D 초고속충전 응축배터리 2025",
    ]


def _run_catl_strategy(user_query: str, market_content: str) -> tuple[str, list[str], bool, list[str]]:
    """
    RAG + Web Search (긍·부정 양방향)로 CATL 전략 분석 수행.
    Returns: (content, sources, quantitative_check, fallback_items)
    """
    retriever = get_retriever()
    all_sources = []
    fallback_items = []

    # RAG 검색 — CATL 전략 항목별
    rag_queries = [
        "CATL 포트폴리오 전략 다각화",
        "CATL LFP 나트륨이온 배터리 기술",
        "CATL 유럽 동남아 생산 전략 공장",
        "CATL ESS 글로벌 OEM 고객 수주",
        "CATL R&D 기술 개발 혁신 로드맵",
    ]
    rag_docs = []
    for q in rag_queries:
        docs = retriever.search("catl", q)
        rag_docs.extend(docs)
        all_sources.extend(extract_sources_from_docs(docs))

    seen_contents = set()
    unique_docs = []
    for doc in rag_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

    # [P1] Agentic RAG: RAG 결과 부족 시 대안 쿼리로 보완 검색
    if len(unique_docs) < 3:
        print("[T3/RAG] 문서 부족 → 대안 쿼리로 재검색")
        fallback_rag_queries = [
            "CATL Contemporary Amperex Technology battery strategy 2025",
            "CATL annual report business overview revenue 2025",
            "CATL 사업 현황 매출 고객 점유율 2025",
        ]
        for q in fallback_rag_queries:
            docs = retriever.search("catl", q)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
                    all_sources.extend(extract_sources_from_docs([doc]))

    # Web 검색 — 확증 편향 방지: 긍·부정 양방향
    positive_results, negative_results = web_search_bidirectional(
        topic="시장점유율 LFP 기술 글로벌 공급망", entity="CATL"
    )
    web_docs_all = positive_results + negative_results
    all_sources.extend(extract_sources_from_web(web_docs_all))

    # 추가 웹 검색 — 최종 보고서 Section 4 구조에 맞춘 보강 쿼리
    extra_results = []
    for q in _build_catl_web_queries(user_query):
        results = web_search(q, max_results=3)
        extra_results.extend(results)
        all_sources.extend(extract_sources_from_web(results))
    web_docs_all.extend(extra_results)

    unique_sources = list(dict.fromkeys(all_sources))
    quantitative_check = len(unique_sources) >= MIN_SOURCES

    if not quantitative_check:
        fallback_items.append(f"CATL 분석 출처 부족 (확보: {len(unique_sources)}/{MIN_SOURCES}건)")

    # LGES 내용 혼입 체크
    lges_keywords = ["LG에너지솔루션", "LGES", "LG Energy Solution"]
    mixed_docs = [d for d in unique_docs if any(kw in d.page_content for kw in lges_keywords)]
    if unique_docs and len(mixed_docs) > len(unique_docs) * 0.3:
        quantitative_check = False
        fallback_items.append("LGES 내용 혼입 비율 > 30%")

    rag_context = format_docs(unique_docs) if unique_docs else "RAG 문서 없음 (data/CATL_annual_report.pdf 배치 필요)"
    web_context = format_searched_docs(web_docs_all) if web_docs_all else "웹 검색 결과 없음"

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=CATL_STRATEGY_SYSTEM),
        HumanMessage(content=f"""{CATL_STRATEGY_HUMAN.format(
            user_query=user_query,
            market_research_content=market_content
        )}

## RAG 검색 결과 (CATL 사업보고서)
{rag_context}

## 웹 검색 결과 (긍·부정 양방향)
### 긍정 쿼리 결과
{format_searched_docs(positive_results) if positive_results else '없음'}

### 부정 쿼리 결과
{format_searched_docs(negative_results) if negative_results else '없음'}
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # [P1] 출력 내용 길이 체크
    if len(content) < MIN_CONTENT_CHARS:
        quantitative_check = False
        fallback_items.append(
            f"T3 출력 내용 너무 짧음 (현재: {len(content)}자 / 최소: {MIN_CONTENT_CHARS}자)"
        )

    return content, unique_sources, quantitative_check, fallback_items


def catl_strategy_node(state: WorkflowState) -> dict:
    """T3: CATL Strategy Agent 노드."""
    print(f"[T3] CATL Strategy Agent 실행 (retry: {state['retry_count'].get('T3', 0)})")

    market_content = state.get("market_research", {}).get("content", "")
    content, sources, quantitative_check, fallback_items = _run_catl_strategy(
        state["user_query"], market_content
    )

    output: AgentOutput = {
        "task_id": "T3",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T3] 완료 — quantitative_check: {quantitative_check}, 출처: {len(sources)}건, 내용: {len(content)}자")

    return {
        "catl_strategy": output,
        "current_task": "T3",
        "retry_count": {"T3": state["retry_count"].get("T3", 0) + 1},
        "total_llm_calls": 1,
    }
