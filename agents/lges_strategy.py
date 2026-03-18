"""
T2: LGES Strategy Agent

LGES 전략 분석 — 포트폴리오·지역·기술 전략 정리.
확증 편향 방지: 긍·부정 양방향 쿼리 사용.
RAG (LGES 문서) + Web Search 활용.

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
from prompts.prompts import LGES_STRATEGY_SYSTEM, LGES_STRATEGY_HUMAN

MODEL_NAME = "gpt-4o-mini"
MIN_SOURCES = 3
MIN_CONTENT_CHARS = 1200  # [P1] 5개 항목 × 최소 200자 + 여유


def _run_lges_strategy(user_query: str, market_content: str) -> tuple[str, list[str], bool, list[str]]:
    """
    RAG + Web Search (긍·부정 양방향)로 LGES 전략 분석 수행.
    Returns: (content, sources, quantitative_check, fallback_items)
    """
    retriever = get_retriever()
    all_sources = []
    fallback_items = []

    # RAG 검색 — LGES 전략 항목별
    rag_queries = [
        "LGES LG에너지솔루션 포트폴리오 전략 다각화",
        "LGES 파우치셀 NCM LFP 배터리 기술",
        "LGES 북미 유럽 생산 전략 공장",
        "LGES ESS HEV 고객 수주",
        "LGES R&D 기술 개발 차세대 배터리",
    ]
    rag_docs = []
    for q in rag_queries:
        docs = retriever.search("lges", q)
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
        print("[T2/RAG] 문서 부족 → 대안 쿼리로 재검색")
        fallback_rag_queries = [
            "LG Energy Solution battery strategy portfolio",
            "LGES annual report business overview",
            "LG에너지솔루션 사업 현황 매출 고객",
        ]
        for q in fallback_rag_queries:
            docs = retriever.search("lges", q)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    unique_docs.append(doc)
                    all_sources.extend(extract_sources_from_docs([doc]))

    # Web 검색 — 확증 편향 방지: 긍·부정 양방향
    positive_results, negative_results = web_search_bidirectional(
        topic="포트폴리오 전략 ESS HEV", entity="LGES LG에너지솔루션"
    )
    web_docs_all = positive_results + negative_results
    all_sources.extend(extract_sources_from_web(web_docs_all))

    # 추가 웹 검색 (CATL 내용 혼입 방지를 위해 명확히 LGES 한정)
    extra_results = web_search("LG에너지솔루션 LGES 2024 2025 전략 실적 생산능력", max_results=3)
    web_docs_all.extend(extra_results)
    all_sources.extend(extract_sources_from_web(extra_results))

    unique_sources = list(dict.fromkeys(all_sources))
    quantitative_check = len(unique_sources) >= MIN_SOURCES

    if not quantitative_check:
        fallback_items.append(f"LGES 분석 출처 부족 (확보: {len(unique_sources)}/{MIN_SOURCES}건)")

    # CATL 내용 혼입 체크
    catl_keywords = ["CATL", "宁德时代", "曾毓群"]
    mixed_docs = [d for d in unique_docs if any(kw in d.page_content for kw in catl_keywords)]
    if unique_docs and len(mixed_docs) > len(unique_docs) * 0.3:
        quantitative_check = False
        fallback_items.append("CATL 내용 혼입 비율 > 30%")

    rag_context = format_docs(unique_docs) if unique_docs else "RAG 문서 없음 (data/LGES_annual_report.pdf 배치 필요)"

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=LGES_STRATEGY_SYSTEM),
        HumanMessage(content=f"""{LGES_STRATEGY_HUMAN.format(
            user_query=user_query,
            market_research_content=market_content
        )}

## RAG 검색 결과 (LGES 사업보고서)
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
            f"T2 출력 내용 너무 짧음 (현재: {len(content)}자 / 최소: {MIN_CONTENT_CHARS}자)"
        )

    return content, unique_sources, quantitative_check, fallback_items


def lges_strategy_node(state: WorkflowState) -> dict:
    """T2: LGES Strategy Agent 노드."""
    print(f"[T2] LGES Strategy Agent 실행 (retry: {state['retry_count'].get('T2', 0)})")

    market_content = state.get("market_research", {}).get("content", "")
    content, sources, quantitative_check, fallback_items = _run_lges_strategy(
        state["user_query"], market_content
    )

    output: AgentOutput = {
        "task_id": "T2",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T2] 완료 — quantitative_check: {quantitative_check}, 출처: {len(sources)}건, 내용: {len(content)}자")

    return {
        "lges_strategy": output,
        "current_task": "T2",
        "retry_count": {"T2": state["retry_count"].get("T2", 0) + 1},
        "total_llm_calls": 1,
    }
