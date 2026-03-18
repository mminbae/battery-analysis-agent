"""
T1: Market Research Agent

시장 환경 파악 — EV 캐즘·HEV·ESS·경쟁 구도 등 외부 변화 정리.
RAG (시장/ESS 문서) + Web Search 활용.

[P1 수정]
- MIN_CONTENT_CHARS: 출력 내용 최소 길이 체크 추가 (기존: 출처 수만 체크)
- Agentic RAG: RAG 결과 부족 시 쿼리를 재작성하여 보완 검색 수행
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from rag.retriever import get_retriever
from rag.utils import format_docs, format_searched_docs, extract_sources_from_docs, extract_sources_from_web
from tools.web_search import web_search
from prompts.prompts import MARKET_RESEARCH_SYSTEM, MARKET_RESEARCH_HUMAN

MODEL_NAME = "gpt-4o-mini"

# 계량 조건
MIN_SOURCES = 3          # 출처 수 최소 기준
MIN_CONTENT_CHARS = 1000  # [P1] 출력 내용 최소 길이 (5개 항목 × 최소 200자)
MIN_RAG_DOCS = 3          # [P1] Agentic RAG: 문서 수 최소 기준


def _agentic_rag_search(retriever, initial_queries: list[str], category: str) -> list:
    """
    [P1] Agentic RAG: 초기 쿼리로 검색 후 결과가 부족하면 대안 쿼리로 재검색.
    Returns: unique deduplicated docs
    """
    rag_docs = []
    all_sources = []

    for q in initial_queries:
        docs = retriever.search(category, q)
        rag_docs.extend(docs)
        all_sources.extend(extract_sources_from_docs(docs))

    # 중복 제거
    seen = set()
    unique_docs = [d for d in rag_docs if not (d.page_content in seen or seen.add(d.page_content))]

    # [P1] 결과 부족 시 대안 쿼리로 재검색
    if len(unique_docs) < MIN_RAG_DOCS:
        print(f"[T1/RAG] 문서 부족 ({len(unique_docs)}건) → 대안 쿼리로 재검색")
        fallback_queries = [
            "electric vehicle market slowdown chasm battery 2024 2025",
            "ESS energy storage system market growth GWh",
            "LFP NCM sodium battery technology competition cost",
            "global battery supply chain geopolitical risk China",
        ]
        for q in fallback_queries:
            docs = retriever.search(category, q)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)
                    all_sources.extend(extract_sources_from_docs([doc]))

    return unique_docs, list(dict.fromkeys(all_sources))


def _run_market_research(user_query: str) -> tuple[str, list[str], bool, list[str]]:
    """
    RAG + Web Search로 시장 환경 분석 수행.
    Returns: (content, sources, quantitative_check, fallback_items)
    """
    retriever = get_retriever()
    all_sources = []
    fallback_items = []

    # [P1] Agentic RAG 검색
    rag_queries = [
        "전기차 캐즘 EV 수요 둔화 원인",
        "HEV 하이브리드 시장 성장 배터리",
        "ESS 에너지저장장치 시장 성장",
        "배터리 LFP NCM 나트륨이온 기술 경쟁",
    ]
    unique_docs, rag_sources = _agentic_rag_search(retriever, rag_queries, "market")
    all_sources.extend(rag_sources)

    # Web 검색 — 최신 시장 동향
    web_queries = [
        "EV 전기차 캐즘 2024 2025 배터리 시장",
        "ESS 배터리 시장 성장 전망 2025",
        "CATL LGES 배터리 시장 경쟁 구도",
        "HEV 하이브리드 배터리 수요 증가 2025",
    ]
    web_docs_all = []
    for q in web_queries:
        results = web_search(q, max_results=3)
        web_docs_all.extend(results)
        all_sources.extend(extract_sources_from_web(results))

    # 계량 조건 체크: 총 출처 수
    unique_sources = list(dict.fromkeys(all_sources))
    if len(unique_sources) < MIN_SOURCES:
        quantitative_check = False
        fallback_items.append(f"시장 분석 출처 부족 (확보: {len(unique_sources)}/{MIN_SOURCES}건)")
    else:
        quantitative_check = True

    # EV/배터리 무관 문서 비율 체크
    battery_keywords = ["배터리", "battery", "EV", "전기차", "ESS", "HEV", "CATL", "LGES", "LFP", "NCM"]
    relevant_docs = [
        d for d in unique_docs
        if any(kw.lower() in d.page_content.lower() for kw in battery_keywords)
    ]
    if unique_docs and len(relevant_docs) / len(unique_docs) < 0.5:
        quantitative_check = False
        fallback_items.append("EV/배터리 무관 문서 비율 > 50%")

    # LLM 호출로 분석 결과 생성
    rag_context = format_docs(unique_docs) if unique_docs else "RAG 문서 없음 (data/ 폴더에 PDF 배치 필요)"
    web_context = format_searched_docs(web_docs_all) if web_docs_all else "웹 검색 결과 없음"

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=MARKET_RESEARCH_SYSTEM),
        HumanMessage(content=f"""{MARKET_RESEARCH_HUMAN.format(user_query=user_query)}

## RAG 검색 결과 (시장·ESS 문서)
{rag_context}

## 웹 검색 결과
{web_context}
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # [P1] 출력 내용 길이 체크
    if len(content) < MIN_CONTENT_CHARS:
        quantitative_check = False
        fallback_items.append(
            f"T1 출력 내용 너무 짧음 (현재: {len(content)}자 / 최소: {MIN_CONTENT_CHARS}자)"
        )

    return content, unique_sources, quantitative_check, fallback_items


def market_research_node(state: WorkflowState) -> dict:
    """T1: Market Research Agent 노드."""
    print(f"[T1] Market Research Agent 실행 (retry: {state['retry_count'].get('T1', 0)})")

    content, sources, quantitative_check, fallback_items = _run_market_research(
        state["user_query"]
    )

    output: AgentOutput = {
        "task_id": "T1",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T1] 완료 — quantitative_check: {quantitative_check}, 출처: {len(sources)}건, 내용: {len(content)}자")

    return {
        "market_research": output,
        "current_task": "T1",
        "retry_count": {"T1": state["retry_count"].get("T1", 0) + 1},
        "total_llm_calls": 1,
    }
