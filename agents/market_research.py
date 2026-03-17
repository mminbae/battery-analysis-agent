"""
T1: Market Research Agent

시장 환경 파악 — EV 캐즘·HEV·ESS·경쟁 구도 등 외부 변화 정리.
RAG (시장/ESS 문서) + Web Search 활용.
"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from rag.retriever import get_retriever
from rag.utils import format_docs, format_searched_docs, extract_sources_from_docs, extract_sources_from_web
from tools.web_search import web_search
from prompts.prompts import MARKET_RESEARCH_SYSTEM, MARKET_RESEARCH_HUMAN

MODEL_NAME = "gpt-4o-mini"

# 계량 조건: 검색 결과 수 < 3건이면 실패
MIN_SOURCES = 3


def _run_market_research(user_query: str) -> tuple[str, list[str], bool, list[str]]:
    """
    RAG + Web Search로 시장 환경 분석 수행.
    Returns: (content, sources, quantitative_check, fallback_items)
    """
    retriever = get_retriever()
    all_sources = []
    fallback_items = []

    # RAG 검색 — 여러 쿼리로 시장 맥락 확보
    rag_queries = [
        "전기차 캐즘 EV 수요 둔화 원인",
        "HEV 하이브리드 시장 성장 배터리",
        "ESS 에너지저장장치 시장 성장",
        "배터리 LFP NCM 나트륨이온 기술 경쟁",
    ]
    rag_docs = []
    for q in rag_queries:
        docs = retriever.search("market", q)
        rag_docs.extend(docs)
        all_sources.extend(extract_sources_from_docs(docs))

    # 중복 제거
    seen_contents = set()
    unique_docs = []
    for doc in rag_docs:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)

    # Web 검색 — 최신 시장 동향
    web_queries = [
        "EV 전기차 캐즘 2024 2025 배터리 시장",
        "ESS 배터리 시장 성장 전망 2025",
        "CATL LGES 배터리 시장 경쟁 구도",
    ]
    web_docs_all = []
    for q in web_queries:
        results = web_search(q, max_results=3)
        web_docs_all.extend(results)
        all_sources.extend(extract_sources_from_web(results))

    # 계량 조건 체크: 총 출처 수
    unique_sources = list(dict.fromkeys(all_sources))  # 순서 유지 중복 제거
    quantitative_check = len(unique_sources) >= MIN_SOURCES

    if not quantitative_check:
        fallback_items.append(f"시장 분석 출처 부족 (확보: {len(unique_sources)}/{MIN_SOURCES}건)")

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

    print(f"[T1] 완료 — quantitative_check: {quantitative_check}, 출처: {len(sources)}건")

    return {
        "market_research": output,
        "current_task": "T1",
        "retry_count": {"T1": state["retry_count"].get("T1", 0) + 1},
        "total_llm_calls": 1,
    }
