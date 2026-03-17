"""
T4: Critic Agent

반론·리스크 탐색 — LGES·CATL 각각의 약점·위협 근거 확보.
확증 편향 방지 2단계: 부정 근거 ≥ 2건 강제.
"""
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from rag.retriever import get_retriever
from rag.utils import format_docs, format_searched_docs, extract_sources_from_docs, extract_sources_from_web
from tools.web_search import web_search
from prompts.prompts import CRITIC_SYSTEM, CRITIC_HUMAN

MODEL_NAME = "gpt-4o-mini"
MIN_NEGATIVE_EACH = 2  # LGES·CATL 각각 부정 근거 ≥ 2건


def _count_negative_evidence(content: str, entity: str) -> int:
    """
    출력 내용에서 특정 기업의 부정 근거 개수를 카운트.
    번호 목록(1. 2. 3.) 또는 불릿(- •) 형식의 항목을 카운트.
    """
    # entity 언급 섹션 추출
    entity_lower = entity.lower()
    lines = content.split("\n")

    in_entity_section = False
    entity_lines = []
    for line in lines:
        if entity_lower in line.lower() or entity in line:
            in_entity_section = True
        if in_entity_section:
            entity_lines.append(line)
        # 다음 기업 섹션이 시작되면 중단
        other_entity = "catl" if entity_lower == "lges" or entity_lower == "lg" else "lges"
        if in_entity_section and other_entity in line.lower() and line != entity_lines[0]:
            break

    entity_text = "\n".join(entity_lines) if entity_lines else content
    # 번호 목록 또는 불릿 항목 카운트
    numbered = re.findall(r"^\s*\d+[\.\)]\s+.+", entity_text, re.MULTILINE)
    bulleted = re.findall(r"^\s*[-•*]\s+.+", entity_text, re.MULTILINE)
    return len(numbered) + len(bulleted)


def _run_critic(lges_content: str, catl_content: str) -> tuple[str, list[str], bool, list[str]]:
    """
    T2/T3 결과를 기반으로 반론 및 리스크 근거 확보.
    Returns: (content, sources, quantitative_check, fallback_items)
    """
    retriever = get_retriever()
    all_sources = []
    fallback_items = []

    # LGES 부정 근거 검색 (RAG)
    lges_negative_queries = [
        "LGES LG에너지솔루션 가동률 저하 실적 부진",
        "LGES 리스크 약점 경쟁 열위 고객 의존",
        "LG에너지솔루션 재무 부채 투자 부담",
    ]
    lges_docs = []
    for q in lges_negative_queries:
        docs = retriever.search("lges", q)
        lges_docs.extend(docs)
        all_sources.extend(extract_sources_from_docs(docs))

    # CATL 부정 근거 검색 (RAG)
    catl_negative_queries = [
        "CATL 지정학 리스크 미국 관세 규제 제재",
        "CATL 약점 중국 의존 공급망 리스크",
        "CATL 특허 분쟁 기술 모방 경쟁",
    ]
    catl_docs = []
    for q in catl_negative_queries:
        docs = retriever.search("catl", q)
        catl_docs.extend(docs)
        all_sources.extend(extract_sources_from_docs(docs))

    # 전체 문서에서 추가 반론 근거 (critic 카테고리)
    critic_docs = retriever.search("critic", "배터리 기업 리스크 약점 위협 요인")
    all_sources.extend(extract_sources_from_docs(critic_docs))

    # Web 검색 — 부정 근거 강화
    lges_web = web_search("LGES LG에너지솔루션 리스크 약점 문제점 2024", max_results=3)
    catl_web = web_search("CATL 지정학 리스크 관세 약점 문제점 2024", max_results=3)
    all_sources.extend(extract_sources_from_web(lges_web + catl_web))

    unique_sources = list(dict.fromkeys(all_sources))

    # 컨텍스트 구성
    lges_rag_context = format_docs(lges_docs) if lges_docs else "LGES RAG 문서 없음"
    catl_rag_context = format_docs(catl_docs) if catl_docs else "CATL RAG 문서 없음"
    lges_web_context = format_searched_docs(lges_web) if lges_web else "없음"
    catl_web_context = format_searched_docs(catl_web) if catl_web else "없음"

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=CRITIC_SYSTEM),
        HumanMessage(content=f"""{CRITIC_HUMAN.format(
            lges_content=lges_content,
            catl_content=catl_content
        )}

## LGES 부정 근거 RAG 검색 결과
{lges_rag_context}

## CATL 부정 근거 RAG 검색 결과
{catl_rag_context}

## LGES 부정 근거 웹 검색 결과
{lges_web_context}

## CATL 부정 근거 웹 검색 결과
{catl_web_context}
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # 계량 조건 체크: 각 기업 부정 근거 ≥ 2건
    lges_count = _count_negative_evidence(content, "LGES")
    catl_count = _count_negative_evidence(content, "CATL")
    quantitative_check = lges_count >= MIN_NEGATIVE_EACH and catl_count >= MIN_NEGATIVE_EACH

    if lges_count < MIN_NEGATIVE_EACH:
        fallback_items.append(f"LGES 부정 근거 부족 (확보: {lges_count}/{MIN_NEGATIVE_EACH}건)")
    if catl_count < MIN_NEGATIVE_EACH:
        fallback_items.append(f"CATL 부정 근거 부족 (확보: {catl_count}/{MIN_NEGATIVE_EACH}건)")

    print(f"[T4] LGES 부정 근거: {lges_count}건, CATL 부정 근거: {catl_count}건")

    return content, unique_sources, quantitative_check, fallback_items


def critic_node(state: WorkflowState) -> dict:
    """T4: Critic Agent 노드."""
    print(f"[T4] Critic Agent 실행 (retry: {state['retry_count'].get('T4', 0)})")

    lges_content = state.get("lges_strategy", {}).get("content", "")
    catl_content = state.get("catl_strategy", {}).get("content", "")

    content, sources, quantitative_check, fallback_items = _run_critic(
        lges_content, catl_content
    )

    output: AgentOutput = {
        "task_id": "T4",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T4] 완료 — quantitative_check: {quantitative_check}")

    return {
        "critic_result": output,
        "current_task": "T4",
        "retry_count": {"T4": state["retry_count"].get("T4", 0) + 1},
        "total_llm_calls": 1,
    }
