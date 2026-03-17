"""
T5: Comparison & SWOT Agent

비교 및 SWOT 구조화 — 양사 전략 차이 비교, S/W/O/T 4분면 작성.
이전 출력(T2, T3, T4)을 통합. 신규 검색 없음.
"""
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from prompts.prompts import SWOT_SYSTEM, SWOT_HUMAN

MODEL_NAME = "gpt-4o-mini"


def _check_swot_completeness(content: str) -> tuple[bool, list[str]]:
    """
    SWOT 출력의 계량 조건 체크:
    - S/W/O/T 각 항목 ≥ 2개
    - 내부(S/W)·외부(O/T) 섹션 모두 존재
    """
    fallback_items = []

    # SWOT 4분면 키워드 존재 여부
    required_sections = ["강점", "약점", "기회", "위협"]
    missing_sections = [s for s in required_sections if s not in content]
    if missing_sections:
        fallback_items.append(f"SWOT 섹션 누락: {missing_sections}")

    # 각 섹션별 항목 수 체크 (최소 2개)
    # 번호 목록 또는 불릿 형식 카운트
    all_items = re.findall(r"^\s*[\d]+[\.\)]\s+.+|^\s*[-•*]\s+.+", content, re.MULTILINE)
    # SWOT 테이블 형식인 경우 | 구분자로 카운트
    table_cells = re.findall(r"\|([^|]+)\|", content)
    non_empty_cells = [c.strip() for c in table_cells if len(c.strip()) > 10]

    total_items = len(all_items) + len(non_empty_cells)
    if total_items < 8:  # S/W/O/T × 각 2항목 = 최소 8개
        fallback_items.append(f"SWOT 항목 부족 (예상 ≥ 8, 확인: {total_items}개)")

    quantitative_check = len(fallback_items) == 0
    return quantitative_check, fallback_items


def swot_node(state: WorkflowState) -> dict:
    """T5: Comparison & SWOT Agent 노드."""
    print(f"[T5] SWOT Agent 실행 (retry: {state['retry_count'].get('T5', 0)})")

    lges_content = state.get("lges_strategy", {}).get("content", "")
    catl_content = state.get("catl_strategy", {}).get("content", "")
    critic_content = state.get("critic_result", {}).get("content", "")

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=SWOT_SYSTEM),
        HumanMessage(content=SWOT_HUMAN.format(
            lges_content=lges_content,
            catl_content=catl_content,
            critic_content=critic_content,
        )),
    ]
    response = llm.invoke(messages)
    content = response.content

    # 계량 조건 체크
    quantitative_check, fallback_items = _check_swot_completeness(content)

    # 출처는 이전 Agent 출처 수집 (신규 검색 없음)
    sources = []
    for key in ["lges_strategy", "catl_strategy", "critic_result"]:
        prev = state.get(key, {})
        if isinstance(prev, dict):
            sources.extend(prev.get("sources", []))
    sources = list(dict.fromkeys(sources))

    output: AgentOutput = {
        "task_id": "T5",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T5] 완료 — quantitative_check: {quantitative_check}")

    return {
        "swot_comparison": output,
        "current_task": "T5",
        "retry_count": {"T5": state["retry_count"].get("T5", 0) + 1},
        "total_llm_calls": 1,
    }
