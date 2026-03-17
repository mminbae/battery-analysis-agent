"""
T6: Report Writing Agent

보고서 생성 — 7개 섹션 완성, 출처 병기, SUMMARY 작성.
T1~T5 전체 결과를 통합. 신규 검색 없음.
"""
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import WorkflowState, AgentOutput
from prompts.prompts import REPORT_SYSTEM, REPORT_HUMAN

MODEL_NAME = "gpt-4o-mini"

# 필수 7개 섹션
REQUIRED_SECTIONS = [
    "SUMMARY",
    "시장 배경",
    "LG에너지솔루션",
    "CATL",
    "SWOT",
    "시사점",
    "REFERENCE",
]


def _check_report_completeness(content: str) -> tuple[bool, list[str]]:
    """
    보고서 완결성 체크:
    - 7개 섹션 모두 존재
    - 출처 병기 (REFERENCE 섹션 존재)
    """
    fallback_items = []

    missing_sections = []
    for section in REQUIRED_SECTIONS:
        if section not in content:
            missing_sections.append(section)

    if missing_sections:
        fallback_items.append(f"누락 섹션: {missing_sections}")

    # REFERENCE 섹션에 URL 또는 출처 있는지 확인
    has_reference = "REFERENCE" in content or "참고" in content
    has_urls = bool(re.search(r"https?://", content))
    if not (has_reference and has_urls):
        fallback_items.append("출처(URL) 병기 부족")

    quantitative_check = len(fallback_items) == 0
    return quantitative_check, fallback_items


def report_node(state: WorkflowState) -> dict:
    """T6: Report Writing Agent 노드."""
    print(f"[T6] Report Writing Agent 실행 (retry: {state['retry_count'].get('T6', 0)})")

    market_content = state.get("market_research", {}).get("content", "")
    lges_content = state.get("lges_strategy", {}).get("content", "")
    catl_content = state.get("catl_strategy", {}).get("content", "")
    critic_content = state.get("critic_result", {}).get("content", "")
    swot_content = state.get("swot_comparison", {}).get("content", "")

    # Fallback 항목 수집 (이전 Task에서 누적된 근거 불충분 항목)
    accumulated_fallbacks = state.get("fallback_items", [])

    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    messages = [
        SystemMessage(content=REPORT_SYSTEM),
        HumanMessage(content=f"""{REPORT_HUMAN.format(
            market_content=market_content,
            lges_content=lges_content,
            catl_content=catl_content,
            critic_content=critic_content,
            swot_content=swot_content,
        )}

{f"## 주의: 다음 항목은 근거 불충분으로 처리되었습니다{chr(10)}{chr(10).join(f'- {item}' for item in accumulated_fallbacks)}" if accumulated_fallbacks else ""}
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # 계량 조건 체크
    quantitative_check, fallback_items = _check_report_completeness(content)

    # 전체 출처 수집
    sources = []
    for key in ["market_research", "lges_strategy", "catl_strategy", "critic_result", "swot_comparison"]:
        prev = state.get(key, {})
        if isinstance(prev, dict):
            sources.extend(prev.get("sources", []))
    sources = list(dict.fromkeys(sources))

    output: AgentOutput = {
        "task_id": "T6",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T6] 완료 — quantitative_check: {quantitative_check}")

    return {
        "final_report": output,
        "current_task": "T6",
        "retry_count": {"T6": state["retry_count"].get("T6", 0) + 1},
        "total_llm_calls": 1,
    }
