"""
T6: Report Writing Agent

보고서 생성 — 7개 섹션 완성, 출처 병기, SUMMARY 작성.
T1~T5 전체 결과를 통합. 신규 검색 없음.

[P2 수정]
- _check_report_completeness: SECTION 6.3 최소 길이, SUMMARY Bullet 수 검증 추가
- _format_reference_section: 수집된 출처를 설계서 양식(기관보고서/웹페이지)으로 정리
- report_node: 정리된 REFERENCE 목록을 Human 메시지에 주입
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

# 기관 보고서로 분류할 키워드 (URL이 아닌 파일명 포함)
INSTITUTIONAL_PATTERNS = [
    r"\.pdf$",
    r"IEA",
    r"McKinsey",
    r"Bloomberg",
    r"Reuters",
    r"Wood Mackenzie",
    r"SK증권",
    r"유진투자",
]


def _format_web_reference(src: str) -> str:
    if src.startswith("[웹] "):
        payload = src[len("[웹] "):]
        parts = payload.split(" | ", 3)
        if len(parts) == 4:
            site, published_at, title, url = parts
            if published_at:
                return f"{site}({published_at}). {title}. {url}"
            return f"{site}. {title}. {url}"
    return src


def _format_reference_section(sources: list[str]) -> str:
    """
    [P2] REFERENCE 초안을 설계서 양식에 맞게 분류·정리.
    - 기관 보고서: 발행기관(YYYY). 보고서명. URL
    - 웹페이지: 기관명(YYYY-MM-DD). 제목. 사이트명, URL
    """
    if not sources:
        return "출처 없음"

    institutional = []
    webpages = []

    for src in sources:
        if src.startswith("[웹] "):
            webpages.append(src)
            continue
        is_institutional = any(re.search(p, src, re.IGNORECASE) for p in INSTITUTIONAL_PATTERNS)
        if is_institutional:
            institutional.append(src)
        else:
            webpages.append(src)

    lines = []
    if institutional:
        lines.append("**기관 보고서**")
        for src in institutional:
            lines.append(f"- {src}")
    if webpages:
        lines.append("\n**웹페이지**")
        for src in webpages:
            lines.append(f"- {_format_web_reference(src)}")

    return "\n".join(lines)


def _check_report_completeness(content: str) -> tuple[bool, list[str]]:
    """
    [P2] 보고서 완결성 체크:
    - 7개 섹션 모두 존재
    - 출처 병기 (REFERENCE 섹션 + URL)
    - SECTION 6.3 최소 길이 (300자)
    - SUMMARY Bullet 3~5개
    """
    fallback_items = []

    # 7개 섹션 존재 여부
    missing_sections = [s for s in REQUIRED_SECTIONS if s not in content]
    if missing_sections:
        fallback_items.append(f"누락 섹션: {missing_sections}")

    # REFERENCE 섹션에 URL 또는 출처 있는지 확인
    has_reference = "REFERENCE" in content or "참고" in content
    has_urls = bool(re.search(r"https?://", content))
    if not (has_reference and has_urls):
        fallback_items.append("출처(URL) 병기 부족")

    # [P2] SECTION 6.3 투자자 시사점 최소 300자 체크
    sec63_match = re.search(r"6\.3.{0,600}", content, re.DOTALL)
    if sec63_match:
        sec63_content = sec63_match.group()
        if len(sec63_content) < 300:
            fallback_items.append(
                f"SECTION 6.3 투자자 시사점 너무 짧음 ({len(sec63_content)}자 / 최소 300자)"
            )
    else:
        fallback_items.append("SECTION 6.3 투자자 시사점 누락")

    # [P2] SUMMARY Bullet 수 체크 (3~5개)
    summary_match = re.search(r"SUMMARY.{0,1000}", content, re.DOTALL)
    if summary_match:
        summary_content = summary_match.group()
        bullets = re.findall(r"^\s*[-•*]\s+.+|^\s*\d+[\.\)]\s+.+", summary_content, re.MULTILINE)
        if len(bullets) < 3:
            fallback_items.append(f"SUMMARY Bullet 부족 ({len(bullets)}개 / 최소 3개)")
        elif len(bullets) > 5:
            fallback_items.append(f"SUMMARY Bullet 초과 ({len(bullets)}개 / 최대 5개)")

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

    # [P2] 전체 출처 수집 후 REFERENCE 형식으로 정리
    sources = []
    for key in ["market_research", "lges_strategy", "catl_strategy", "critic_result", "swot_comparison"]:
        prev = state.get(key, {})
        if isinstance(prev, dict):
            sources.extend(prev.get("sources", []))
    sources = list(dict.fromkeys(sources))
    formatted_references = _format_reference_section(sources)

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

## [P2] REFERENCE 초안 (SECTION 7 REFERENCE에 이 형식으로 정리)
{formatted_references}

{f"## 주의: 다음 항목은 근거 불충분으로 처리되었습니다{chr(10)}{chr(10).join(f'- {item}' for item in accumulated_fallbacks)}" if accumulated_fallbacks else ""}
"""),
    ]
    response = llm.invoke(messages)
    content = response.content

    # 계량 조건 체크
    quantitative_check, fallback_items = _check_report_completeness(content)

    output: AgentOutput = {
        "task_id": "T6",
        "content": content,
        "sources": sources,
        "quantitative_check": quantitative_check,
        "fallback_items": fallback_items,
    }

    print(f"[T6] 완료 — quantitative_check: {quantitative_check}, 내용: {len(content)}자")

    return {
        "final_report": output,
        "current_task": "T6",
        "retry_count": {"T6": state["retry_count"].get("T6", 0) + 1},
        "total_llm_calls": 1,
    }
