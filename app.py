"""
배터리 시장 전략 분석 Agent — 실행 엔트리포인트.

사용법:
    python app.py
    python app.py --query "LGES와 CATL의 포트폴리오 전략을 비교해줘"
    python app.py --data-dir /path/to/pdfs

환경변수 (.env):
    OPENAI_API_KEY=...
    TAVILY_API_KEY=...
"""
import argparse
import sys
import os
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트를 sys.path에 추가 (battery-agent/ 디렉토리에서 실행 시)
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

PDF_STYLESHEET = Path(__file__).parent / "styles" / "markdown-preview.css"


DEFAULT_QUERY = (
    "전기차 캐즘 심화 환경에서 LG에너지솔루션과 CATL의 포트폴리오 다각화 전략을 비교 분석하고, "
    "투자자·전략기획 담당자가 기업 간 우위 요소를 판단할 수 있는 인사이트를 제공해주세요."
)


def check_env():
    """필수 환경변수 체크."""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        print(f"[오류] 다음 환경변수가 설정되지 않았습니다: {missing}")
        print("  .env 파일을 생성하고 API 키를 설정하세요.")
        sys.exit(1)


def export_pdf_from_markdown(markdown_path: str) -> str | None:
    """md-to-pdf로 Markdown을 PDF로 변환."""
    if shutil.which("npx") is None:
        print("[경고] npx를 찾을 수 없어 PDF 변환을 건너뜁니다.")
        return None

    markdown_file = Path(markdown_path)
    pdf_path = markdown_file.with_suffix(".pdf")

    command = [
        "npx",
        "-y",
        "md-to-pdf",
        str(markdown_file),
        "--body-class",
        "markdown-body",
        "--stylesheet",
        str(PDF_STYLESHEET),
        "--highlight-style",
        "github",
        "--pdf-options",
        json.dumps(
            {
                "format": "A4",
                "margin": {
                    "top": "16mm",
                    "right": "14mm",
                    "bottom": "18mm",
                    "left": "14mm",
                },
                "printBackground": True,
            }
        ),
    ]

    try:
        subprocess.run(command, check=True, cwd=Path(__file__).parent)
    except subprocess.CalledProcessError as exc:
        print(f"[경고] PDF 변환 실패: {exc}")
        return None

    if pdf_path.exists():
        print(f"[완료] PDF 저장됨: {pdf_path}")
        return str(pdf_path)

    print("[경고] PDF 파일이 생성되지 않았습니다.")
    return None


def save_report(content: str, sources: list[str], termination_reason: str, total_calls: int):
    """outputs/ 디렉토리에 보고서 저장."""
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"battery_analysis_{timestamp}.md"

    separator = "━" * 60
    report_text = f"""# 글로벌 배터리 시장 캐즘 대응 전략 비교 분석
## LG에너지솔루션 vs CATL 포트폴리오 다각화 전략을 중심으로

> 생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 종료 사유: {termination_reason}
> 총 LLM 호출 수: {total_calls}회

{separator}

{content}

{separator}

## 수집된 출처 목록

{chr(10).join(f"- {s}" for s in sources) if sources else "출처 없음"}
"""

    filename.write_text(report_text, encoding="utf-8")
    print(f"\n[완료] 보고서 저장됨: {filename}")
    export_pdf_from_markdown(str(filename))
    return str(filename)


def run(query: str, data_dir: str = "data/"):
    """메인 실행 함수."""
    print("=" * 60)
    print("배터리 시장 전략 분석 Multi-Agent 시스템")
    print("=" * 60)
    print(f"[분석 질의] {query}\n")

    # 1. RAG retriever 초기화
    print("[초기화] RAG retriever 초기화 중...")
    from rag.retriever import init_retriever
    init_retriever(data_dir=data_dir)

    # 2. 그래프 로드
    print("[초기화] LangGraph 그래프 로드 중...")
    from graph.graph import graph, get_initial_state

    # 3. 초기 상태 설정
    initial_state = get_initial_state(query)

    # 4. 그래프 실행
    print("\n[실행] 분석 시작...\n")
    print("-" * 60)

    final_state = None
    try:
        for event in graph.stream(initial_state, stream_mode="values"):
            final_state = event
    except Exception as e:
        print(f"\n[오류] 그래프 실행 중 오류 발생: {e}")
        raise

    if final_state is None:
        print("[오류] 최종 상태가 없습니다.")
        sys.exit(1)

    # 5. 결과 출력
    print("\n" + "=" * 60)
    termination_reason = final_state.get("termination_reason", "unknown")
    total_calls = final_state.get("total_llm_calls", 0)
    is_completed = final_state.get("is_completed", False)

    print(f"[종료] 사유: {termination_reason} | LLM 호출: {total_calls}회 | 완료: {is_completed}")

    final_report = final_state.get("final_report", {})
    report_content = final_report.get("content", "보고서 생성 실패")
    report_sources = final_report.get("sources", [])
    fallback_items = final_state.get("fallback_items", []) + final_report.get("fallback_items", [])

    if fallback_items:
        print(f"\n[주의] 근거 불충분 항목:")
        for item in fallback_items:
            print(f"  - {item}")

    # 6. 보고서 저장
    output_path = save_report(report_content, report_sources, termination_reason, total_calls)

    # 7. 콘솔 미리보기 (첫 500자)
    print("\n[보고서 미리보기]")
    print("-" * 60)
    preview = report_content[:500] + ("..." if len(report_content) > 500 else "")
    print(preview)
    print("-" * 60)
    print(f"\n전체 보고서: {output_path}")

    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="배터리 시장 전략 분석 Multi-Agent 시스템"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="분석 질의 (기본값: LGES vs CATL 포트폴리오 전략 비교)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="RAG용 PDF 파일 디렉토리 (기본값: data/)",
    )
    args = parser.parse_args()

    check_env()
    run(query=args.query, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
