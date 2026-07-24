"""Part A · Step 3(서빙) — build-vs-buy 손익분기 분석.

품질로 안 싸운다는 논지의 정직한 실탄: **언제 자체 서빙(self-host)이 프론티어
배치 API보다 싼지**를 실측 $로 계산한다. "작은 모델이 무조건 좋다"가 아니라
"우리 스케일에선 손익분기가 여기고, 그 아래선 API가 이긴다"를 숫자로 보여주는 게
목적(steam-part-a-value-reframe).

- **self-host(측정치)**: 학습 1회 $2.10(T4 11 GPU-hr, finetune_lora) + 서빙 건당
  $3.26/24만(AWQ, bench_vllm 실측). 학습은 고정비, 서빙은 건당 변동비.
- **프론티어 배치 API**: 건당 (입력토큰×입력가 + 출력토큰×출력가). 고정비 0, 대신
  건당 원가가 self-host보다 훨씬 큼. Batch API = 표준가의 50%.

토큰 수(입력~475·출력~166/게임)는 vLLM 벤치에서 실측한 **Qwen 토크나이저** 기준
근사치다. 프론티어(Claude) 토크나이저는 조금 다르므로 정밀값은 count_tokens로
재보정할 것 — 하지만 크로스오버는 배수 차이가 커서 이 근사로도 결론이 안 흔들린다.

usage:
    uv run --with matplotlib serve/cost_crossover.py
    # 차트 없이 표만: uv run serve/cost_crossover.py
"""

from __future__ import annotations

from pathlib import Path

# ── self-host 측정치 (serve/bench_awq.json, finetune_lora GPU-시간 로깅) ──
TRAIN_ONCE_USD = 2.10          # LoRA 학습 1회(고정비). T4 11 GPU-hr @ $0.19/hr.
SERVE_PER_GAME_USD = 3.26 / 240_000   # AWQ 서빙 건당(변동비) = $1.358e-5

# ── 태스크 토큰 수/게임 (vLLM 벤치 실측, Qwen 토크나이저) ──
IN_TOK = 475    # detailed_description(중앙 929자) + 프롬프트 템플릿
OUT_TOK = 166   # 3필드 한국어 요약 JSON

CATALOG = 240_000   # 스팀 전체 게임 수(Zenodo)

# ── 프론티어 "3대장" 싼 대량처리 티어, Batch API(표준가 50%). ($/1M 토큰) ──
# 가격 확인 2026-07-10 (WebSearch). ⚠️변동 잦음(예: Gemini Flash 2026-07-02 인상)
# → 파라미터이니 pricing 페이지에서 재확인·교체 가능.
# 최저가 모델(Flash-Lite/Nano)은 그만큼 약함 → 품질 맞는 비교는 Flash/Haiku 티어.
FRONTIER = {
    "Gemini 2.5 Flash-Lite (Batch)": (0.05, 0.20),   # 전체 최저(그러나 최약)
    "GPT-5.4 Nano (Batch)":          (0.10, 0.625),  # OpenAI 최저
    "Gemini 2.5 Flash (Batch)":      (0.15, 1.25),   # 품질 맞는 중간
    "Claude Haiku 4.5 (Batch)":      (0.50, 2.50),   # 품질 맞는 중간
}


def per_game_usd(price_in: float, price_out: float) -> float:
    """프론티어 건당 원가 = 입력토큰×입력가 + 출력토큰×출력가."""
    return IN_TOK * price_in / 1e6 + OUT_TOK * price_out / 1e6


def crossover_games(api_per_game: float) -> float:
    """self-host 누적($2.10 + 서빙·N)이 API 누적(건당·N)보다 싸지는 N.

    2.10 + SERVE·N = API·N  →  N = 2.10 / (API − SERVE)
    """
    denom = api_per_game - SERVE_PER_GAME_USD
    return TRAIN_ONCE_USD / denom if denom > 0 else float("inf")


def selfhost_cumulative(n: int) -> float:
    return TRAIN_ONCE_USD + SERVE_PER_GAME_USD * n


def main() -> None:
    print(f"[가정] 입력 {IN_TOK}tok · 출력 {OUT_TOK}tok/게임 (vLLM 실측, Qwen 토크나이저)")
    print(f"[self-host] 학습 1회 ${TRAIN_ONCE_USD:.2f} + 서빙 ${SERVE_PER_GAME_USD*1e6:.2f}/1M게임"
          f" (= ${SERVE_PER_GAME_USD:.2e}/게임, AWQ 실측)")
    sh_240k = selfhost_cumulative(CATALOG)
    print(f"           → 24만 전량: ${sh_240k:.2f}\n")

    print(f"{'프론티어 옵션':<22} {'건당$':>10} {'24만$':>10} {'손익분기(게임)':>14} {'24만서 배수':>10}")
    print("-" * 70)
    rows = []
    for name, (pin, pout) in FRONTIER.items():
        pg = per_game_usd(pin, pout)
        total = pg * CATALOG
        xo = crossover_games(pg)
        ratio = total / sh_240k
        rows.append((name, pin, pout, pg, total, xo))
        print(f"{name:<22} ${pg:>8.6f} ${total:>8.0f} {xo:>13,.0f} {ratio:>9.0f}x")

    cheapest = min(rows, key=lambda r: r[3])   # 전체 최저(약함)
    fair = next(r for r in rows if r[0].startswith("Claude Haiku"))  # 품질 맞는 티어
    print("\n[결론] 상대에 따라 그림이 크게 다름 — 정직하게:")
    print(f"  · 최저가({cheapest[0]}): 손익분기 **{cheapest[5]:,.0f}게임**, 24만서 "
          f"${sh_240k:.2f} vs ${cheapest[4]:.0f} = **{cheapest[4]/sh_240k:.1f}배**. "
          f"근데 이 티어는 그만큼 약함(품질 동급 아님).")
    print(f"  · 품질맞는 티어({fair[0]}): 손익분기 **{fair[5]:,.0f}게임**, 24만서 "
          f"**{fair[4]/sh_240k:.0f}배**.")
    print("  ⚠️ 핵심 정직: **최저가 프론티어(Flash-Lite급)와는 24만서도 2~3배로 좁혀짐** — "
          "'무조건 self-host'는 과장. self-host가 확실히 이기는 건 (a)품질 맞는 티어 비교거나 "
          "(b)카탈로그를 넘어 재처리·다운스트림까지 누적할 때. 소량·애드혹은 API가 이김.")

    _plot(rows, sh_240k)


def _plot(rows, sh_240k: float) -> None:
    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception:
        print("\n[chart] matplotlib 없음 → 표만. 차트는 `uv run --with matplotlib`로.")
        return

    import numpy as np  # noqa: PLC0415
    ns = np.logspace(1, np.log10(CATALOG * 2), 200)

    # 차트 라벨은 영어(WSL 기본폰트에 한글 글리프 없음 → 네모박스 방지)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, TRAIN_ONCE_USD + SERVE_PER_GAME_USD * ns,
            label="Self-host (train $2.10 + AWQ serving)", lw=2.4, color="#1b2838")
    labels_en = {
        "Haiku 4.5 (Batch)": "Claude Haiku 4.5 (Batch API)",
        "Haiku 4.5 (표준)": "Claude Haiku 4.5 (standard)",
        "Sonnet 4.6 (Batch)": "Claude Sonnet 4.6 (Batch API)",
    }
    for name, pin, pout, pg, _total, xo in rows:
        ax.plot(ns, pg * ns, "--", lw=1.6, label=labels_en.get(name, name))
        if xo < ns[-1]:
            ax.axvline(xo, color="gray", ls=":", lw=0.8)

    ax.axvline(CATALOG, color="#66c0f4", lw=1.2, alpha=0.7)
    ax.text(CATALOG, ax.get_ylim()[0], " catalog 240k", rotation=90,
            va="bottom", ha="right", fontsize=8, color="#2a6f97")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("games processed (cumulative)")
    ax.set_ylabel("cumulative cost (USD)")
    ax.set_title("Build vs Buy — self-hosted small model vs frontier Batch API")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls=":", alpha=0.3)

    out = Path(__file__).resolve().parent / "cost_crossover.png"
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"\n[chart] → {out}")


if __name__ == "__main__":
    main()
