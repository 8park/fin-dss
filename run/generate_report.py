import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 평가 함수 예시 (실제 논문용 기준에 맞게 확장 가능)
def evaluate_ttf(strategy_summary):
    if "보수" in strategy_summary:
        return "높음"
    elif "공격" in strategy_summary:
        return "중간"
    else:
        return "보통"

def evaluate_is_success(info_quality, usability):
    if info_quality > 0.8 and usability > 0.8:
        return "매우 우수"
    elif info_quality > 0.6:
        return "우수"
    else:
        return "보통"

def evaluate_trust(risk_score):
    if risk_score < 0.3:
        return "매우 우수"
    elif risk_score < 0.5:
        return "우수"
    else:
        return "보통"

def plot_return_curve(csv_path, label, out_path):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.plot(df['date'], df['cumulative_return'], label=label)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title(f'Cumulative Return Curve: {label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_sharpe_vs_mdd(csv1, csv2, out_path):
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    plt.figure(figsize=(6,6))
    plt.scatter(df1['sharpe_ratio'].iloc[-1], df1['max_drawdown'].min(), label='S1', color='blue')
    plt.scatter(df2['sharpe_ratio'].iloc[-1], df2['max_drawdown'].min(), label='S2', color='red')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Max Drawdown')
    plt.title('Sharpe Ratio vs Max Drawdown')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def summarize_results():
    s1_path = "../results/s1_ppo_results.csv"
    s2_path = "../results/s2_risk_results.csv"
    s1 = pd.read_csv(s1_path)
    s2 = pd.read_csv(s2_path)

    # 전략 요약(예시)
    strategy_s1 = "보수적 투자 전략 (채권 60%, 주식 30%, 현금 10%)"
    strategy_s2 = "리스크 기반 중립 전략 (주식 50%, 채권 30%, 현금 20%)"
    # 실제 전략 요약은 configs/*.json에서 불러올 수 있음

    # 정량 지표
    summary = {
        "S1 - Return": s1["cumulative_return"].iloc[-1],
        "S2 - Return": s2["cumulative_return"].iloc[-1],
        "S1 - Sharpe": s1["sharpe_ratio"].iloc[-1],
        "S2 - Sharpe": s2["sharpe_ratio"].iloc[-1],
        "S1 - MDD": s1["max_drawdown"].min(),
        "S2 - MDD": s2["max_drawdown"].min()
    }

    # 정성 평가 (예시)
    ttf = evaluate_ttf(strategy_s1)
    is_success = evaluate_is_success(0.9, 0.85)  # 예시값
    trust = evaluate_trust(0.35)  # 예시값

    # 그래프 생성
    os.makedirs("../results/plots", exist_ok=True)
    plot_return_curve(s1_path, "S1: LLM+PPO", "../results/plots/s1_return_curve.png")
    plot_return_curve(s2_path, "S2: LLM Risk+CPPO", "../results/plots/s2_return_curve.png")
    plot_sharpe_vs_mdd(s1_path, s2_path, "../results/plots/sharpe_vs_mdd.png")

    # Markdown 보고서 생성
    with open("../reports/experiment_summary.md", "w") as f:
        f.write(f"# IS 논문용 실험 결과 ({datetime.now().strftime('%Y.%m')})\n\n")
        f.write("## TTF 기반 전략 분류\n")
        f.write(f"- 전략: {strategy_s1}  \n- 업무 적합도(TTF): {ttf}\n\n")
        f.write("## IS Success Model 평가\n")
        f.write(f"- 정보 품질: {is_success}  \n- 시스템 사용성: 높음\n\n")
        f.write("## Algorithmic Trust 분석\n")
        f.write(f"- 시스템 신뢰성: {trust}\n\n")
        f.write("## 정량적 평가 요약\n")
        f.write("|지표| 값 |\n|---|---|\n")
        f.write(f"|누적 수익률(CR)|{summary['S1 - Return']*100:.2f}%|\n")
        f.write(f"|Sharpe Ratio(SR)|{summary['S1 - Sharpe']:.2f}|\n")
        f.write(f"|최대낙폭(MDD)|{summary['S1 - MDD']*100:.2f}%|\n\n")
        f.write("## 그래프\n")
        f.write("- ![S1 Return](../results/plots/s1_return_curve.png)\n")
        f.write("- ![S2 Return](../results/plots/s2_return_curve.png)\n")
        f.write("- ![Sharpe vs MDD](../results/plots/sharpe_vs_mdd.png)\n")
        f.write("\n---\n")
        f.write("*본 보고서는 자동 생성되었습니다.*\n")

if __name__ == "__main__":
    summarize_results() 