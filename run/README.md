# FinDSS 실험 자동화 매뉴얼

## 1. 디렉토리 구조

```
fin_dss_experiment/
├── data/            # 원시/정제 데이터 저장
├── configs/         # 전략 입력 JSON
├── logs/            # 실행 로그
├── results/         # 실험 결과(CSV, 그래프)
├── reports/         # 논문용 자동 리포트
├── run/             # 실행 스크립트
├── FinRL_DeepSeek-main/  # RL 모듈(참조)
├── FinRobot-master/      # LLM 모듈(참조)
```

## 2. 실험 실행 방법

1. 환경 준비
   ```bash
   pip install -r requirements.txt
   ```
2. 전략 JSON 준비: `configs/`에 aapl_strategy.json, tsla_risk_strategy.json 등 생성
3. 실험 실행
   - S1: LLM+PPO
     ```bash
     bash run/run_s1.sh
     ```
   - S2: LLM Risk+CPPO
     ```bash
     bash run/run_s2.sh
     ```
   - 전체 일괄 실행
     ```bash
     bash run/run_all.sh
     ```
4. 결과 요약/리포트 생성
   ```bash
   python run/generate_report.py
   ```

## 3. 결과물 위치
- 로그: `logs/`
- 실험 결과(CSV): `results/`
- 그래프: `results/plots/`
- 논문용 리포트: `reports/experiment_summary.md`

## 4. 논문용 자동 리포트 예시
- TTF, IS Success Model, Algorithmic Trust 평가 자동 포함
- 정량/정성 지표, 표, 그래프 자동 생성

## 5. 참고
- FinRL_DeepSeek-main, FinRobot-master는 절대 복사/이동하지 않고 직접 참조
- 모든 실험 자동화 및 재현 가능 