#!/bin/bash
set -e

echo "[S2] LLM Risk + CPPO 실행 시작..."
cd ../FinRL_DeepSeek-main
python train_cppo_llm_risk.py --config_path ../configs/tsla_risk_strategy.json > ../logs/s2_log.txt 2>&1
cd - > /dev/null
echo "[S2] 완료! 결과는 logs/s2_log.txt 및 results/ 디렉토리 확인" 