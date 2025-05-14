#!/bin/bash
set -e

echo "[S1] LLM + PPO 전략 실행 시작..."
cd ../FinRL_DeepSeek-main
python train_ppo_llm.py --config_path ../configs/aapl_strategy.json > ../logs/s1_log.txt 2>&1
cd - > /dev/null
echo "[S1] 완료! 결과는 logs/s1_log.txt 및 results/ 디렉토리 확인" 