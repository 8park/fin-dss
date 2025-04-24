#!/usr/bin/env python3
"""
pipeline.py: Hierarchical AI DSS pipeline integrating FinRobot, FinEmotion, and FinRL-LLM.
This script orchestrates:
  1. FinRobot strategy generation via existing workflow script
  2. Optional sentiment-based adjustment using FinEmotion
  3. RL training/backtest execution via FinRL-LLM's train_ppo_llm.py

Usage:
  python pipeline.py --role conservative --market-state path/to/state.json \
                    [--use-emotion --sentiment-data path/to/text --alpha 0.1] \
                    --output-dir path/to/output \
                    --n-procs 8
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from finemotion.emotion import get_sentiment  # FinEmotion

# Paths to module entry scripts
FINROBOT_SCRIPT = os.path.join('FinRobot', 'experiments', 'multi_factor_agents.py')
FINRL_SCRIPT     = os.path.join('FinRL_llm', 'train_ppo_llm.py')


def generate_strategy(role: str, state_path: str, out_path: str) -> None:
    """
    Call FinRobot workflow to generate strategy JSON at out_path.
    """
    cmd = [
        sys.executable,
        FINROBOT_SCRIPT,
        '--role', role,
        '--state-file', state_path,
        '--output', out_path
    ]
    logging.info(f"Generating strategy: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def adjust_strategy(path: str, alpha: float, text_path: str) -> None:
    """
    Load strategy JSON, adjust risk_pref by sentiment score, and overwrite JSON.
    """
    with open(path, 'r') as f:
        strategy = json.load(f)

    with open(text_path, 'r') as f:
        text = f.read()
    sentiment_label = get_sentiment(text)
    mapping = {'positive': 1.0, 'neutral': 0.0, 'negative': -1.0}
    score = mapping.get(sentiment_label, 0.0)
    original = strategy.get('risk_pref', 0.0)
    adjusted = original + alpha * score
    strategy['risk_pref'] = adjusted
    logging.info(f"Sentiment: {sentiment_label} ({score}), risk_pref: {original} -> {adjusted}")

    with open(path, 'w') as f:
        json.dump(strategy, f, indent=4)


def run_rl(strategy_path: str, output_dir: str, n_procs: int) -> None:
    """
    Execute RL training/backtest via mpirun on FinRL-LLM script.
    """
    env = os.environ.copy()
    env['STRATEGY_PATH'] = os.path.abspath(strategy_path)
    env['OUTPUT_DIR']    = os.path.abspath(output_dir)

    cmd = [
        'mpirun', '-np', str(n_procs),
        sys.executable, FINRL_SCRIPT
    ]
    logging.info(f"Running RL: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical AI DSS pipeline")
    parser.add_argument('--role', required=True, help='Investor role for strategy')
    parser.add_argument('--market-state', required=True,
                        help='Path to market state JSON')
    parser.add_argument('--use-emotion', action='store_true',
                        help='Enable sentiment-based strategy adjustment')
    parser.add_argument('--sentiment-data', help='Path to text input for sentiment')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Weight for sentiment adjustment')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for strategy.json and results')
    parser.add_argument('--n-procs', type=int, default=8,
                        help='Number of processes for mpirun')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    os.makedirs(args.output_dir, exist_ok=True)
    strategy_path = os.path.join(args.output_dir, 'strategy.json')

    # 1. Generate strategy
    generate_strategy(args.role, args.market_state, strategy_path)

    # 2. Sentiment adjustment
    if args.use_emotion:
        if not args.sentiment_data:
            parser.error('--sentiment-data is required with --use-emotion')
        adjust_strategy(strategy_path, args.alpha, args.sentiment_data)

    # 3. Run RL training/backtest
    run_rl(strategy_path, args.output_dir, args.n_procs)


if __name__ == '__main__':
    main()

