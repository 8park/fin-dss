import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def create_directory_structure():
    """Create necessary directories for the dataset."""
    directories = ['configs', 'results', 'reports']
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)

def generate_strategy_jsons():
    """Generate strategy JSON files for different risk levels."""
    strategies = {
        'conservative': [
            {
                'strategy_id': 'strategy_conservative_1',
                'company': 'AAPL',
                'strategy_summary': 'Conservative strategy (60% bonds, 30% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.32, 'eps': 5.4},
                'sentiment_score': 0.15,
                'risk_score': 0.20
            },
            {
                'strategy_id': 'strategy_conservative_2',
                'company': 'MSFT',
                'strategy_summary': 'Conservative strategy (55% bonds, 35% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.35, 'eps': 6.2},
                'sentiment_score': 0.18,
                'risk_score': 0.22
            },
            {
                'strategy_id': 'strategy_conservative_3',
                'company': 'GOOGL',
                'strategy_summary': 'Conservative strategy (50% bonds, 40% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.28, 'eps': 4.8},
                'sentiment_score': 0.12,
                'risk_score': 0.25
            }
        ],
        'neutral': [
            {
                'strategy_id': 'strategy_neutral_1',
                'company': 'AMZN',
                'strategy_summary': 'Neutral strategy (40% bonds, 50% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.25, 'eps': 3.2},
                'sentiment_score': 0.35,
                'risk_score': 0.45
            },
            {
                'strategy_id': 'strategy_neutral_2',
                'company': 'META',
                'strategy_summary': 'Neutral strategy (35% bonds, 55% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.30, 'eps': 4.5},
                'sentiment_score': 0.42,
                'risk_score': 0.48
            },
            {
                'strategy_id': 'strategy_neutral_3',
                'company': 'NVDA',
                'strategy_summary': 'Neutral strategy (30% bonds, 60% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.45, 'eps': 8.2},
                'sentiment_score': 0.38,
                'risk_score': 0.52
            }
        ],
        'aggressive': [
            {
                'strategy_id': 'strategy_aggressive_1',
                'company': 'TSLA',
                'strategy_summary': 'Aggressive strategy (20% bonds, 70% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.22, 'eps': 2.8},
                'sentiment_score': 0.65,
                'risk_score': 0.75
            },
            {
                'strategy_id': 'strategy_aggressive_2',
                'company': 'AMD',
                'strategy_summary': 'Aggressive strategy (15% bonds, 75% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.35, 'eps': 3.5},
                'sentiment_score': 0.58,
                'risk_score': 0.68
            },
            {
                'strategy_id': 'strategy_aggressive_3',
                'company': 'COIN',
                'strategy_summary': 'Aggressive strategy (10% bonds, 80% stocks, 10% cash)',
                'financial_ratios': {'roe': 0.18, 'eps': 1.2},
                'sentiment_score': 0.72,
                'risk_score': 0.82
            }
        ]
    }
    
    for risk_level, strategy_list in strategies.items():
        for strategy in strategy_list:
            with open(f'configs/{strategy["strategy_id"]}.json', 'w') as f:
                json.dump(strategy, f, indent=2)

def generate_rl_results():
    """Generate synthetic RL results for each strategy."""
    start_date = datetime(2025, 1, 1)
    days = 90
    
    for strategy_id in [f'strategy_{level}_{i}' for level in ['conservative', 'neutral', 'aggressive'] for i in range(1, 4)]:
        # Generate synthetic returns based on strategy type
        if 'conservative' in strategy_id:
            daily_returns = np.random.normal(0.0005, 0.008, days)
        elif 'neutral' in strategy_id:
            daily_returns = np.random.normal(0.0008, 0.012, days)
        else:  # aggressive
            daily_returns = np.random.normal(0.0012, 0.018, days)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(daily_returns)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        
        # Calculate max drawdown
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (rolling_max - cumulative_returns) / rolling_max
        max_drawdown = np.max(drawdowns)
        
        # Create DataFrame
        dates = [start_date + timedelta(days=i) for i in range(days)]
        df = pd.DataFrame({
            'date': dates,
            'cumulative_return': cumulative_returns,
            'sharpe_ratio': [sharpe_ratio] * days,
            'max_drawdown': [max_drawdown] * days
        })
        
        # Save to CSV
        df.to_csv(f'results/{strategy_id}_results.csv', index=False)

def generate_evaluation_dataset():
    """Generate unified evaluation dataset."""
    evaluation_data = []
    
    for strategy_id in [f'strategy_{level}_{i}' for level in ['conservative', 'neutral', 'aggressive'] for i in range(1, 4)]:
        # Read results
        results = pd.read_csv(f'results/{strategy_id}_results.csv')
        
        # Determine strategy type and evaluation metrics
        if 'conservative' in strategy_id:
            strategy_type = 'Conservative'
            ttf_score = 'High'
            info_quality = np.random.uniform(0.85, 0.95)
            usability = np.random.uniform(0.80, 0.90)
            trust_score = 'Good'
        elif 'neutral' in strategy_id:
            strategy_type = 'Neutral'
            ttf_score = 'Medium'
            info_quality = np.random.uniform(0.70, 0.80)
            usability = np.random.uniform(0.70, 0.80)
            trust_score = 'Medium'
        else:  # aggressive
            strategy_type = 'Aggressive'
            ttf_score = 'Low'
            info_quality = np.random.uniform(0.60, 0.70)
            usability = np.random.uniform(0.60, 0.70)
            trust_score = 'Low'
        
        # Get final metrics
        final_return = results['cumulative_return'].iloc[-1]
        final_sharpe = results['sharpe_ratio'].iloc[-1]
        final_mdd = results['max_drawdown'].iloc[-1]
        
        # Read strategy JSON for additional metrics
        with open(f'configs/{strategy_id}.json', 'r') as f:
            strategy = json.load(f)
        
        evaluation_data.append({
            'strategy_id': strategy_id,
            'strategy_type': strategy_type,
            'risk_score': strategy['risk_score'],
            'sentiment_score': strategy['sentiment_score'],
            'cumulative_return': final_return,
            'sharpe_ratio': final_sharpe,
            'max_drawdown': final_mdd,
            'ttf_score': ttf_score,
            'info_quality': info_quality,
            'usability': usability,
            'trust_score': trust_score
        })
    
    # Create and save DataFrame
    df = pd.DataFrame(evaluation_data)
    df.to_csv('results/evaluation_dataset.csv', index=False)
    return df

def generate_graph_data(evaluation_df):
    """Generate data for visualization."""
    # Cumulative return over time
    all_returns = pd.DataFrame()
    for strategy_id in evaluation_df['strategy_id']:
        results = pd.read_csv(f'results/{strategy_id}_results.csv')
        all_returns[strategy_id] = results['cumulative_return']
    all_returns['date'] = pd.date_range(start='2025-01-01', periods=90)
    all_returns.to_csv('results/cumulative_return_over_time.csv', index=False)
    
    # Summary scatter plot data
    scatter_data = evaluation_df[['strategy_id', 'sharpe_ratio', 'max_drawdown']]
    scatter_data.to_csv('results/summary_scatter.csv', index=False)

def generate_reports(evaluation_df):
    """Generate individual strategy reports and summary report."""
    # Generate individual strategy reports
    for _, row in evaluation_df.iterrows():
        strategy_id = row['strategy_id']
        report_content = f"""# Strategy Performance Report: {strategy_id}

## Quantitative Metrics
- Cumulative Return: {row['cumulative_return']*100:.2f}%
- Sharpe Ratio: {row['sharpe_ratio']:.2f}
- Maximum Drawdown: {row['max_drawdown']*100:.2f}%

## IS Theoretical Evaluation
- Task-Technology Fit (TTF): {row['ttf_score']}
- Information Quality: {row['info_quality']:.2f}
- System Usability: {row['usability']:.2f}
- Algorithmic Trust: {row['trust_score']}

## Risk and Sentiment Analysis
- Risk Score: {row['risk_score']:.2f}
- Sentiment Score: {row['sentiment_score']:.2f}
"""
        with open(f'reports/{strategy_id}.md', 'w') as f:
            f.write(report_content)
    
    # Generate summary report
    summary_content = """# IS Research Strategy Summary

## Quantitative Metrics
| Strategy | Return | Sharpe | MDD |
|----------|--------|--------|-----|
"""
    
    for _, row in evaluation_df.iterrows():
        summary_content += f"| {row['strategy_id']} | {row['cumulative_return']*100:.1f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.1f}% |\n"
    
    summary_content += "\n## IS Theoretical Evaluation\n"
    for _, row in evaluation_df.iterrows():
        summary_content += f"- {row['strategy_id']}: TTF = {row['ttf_score']}, Info Quality = {row['info_quality']:.2f}, Trust = {row['trust_score']}\n"
    
    with open('reports/summary.md', 'w') as f:
        f.write(summary_content)

def main():
    """Main function to generate the complete dataset."""
    print("Creating directory structure...")
    create_directory_structure()
    
    print("Generating strategy JSON files...")
    generate_strategy_jsons()
    
    print("Generating RL results...")
    generate_rl_results()
    
    print("Generating evaluation dataset...")
    evaluation_df = generate_evaluation_dataset()
    
    print("Generating graph data...")
    generate_graph_data(evaluation_df)
    
    print("Generating reports...")
    generate_reports(evaluation_df)
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main() 