#!/usr/bin/env python3
"""
Fraud Detection Analytics Report Generator
==========================================

This script generates a comprehensive analytics report for the fraud detection case study.
It creates visualizations and saves them to the 'report_images' directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle

# Create directory for report images
os.makedirs('report_images', exist_ok=True)

# Visualization settings
plt.style.use('ggplot') 
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set(style="whitegrid")
sns.set_palette('viridis')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load transaction data and related profiles"""
    transactions = pd.read_csv('data/transactions_df.csv')
    customer_profiles = pd.read_csv('data/customer_profiles_table.csv')
    terminal_profiles = pd.read_csv('data/terminal_profiles_table.csv')
    
    # Convert TX_DATETIME to datetime
    transactions['TX_DATETIME'] = pd.to_datetime(transactions['TX_DATETIME'])
    
    # Fix available_terminals (convert string representation to actual list)
    customer_profiles['available_terminals'] = customer_profiles['available_terminals'].apply(
        lambda x: [int(i) for i in x.strip('[]').split(',')] if isinstance(x, str) and len(x) > 2 else [])
    
    print(f"Dataset information:")
    print(f"Transactions: {transactions.shape[0]} rows, {transactions.shape[1]} columns")
    print(f"Customers: {customer_profiles.shape[0]}")
    print(f"Terminals: {terminal_profiles.shape[0]}")
    
    return transactions, customer_profiles, terminal_profiles

def visualize_fraud_distribution(transactions):
    """Visualize the distribution of fraudulent vs legitimate transactions"""
    fraud_count = transactions['TX_FRAUD'].sum()
    total_count = transactions.shape[0]
    fraud_percentage = fraud_count/total_count * 100
    
    print(f"\nFraud statistics:")
    print(f"Fraudulent transactions: {fraud_count} ({fraud_percentage:.2f}%)")
    print(f"Legitimate transactions: {total_count - fraud_count} ({100-fraud_percentage:.2f}%)")
    
    # Visualize fraud distribution
    plt.figure(figsize=(10, 6))
    fraud_labels = ['Legitimate', 'Fraudulent']
    fraud_counts = [total_count - fraud_count, fraud_count]
    
    colors = sns.color_palette('viridis', n_colors=2)
    plt.pie(fraud_counts, labels=fraud_labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=[0, 0.1])
    plt.title('Distribution of Fraudulent vs. Legitimate Transactions', fontsize=16)
    plt.axis('equal')
    plt.savefig('report_images/fraud_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_fraud_scenarios(transactions):
    """Analyze the distribution of fraud scenarios"""
    fraud_data = transactions[transactions['TX_FRAUD'] == 1]
    fraud_scenarios = fraud_data['TX_FRAUD_SCENARIO'].value_counts().sort_index()
    scenario_names = {
        1: 'Anomalous Amount',
        2: 'Compromised Terminal',
        3: 'Card-Not-Present',
        4: 'Quick Cash-Out'
    }
    
    print("\nFraud Scenario Distribution:")
    for scenario_id, count in fraud_scenarios.items():
        scenario_name = scenario_names.get(scenario_id, f'Unknown ({scenario_id})')
        percentage = count / fraud_data.shape[0] * 100
        print(f"  {scenario_name}: {count} transactions ({percentage:.1f}%)")
    
    # Create a visualization for fraud scenarios
    plt.figure(figsize=(12, 6))
    scenario_df = pd.DataFrame({
        'scenario': [scenario_names.get(i, f'Unknown ({i})') for i in fraud_scenarios.index],
        'count': fraud_scenarios.values,
        'percentage': [count / fraud_data.shape[0] * 100 for count in fraud_scenarios.values]
    })
    
    ax = sns.barplot(x='scenario', y='count', data=scenario_df, palette='viridis')
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{scenario_df["percentage"].iloc[i]:.1f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=11)
    
    plt.title('Distribution of Fraud Scenarios', fontsize=14)
    plt.xlabel('Fraud Scenario Type', fontsize=12)
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('report_images/fraud_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_terminal_fraud(transactions, terminal_profiles):
    """Analyze fraud rates by terminal type"""
    # Merge transactions with terminal data
    tx_with_terminal = pd.merge(transactions, 
                               terminal_profiles[['TERMINAL_ID', 'terminal_type']], 
                               on='TERMINAL_ID', how='left')
    
    # Calculate fraud statistics by terminal type
    terminal_stats = tx_with_terminal.groupby('terminal_type').agg(
        total_count=('TX_FRAUD', 'count'),
        fraud_count=('TX_FRAUD', 'sum')
    ).reset_index()
    terminal_stats['fraud_rate'] = terminal_stats['fraud_count'] / terminal_stats['total_count'] * 100
    
    print("\nFraud Rate by Terminal Type:")
    for _, row in terminal_stats.sort_values('fraud_rate', ascending=False).iterrows():
        print(f"  {row['terminal_type']}: {row['fraud_rate']:.2f}% ({row['fraud_count']} out of {row['total_count']})")
    
    # Create a visualization for terminal fraud rates
    plt.figure(figsize=(12, 6))
    terminal_stats_sorted = terminal_stats.sort_values('fraud_rate', ascending=False)
    
    ax = sns.barplot(x='terminal_type', y='fraud_rate', data=terminal_stats_sorted, palette='viridis')
    
    # Add percentage labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.2f}%', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom', fontsize=11)
    
    plt.title('Fraud Rate by Terminal Type', fontsize=14)
    plt.xlabel('Terminal Type', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig('report_images/terminal_fraud_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return tx_with_terminal

def analyze_transaction_amounts(transactions):
    """Analyze transaction amount patterns for fraudulent vs legitimate transactions"""
    # Transaction amount statistics
    legitimate_amount = transactions[transactions['TX_FRAUD'] == 0]['TX_AMOUNT']
    fraud_amount = transactions[transactions['TX_FRAUD'] == 1]['TX_AMOUNT']
    
    print("\nTransaction Amount Statistics:")
    print(f"  Legitimate transactions: Avg=${legitimate_amount.mean():.2f}, Median=${legitimate_amount.median():.2f}, Max=${legitimate_amount.max():.2f}")
    print(f"  Fraudulent transactions: Avg=${fraud_amount.mean():.2f}, Median=${fraud_amount.median():.2f}, Max=${fraud_amount.max():.2f}")
    
    # Create a visualization for amount distribution
    plt.figure(figsize=(12, 6))
    
    # Create a DataFrame for easier plotting
    amount_data = pd.DataFrame({
        'Amount': pd.concat([legitimate_amount, fraud_amount]),
        'Type': ['Legitimate'] * len(legitimate_amount) + ['Fraudulent'] * len(fraud_amount)
    })
    
    # Plot boxplot with log scale
    sns.boxplot(x='Type', y='Amount', data=amount_data, palette='viridis')
    plt.yscale('log')
    plt.title('Transaction Amount Distribution (Log Scale)', fontsize=14)
    plt.xlabel('Transaction Type', fontsize=12)
    plt.ylabel('Amount ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('report_images/amount_distribution_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # KDE plot for amount distribution
    plt.figure(figsize=(12, 6))
    sns.kdeplot(legitimate_amount, label='Legitimate', shade=True)
    sns.kdeplot(fraud_amount, label='Fraudulent', shade=True)
    plt.title('Distribution of Transaction Amounts', fontsize=14)
    plt.xlabel('Amount ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0, 1000)  # Limiting x-axis for better visualization
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_images/amount_distribution_kde.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_temporal_patterns(transactions):
    """Analyze temporal patterns of fraud"""
    # Extract time features
    transactions['hour'] = transactions['TX_DATETIME'].dt.hour
    transactions['day_of_week'] = transactions['TX_DATETIME'].dt.dayofweek
    transactions['day_name'] = transactions['TX_DATETIME'].dt.day_name()
    transactions['month'] = transactions['TX_DATETIME'].dt.month
    
    # Fraud by hour of day
    hourly_stats = transactions.groupby('hour').agg(
        total=('TX_FRAUD', 'count'),
        fraud=('TX_FRAUD', 'sum')
    ).reset_index()
    hourly_stats['fraud_rate'] = hourly_stats['fraud'] / hourly_stats['total'] * 100
    
    # Fraud by day of week
    day_stats = transactions.groupby(['day_of_week', 'day_name']).agg(
        total=('TX_FRAUD', 'count'),
        fraud=('TX_FRAUD', 'sum')
    ).reset_index()
    day_stats['fraud_rate'] = day_stats['fraud'] / day_stats['total'] * 100
    day_stats = day_stats.sort_values('day_of_week')
    
    # Monthly pattern
    monthly_stats = transactions.groupby('month').agg(
        total=('TX_FRAUD', 'count'),
        fraud=('TX_FRAUD', 'sum')
    ).reset_index()
    monthly_stats['fraud_rate'] = monthly_stats['fraud'] / monthly_stats['total'] * 100
    
    # Plot temporal patterns - Hour of day
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='hour', y='fraud_rate', data=hourly_stats, marker='o', linewidth=2)
    plt.title('Fraud Rate by Hour of Day', fontsize=14)
    plt.xlabel('Hour of Day (24h)', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_images/fraud_by_hour.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot temporal patterns - Day of week
    plt.figure(figsize=(12, 6))
    sns.barplot(x='day_name', y='fraud_rate', data=day_stats, palette='viridis')
    plt.title('Fraud Rate by Day of Week', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_images/fraud_by_day.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot temporal patterns - Month
    plt.figure(figsize=(12, 6))
    sns.barplot(x='month', y='fraud_rate', data=monthly_stats, palette='viridis')
    plt.title('Fraud Rate by Month', fontsize=14)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report_images/fraud_by_month.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return transactions

def analyze_customer_behavior(transactions, customer_profiles):
    """Analyze customer behavior patterns related to fraud"""
    # Merge transaction data with customer profiles
    tx_with_customer = pd.merge(transactions, 
                              customer_profiles[['CUSTOMER_ID', 'mean_amount', 'std_amount']], 
                              on='CUSTOMER_ID', how='left')
    
    # Calculate amount deviation from customer average
    tx_with_customer['amount_deviation'] = (tx_with_customer['TX_AMOUNT'] - tx_with_customer['mean_amount']) / tx_with_customer['std_amount']
    
    # Visualize amount deviation distribution
    plt.figure(figsize=(12, 6))
    fraud_deviations = tx_with_customer[tx_with_customer['TX_FRAUD'] == 1]['amount_deviation']
    legit_deviations = tx_with_customer[tx_with_customer['TX_FRAUD'] == 0]['amount_deviation']
    
    sns.kdeplot(legit_deviations, label='Legitimate', shade=True)
    sns.kdeplot(fraud_deviations, label='Fraudulent', shade=True)
    plt.title('Distribution of Transaction Amount Deviation from Customer Average', fontsize=14)
    plt.xlabel('Amount Deviation (z-score)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-5, 15)  # Limit x-axis for better visualization
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_images/amount_deviation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return tx_with_customer

def create_geographic_visualization(transactions, customer_profiles, terminal_profiles):
    """Create a geographic visualization of transactions and fraud"""
    # Merge transaction data with customer and terminal profiles
    tx_geo = pd.merge(transactions, customer_profiles[['CUSTOMER_ID', 'x_customer_id', 'y_customer_id']], on='CUSTOMER_ID', how='left')
    tx_geo = pd.merge(tx_geo, terminal_profiles[['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id']], on='TERMINAL_ID', how='left')
    
    # Calculate transaction distance
    tx_geo['distance'] = np.sqrt((tx_geo['x_customer_id'] - tx_geo['x_terminal_id'])**2 + 
                                 (tx_geo['y_customer_id'] - tx_geo['y_terminal_id'])**2)
    
    # Plot distance distribution
    plt.figure(figsize=(12, 6))
    fraud_distances = tx_geo[tx_geo['TX_FRAUD'] == 1]['distance']
    legit_distances = tx_geo[tx_geo['TX_FRAUD'] == 0]['distance']
    
    sns.kdeplot(legit_distances, label='Legitimate', shade=True)
    sns.kdeplot(fraud_distances, label='Fraudulent', shade=True)
    plt.title('Distribution of Transaction Distances', fontsize=14)
    plt.xlabel('Distance between Customer and Terminal', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig('report_images/transaction_distances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a scatter plot of terminal locations with fraud rate as color
    terminal_fraud = tx_geo.groupby('TERMINAL_ID').agg(
        total=('TX_FRAUD', 'count'),
        fraud=('TX_FRAUD', 'sum'),
        x=('x_terminal_id', 'first'),
        y=('y_terminal_id', 'first')
    ).reset_index()
    
    terminal_fraud['fraud_rate'] = terminal_fraud['fraud'] / terminal_fraud['total'] * 100
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(terminal_fraud['x'], terminal_fraud['y'], 
                         c=terminal_fraud['fraud_rate'], cmap='viridis', 
                         s=terminal_fraud['total']/5, alpha=0.7)
    
    plt.colorbar(scatter, label='Fraud Rate (%)')
    plt.title('Geographic Distribution of Terminals and Fraud Rates', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.tight_layout()
    plt.savefig('report_images/terminal_geography.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return tx_geo

def analyze_model_results():
    """Analyze and visualize model performance results"""
    try:
        # Try to load the trained model
        with open('models/xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
            
        # If model exists, create a feature importance plot
        plt.figure(figsize=(12, 8))
        features = model[2].feature_names_in_
        importances = model[2].feature_importances_
        
        indices = np.argsort(importances)[::-1]
        top_n = 15
        
        plt.barh(range(top_n), importances[indices][:top_n], align='center')
        plt.yticks(range(top_n), [features[i] for i in indices[:top_n]])
        plt.title('Top 15 Feature Importances for Fraud Detection', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.tight_layout()
        plt.savefig('report_images/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nModel analysis complete.")
    except Exception as e:
        print(f"Model analysis skipped: {str(e)}")

def generate_html_report():
    """Generate an HTML report with all visualizations"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Analytics Report</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1 { color: #2c3e50; text-align: center; }
            h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; }
            .figure { margin: 20px 0; text-align: center; }
            .figure img { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .figure p { font-style: italic; color: #7f8c8d; }
        </style>
    </head>
    <body>
        <h1>Payment Fraud Detection Analytics Report</h1>
        
        <h2>1. Fraud Distribution Overview</h2>
        <div class="figure">
            <img src="report_images/fraud_distribution.png" alt="Fraud Distribution">
            <p>Distribution of fraudulent vs. legitimate transactions in the dataset.</p>
        </div>
        
        <h2>2. Fraud Scenario Analysis</h2>
        <div class="figure">
            <img src="report_images/fraud_scenarios.png" alt="Fraud Scenarios">
            <p>Distribution of different fraud scenarios in the dataset.</p>
        </div>
        
        <h2>3. Terminal Type Analysis</h2>
        <div class="figure">
            <img src="report_images/terminal_fraud_rates.png" alt="Terminal Fraud Rates">
            <p>Fraud rates across different terminal types.</p>
        </div>
        
        <h2>4. Transaction Amount Analysis</h2>
        <div class="figure">
            <img src="report_images/amount_distribution_boxplot.png" alt="Amount Distribution Boxplot">
            <p>Boxplot showing the distribution of transaction amounts for legitimate and fraudulent transactions.</p>
        </div>
        <div class="figure">
            <img src="report_images/amount_distribution_kde.png" alt="Amount Distribution KDE">
            <p>Density plot showing the distribution of transaction amounts.</p>
        </div>
        
        <h2>5. Temporal Patterns of Fraud</h2>
        <div class="figure">
            <img src="report_images/fraud_by_hour.png" alt="Fraud by Hour">
            <p>Fraud rates by hour of day.</p>
        </div>
        <div class="figure">
            <img src="report_images/fraud_by_day.png" alt="Fraud by Day">
            <p>Fraud rates by day of week.</p>
        </div>
        <div class="figure">
            <img src="report_images/fraud_by_month.png" alt="Fraud by Month">
            <p>Fraud rates by month.</p>
        </div>
        
        <h2>6. Customer Behavior Analysis</h2>
        <div class="figure">
            <img src="report_images/amount_deviation.png" alt="Amount Deviation">
            <p>Distribution of transaction amount deviations from customer averages.</p>
        </div>
        
        <h2>7. Geographic Analysis</h2>
        <div class="figure">
            <img src="report_images/transaction_distances.png" alt="Transaction Distances">
            <p>Distribution of distances between customers and terminals.</p>
        </div>
        <div class="figure">
            <img src="report_images/terminal_geography.png" alt="Terminal Geography">
            <p>Geographic distribution of terminals with fraud rates.</p>
        </div>
        
        <h2>8. Model Feature Importance</h2>
        <div class="figure">
            <img src="report_images/feature_importance.png" alt="Feature Importance">
            <p>Top features for fraud detection based on the XGBoost model.</p>
        </div>
    </body>
    </html>
    """
    
    with open('fraud_detection_report.html', 'w') as f:
        f.write(html_content)
    
    print("\nHTML report generated as 'fraud_detection_report.html'")

def main():
    """Main function to run the entire analysis"""
    print("=== Fraud Detection Analytics Report Generator ===\n")
    
    # Load data
    transactions, customer_profiles, terminal_profiles = load_data()
    
    # Run all analyses and create visualizations
    visualize_fraud_distribution(transactions)
    analyze_fraud_scenarios(transactions)
    tx_with_terminal = analyze_terminal_fraud(transactions, terminal_profiles)
    analyze_transaction_amounts(transactions)
    transactions_with_time = analyze_temporal_patterns(transactions)
    tx_with_customer = analyze_customer_behavior(transactions, customer_profiles)
    tx_geo = create_geographic_visualization(transactions, customer_profiles, terminal_profiles)
    analyze_model_results()
    
    # Generate HTML report
    generate_html_report()
    
    print("\n=== Analysis Complete ===")
    print(f"All visualizations saved to the 'report_images' directory")

if __name__ == "__main__":
    main() 