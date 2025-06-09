"""
Trade Journal Analysis

This script loads trade data from a CSV file, cleans it, computes performance metrics,
and visualises cumulative profit/loss, monthly performance, best days, and most profitable pairs.

Author: Ali Huqoqi
Date: 09/06/2025
"""

import pandas as pd
import matplotlib.pyplot as plt

def main():
    """
    Load trade data, clean it, analyse key metrics, and create visualisations
    to help understand trading performance over time, by trade size,
    best performing days, and most profitable currency pairs.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv('Trades.csv')
    print(df.info())   # Summary info about data types and missing values
    print(df.head())   # First 5 rows preview

    # Convert 'Entry Time' and 'Exit Time' to datetime for time-based analysis
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])

    # Clean 'P&L' column: remove dollar signs and commas, then convert to float
    df['P&L'] = df['P&L'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(df.head())  # Confirm data cleaning

    # Overall trade statistics
    total_trades = len(df)
    winning_trades = len(df[df["P&L"] > 0])
    losing_trades = len(df[df["P&L"] <= 0])
    win_rate = (winning_trades / total_trades) * 100
    total_pnl = df["P&L"].sum()
    avg_pnl = df["P&L"].mean()

    print(f"Total trades: {total_trades}")
    print(f"Winning trades: {winning_trades}")
    print(f"Losing trades: {losing_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Average P&L per trade: ${avg_pnl:.2f}")

    # Sort by 'Exit Time' for cumulative P&L calculation and visualisation
    df_sorted = df.sort_values("Exit Time")
    df_sorted["Cumulative P&L"] = df_sorted["P&L"].cumsum()

    # Plot cumulative profit/loss over time
    plt.figure(figsize=(10,6))
    plt.plot(df_sorted["Exit Time"], df_sorted["Cumulative P&L"], marker="o", linestyle="-")
    plt.title("Cumulative P&L Over Time")
    plt.xlabel("Exit Time")
    plt.ylabel("Cumulative P&L ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Analyse trade size impact on P&L
    size_summary = df.groupby('Size').agg(
        Average_PnL=('P&L', 'mean'),
        Total_Trades=('P&L', 'count')
    ).reset_index()

    print("\nTrade Size Impact on P&L:")
    print(size_summary)

    # Monthly performance summary
    monthly_summary = df.groupby(df['Exit Time'].dt.to_period('M')).agg(
        Total_PnL=('P&L', 'sum'),
        Average_PnL=('P&L', 'mean'),
        Trades_Count=('P&L', 'count'),
        Win_Rate=('P&L', lambda x: (x > 0).mean() * 100)
    ).reset_index()

    monthly_summary['Exit Time'] = monthly_summary['Exit Time'].dt.strftime('%Y-%m')

    print("\nMonthly Performance Summary:")
    print(monthly_summary)

    # Plot monthly total P&L bar chart
    plt.figure(figsize=(10,6))
    plt.bar(monthly_summary['Exit Time'], monthly_summary['Total_PnL'], color='skyblue')
    plt.title('Monthly Total P&L')
    plt.xlabel('Month')
    plt.ylabel('Total P&L ($)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Analyse best performing days by summing P&L per day
    df['Exit Date'] = df['Exit Time'].dt.date
    daily_performance = df.groupby('Exit Date')['P&L'].sum().sort_values(ascending=False)
    print("\nBest Performing Days (Top 5):")
    print(daily_performance.head())

    # Analyse most profitable currency pairs
    pair_performance = df.groupby('Symbol')['P&L'].sum().sort_values(ascending=False)
    print("\nMost Profitable Currency Pairs:")
    print(pair_performance)

if __name__ == "__main__":
    main()