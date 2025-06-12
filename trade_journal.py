"""
Trade Journal Analysis

This script loads trade data from a CSV file, cleans it, computes performance metrics,
and visualises cumulative profit/loss, monthly performance, best days, and most profitable pairs.

Author: Ali Huqoqi
Date: 09/06/2025
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


class TradeJournal:
    """
    Manages trades loading, analysis, and visualisations.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Load and clean trade data from CSV."""
        df = pd.read_csv(self.csv_path)
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        df['Exit Time'] = pd.to_datetime(df['Exit Time'])
        df['P&L'] = (
            df['P&L']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
            .astype(float)
        )
        self.df = df

    def print_summary(self) -> None:
        """Print info and missing values summary."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        print(self.df.info())
        print(self.df.head())
        print("\nMissing values per column:")
        print(self.df.isnull().sum())

    def print_stats(self) -> None:
        """Calculate and print overall trade stats."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        total_trades = len(self.df)
        winning_trades = len(self.df[self.df["P&L"] > 0])
        losing_trades = len(self.df[self.df["P&L"] <= 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades else 0.0
        total_pnl = self.df["P&L"].sum()
        avg_pnl = self.df["P&L"].mean()

        print(f"Total trades: {total_trades}")
        print(f"Winning trades: {winning_trades}")
        print(f"Losing trades: {losing_trades}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Average P&L per trade: ${avg_pnl:.2f}")

    def plot_cumulative_pnl(self) -> None:
        """Plot cumulative profit/loss over time."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        df_sorted = self.df.sort_values("Exit Time")
        df_sorted["Cumulative P&L"] = df_sorted["P&L"].cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted["Exit Time"], df_sorted["Cumulative P&L"], marker="o", linestyle="-")
        plt.title("Cumulative P&L Over Time")
        plt.xlabel("Exit Time")
        plt.ylabel("Cumulative P&L ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def trade_size_summary(self) -> pd.DataFrame:
        """Return trade size impact summary."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        return self.df.groupby('Size').agg(
            Average_PnL=('P&L', 'mean'),
            Total_Trades=('P&L', 'count')
        ).reset_index()

    def monthly_performance_summary(self) -> pd.DataFrame:
        """Return monthly performance summary."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        monthly_summary = self.df.groupby(self.df['Exit Time'].dt.to_period('M')).agg(
            Total_PnL=('P&L', 'sum'),
            Average_PnL=('P&L', 'mean'),
            Trades_Count=('P&L', 'count'),
            Win_Rate=('P&L', lambda x: (x > 0).mean() * 100)
        ).reset_index()
        monthly_summary['Exit Time'] = monthly_summary['Exit Time'].dt.strftime('%Y-%m')
        return monthly_summary

    def plot_monthly_pnl(self) -> None:
        """Plot monthly total P&L."""
        monthly_summary = self.monthly_performance_summary()
        plt.figure(figsize=(10, 6))
        plt.bar(monthly_summary['Exit Time'], monthly_summary['Total_PnL'], color='skyblue')
        plt.title('Monthly Total P&L')
        plt.xlabel('Month')
        plt.ylabel('Total P&L ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def best_performing_days(self, top_n: int = 5) -> pd.Series:
        """Return top N best performing days by P&L."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        self.df['Exit Date'] = self.df['Exit Time'].dt.date
        daily_performance = self.df.groupby('Exit Date')['P&L'].sum().sort_values(ascending=False)
        return daily_performance.head(top_n)

    def most_profitable_pairs(self) -> pd.Series:
        """Return currency pairs ranked by total P&L."""
        if self.df is None:
            raise ValueError("Data not loaded yet.")
        return self.df.groupby('Symbol')['P&L'].sum().sort_values(ascending=False)


def main() -> None:
    journal = TradeJournal('Trades.csv')
    journal.load_data()
    journal.print_summary()
    journal.print_stats()
    journal.plot_cumulative_pnl()

    print("\nTrade Size Impact on P&L:")
    print(journal.trade_size_summary())

    print("\nMonthly Performance Summary:")
    print(journal.monthly_performance_summary())

    journal.plot_monthly_pnl()

    print("\nBest Performing Days (Top 5):")
    print(journal.best_performing_days())

    print("\nMost Profitable Currency Pairs:")
    print(journal.most_profitable_pairs())


if __name__ == "__main__":
    main()
