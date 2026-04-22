import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from lifelines import KaplanMeierFitter


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def get_ticker_order_by_volatility(df):
    return (
        df.groupby("Ticker")["Return"]
        .std()
        .sort_values()
        .index
    )


# --------------------------------------------------
# EDA Plots
# --------------------------------------------------

def plot_return_boxplot(df):
    order = get_ticker_order_by_volatility(df)

    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(10, 5))

    sns.boxplot(
        x="Ticker",
        y="Return",
        data=df,
        order=order
    )

    plt.title("Return Distribution by Ticker (Ordered by Volatility)")
    plt.xlabel("Stock")
    plt.ylabel("Daily Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_return_histogram(df):
    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(10, 5))

    sns.histplot(df["Return"], bins=100, kde=True)

    plt.title("Distribution of Daily Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_average_returns(df):
    avg_returns = df.groupby("Ticker")["Return"].mean()

    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(8, 4))

    sns.barplot(
        x=avg_returns.index,
        y=avg_returns.values
    )

    plt.title("Average Daily Return by Stock")
    plt.xlabel("Ticker")
    plt.ylabel("Average Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_price_trends(df, price_col="Adj Close"):
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))

    for ticker in df["Ticker"].unique():
        data = df[df["Ticker"] == ticker]
        plt.plot(data["Date"], data[price_col], label=ticker)

    plt.title("Stock Price Trends Over Time")
    plt.xlabel("Date")
    plt.ylabel(price_col)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_volatility_boxplot(df):
    order = get_ticker_order_by_volatility(df)

    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(12, 5))

    sns.boxplot(
        x="Ticker",
        y="Volatility_10d",
        data=df,
        order=order
    )

    plt.title("Volatility Comparison by Stock")
    plt.xlabel("Stock")
    plt.ylabel("10-Day Rolling Volatility")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_volatility_facets(df):
    sns.set_theme(style="darkgrid")

    g = sns.FacetGrid(df, col="Ticker", col_wrap=3, height=3, sharey=True)
    g.map_dataframe(
        sns.lineplot,
        x="Date",
        y="Volatility_10d",
        color="green"
    )

    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", rotation=45)

    g.set_titles("{col_name}")
    g.set_axis_labels("Date", "Volatility")

    plt.tight_layout()
    plt.show()


def plot_return_acf_pair(df, ticker):
    series = df.loc[df["Ticker"] == ticker, "Return"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    plot_acf(series, lags=20, ax=axes[0])
    axes[0].set_title(f"ACF: {ticker} Returns")

    plot_acf(series**2, lags=20, ax=axes[1])
    axes[1].set_title(f"ACF: {ticker} Squared Returns")

    plt.tight_layout()
    plt.show()


def plot_return_correlation_heatmap(df):
    pivot_df = df.pivot(index="Date", columns="Ticker", values="Return")
    corr_matrix = pivot_df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.set_theme(style="white")
    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix of Stock Returns")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Risk / Stats Plots
# --------------------------------------------------

def plot_var_comparison(tsla, wmt, var_95_tsla, var_95_wmt):
    sns.set_theme(style="darkgrid")

    plt.figure(figsize=(10, 5))

    sns.histplot(tsla, bins=100, stat="density", alpha=0.5, label="TSLA")
    sns.histplot(wmt, bins=100, stat="density", alpha=0.5, label="WMT")

    plt.axvline(var_95_tsla, linestyle="--")
    plt.axvline(var_95_wmt, linestyle="--")

    plt.title("Value-at-Risk (95%) Comparison")
    plt.xlabel("Daily Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_return_to_risk_ratio(summary):
    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(8, 4))

    sns.barplot(
        x=summary.index,
        y=summary["return_to_risk"]
    )

    plt.title("Return-to-Risk Ratio by Stock")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_return_vs_risk(summary_table):
    sns.set_theme(style="darkgrid", palette="Set2")
    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=summary_table,
        x="std_return",
        y="mean_return",
        hue=summary_table.index,
        s=100
    )

    for ticker in summary_table.index:
        plt.text(
            summary_table.loc[ticker, "std_return"],
            summary_table.loc[ticker, "mean_return"],
            ticker
        )

    plt.title("Return vs Risk (Volatility)")
    plt.xlabel("Volatility")
    plt.ylabel("Average Return")

    plt.tight_layout()
    plt.show()

def plot_qq_comparison(tsla, wmt):
    """
    QQ plot comparison for TSLA and WMT with consistent colors.
    """
    palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    stats.probplot(tsla, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set_color(palette[0])   
    axes[0].get_lines()[1].set_color("black")      
    axes[0].set_title("TSLA")

    # WMT → orange
    stats.probplot(wmt, dist="norm", plot=axes[1])
    axes[1].get_lines()[0].set_color(palette[1])
    axes[1].get_lines()[1].set_color("black")
    axes[1].set_title("WMT")

    plt.tight_layout()
    plt.show()


def plot_mean_difference_ci(mean_diff, ci_low, ci_high):
    plt.figure(figsize=(6, 2))

    plt.errorbar(
        x=mean_diff,
        y=0,
        xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
        fmt="o"
    )

    plt.axvline(0, linestyle="--")

    plt.title("95% CI for Mean Return Difference (TSLA - WMT)")
    plt.yticks([])
    plt.xlabel("Return Difference")

    plt.tight_layout()
    plt.show()


def plot_positive_negative_proportion(prop):
    """
    Plot proportion of positive vs negative returns using Set2 colors.
    """
    colors = sns.color_palette("Set2")

    plt.figure(figsize=(6, 4))

    plt.bar(
        ["Positive", "Negative"],
        [prop, 1 - prop],
        color=[colors[0], colors[1]]
    )

    plt.title("Proportion of Positive vs Negative Returns")
    plt.ylabel("Proportion")

    plt.tight_layout()
    plt.show()


def plot_growth_vs_other_volatility(df):
    """
    Compare volatility of growth stocks vs others with Set2 colors.
    """
    colors = sns.color_palette("Set2")

    plot_df = df.copy()
    plot_df["Group"] = plot_df["Ticker"].isin(
        ["TSLA", "NVDA", "META"]
    ).map({True: "Growth", False: "Other"})

    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=plot_df,
        x="Group",
        y="Volatility_10d",
        palette=[colors[0], colors[1]]
    )

    plt.title("Volatility: Growth vs Other Stocks")
    plt.xlabel("")
    plt.ylabel("10-Day Rolling Volatility")

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Model / Survival / Ranking Plots
# --------------------------------------------------

def plot_model_comparison(model_comparison):
    best_model_name = model_comparison.loc[
        model_comparison["Test_MAE"].idxmin(), "Model"
    ]

    colors = [
        "#66c2a5" if m == best_model_name else "#bdbdbd"
        for m in model_comparison["Model"]
    ]

    plt.figure(figsize=(8, 4))

    sns.barplot(
        y="Model",
        x="Test_MAE",
        data=model_comparison,
        palette=colors
    )

    plt.title("Model Comparison (Test MAE)")
    plt.xlabel("MAE")
    plt.ylabel("Model")

    plt.tight_layout()
    plt.show()


def plot_survival_curves(df_surv_cycles):
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(8, 5))

    for ticker in df_surv_cycles["Ticker"].unique():
        subset = df_surv_cycles[df_surv_cycles["Ticker"] == ticker]

        kmf.fit(
            subset["duration"].values,
            event_observed=subset["event"].values,
            label=ticker
        )
        kmf.plot_survival_function()

    plt.title("Survival Curves (Time to ≥5% Return)")
    plt.xlabel("Days")
    plt.ylabel("Survival Probability")

    plt.tight_layout()
    plt.show()


def plot_top1_vs_market_distribution(top1_returns, market_returns):
    plt.figure(figsize=(8, 4))

    sns.kdeplot(top1_returns, label="Top-1 Strategy", fill=True)
    sns.kdeplot(market_returns, label="Market", fill=True)

    plt.title("Return Distribution: Top-1 Strategy vs Market")
    plt.xlabel("Return")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_forecast_vs_actual(df_test, y_test, test_preds):
    plt.figure(figsize=(12, 5))

    plt.plot(df_test["Date"], y_test, label="Actual", color="black")
    plt.plot(df_test["Date"], test_preds, label="Predicted", color="red")

    plt.legend()
    plt.title("Forecast vs Actual (Test Period)")

    plt.tight_layout()
    plt.show()

def plot_var_comparison(tsla, wmt, var_95_tsla, var_95_wmt):
    """
    Plot 95% Value-at-Risk comparison for TSLA and WMT daily returns.
    """
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 5))

    sns.histplot(tsla, bins=100, stat="density", color="green", alpha=0.5, label="TSLA")
    sns.histplot(wmt, bins=100, stat="density", color="orange", alpha=0.5, label="WMT")

    plt.axvline(var_95_tsla, color="green", linestyle="--")
    plt.axvline(var_95_wmt, color="orange", linestyle="--")

    plt.title("Value-at-Risk (95%) Comparison")
    plt.xlabel("Daily Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()