import pandas as pd 
import yfinance as yf
import matplotlib.pyplot as plt
import time 
from datetime import datetime
import requests
from matplotlib.backends.backend_pdf import PdfPages
from io import StringIO

def get_benchmark():
    while True:
        valid = ["SPY", "QQQ", "IWM", "DIA"]
        benchmark = input("Enter a benchmark (SPY, QQQ, IWM, or DIA): ").strip().upper()
        if benchmark not in valid:
                print(f"Enter a valid ticker {', '.join(valid)}")
                continue 
        return benchmark
    
def get_user_portfolio():
    while True:
        try:
            num_assets = int(input("How many assets are in your portfolio?: "))
            if num_assets <= 0:
                print("At least enter 1 asset")
                continue
            break
        except ValueError:
            print("Please enter a whole number.")
           
    portfolio = {}

    for i in range(num_assets):
        while True: 
            ticker = input(f"Enter ticker #{i+1}: ").upper().strip()
            if not ticker.isalpha():
                print("Ticker must contain letters only. Please try again.")
                continue
            if ticker in portfolio:
                print("You already entered that ticker.")
                continue        
            break              
        
        while True:           
            try:
                weight = float(input(f"Enter {ticker} weight: "))
                if weight < 0:
                    print("Please enter a non-negative number.")
                    continue
                break
            except ValueError:
                print("Please enter a numerical value.")
        
        portfolio[ticker] = weight 

    return portfolio

def save_weights(portfolio):
    weights_df = pd.DataFrame(list(portfolio.items()), columns = ["Ticker", "Weight"])
    weights_df.to_csv('weights_user.csv', index = False)
    print('Weights saved to weights_user.csv')

def get_date_range():
    while True:
        try:
            start_date = input("Enter a start date in (YYYY-MM-DD): ")
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid Format. Please enter a date in YYYY-MM-DD")
            continue
        try:
            end_date = input("Enter the end date in (YYYY-MM-DD): ")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid Format. Please enter a date in YYYY-MM-DD")
            continue

        if end_date <= start_date:
            print("End date must be after start date.")
            continue
        return start_date, end_date

def download_single_ticker(ticker, start_date, end_date, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(tickers=ticker, start=start_date, 
                               end=end_date, auto_adjust=True, progress=False)
            if not data.empty:
                print(f"{ticker} successfully downloaded.")
                return data['Close']
            print(f"Attempt {attempt} for {ticker} not successfully downloaded")
        except Exception as e:
            print(f"Attempt #{attempt} for {ticker} {e} failed.")
        if attempt < max_retries:
            print(f"Retrying download for {ticker} in 15 seconds.")
            time.sleep(15)
    return None

def download_price_data(tickers, start_date, end_date):    # ← rewritten
    try:
        data = yf.download(
            tickers=" ".join(tickers),
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="ticker"
        )
        if data.empty:
            print("Download failed — no data returned.")
            return None
        if len(tickers) == 1:
            price_data = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            price_data = data.xs("Close", axis=1, level=1)
        print(f"Downloaded: {', '.join(tickers)}")
        return price_data
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def download_benchmark(benchmark, start_date, end_date):
    time.sleep(15)
    benchmark_prices = download_single_ticker(benchmark, start_date, end_date)
    if benchmark_prices is None:
        print(f"Could not download benchmark {benchmark}.")
        return None
    return benchmark_prices

def load_from_csv(benchmark):
    price_data = pd.read_csv('price_data_user.csv', index_col=0, parse_dates=True)
    benchmark_prices = pd.read_csv('benchmark_prices.csv', index_col=0, parse_dates=True).squeeze()
    weights_series = load_weights()
    print("Loaded data from existing CSVs.")
    return price_data, benchmark_prices, weights_series

def load_weights():
    weights_df = pd.read_csv('weights_user.csv')
    weights_series = pd.Series(weights_df["Weight"].values, index=weights_df["Ticker"])
    weights_series = weights_series / weights_series.sum()
    return weights_series

def get_initial_investment():
    while True:
        try:
            initial_investment = float(input("What is your initial investment (in dollars)?: "))
            if initial_investment <= 0:
                print("Please enter a number greater than zero")
                continue
            return initial_investment
        except ValueError: 
           print("Please enter a numerical value.")

def load_price_data(weights_series):
    price_data = pd.read_csv('price_data_user.csv', index_col=0, parse_dates=True)
    price_data = price_data.sort_index()
    price_data = price_data.dropna()
    missing_tickers = set(weights_series.index) - set(price_data.columns)
    if missing_tickers:
        raise ValueError(f"Missing tickers in price data: {missing_tickers}")
    return price_data

def calculate_portfolio(price_data, weights_series, initial_investment):
    returns_data = price_data.pct_change().dropna()
    portfolio_returns = returns_data.dot(weights_series)
    portfolio_value = initial_investment * (1 + portfolio_returns).cumprod()
    return portfolio_returns, portfolio_value

def calculate_benchmark(benchmark_prices, initial_investment):
    benchmark_prices = benchmark_prices.squeeze()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    benchmark_value = initial_investment * (1 + benchmark_returns).cumprod()
    return benchmark_returns, benchmark_value

def calculate_drawdown(portfolio_value):
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def get_risk_free_rate():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
        df = df.dropna()
        risk_free_rate = float(df['DGS3MO'].iloc[-1]) / 100
        return risk_free_rate
    except:
        print("Could not fetch risk-free rate. Using 4% default.")
        return 0.04
    
def calc_beta_and_corr(portfolio_returns, benchmark_returns):
    portfolio_returns, benchmark_returns = portfolio_returns.align(benchmark_returns, join='inner')
    correlation = portfolio_returns.corr(benchmark_returns)
    covarience = portfolio_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    if variance == 0: 
        beta = None
    else:
        beta = covarience/variance
    return beta, correlation

def calculate_metrics(portfolio_returns, portfolio_value, risk_free_rate):
    current_value = float(portfolio_value.iloc[-1])
    growth_multiple = float(portfolio_value.iloc[-1]) / float(portfolio_value.iloc[0])
    num_days = len(portfolio_value)
    annualized_return = growth_multiple ** (252 / num_days) - 1
    annualized_volatility = portfolio_returns.std() * (252 ** 0.5)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_deviation = downside_returns.std() * (252 ** 0.5)
    if downside_deviation == 0:
        sortino_ratio = None
    else:
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    return current_value, annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio

def print_results(current_value, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio,
                  bench_current_value, bench_annualized_return, bench_annualized_volatility, 
                  bench_max_drawdown, bench_sharpe_ratio, risk_free_rate, sortino_ratio, bench_sortino_ratio, beta, correlation):
    print(f"{'Metric':<25} {'Portfolio':>12} {'Benchmark':>12}")
    print("-" * 50)
    print(f"{'Current Value':<25} ${current_value:>11,.2f} ${bench_current_value:>11,.2f}")
    print(f"{'Annualized Return':<25} {annualized_return:>11.2%} {bench_annualized_return:>11.2%}")
    print(f"{'Annualized Volatility':<25} {annualized_volatility:>11.2%} {bench_annualized_volatility:>11.2%}")
    print(f"{'Max Drawdown':<25} {max_drawdown:>11.2%} {bench_max_drawdown:>11.2%}")
    print(f"{'Sharpe Ratio':<25} {sharpe_ratio:>11.2f} {bench_sharpe_ratio:>11.2f}")
    print(f"{'Sortino Ratio':<25} {sortino_ratio:>11.2f} {bench_sortino_ratio:>11.2f}")
    beta_str = f"{beta:>11.2f}" if beta is not None else "        N/A"
    corr_str = f"{correlation:>11.2f}" if correlation is not None else "        N/A"
    print(f"{'Beta':<25} {beta_str} {'N/A':>12}")
    print(f"{'Correlation':<25} {corr_str} {'N/A':>12}")
    print(f"Risk-Free Rate Used: {risk_free_rate:.2%}")

def plot_dashboard(portfolio_value, benchmark_value, drawdown, portfolio_returns, start_date, end_date):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 12), sharex=True)
    ax1.plot(portfolio_value, label="Portfolio Value")
    ax1.plot(benchmark_value, label="Benchmark Value")
    ax1.set_title("Portfolio vs Benchmark")
    ax1.set_ylabel("Value ($)")
    ax1.legend()
    ax1.grid(True)
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.plot(drawdown, color='red', linewidth=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend()
    ax2.grid(True)
    rolling_vol = portfolio_returns.rolling(30).std() * (252 ** 0.5)
    ax3.plot(rolling_vol, color='navy', label='Rolling 30-Day Volatility')
    ax3.set_title("Rolling Volatility")
    ax3.set_ylabel("Volatility (Annualized)")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True)
    plt.suptitle(f"Portfolio Dashboard: {start_date} to {end_date}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def save_report(portfolio_value, benchmark_value, drawdown, portfolio_returns,
                start_date, end_date, current_value, annualized_return,
                annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio,
                beta, correlation, bench_current_value, bench_annualized_return,
                bench_annualized_volatility, bench_max_drawdown, bench_sharpe_ratio,
                bench_sortino_ratio, risk_free_rate, benchmark, portfolio, weights_series):
    filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with PdfPages(filename) as pdf:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.text(0.5, 0.97, "Portfolio Report",
                ha='center', fontsize=22, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.92, f"{start_date}  to  {end_date}",
                ha='center', fontsize=13, color='gray', transform=ax.transAxes)
        ax.plot([0.05, 0.95], [0.89, 0.89], color='#cccccc', linewidth=1, transform=ax.transAxes)
        ax.text(0.5, 0.84, "Portfolio Holdings", ha='center', fontsize=12,
                fontweight='bold', transform=ax.transAxes)
        y_h = 0.79
        for ticker, weight in zip(weights_series.index, weights_series.values):
            ax.text(0.5, y_h, f"{ticker}  —  {weight:.1%}",
                    ha='center', fontsize=11, transform=ax.transAxes)
            y_h -= 0.05
        ax.text(0.5, y_h - 0.01, f"Benchmark: {benchmark}",
                ha='center', fontsize=11, color='gray', transform=ax.transAxes)
        divider_y = y_h - 0.06
        ax.plot([0.05, 0.95], [divider_y, divider_y], color='#cccccc', linewidth=1, transform=ax.transAxes)
        metrics = [
            ["Current Value",   f"${current_value:,.2f}",       f"${bench_current_value:,.2f}"],
            ["Annual Return",   f"{annualized_return:.2%}",      f"{bench_annualized_return:.2%}"],
            ["Annual Vol",      f"{annualized_volatility:.2%}",  f"{bench_annualized_volatility:.2%}"],
            ["Max Drawdown",    f"{max_drawdown:.2%}",           f"{bench_max_drawdown:.2%}"],
            ["Sharpe Ratio",    f"{sharpe_ratio:.2f}",           f"{bench_sharpe_ratio:.2f}"],
            ["Sortino Ratio",   f"{sortino_ratio:.2f}",          f"{bench_sortino_ratio:.2f}"],
            ["Beta",            f"{beta:.2f}" if beta is not None else "N/A", "N/A"],
            ["Correlation",     f"{correlation:.2f}" if correlation is not None else "N/A", "N/A"],
            ["Risk-Free Rate",  f"{risk_free_rate:.2%}",         f"{risk_free_rate:.2%}"],
        ]
        table_bottom = max(0.02, divider_y - 0.52)
        table_height = divider_y - table_bottom - 0.02
        table = ax.table(
            cellText=metrics,
            colLabels=["Metric", "Portfolio", "Benchmark"],
            cellLoc='center',
            colLoc='center',
            bbox=[0.05, table_bottom, 0.90, table_height]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#cccccc')
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1f3f6e')
            elif row % 2 == 0:
                cell.set_facecolor('#f2f6fc')
            else:
                cell.set_facecolor('white')
        pdf.savefig(fig)
        plt.close()
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 12), sharex=True)
        ax1.plot(portfolio_value, label="Portfolio Value")
        ax1.plot(benchmark_value, label="Benchmark Value")
        ax1.set_title("Portfolio vs Benchmark")
        ax1.set_ylabel("Value ($)")
        ax1.legend()
        ax1.grid(True)
        ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
        ax2.plot(drawdown, color='red', linewidth=0.8)
        ax2.axhline(y=0, color='black', linewidth=0.8)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend()
        ax2.grid(True)
        rolling_vol = portfolio_returns.rolling(30).std() * (252 ** 0.5)
        ax3.plot(rolling_vol, color='navy', label='Rolling 30-Day Volatility')
        ax3.set_title("Rolling Volatility")
        ax3.set_ylabel("Volatility (Annualized)")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True)
        plt.suptitle(f"Portfolio Dashboard: {start_date} to {end_date}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        print(f"Report saved to {filename}.")

def main():
    mode = input("Download fresh data or load from CSV? (download/load): ").strip().lower()
    benchmark = get_benchmark()
    portfolio = get_user_portfolio()
    start_date, end_date = get_date_range()
    if mode in ["load", "l"]:
        price_data, benchmark_prices, weights_series = load_from_csv(benchmark)
    else:
        tickers = list(portfolio.keys())
        price_data = download_price_data(tickers, start_date, end_date)
        if price_data is None:
            print("Download process failed. Please try again.")
            return
        benchmark_prices = download_benchmark(benchmark, start_date, end_date)
        if benchmark_prices is None:
            print("Benchmark download failed. Please try again.")
            return
        save_weights(portfolio)
        price_data.to_csv('price_data_user.csv')
        benchmark_prices.to_csv('benchmark_prices.csv')
        print("CSVs successfully downloaded.")
        weights_series = load_weights()
    initial_investment = get_initial_investment()
    price_data = load_price_data(weights_series)
    portfolio_returns, portfolio_value = calculate_portfolio(price_data, weights_series, initial_investment)
    benchmark_returns, benchmark_value = calculate_benchmark(benchmark_prices, initial_investment)
    drawdown, max_drawdown = calculate_drawdown(portfolio_value)
    bench_drawdown, bench_max_drawdown = calculate_drawdown(benchmark_value)
    risk_free_rate = get_risk_free_rate()
    beta, correlation = calc_beta_and_corr(portfolio_returns, benchmark_returns)
    current_value, annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio = (
        calculate_metrics(portfolio_returns, portfolio_value, risk_free_rate)
    )
    bench_current_value, bench_annualized_return, bench_annualized_volatility, bench_sharpe_ratio, bench_sortino_ratio = (
        calculate_metrics(benchmark_returns, benchmark_value, risk_free_rate)
    )
    print_results(current_value, annualized_return, annualized_volatility, max_drawdown, sharpe_ratio,
                  bench_current_value, bench_annualized_return, bench_annualized_volatility,
                  bench_max_drawdown, bench_sharpe_ratio, risk_free_rate, sortino_ratio, bench_sortino_ratio, beta, correlation)
    plot_dashboard(portfolio_value, benchmark_value, drawdown, portfolio_returns, start_date, end_date)
    save_report(portfolio_value, benchmark_value, drawdown, portfolio_returns,
                start_date, end_date, current_value, annualized_return,
                annualized_volatility, max_drawdown, sharpe_ratio, sortino_ratio,
                beta, correlation, bench_current_value, bench_annualized_return,
                bench_annualized_volatility, bench_max_drawdown, bench_sharpe_ratio,
                bench_sortino_ratio, risk_free_rate, benchmark, portfolio, weights_series)

main()