# Portfolio Optimization Project

## 📜 Project Overview

This project provides a comprehensive analysis, visualization, and optimization of a stock portfolio. The core objective is to select an optimal portfolio from a set of possibilities to **maximize returns and minimize risks**. It leverages key Python libraries like pandas, numpy, matplotlib, seaborn, and plotly to handle data processing, analysis, and visualization. [cite_start]The project also introduces the concept of the **Sharpe Ratio** as a critical metric for evaluating risk-adjusted returns and uses **Monte Carlo simulations** to find the most efficient portfolio allocation[cite: 15, 607, 684].

---

## 🛠️ Key Concepts & Tools

### **Financial Concepts**
* [cite_start]**Portfolio:** A collection of financial assets, such as stocks, bonds, cash, and real estate, used to diversify investments and manage risk[cite: 13, 14].
* **Percentage Daily Return:** The percentage gain or loss for a stock from one day to the next. [cite_start]It's calculated using the formula: $r_{t}=\frac{p_{t}-p_{t-1}}{p_{t-1}}\times100$[cite: 138, 139, 141].
* **Sharpe Ratio:** A measure of risk-adjusted return. It helps investors determine the return of an investment in comparison to its risk. [cite_start]A higher Sharpe Ratio indicates a better risk-adjusted return[cite: 606, 614, 630]. [cite_start]The formula is: $Sharpe~Ratio=\frac{R_{p}-R_{f}}{\sigma_{p}}$, where $R_{p}$ is the portfolio return, $R_{f}$ is the risk-free rate, and $\sigma_{p}$ is the portfolio's standard deviation (volatility)[cite: 607, 609, 610, 612].
* **Monte Carlo Simulation:** A computational technique that runs multiple trials with random inputs to model the probability of different outcomes. [cite_start]In this project, it is used to generate random portfolio weights and observe the resulting returns, risks, and Sharpe Ratios[cite: 684, 686].
* **Markowitz Efficient Frontier:** A set of optimal portfolios that offer the highest expected return for a given level of risk. [cite_start]This theory, introduced by Harry Markowitz in 1952, is visualized by plotting volatility (risk) on the x-axis and expected return on the y-axis[cite: 727, 728, 729, 730].

---

### **Python Libraries**
The project utilizes the following libraries for its functionality:

* [cite_start]**pandas:** For data manipulation and analysis, particularly for handling stock data in DataFrames[cite: 68].
* [cite_start]**numpy:** For numerical operations and array manipulation, especially for calculations involving portfolio weights and metrics[cite: 69].
* [cite_start]**matplotlib:** A foundational library for creating static and interactive data visualizations[cite: 226, 227].
* [cite_start]**seaborn:** Built on top of matplotlib, it provides a high-level interface for drawing aesthetically pleasing statistical graphics[cite: 233, 236].
* [cite_start]**plotly express:** A user-friendly, high-level interface for plotly that creates interactive figures with minimal code, perfect for financial data visualization[cite: 240, 251].
* [cite_start]**cufflinks:** A library that links pandas with plotly, enabling interactive plots to be generated directly from pandas DataFrames[cite: 347].

---

## 🚀 Project Steps

1.  **Data Acquisition & Preparation:** Import necessary libraries and datasets. Stock data (e.g., `AMZN.csv`, `stock_prices.csv`) is read into a pandas DataFrame. [cite_start]The project defines key stock market terms like Open, High, Low, Close, Volume, and Adjusted Close prices[cite: 48, 49, 53, 54, 55, 56, 57].

2.  [cite_start]**Daily Returns Calculation:** Calculate the percentage daily return for each stock using the `pct_change()` method[cite: 150, 151].

3.  [cite_start]**Single Stock Visualization:** Visualize single-stock data using various plots, including line plots for adjusted close prices and daily returns, pie charts to classify returns into trends, and candlestick graphs to show price points (open, high, low, and close) for a given period[cite: 271, 299, 328, 329].

4.  **Multi-Stock Analysis & Visualization:** Extend the analysis to a portfolio of multiple stocks. Visualize closing prices and daily returns for the entire portfolio. [cite_start]This section also explores correlations between stocks using a heatmap and a pair plot[cite: 385, 388, 393, 398].

5.  [cite_start]**Random Weights Generation:** Define a function to generate random portfolio weights that sum up to 1, representing different allocation scenarios[cite: 447, 451].

6.  **Asset Allocation & Metrics Calculation:** A function is created to allocate an initial investment based on the random weights. It calculates the portfolio's daily value and daily return. [cite_start]The project also defines a **simulation engine** to compute key metrics like expected annual return, volatility, and Sharpe Ratio for each portfolio[cite: 544, 646].

7.  **Monte Carlo Simulation:** Run thousands of simulations with randomly generated weights to explore a wide range of portfolio combinations. [cite_start]The results (returns, volatility, and Sharpe Ratios) are stored for further analysis[cite: 697].

8.  **Portfolio Optimization:** Identify the optimal portfolio by finding the one with the **highest Sharpe Ratio** from all the simulation runs. [cite_start]The results are visualized on a scatter plot, illustrating the efficient frontier and highlighting the optimal portfolio[cite: 727, 731, 767, 773].

---

## 📈 Sample Results

The following are examples of statistical metrics obtained from a sample simulation run.

* **Expected Portfolio Annual Return:** 16.30%
* **Portfolio Standard Deviation (Volatility):** 18.02%
* **Sharpe Ratio:** 0.74
* **Portfolio Final Value:** $3,203,874.76
* **Return on Investment:** 220.39%
* [cite_start]*Note: These values are for a portfolio with equal weights and a risk-free rate of 3%[cite: 834, 835, 836, 837, 838]. The results may vary based on the data and random simulations.*
