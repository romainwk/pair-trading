# from settings import FILE_PATH
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import strategy_runner
import settings

def lineplot(df, default, key, type="Scatter"):
    clist = df.columns.tolist()
    strategies = st.multiselect("Select strategy", clist, default =default, key=key)
    st.text("You selected: {}".format(", ".join(strategies)))
    dfs = {strat: df[strat] for strat in strategies}
    fig = go.Figure()
    for strat, x in dfs.items():
        fig = fig.add_trace(getattr(go,type)(x=df.index, y=x, name=strat))
    return st.plotly_chart(fig)

def get_df(iteration):
    f = lambda x: x[~x.index.duplicated(keep="first")]
    names = [f"{s.get("folder")}/{s.get("strategy_name")}" for s in iteration]
    df = pd.concat(
        [f(pd.read_csv(f"strategies/{name}/index.csv", index_col=0, parse_dates=True)["Index"]) for name in
         names], axis=1, keys=[n.split("/")[-1] for n in names])
    return df

def performance_metrics(df):
    def _sr(w):
        log_r = np.log(df.div(df.shift(w))).mean()
        vol = np.log(df.div(df.shift(w))).std()
        sr = log_r/vol * np.sqrt(252/w)
        return sr

    def _annualised_ret():
        T = (df.index[-1] - df.index[0]).days/365.25
        r = (df.iloc[-1] / df.iloc[0])**(1/T)-1
        return r

    def _calmar_ratio(w):
        log_r = np.log(df.div(df.shift(w)))
        cond_vol = log_r[log_r<=0].std()
        calmar = log_r.mean() / cond_vol * np.sqrt(252 / w)
        return calmar

    def _cvar(w, q):
        r= df.pct_change(w)
        q = r.quantile(q)
        cvar = r[r <= q].mean()
        return cvar

    def _max_dd_over_w(w):
        return df.pct_change(w).min()

    def __max_dd_over_hist(s):
        mdd = 0
        peak = s.iloc[0]
        for x in s:
            if x > peak:
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return -mdd

    def _max_dd_over_hist():
        return pd.Series({u: __max_dd_over_hist(df[u]) for u in df.columns})

    metrics = {"Annualised Ret":_annualised_ret(),
               "Realised Vol": np.log(df.div(df.shift(1))).std()*np.sqrt(252),
               "Sharpe Ratio (daily)": _sr(w=1),
               "Sharpe Ratio (weekly)":_sr(w=5),
               "Sharpe Ratio (monthly)": _sr(w=20),
               "Calmar Ratio":_calmar_ratio(w=20),
               "Max DD over 1W": _max_dd_over_w(w=5),
               "Max DD (peak to trough)": _max_dd_over_hist(),
               "C-VaR(5%, 1M)": _cvar(w=20, q=0.05),
    }
    metrics = pd.DataFrame(metrics).T.round(2)
    return metrics


class WebApp(object):
    def __init__(self):
        pass

        self.run()

    def sensitivity_analysis(self):

        st.header("Parameter Sensitivity")

        st.markdown(
            r"""
            Exploring below the pair-trading strategy sensitivity to a range of parameters:
            - Rolling window $T_{L}, T_{S}$ used to estimate pairwise correlations
            - Model used to estimate the covariance matrix  $\hat{\Sigma}$
            - Model used to estimate the hedge ratios $\beta$
            - Sensitivity to the quantile threshold $q$
            - Sensitivity to the rebalancing frequency
            - Exit conditions: Max holding period, take profit, stop loss..
            """
        )


        # Rolling window
        st.subheader(r"Sensitivity to the rolling windows " + r"$T_{L}$" + ", " + "$T_{S}$")
        st.write(r"$T_{L}$" + " is used to estimate the correlation matrix $\hat{\Sigma}$. To remove a degree of freedom, I fix the dislocation/mean-reversion signal $T^{S}=1/2T^{L}$")

        df = get_df(iteration=settings.iterations1)
        lineplot(df, default=df.columns[::4],key ="Window")

        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=["Sharpe Ratio (weekly)", "Calmar Ratio"],key ="Window Perf Metrics", type="Bar")
        st.dataframe(perf_metrics)

        # Covariance matrix
        st.subheader(r"Sensitivity to the Var-Covar matrix estimation " +r"$\hat{\Sigma}$")

        st.markdown(r"$\hat{\Sigma}^{Sample}$ is the MLE estimator of covariance (unbiased) but suffers from high variance.")
        st.markdown(r"$\hat{\Sigma}^{EWM}$ can be viewed as a Kalman filter representation (see Roncalli & Bruder, Trend Filtering Methods for Momentum Strategies, 2010).")
        st.markdown(r"$\hat{\Sigma}^{Ledoit}$ is a linear shrinkage estimator that regularise correlations towards the mean")
        st.markdown(r"$\hat{\Sigma}^{OAS}$ is another linear shrinkage estimator improving Ledoit estimator (assuming Gaussian distr)")

        df = get_df(iteration=settings.iterations2)
        lineplot(df, default=df.columns,key ="Covariance Matrix")

        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=["Sharpe Ratio (weekly)", "Calmar Ratio"],key ="Covariance Matrix Metrics", type="Bar")
        st.dataframe(perf_metrics)

        # beta estimation
        st.subheader(r"Sensitivity to the hedge ratio " + r"$\beta$")
        st.write(r"For each pair $(i,j)$ the hedge ratio $\beta_(i,j)$ is estimated to determine the units of stock $i$ to be traded against stock $j$")
        st.write(r"The rolling OLS approach uses the same window $T_{L}$ used to estimate the variance covariance matrix")
        st.write(r"The Kalman filtering approach embeds a prior on $\beta_(i,j)$ ~N(1,1) [returns are standardised prior to running the filter]")

        df = get_df(iteration=settings.iterations3)
        lineplot(df, default=df.columns, key="Hedge Ratio")

        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=["Sharpe Ratio (daily)", "Sharpe Ratio (weekly)", "Sharpe Ratio (monthly)"], key="Hedge Ratio Metrics", type="Bar")
        # st.dataframe(perf_metrics)

        # quantile threshold
        st.subheader(r"Sensitivity to the quantile threshold $q$")
        st.write(r"The quantile $q$ is used in the 'triple quantile filtering' recipe used to reduce the dimension of the correl matrices")

        df = get_df(iteration=settings.iterations4)
        # lineplot(df, default=df.columns, key="Quantile threshold")

        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=["Sharpe Ratio (daily)", "Sharpe Ratio (weekly)", "Sharpe Ratio (monthly)"], key="quantile_threshold", type="Bar")
        # st.dataframe(perf_metrics)

        default = ["Sharpe Ratio (daily)", "Sharpe Ratio (weekly)"]
        st.subheader(r"Sensitivity to the rebalancing frequency")
        key = "reb_freq"

        df = get_df(iteration=settings.iterations5)
        # lineplot(df, default=df.columns, key=key)
        #
        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=default, key=key, type="Bar")
        #
        # # max holding period
        st.subheader(r"Sensitivity to the max holding period")
        key = "max_holding_period"
        df = get_df(iteration=settings.iterations6)
        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=default, key=key, type="Bar")

        st.subheader(r"Sensitivity to the profit taking threshold")
        key = "profit_taking"
        df = get_df(iteration=settings.iterations7)
        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=default, key=key, type="Bar")

        st.subheader(r"Sensitivity to the stop loss threshold")
        key = "stop_loss"
        df = get_df(iteration=settings.iterations8)
        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=default, key=key, type="Bar")

        st.subheader(r"Sensitivity to cost assumption")
        key = "cost"
        df = get_df(iteration=settings.iterations9)
        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=default, key=key, type="Bar")

        # perf_metrics = performance_metrics(df)
        # lineplot(perf_metrics.T, default=["Sharpe Ratio (weekly)", "Calmar Ratio"], key=key, type="Bar")
        # # st.dataframe(perf_metrics)


    def methodology(self):
        st.header("Methodology")

        st.subheader(r"Dataset")
        st.text("")
        st.markdown(r"""
        The reference index is the S&P 500.
        - The constituent list is borrowed from Wikipedia
        - Historical prices are downloaded from Yahoo via the API (see data_downloader.py) 
        - Prices are adjusted for splits and dividends
        - However due to data access constraint, the dataset doesn't account for index composition change (normally retrieved via BBG MEMB)
        - The analysis can be easily expanded to other indices by populating the relevant index settings in data_downloader.py  
            """
        )

        st.subheader(r"Step1: Pair Candidates")

        st.markdown(r"""
                   **Correlation Filtering**
                   """)

        st.markdown(
            r"""The first step is to find stock pairs that are candidate for a pair trading strategy - e.g. pairs that tend to co-move together."""
        )
        st.markdown(r"""
            There are many ways to do this (mininum distance metrics, cointegration tests)... For simplicity I chose to focus here on estimating correlations.  
            Due to the large dimension of the problem [pair combinations: $N*(N-1)/2$ ~ order 100,000] relative to the size of the training period (typically few weeks or months), 
            I used the following strategy to reduce the number of candidates: 
            - Stocks in the universe are grouped in clusters. A cluster is defined here as GIC sector groups 
            - Correlations of demeaned daily log returns over a period $T_{L}$ are estimated on a rolling basis
            - A "triple-quantile" filter is applied to each correlation matrix at each time $t$:
                - For a stock $i$, consider stocks $j$ within the top $q$ quantile of its pairwise correlation 
                - For a given cluster $k \in K$, retain the pairs within the top $q$ quantile of correlations in the cluster
                - Aggregate all clusters and retain the pairs within the top $q$ quantile of correlations across $K$ clusters
                """)

        st.markdown(r"""
            Rationale is: 
            - 1st quantile pass: Avoid running too many correlated trades
            - 2nd quantile pass: select best in class
            - 3rd quantile pass: retain sectorial diversification
            The same $q$ is kept through all 3 passes to reduce overparametrisation.
            """
        )

        # universe of eligible pairs are identified by means of long term correlation among each industry group
        df = pd.read_csv(f"data//S&P500_classification.csv", index_col=0)
        x = df["GIC_sector"].groupby(df.GIC_sector).count()
        fig, axes = plt.subplots(figsize=(10, 4))
        x.sort_values(ascending=False).plot.bar(ax=axes)
        axes.set_title("Number of stocks per GIC sector")
        st.pyplot(fig)

        st.markdown(r"""
                **Correlation Estimators**.
                """)
        st.markdown(r"""
        Sample correlation is a notoriously noisy measure, especially as $N$ becomes large relative to $T$. The backtester supports smoother correlation estimates: 
        - Exponentially weighted moving average correlation (EWMCorrelation)
        - Linear shrinkage estimators (LedoitWolfShrinkage, OracleApproximatingShrinkage) 
        """)

        st.subheader(r"Step2: Entry Signal")

        st.markdown(r"""
                        **Dislocation / Mean-reversion signal**.
                        """)
        st.markdown(r"""
        The pairs obtained in the previous step (high correlation) are candidates for inclusion in the strategy. 
        """)
        st.markdown(r"""
        A signal (Entry Signal) assesses dislocation among highly correlated pairs and decides which pairs are to be included. The signal is obtained through these steps:   
        - $\forall (i,j) \in Candidates_{t}$, the log returns of pair $(i,j)$ are regressed on the rolling window $T_{L}$:    
        $S_{i,t} = \beta_{i,j,t}*S_{j,t} + \varepsilon_{i,j,t}$  
        - An EWMA of the short-term spread (e.g. residuals) $\hat{X_{t}} = EWMA(S_{i,t} - \beta_{i,j,t}*S_{j,t} )$ gives an indication of dislocation between pair $i$ and $j$. The EWMA is estimated using a window $T_{S}=0.5*T_{L}$   
        - The spread is standardised on a rolling basis ($window=T_{S}$) to form a z-score $\hat{Z_{t}}$
        - The z-score $\hat{Z_{t}}$ is mapped to a continuous entry signal $\widehat{V_{t}} = f(\hat{Z_{t}}) \in [0,1]$ via a logistic transformation $f$
                        """)

        st.markdown(r"""
                       **Illustration of the entry signal**.
                       """)

        st.markdown(r"""$\beta_{i,j,t}$ is estimated by Kalman filtering:""")

        path = f"strategies/base/baseline"
        df = pd.read_csv(f"{path}/HR.csv", index_col=0, parse_dates=True, header=[0,1]).iloc[100:]
        df.columns = [f"{i}, {j}" for i, j in zip(df.columns.get_level_values(0), df.columns.get_level_values(1))]
        lineplot(df, default=["BAC, JPM", "BAC, C"], key="kalman")

        st.markdown(r"""The z-score $\hat{Z_{t}}$ of the spread (regression residuals) is mapped to a dislocation / mean-reversion signal $\in [0,1]$""")
        st.markdown(r"""$L_t=2\Phi(\hat{Z_{t}})-1$""")
        st.markdown(r"""$F_t=(1-\phi(\hat{Z_{t}})/(\phi(0)))$""")
        st.markdown(r"""$V_t=L_tF_t$""")

        st.markdown(r"""This applies a band-stop filter so that low absolute values of z-scores are discarded""")

        fig, axes = plt.subplots(figsize=(8, 4))
        axes.set_title("Filter")

        domain = np.arange(-4, 4, 0.10)
        L = pd.Series({d1: (2 * scipy.stats.norm.cdf(d1) - 1) for d1 in domain})
        F = pd.Series({d1: (1 - scipy.stats.norm.pdf(d1) / scipy.stats.norm.pdf(0)) for d1 in domain})
        df = pd.DataFrame(dict(L=L, F=F, V=L * F))
        df.plot(ax=axes)
        axes.set_xlabel(r"z-score $\hat{Z_{t}}$")
        axes.axhline(0, linewidth=0.5, linestyle="--")
        axes.set_ylabel(r"$Entry signal$")
        st.pyplot(fig)

        st.markdown(r"""**Entry Signal example among correlated pairs:**""")
        df = pd.read_csv(f"{path}/mean_reversion_signal.csv", index_col=0, parse_dates=True, header=[0, 1]).iloc[100:]
        df.columns = [f"{i}, {j}" for i, j in zip(df.columns.get_level_values(0), df.columns.get_level_values(1))]
        lineplot(df, default=["BAC, JPM", "BAC, C"], key="mr_signal")

        st.subheader(r"Step3: Exit Signal and Portfolio Construction")

        st.markdown(r"""**Exit Signal:**""")
        st.markdown(r"""At each portfolio rebalance date, the portfolio includes a new set of trades given by the Entry Signal $V_t$ from Step2""")
        st.markdown(r"""
        Exit is triggered when the earliest of 4 conditions applies:
        - The dislocation signal $V_t$ reaches **signal_threshold_exit**
        - The **max_holding_period** is reached 
        - The **profit_taking** target is reached
        - The **stop_loss** target is reached
        
        """)

        st.markdown(r"""**Portfolio Construction:**""")
        st.markdown(r"""
        - The entry size is proportional to the strength of the Entry Signal $V_t$
        - The leverage can be set to hit a gross notional target (**notional_sizing="TargetNotional"**) or a realised volatility target (**notional_sizing="TargetVol"**)
        - Trades are generally overlapping (the next rebalance date precedes the exit date from the previous trades)
        """)

        st.markdown(r"""**Transaction costs:**""")
        st.markdown(r"""
        In the absence of order book level data, costs are estimated using an estimate of realised volatility (EWM, $window=T_{L}$
        """)
        st.markdown(r"""
        Costs are set with **transaction_cost** as a multiple of daily standard deviation.
        While this is a very rough estimate, Sarkissian (2016) gives some theoretical support by showing how the bid-ask spread can be related to the underlying volatility by expressing it as a function of straddle premium
        """)

    def interactive_strategy(self):
        st.header("Interactive Strategy")
        st.subheader("Parameters choice")

        param_choices = dict(correlation_estimate=("SampleCorrelation", "EWMCorrelation", "LedoitWolfShrinkage", "OracleApproximatingShrinkage"),
                             hedge_ratio_estimate=("RollingOLS", "KalmanFilter"),
                             correlation_window=(60,90,120,150),
                             correlation_quantile=(0.05, 0.10, 0.15, 0.20),
                             mean_reversion_window=(10,20,30,45,60),
                             rebal_frequency=(5,10,15,20,),  # how frequently a new set of pairs is considered
                             max_holding_period=(1,5,10,15,20),
                             profit_taking=(None, 0.01,0.02,0.03,0.04,0.05),
                             stop_loss=(None, 0.01,0.02,0.03,0.04,0.05),

                             notional_sizing=("TargetNotional", "TargetVol"),  # TargetNotional, TargetVol
                             leverage=(0.5,1,2,4),  # gross leverage of L/S strategy if sizing by TargetNotional
                             transaction_cost=(0, 0.5 * 1 / np.sqrt(252),  1 / np.sqrt(252), 2 / np.sqrt(252)),
                             )

        default =  dict(correlation_estimate=1,
                             hedge_ratio_estimate=1,
                             correlation_window=1,
                             correlation_quantile=1,
                             mean_reversion_window=3,
                             rebal_frequency=0,  # how frequently a new set of pairs is considered
                             max_holding_period=2,
                             profit_taking=0,
                             stop_loss=0,

                             notional_sizing=0,  # TargetNotional, TargetVol
                             leverage=2,  # gross leverage of L/S strategy if sizing by TargetNotional
                             transaction_cost=0,
                             )
        options = {}
        for param, choice in param_choices.items():
            options[param] = st.selectbox(param, choice, index=default[param])

        strat_settings = settings.get_settings(dict(strategy_name="online_strategy", folder="online_strategy", debug = True))
        strat_settings.update(options)
        strategy_runner.strategy_runner(strat_settings)

        st.subheader("Strategy performance")

        path = f"strategies/online_strategy/online_strategy"
        df = pd.read_csv(f"{path}/index.csv", index_col=0, parse_dates=True)["Index"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df))
        fig.update_layout(
            title="Strategy performance"
        )
        st.plotly_chart(fig)

        perf_metrics = performance_metrics(pd.DataFrame(df).rename({0:"Index"},axis=1))
        st.dataframe(perf_metrics)

        st.subheader("Strategy features")
        df = pd.read_csv(f"{path}/portfolio_composition.csv", index_col=0, parse_dates=True)
        x= pd.DataFrame(df["GIC_sector"].value_counts()).rename(dict(count="Number of pairs traded"))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x.index, y=x["count"]))
        fig.update_layout(
            title="Number of pair traded per sector"
        )
        st.plotly_chart(fig)

        df["Pair"] = [f"{x}, {y}" for x, y in zip(df["Pair1"], df["Pair2"])]
        top = df["Pair"].groupby(df.Pair).count().sort_values(ascending=False).iloc[:10]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=top.index, y=top))
        fig.update_layout(
            title="Top 10 most traded pairs"
        )
        st.plotly_chart(fig)

        fig, axes = plt.subplots(figsize=(10, 4))
        df["HR"].hist(ax=axes, bins=250, alpha=0.75, color="darkblue")
        axes.set_title("Distribution of Hedge Ratio at Entry Date")
        st.pyplot(fig)

        fig, axes = plt.subplots(figsize=(10, 4))
        df["Correlation"].hist(ax=axes, bins=250, alpha=0.75)
        axes.set_title("Distribution of Correlation at Entry Date")
        st.pyplot(fig)

        # portfolio
        def _count_exit(field):
            x = df.groupby(["Date", "Pair1", "Pair2"])[field].sum()
            return x[x > 0].count()

        df = pd.read_csv(f"{path}/portfolio.csv", index_col=0, parse_dates=True)

        exits = {"Take Profit trigger":_count_exit("ProfitTaking"),
        "Stop Loss trigger": _count_exit("StopLoss"),
         "Exit Signal": _count_exit("ExitSignal"),
         "MaxHoldingPeriod": _count_exit("MaxHoldingPeriod"),

         }

        exits = pd.Series(exits).sort_values(ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=exits.index, y=exits))
        fig.update_layout(
            title="Trade Exit breakdown"
        )
        st.plotly_chart(fig)

        st.subheader("Portfolio view")
        st.dataframe(df.iloc[-50:])


    def limitations(self):
        st.write("No access to historical comp of indices \n "
                 "Transaction cost v rough"
                 "GIC industry -> would be better to use"
                 "")

    def run(self):

        pg = st.navigation([st.Page(self.methodology),
                            st.Page(self.interactive_strategy),
                            st.Page(self.sensitivity_analysis),
                            ],
                           )

        pg.run()

def main():
    WebApp()

if __name__ == '__main__':
    main()