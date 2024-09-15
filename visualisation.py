from settings import FILE_PATH
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import settings
import plotly.express as px
import plotly.graph_objects as go

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
        [f(pd.read_csv(f"{FILE_PATH}//strategies/{name}/index.csv", index_col=0, parse_dates=True)["Index"]) for name in
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

    def summary(self):
        st.write("Text that explains the approach being taken")
        pass

    def results(self):

        st.header("Parameter Sensitivity")

        st.markdown(
            r"""
            Exploring below the pair-trading strategy sensitivity to a range of parameters:
            - Rolling window $w$ used to estimate pairwise correlations
            - Model used to estimate the covariance matrix 
            - Model used to estimate $\beta$
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
        - It is possible to easily expand the analysis to other indices by populating the relevant index settings in data_downloader.py  
            """
        )

        st.subheader(r"Step1: Pair Candidates")

        st.markdown(r"""
                   **Filtering**.
                   """)

        st.markdown(
            r"""The first step is to find stock pairs that are candidate for a pair trading strategy - e.g. pairs that tend to co-move together."""
        )
        st.markdown(r"""
            There are many ways to do this (mininum distance metrics, cointegration tests)... For simplicity I chose to focus here on estimating correlations.  
            Due to the large dimension of the problem [pair combinations: $N*(N-1)/2$ ~ order 100,000] relative to the size of the training period (typically few months), 
            I used the following strategy to reduce the number of candidates: 
            - Stocks in the universe are grouped in clusters. A cluster is defined here as GIC sector groups 
            - Correlations of demeaned daily log returns over a period $T^{L}$ are estimated on a rolling basis
            - A "triple-quantile" filter is applied to each correlation matrix at each time $t$:
                - For a stock $i$, consider stocks $j$ within the top $q$ quantile of its pairwise corr 
                - For a given cluster $k$, retain the pairs within the top $q$ quantile of correlations in the cluster
                - Aggregate all clusters and retain the pairs within the top $q$ quantile of correlations across clusters
                """)

        st.markdown(r"""
            Rationale is: 
            - 1st quantile pass: Avoid running too many correlated trades
            - 2nd quantile pass: select best in class
            - 3rd quantile pass: retain sectorial diversification
            """
        )

        # universe of eligible pairs are identified by means of long term correlation among each industry group
        df = pd.read_csv(f"{FILE_PATH}\\data\\S&P500_classification.csv", index_col=0)
        x = df["GIC_sector"].groupby(df.GIC_sector).count()
        fig, axes = plt.subplots(figsize=(10, 4))
        x.sort_values(ascending=False).plot.bar(ax=axes)
        axes.set_title("Number of stocks per GIC sector")
        st.pyplot(fig)

        st.markdown(r"""
                **Correlation Estimate**.
                """)
        st.markdown(r"""
        Sample correlation is a notoriously noisy measure, especially as $N$ becomes large relative to $T$. The backtester supports more robust correlation estimates: 
        - Exponentially weighted moving average correlation (EWMCorrelation)
        - Linear shrinkage estimators (LedoitWolfShrinkage, OracleApproximatingShrinkage) 
        """)

        st.subheader(r"Step2: Entry Signal")

        st.markdown(r"""
                        **Dislocation / Mean-reversion signal**.
                        """)
        st.markdown(r"""
        The pairs obtained in the previous step are candidates for inclusion in the strategy. A signal assesses dislocation among highly correlated pairs: 
        - Log returns of pair $(i,j)$ are regressed on the rolling window $T_{L}$: 
        $S_{i,t} = \beta_{i,j,t}*S_{j,t} + \varepsilon_{i,j,t}$  
        - An EWMA of the short-term spread (e.g. residuals) $\hat{X_{t}} = EWMA(S_{i,t} - \beta_{i,j,t}*S_{j,t} )$ gives an indication of dislocation between pair $i$ and $j$. The EWMA is estimated using a window $T_{S}=0.5*T_{L}$   
        - The spread is standardised on a rolling basis (window=$T_{S}$) to form a z-score
        - The z-score is mapped to a continuous signal $\widehat{MR_{t}} = f(\hat{X_{t}}) \in [0,1]$ via a logistic transformation $f$ 
                        """)

        st.markdown(r"""
                       **Illustration of the entry signal**.
                       """)

        st.write("This is a band-stop filter on the z-score")

        fig, axes = plt.subplots(figsize=(8, 4))

        axes.set_title("Filter")

        domain = np.arange(-4, 4, 0.10)
        L = pd.Series({d1: (2 * scipy.stats.norm.cdf(d1) - 1) for d1 in domain})
        F = pd.Series({d1: (1 - scipy.stats.norm.pdf(d1) / scipy.stats.norm.pdf(0)) for d1 in domain})
        df = pd.DataFrame(dict(L=L, F=F, S=L * F))
        df.plot(ax=axes)
        st.pyplot(fig)


    def example(self):
        pass

    def limitations(self):
        st.write("No access to historical comp of indices \n "
                 "Transaction cost v rough"
                 "GIC industry -> would be better to use"
                 "")

    def run(self):

        pg = st.navigation([st.Page(self.methodology),
                            st.Page(self.example),
                            st.Page(self.results),
                            # st.Page(self.limitations)
                            ],
                           )
        #
        pg.run()
        # self.methodology()
        # self.results()

def main():
    WebApp()

if __name__ == '__main__':
    main()