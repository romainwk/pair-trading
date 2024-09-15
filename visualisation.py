from settings import FILE_PATH
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import settings
import plotly.express as px
import plotly.graph_objects as go

def lineplot(df, default):
    clist = df.columns.tolist()
    strategies = st.multiselect("Select strategy", clist, default =default)
    st.text("You selected: {}".format(", ".join(strategies)))
    dfs = {strat: df[strat] for strat in strategies}
    fig = go.Figure()
    for strat, x in dfs.items():
        fig = fig.add_trace(go.Scatter(x=df.index, y=x, name=strat))
    return st.plotly_chart(fig)

def get_df(iteration):
    names = [f"{s.get("folder")}/{s.get("strategy_name")}" for s in iteration]
    df = pd.concat(
        [pd.read_csv(f"{FILE_PATH}//strategies/{name}/index.csv", index_col=0, parse_dates=True)["Index"] for name in
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

        st.write("Showing here the main results of the strat")

        st.header("Parameter Sensitivity")

        st.subheader(r"Sensitivity to rolling window " + r"$w$")
        st.write(f"Sensitivity to the window " + r"$w$" + " used to estimate the correlation")

        df = get_df(iteration=settings.iterations1)
        lineplot(df, default=df.columns[::4])

        perf_metrics = performance_metrics(df)
        lineplot(perf_metrics.T, default=["Sharpe Ratio (weekly)", "Calmar Ratio"])
        st.dataframe(perf_metrics)


    def methodology(self):


        # universe of eligible pairs are identified by means of long term correlation among each industry group
        df = pd.read_csv(f"{FILE_PATH}\\data\\S&P500_classification.csv", index_col=0)

        # stocks are divided into sectors (GICS) to reduce dimensionality / spurious corr
        x = df["GIC_sub_industry"].groupby(df.GIC_sub_industry).count()

        fig, axes = plt.subplots(figsize=(25, 4))
        x.sort_values(ascending=False).plot.bar(ax=axes)
        # df.plot(ax=axes)
        st.pyplot(fig)

        # triple quantile sorting
        # # To reduce dimensionality, for each stock i, only consider a pair with stock j in the top quantile of its pairwise correlation
        # # Selected pair is given by top quantile among the group at each date
        # selected pairs to test for MR: top quantile among all groups each day
        print("")
        st.write("This is a band-stop filter on the z-score")

        fig, axes = plt.subplots(figsize=(8, 4))

        axes.set_title("Filter")

        domain = np.arange(-4, 4, 0.10)
        L = pd.Series({d1: (2 * scipy.stats.norm.cdf(d1) - 1) for d1 in domain})
        F = pd.Series({d1: (1 - scipy.stats.norm.pdf(d1) / scipy.stats.norm.pdf(0)) for d1 in domain})
        df = pd.DataFrame(dict(L=L, F=F, S=L * F))
        df.plot(ax=axes)
        st.pyplot(fig)



    def limitations(self):
        st.write("No access to historical comp of indices \n "
                 "Transaction cost v rough"
                 "GIC industry -> would be better to use"
                 "")

    def run(self):

        # pg = st.navigation([st.Page(self.summary),
        #                     st.Page(self.methodology),
        #                     st.Page(self.results),
        #                     st.Page(self.limitations)
        #                     ],
        #                    )
        #
        # pg.run()
        self.results()

def main():
    WebApp()

if __name__ == '__main__':
    main()