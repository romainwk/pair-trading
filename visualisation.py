from settings import FILE_PATH
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import settings
import plotly.express as px
import plotly.graph_objects as go

class WebApp(object):
    def __init__(self):
        pass

        self.run()

    def summary(self):
        st.write("Text that explains the approach being taken")
        pass

    def plot(self, df):
        df = pd.DataFrame(px.data.gapminder())

        clist = df["country"].unique().tolist()

        countries = st.multiselect("Select country", clist)
        st.header("You selected: {}".format(", ".join(countries)))

        dfs = {country: df[df["country"] == country] for country in countries}

        fig = go.Figure()
        for country, df in dfs.items():
            fig = fig.add_trace(go.Scatter(x=df["year"], y=df["gdpPercap"], name=country))

        st.plotly_chart(fig)

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

    def results(self):
        st.write("Showing here the main results of the strat")

        st.write("This is the backtesting results")

        names = [s.get("strategy_name") for s in settings.iterations1[:5]]
        df = pd.concat([pd.read_csv(f"{FILE_PATH}//strategies/{name}/index.csv", index_col=0, parse_dates=True)["Index"] for name in names], axis=1, keys=names)
        # df = pd.read_excel(f"{FILE_PATH}//strategies//index_test4.xlsx", index_col=0, parse_dates=True)
        st.line_chart(df, width =2)

        clist = df.columns.tolist()

        strategies = st.multiselect("Select strategy", clist)
        st.header("You selected: {}".format(", ".join(strategies)))

        dfs = {strat: df[strat] for strat in strategies}

        fig = go.Figure()
        for strat, df in dfs.items():
            fig = fig.add_trace(go.Scatter(x=df.index, y=df[strat], name=strat))

        st.plotly_chart(fig)

        x = st.slider('x')  # 👈 this is a widget
        st.write(x, 'squared is', x * x)

        df = pd.DataFrame({
            'first column': [1, 2, 3, 4],
            'second column': [10, 20, 30, 40]
        })

        option = st.selectbox(
            'Which number do you like best?',
            df['first column'])

        fig, axes = plt.subplots(figsize=(8, 4))
        axes.plot(df)
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