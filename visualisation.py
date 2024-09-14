from settings import FILE_PATH
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class WebApp(object):
    def __init__(self):
        pass

        self.run()

    def summary(self):
        st.write("Text that explains the approach being taken")

    def methodology(self):

        # stocks are divided into sectors (GICS) to reduce dimensionality / spurious corr

        # universe of eligible pairs are identified by means of long term correlation among each sector group


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
        df = pd.read_excel(f"{FILE_PATH}//strategies//index_test4.xlsx", index_col=0, parse_dates=True)
        st.line_chart(df)

    def limitations(self):
        st.write("No access to historical comp of indices \n "
                 "Transaction cost v rough"
                 "")

    def run(self):

        pg = st.navigation([st.Page(self.summary),
                            st.Page(self.methodology),
                            st.Page(self.results),
                            st.Page(self.limitations)
                            ],
                           )

        pg.run()
