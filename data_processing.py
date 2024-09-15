# from settings import FILE_PATH
import pandas as pd
import streamlit as st

class Data(object):
    '''
    Data processing class - very light here given time constraint, but in practice would handle:
        - corporate actions, splits, mergers etc
        - chg in index composition over time + listing, delisting etc
        - GIC industry classification

    '''
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)
        with st.status("Processing data..."):
            self.run()

    def _load(self):
        price = pd.read_csv(f"data//{self.index_universe}_data.csv", index_col=0, parse_dates=True)
        price.index = price.index.tz_localize(None)
        price = price.reindex(self.schedule.training_dates)
        self.classification = pd.read_csv(f"data//{self.index_universe}_classification.csv")
        self.price = price

    def _create_clusters(self):
        classification = self.classification[self.cluster_by]
        stocks_per_cluster = classification.value_counts()
        retained = stocks_per_cluster[stocks_per_cluster>=self.min_stock_per_cluster]
        self.clusters = {s: sorted(self.classification.query(f"{self.cluster_by}=='{s}'").Ticker) for s in retained.index}

    def run(self):
        self._load()
        self._create_clusters()





