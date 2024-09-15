import pandas as pd
import numpy as np
from settings import FILE_PATH
from sklearn.covariance import OAS, LedoitWolf
import datetime
from numpy.lib.stride_tricks import sliding_window_view
import os
import logging
from joblib import Parallel, delayed
from collections import ChainMap

class CorrelationEstimator(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        self.clusters = self.data.clusters
        self.trading_dates = self.schedule.trading_dates
        self.rebal_dates = self.schedule.rebal_dates

        self.run()

    def _handle_missing_data(self, A):
        A = A.dropna(how="all", axis=1)
        A = A.where(A.notna(), A.median(), axis=1) # estimate remaining NAs with median over obs window
        return A

    def _gen_sliding_windows(self, X):
        sliding_windows = list(zip([t - datetime.timedelta(self.correlation_window) for t in self.rebal_dates],  self.rebal_dates))
        M = {t1: X.loc[t0:t1] for t0, t1 in sliding_windows}
        M = {t1: self._handle_missing_data(A) for t1, A in M.items()}
        M = {t: A for t, A in M.items() if len(A.columns)!=0}
        return M

    def SampleCorrelation(self, M):
        # return X.rolling(window=self.correlation_window, min_periods=self.correlation_window).corr()
        R = {t: X.corr() for t, X in M.items()}
        return R

    def EWMCorrelation(self, M):
        R = self.X.ewm(halflife=self.correlation_window, min_periods=self.correlation_window).corr()
        R = {t: R.xs(t) for t in M}
        R = {t: r.dropna(how="all", axis=1).dropna(how="all", axis=0) for t, r in R.items()}
        return R

    def LedoitWolfShrinkage(self, M):
        # def ledoit_wolf_shrinkage(A):
        #       '''Homemade estimation discarded since supported by scikit learn''
        #     # Ledoit Shrinking framework using the Constant Correlation Model (2004)
        #     # prior is that all pairwise correlations are identical. Shrinkage matrix F is simply the covariance matrix implied by constant correl
        #     A = A.dropna(how="all", axis=1)
        #     S = A.cov()
        #     var_X = np.diag(S).reshape(-1, 1)
        #     std_X = np.sqrt(var_X)
        #     rho = X.corr()
        #
        #     rho_upper_triu = rho.where(np.triu(np.ones(rho.shape)).astype(np.bool))
        #     avg_rho = rho_upper_triu.stack().mean()
        #
        #     # prior covariance matrix (avg corr shrinks upper and lower values of covariance)
        #     F = avg_rho * std_X * std_X.T
        #     np.fill_diagonal(F, var_X)
        #
        #     F = pd.DataFrame(F, index=S.index, columns=S.columns)
        #
        #     U = self.shrink_factor * F + (1 - self.shrink_factor) * S
        #     shrunk_corr = U / (std_X * std_X.T)
        #     return shrunk_corr

        def ledoit_wolf_shrinkage(X):
            lw = LedoitWolf(store_precision=False, assume_centered=True)
            lw.fit(X)
            S_hat = lw.covariance_
            S_hat = pd.DataFrame(S_hat, index=X.columns, columns=X.columns)
            V = np.diag(S_hat).reshape(-1, 1)
            sqrt_V = np.sqrt(V)
            rho = S_hat/(sqrt_V*sqrt_V.T)
            return rho

        R = {t: ledoit_wolf_shrinkage(X) for t, X in M.items()}
        return R

    def OracleApproximatingShrinkage(self, M):
        def oas_shrinkage(X):
            lw = OAS(store_precision=False, assume_centered=True)
            lw.fit(X)
            S_hat = lw.covariance_
            S_hat = pd.DataFrame(S_hat, index=X.columns, columns=X.columns)
            V = np.diag(S_hat).reshape(-1, 1)
            sqrt_V = np.sqrt(V)
            rho = S_hat/ (sqrt_V * sqrt_V.T)
            return rho
        R = {t: oas_shrinkage(X) for t, X in M.items()}
        return R

    def _filter_correlation(self, R):

        # upper triangular matrix above the diagonal to keep unique pairwise corr
        U = [np.triu(np.ones(X.shape[-2:]), k=1).astype(np.bool) for X in R.values()]
        A = [u*x for u, x in zip(U, R.values())]

        A = pd.concat(A, keys=R).replace({0:np.nan})

        A_flat = A.stack()
        A_flat = A_flat[A_flat>0]
        A_flat.name="Correlation"
        A_flat = pd.DataFrame(A_flat)

        # To reduce dimensionality, for each stock i, only consider a pair with stock j in the top quantile of its pairwise correlation
        R = pd.concat(R)

        top_corr_q_per_pair1 = R.quantile(axis=1, q=1 - self.correlation_quantile)
        # max_corr_per_pair = R[R < 1-epsilon].max(axis=1).dropna()
        top_corr_q_per_pair1.name = "TopQuantileCorr1"
        top_corr_q_per_pair2 = top_corr_q_per_pair1.copy()
        top_corr_q_per_pair2.name = "TopQuantileCorr2"

        A_flat = A_flat.join(top_corr_q_per_pair1, [A_flat.index.get_level_values(0), A_flat.index.get_level_values(1)], how="left").iloc[:,2:]
        A_flat = A_flat.join(top_corr_q_per_pair2, [A_flat.index.get_level_values(0), A_flat.index.get_level_values(2)], how="left").iloc[:,2:]
        A_flat = A_flat.query("Correlation>=TopQuantileCorr1 or Correlation>=TopQuantileCorr2")

        # Selected pair is given by top quantile among the group at each date
        lower_bound = A_flat.groupby(A_flat.index.get_level_values(0)).apply(lambda x: np.quantile(x, q=1 - self.correlation_quantile))
        lower_bound.name="LowBoundCorr"

        A_flat = A_flat.join(lower_bound, A_flat.index.get_level_values(0), how="left").iloc[:,1:]

        # Expected selected number of pairs is of order N*q/2 since each stock can  be matched with another one
        A_flat = A_flat.query("Correlation>=LowBoundCorr")
        return A_flat


    def _get_correlation_cluster(self, cluster):
        print(cluster)
        logging.info(f"computing correlation for {cluster}")
        stocks = self.data.clusters[cluster]
        price = self.data.price[stocks].ffill()

        X = np.log(price / price.shift(1))
        X = (X - X.rolling(self.correlation_window).mean())# center returns
        M = self._gen_sliding_windows(X)
        self.X=X

        R = getattr(self, self.correlation_estimate)(M)
        R = self._filter_correlation(R)
        # self._save(cluster, R)
        return {cluster: R}

    def _filter_across_clusters(self, rho):
        top_quantile =rho["Correlation"].groupby("Date").apply(lambda x: np.quantile(x, q=1 - self.correlation_quantile))
        top_quantile.name="TopQuantileCorrAllClusters"

        rho = rho.join(top_quantile, on="Date")
        rho = rho.query("Correlation>=TopQuantileCorrAllClusters")
        return rho

    def _format(self, rho):
        rho = rho.sort_values(by=["Date", self.cluster_by])
        rho[self.cluster_by] = rho.index.get_level_values(0)
        rho.index= rho.index.droplevel(0)
        return rho

    def _save(self, rho):
        directory = f"{FILE_PATH}\\strategies\\{self.strategy_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.rho.to_csv(f"{directory}\\{self.correlation_estimate}_{self.correlation_window}.csv")

    def run(self):
        # [self._get_correlation_cluster(c) for c in self.clusters]
        # R = {c: self._get_correlation_cluster(c) for c in self.clusters}
        # self.clusters=["Industrials"] #FIXME remove
        # self.n_parallel_jobs = 1

        res = Parallel(n_jobs=self.n_parallel_jobs)(delayed(self._get_correlation_cluster)(c) for c in self.clusters)
        res = dict(ChainMap(*res))
        rho = pd.concat(res.values(), keys=res)
        rho.index.names = [self.cluster_by, "Date", "Pair1", "Pair2"]
        rho = self._filter_across_clusters(rho)
        self.rho= self._format(rho)
        if self.export_data: self._save()
