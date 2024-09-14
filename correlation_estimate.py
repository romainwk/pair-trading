import pandas as pd
import numpy as np
from settings import FILE_PATH

class CorrelationEstimator(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        self.sectors = self.data.clusters
        self.trading_dates = self.schedule.trading_dates
        self.rebal_dates = self.schedule.rebal_dates

        self.run()

    def SampleCorrelation(self, X):
        return X.rolling(window=self.correlation_window, min_periods=self.correlation_window).corr()

    def EWMCorrelation(self, X):
        return X.ewm(halflife=self.correlation_window, min_periods=self.correlation_window).corr()

    def _ledoit_wolf_shrinkage(self, X, t0, t1):
        # Ledoit Shrinking framework using the Constant Correlation Model (2004)
        # prior is that all pairwise correlations are identical. Shrinkage matrix F is simply the covariance matrix implied by constant correl

        X = X.loc[t0:t1].dropna(how="all", axis=1)
        S = X.cov()
        var_X = np.diag(S).reshape(-1, 1)
        std_X = np.sqrt(var_X)
        rho = X.corr()

        rho_upper_triu = rho.where(np.triu(np.ones(rho.shape)).astype(np.bool))
        avg_rho = rho_upper_triu.stack().mean()

        # prior covariance matrix (avg corr shrinks upper and lower values of covariance)
        F = avg_rho * std_X * std_X.T
        np.fill_diagonal(F, var_X)

        F = pd.DataFrame(F, index=S.index, columns=S.columns)

        U = self.shrink_factor * F + (1 - self.shrink_factor) * S
        shrunk_corr = U / (std_X * std_X.T)
        return shrunk_corr

    def ShrunkCorrelation(self, X):
        sliding_windows = list(zip(self.trading_days - datetime.timedelta(self.correlation_window),  self.trading_days))
        corr_matrices = {t1: self._ledoit_wolf_shrinkage(X, t0, t1) for t0, t1 in sliding_windows}
        return corr_matrices

    def _get_corr_matrix(self, sector):

        stocks = self.data.clusters[sector]
        price = self.data.data[stocks]

        r = np.log(price / price.shift(1))
        r = (r - r.rolling(self.correlation_window).mean())  # / r.expanding().std() # returns are standardised
        corr_matrices = getattr(self, self.pair_selection_method)(X=r)
        return corr_matrices

    def _select_pairs(self, corr_matrix, epsilon=1e-2):

        corr_matrix = corr_matrix.loc[self.rebal_dates]

        # X = collection of corr matrices each day
        X = np.array([np.array(corr_matrix.xs(t)) for t in self.rebal_dates])
        # upper triangular matrix above the diagonal to keep unique pairwise corr
        U = np.triu(np.ones(X.shape[-2:]), k=1).astype(np.bool)
        A = X * U

        A = pd.DataFrame(A.reshape(corr_matrix.shape), index=corr_matrix.index, columns=corr_matrix.columns)

        B = A.stack()
        B = B[B>0]
        B.name="Correlation"
        B = pd.DataFrame(B)

        # To reduce dimensionality, for each stock i, only consider a pair with stock j == max(corr(i,k))
        max_corr_per_pair = corr_matrix[corr_matrix < 1-epsilon].max(axis=1).dropna()
        max_corr_per_pair.name = "MaxPairWiseCorr1"
        max_corr_per_pair2 = max_corr_per_pair.copy()
        max_corr_per_pair2.name = "MaxPairWiseCorr2"

        B = B.join(max_corr_per_pair, [B.index.get_level_values(0), B.index.get_level_values(1)], how="left").iloc[:,2:]
        B = B.join(max_corr_per_pair2, [B.index.get_level_values(0), B.index.get_level_values(2)], how="left").iloc[:,2:]
        B = B.query("Correlation==MaxPairWiseCorr1 or Correlation==MaxPairWiseCorr2")

        # Selected pair is given by top quantile at each date
        lower_bound = B.groupby(B.index.get_level_values(0)).apply(lambda x: np.quantile(x, q=1 - self.correlation_quantile))
        lower_bound.name="LowerCorrBound"

        B = B.join(lower_bound, B.index.get_level_values(0), how="left").iloc[:,1:]

        # Expected selected number of pairs is of order N*q/2 since each stock can  be matched with another one
        B = B.query("Correlation>LowerCorrBound")

        return B

    def _aggregate(self, eligible_pairs):
        eligible_pairs = pd.concat([eligible_pairs[sector]["Correlation"] for sector in eligible_pairs], axis=0, keys=eligible_pairs)
        eligible_pairs = eligible_pairs.sort_index(level=1)

        eligible_pairs = eligible_pairs.reset_index()

        eligible_pairs["level_1"] = eligible_pairs["level_1"].dt.tz_localize(None)
        eligible_pairs.to_excel(f"{FILE_PATH}//strategies\\eligible_pairs.xlsx")

    def run(self):

        corr_matrices = {sector: self._get_corr_matrix(sector) for sector in self.sectors}
        print("")
        # logging.info("Selecting Pairs...")
        # # step 2 - select pairs based on correlations
        # eligible_pairs = {sector: self._select_pairs(corr_matrix) for sector, corr_matrix in corr_matrices.items()}
        #
        # # step 3 - aggregate pairs from all sectors
        # self._aggregate(eligible_pairs)