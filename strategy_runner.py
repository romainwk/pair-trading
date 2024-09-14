import datetime
import numpy as np
# from pandas.tseries.holiday import get_calendar, HolidayCalendarFactory, GoodFriday
# from datetime import datetime
import scipy
import pandas as pd
import statsmodels.api as sm
from pydeck import settings
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
import itertools
import logging

import scheduler
import settings
import data_processing
import correlation_estimate
import visualisation
from settings import FILE_PATH

from importlib import reload
reload(scheduler)
reload(settings)
reload(data_processing)
reload(visualisation)
reload(correlation_estimate)

class PairScreening(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        self.schedule_obj = Schedule(settings)
        self.data_obj = Data(settings)
        self.sectors = self.data_obj.clusters

        self.trading_dates = self.schedule_obj.trading_dates
        self.rebal_dates = self.schedule_obj.rebal_dates
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

        stocks = self.data_obj.clusters[sector]
        price = self.data_obj.data[stocks]

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

        # step 1 - compute correlation matrices
        logging.info("Computing Corr Matrices...")
        corr_matrices = {sector: self._get_corr_matrix(sector) for sector in self.sectors}

        logging.info("Selecting Pairs...")
        # step 2 - select pairs based on correlations
        eligible_pairs = {sector: self._select_pairs(corr_matrix) for sector, corr_matrix in corr_matrices.items()}

        # step 3 - aggregate pairs from all sectors
        self._aggregate(eligible_pairs)

class PairSelection(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        # FIXME pass to settings instead
        self.schedule_obj = Schedule(settings)
        self.data_obj = Data(settings)

        self.trades_schedule = self.schedule_obj.trades_schedule
        self.rebal_dates = self.schedule_obj.rebal_dates

        self.run()

    def _load_eligible_pairs(self):
        eligible_pairs = pd.read_excel(f"{FILE_PATH}//strategies//eligible_pairs.xlsx", index_col=[2,3,4])[["level_0", "Correlation"]].rename(dict(level_0="GIC_sector",), axis=1)
        eligible_pairs.index.names = ["Date", "Pair1", "Pair2"]
        self.eligible_pairs=eligible_pairs

    def _get_returns(self):
        price = self.data_obj.data
        R = np.log(price / price.shift(1))
        R = (R - R.rolling(self.correlation_window).mean())  # / r.expanding().std() # returns are standardised
        self.R = R

    def _get_rolling_OLS_beta(self,pair):
        r = self.R[list(pair)].dropna()
        x, y = r[pair[1]], r[pair[0]]
        rols = RollingOLS(y, x, window=min(self.correlation_window, len(r)))
        rres = rols.fit(params_only=True)
        beta = rres.params.copy()[pair[1]]
        return beta

    def _get_hedge_ratio(self):
        # unique_pairs = set(zip(self.eligible_pairs.Pair1, self.eligible_pairs.Pair2))
        unique_pairs = set(zip(self.eligible_pairs.index.get_level_values(1), self.eligible_pairs.index.get_level_values(2)))

        HR = {pair: self._get_rolling_OLS_beta(pair) for pair in unique_pairs}  # Make this FASTER - too slow
        HR = pd.DataFrame(HR)
        self.HR = HR

    def _get_spread(self):
        # spread defined as y_t - beta_t * x_t - in vectorised form for speed
        pair1, pair2 = self.HR.columns.get_level_values(0), self.HR.columns.get_level_values(1)
        idx = self.HR.index
        Y1, Y2 = np.array(self.R.loc[idx, pair1]), np.array(self.R.loc[idx,pair2])

        S = Y1 - np.array(self.HR) * Y2
        S = pd.DataFrame(S, index=self.HR.index, columns=self.HR.columns)
        self.S = S

    def _get_mean_reversion_signal(self):

        # Z-score as EWM filter - e.g. short term trending signal on the spread
        Z = self.S.ewm(halflife=self.correlation_window / 2).mean() / (self.S.ewm(halflife=self.correlation_window / 4).std())

        # translate z-score into a signal with logistic transformation + band-stop filter controlling for weak dislocations
        d1 = Z * np.sqrt(252)  # z_spread ~N(0,1) - review scaling factor
        L = 2 * scipy.stats.norm.cdf(d1) - 1
        F = (1 - scipy.stats.norm.pdf(d1) / scipy.stats.norm.pdf(0))

        self.mr_signal = pd.DataFrame(L * F, index=self.S.index, columns=self.S.columns)

    def _get_portfolio(self):
        mr_signal = self.mr_signal.copy()
        # signal.index = signal.index.tz_localize(None) #FIXME remove if not load fro m xl
        mr_signal = mr_signal.T.stack()
        mr_signal.name = "EntrySignal"
        mr_signal.index.names = ["Pair1", "Pair2", "Date"]

        HR = self.HR.copy()
        # HR.index = HR.index.tz_localize(None)  # FIXME remove if not load fro m xl
        HR=HR.T.stack()
        HR.name = "HR"
        HR.index.names = ["Pair1", "Pair2", "Date"]

        portfolio = self.eligible_pairs.copy()
        portfolio = portfolio.join(mr_signal, how="left")
        portfolio = portfolio.join(HR, how="left")
        self.portfolio=portfolio

    def _reindex_portfolio(self):
        def _add_trading_dates(p, t):
            trades_dt = self.trades_schedule[t]
            trades = p.loc[t]
            trades = pd.concat([trades]*len(trades_dt), keys=trades_dt)
            return trades
        p = self.portfolio.copy()
        roll_number = pd.Series(index=self.rebal_dates, data=range(len(self.rebal_dates)), name="RollNumber")
        p =p.join(roll_number, on=p.index.get_level_values(0)).iloc[:,1:]
        p = pd.concat([_add_trading_dates(p,t) for t in self.rebal_dates])
        self.portfolio = p

    def _size_portfolio(self):
        def custom_round(x, base):
            return int(base * round(float(100*x) / base))*0.01

        p = self.portfolio.copy()
        p.index.names = ["Date", "Pair1", "Pair2"]

        # go long/short the spread depending on sgn(EntrySignal)
        p["SizePair1"] = p["EntrySignal"] * -1
        p["SizePair2"] = p["EntrySignal"] * p["HR"]

        # discretise signal according to its strength
        p["SizePair1"] = p["SizePair1"].apply(lambda x: custom_round(x, base=10))
        p["SizePair2"] = p["SizePair2"].apply(lambda x: custom_round(x, base=10))
        # weak signal discarded
        p = p.query("abs(SizePair1)>0 and abs(SizePair2)>0")

        # rescale according to notional target
        gross_leverage = (p["SizePair1"].abs() + p["SizePair2"].abs()).groupby("Date").sum()*0.01
        gross_leverage.name = "GrossLeverage"
        p = p.join(gross_leverage, on="Date")
        p["ScalingFactor"] = 1/p["GrossLeverage"]

        p["UnitPair1"] = p["ScalingFactor"] * p["SizePair1"]
        p["UnitPair2"] = p["ScalingFactor"] * p["SizePair2"]
        self.portfolio = p

    def _get_prices(self):
        p = self.portfolio.copy()

        s1 = self.data_obj.data.copy()
        s1 = s1.stack()
        s1.index.names = ["Date", "Pair1"]
        s1.name = "ClosePricePair1"

        s2 = s1.copy()
        s2.name="ClosePricePair2"

        p = p.join(s1, on=[p.index.get_level_values(0), p.index.get_level_values(1)]).iloc[:,2:]
        p = p.join(s2, on=[p.index.get_level_values(0), p.index.get_level_values(2)]).iloc[:,2:]

        p["Pair"] = list(zip(p.index.get_level_values(1), p.index.get_level_values(2)))
        p = p.sort_values(by=["RollNumber", "Pair", "Date"])

        self.portfolio = p

    def _get_pair_pnl(self):
        p = self.portfolio.copy()
        p["EntryDate"] =p.Pair != p.Pair.shift(1)

        p["EntryPricePair1"] = (p["EntryDate"] * p["ClosePricePair1"]).replace({0: np.nan}).ffill()
        p["EntryPricePair2"] = (p["EntryDate"] * p["ClosePricePair2"]).replace({0: np.nan}).ffill()

        # normalise by entry price here
        p["PnLPair1"] =  (p["ClosePricePair1"].diff()/p["EntryPricePair1"]  * (1-p.EntryDate)).fillna(0)
        p["PnLPair2"] = (p["ClosePricePair2"].diff()/p["EntryPricePair2"] * (1 - p.EntryDate)).fillna(0)

        p["PnLPair"] = p["UnitPair1"] * p["PnLPair1"] + p["UnitPair2"] * p["PnLPair2"]

        cum_pnl_pair = p[["RollNumber", "Pair", "PnLPair"]].groupby(["Pair", "RollNumber"]).cumsum()
        p["CumPairPnLHPeriod"] = cum_pnl_pair # p["PnLPair"].groupby(["Pair1", "Pair2"]).cumsum()
        self.portfolio = p

    def _add_exit_conditions(self):
        p = self.portfolio.copy()
        if not self.profit_taking: self.profit_taking = np.inf
        if not self.stop_loss: self.stop_loss = np.inf

        p["MaxHoldingPeriod"] = p.Pair != p.Pair.shift(-1)
        p["ProfitTaking"] = p["CumPairPnLHPeriod"] > self.profit_taking
        p["StopLoss"] = p["CumPairPnLHPeriod"] < -self.stop_loss
        p["Exit"] = p["MaxHoldingPeriod"] | p["ProfitTaking"] | p["StopLoss"]
        p["DaysPostExit"] = p[["RollNumber", "Pair", "Exit"]].groupby(["Pair", "RollNumber"]).cumsum()
        p["ExitDate"] = p["DaysPostExit"]==1
        p["Exited"] = p["DaysPostExit"].where(p["DaysPostExit"] == 0, 1)

        p["PairDailyPnL"] = p["PnLPair"] * (1 - p["Exited"].shift(1)).fillna(0)
        p["CumPairPnL"] = p["CumPairPnLHPeriod"] * (1 - p["Exited"].shift(1)).fillna(0)

        self.portfolio = p

    def _add_entry_exit_costs(self):
        def _add_asset_volatility(p):
            s1 = self.data_obj.data.copy()
            s1 = np.log(s1.div(s1.shift(1))).ewm(halflife=self.correlation_window).std()*np.sqrt(252)
            s1 = s1.stack()
            s1.index.names = ["Date", "Pair1"]
            s1.name = "RealisedVolPair1"

            s2 = s1.copy()
            s2.name = "RealisedVolPair2"

            p = p.join(s1, on=[p.index.get_level_values(0), p.index.get_level_values(1)]).iloc[:, 2:]
            p = p.join(s2, on=[p.index.get_level_values(0), p.index.get_level_values(2)]).iloc[:, 2:]
            return p

        p = self.portfolio
        p = _add_asset_volatility(p)

        p["EntryCostPair1"] = self.transaction_cost * p["RealisedVolPair1"] * p["EntryDate"].shift(1).fillna(0)
        p["EntryCostPair2"] = self.transaction_cost * p["RealisedVolPair2"] * p["EntryDate"].shift(1).fillna(0)

        p["NetPnLPair"] = p["PnLPair"] - abs(p["UnitPair1"]) * p["EntryCostPair1"] - abs(p["UnitPair2"]) * p["EntryCostPair2"]
        p["PairNetDailyPnL"] = p["NetPnLPair"] * (1 - p["Exited"].shift(1)).fillna(0)

        self.portfolio=p

    def _get_portfolio_pnl(self):

        p = self.portfolio.copy()

        T = pd.to_datetime(self.end_date)
        reb_dates = list(zip(self.rebal_dates[:-1], self.rebal_dates[1:])) + [(self.rebal_dates[-1], T)]

        roll_number = {t: i for t, i in zip(self.rebal_dates, range(len(self.rebal_dates)))}

        notional = {}
        index = {}

        I = pd.Series(index=[self.rebal_dates[0]], data=self.start_value)

        # portfolio PnL logic accounts for overlapping trades with associated notionals
        # iterative logic vectorised at the rebal_freq level
        for t0, t1 in reb_dates:
            active_trades = p.query(f"Date>='{t0}' & Date<'{t1}'")

            if t0 == self.rebal_dates[0]: tm1 = t0
            else: tm1 = I.index[-1]

            n = roll_number[t0]
            if self.notional_sizing == "TargetNotional":
                notional[n] = self.leverage * I.loc[tm1] * 0.01
            elif self.notional_sizing == "TargetVol":
                realised_vol = (I.pct_change().ewm(halflife=self.correlation_window).std() * np.sqrt(252)).fillna(self.target_vol_level).loc[tm1]
                notional[n] = self.target_vol_level/realised_vol * I.loc[tm1] * 0.01
            else: raise Exception(f"{self.notional_sizing} unsupported")

            x = pd.Series(notional, name="Notional")
            x.index.name="RollNumber"

            active_trades = active_trades.join(x, "RollNumber", how="left")
            active_trades["NotionalDailyPnL"] = active_trades["Notional"]*active_trades["PairNetDailyPnL"]

            pnl = active_trades["NotionalDailyPnL"].groupby("Date").sum()

            I = pd.concat([I, I.loc[tm1] + pnl])

        self.I = I

        self.I.to_excel(f"{FILE_PATH}//strategies//index_test4.xlsx")

    def _get_portfolio_stats(self):
        ts = self.I
        # r = np.log(ts.div(ts.shift(1)))
        # sr = r.mean()/r.std()*np.sqrt(252)
        # print("")

    def run(self):

        self._load_eligible_pairs()
        self._get_returns()
        self._get_hedge_ratio()
        self._get_spread()
        self._get_mean_reversion_signal()

        self._get_portfolio()
        self._reindex_portfolio()
        self._size_portfolio()
        self._get_prices()
        self._get_pair_pnl()
        self._add_exit_conditions()
        self._add_entry_exit_costs()
        self._get_portfolio_pnl()
        self._get_portfolio_stats()

# @st.cache_data
def strategy_runner(settings):

    schedule = scheduler.Schedule(settings)
    settings.update(schedule=schedule)
    data = data_processing.Data(settings)
    settings.update(data=data)

    correlation_estimate.CorrelationEstimator(settings)

    print("")
    # PairScreening(settings)
    # PairSelection(settings)


def main():
    # add something that checks enough data before computing corr
    # plot SR function of window (e.g. weekly SR)
    # add costs as func of vol

    # correlations computed once at start
    # and then loaded by all strategies ??

    strategy_runner(settings.test_strategy)
    # WebApp()

if __name__ == '__main__':
    main()