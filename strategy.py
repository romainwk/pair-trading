from streamlit import columns

from settings import FILE_PATH
import pandas as pd
import numpy as np
from statsmodels.regression.rolling import RollingOLS
from joblib import Parallel, delayed
from collections import ChainMap
import scipy
import os
from pykalman import KalmanFilter

class MeanReversionSignal(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        self.trades_schedule = self.schedule.trades_schedule
        self.rebal_dates = self.schedule.rebal_dates
        self.rho=self.correlations.rho
        self.run()

    # def _load(self):
    #     directory = f"{FILE_PATH}\\strategies\\{self.strategy_name}"
    #     self.rho = pd.read_csv(f"{directory}\\{self.correlation_estimate}_{self.correlation_window}.csv", index_col=[1,2,3])

    def _get_returns(self):
        price = self.data.price
        R = np.log(price / price.shift(1))
        R = (R - R.rolling(self.correlation_window).mean())  # / r.expanding().std() # returns are standardised
        self.R = R

    def RollingOLS(self,pair):
        r = self.R[list(pair)].dropna()
        x, y = r[pair[1]], r[pair[0]]
        rols = RollingOLS(y, x, window=min(self.correlation_window, len(r)))
        rres = rols.fit(params_only=True)
        beta = rres.params.copy()[pair[1]]
        return {pair:beta}

    def KalmanFilter(self, pair, beta_prior=1, delta = 1e-5):

        r = self.R[list(pair)].dropna()
        r = (r / r.rolling(self.correlation_window).std()).dropna() # standardise obs so residuals ~N(0,1)
        x, y = pd.DataFrame(r[pair[1]]), r[pair[0]]

        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=1,
            initial_state_mean=beta_prior,
            initial_state_covariance=np.ones(1),
            transition_matrices=np.eye(1),
            observation_matrices=np.array(x)[:, np.newaxis],
            observation_covariance=1.0,
            transition_covariance=delta
        )

        beta, cov = kf.filter(y)
        beta = pd.DataFrame(beta, index=r.index)[0]
        return {pair: beta}

    def _get_hedge_ratio(self):
        # estimate t/s of HR across all pairs in the strategy
        unique_pairs = set(zip(self.rho.index.get_level_values(1), self.rho.index.get_level_values(2)))
        batch_size=int(len(unique_pairs)/self.n_parallel_jobs)
        hedge_ratio_func = getattr(self, self.hedge_ratio_estimate)
        res = Parallel(n_jobs=self.n_parallel_jobs, verbose=1, batch_size=batch_size)(delayed(hedge_ratio_func)(p) for p in unique_pairs)
        res = dict(ChainMap(*res))
        HR = pd.concat(res.values(), keys=res).dropna()
        HR = HR.unstack().T
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
        Z = self.S.ewm(halflife=self.mean_reversion_window).mean() / (self.S.ewm(halflife=self.mean_reversion_window).std())
        # translate z-score into a signal with logistic transformation + band-stop filter controlling for weak dislocations
        d1 = Z * np.sqrt(252)  # z_spread ~N(0,1) - review scaling factor
        L = 2 * scipy.stats.norm.cdf(d1) - 1
        F = (1 - scipy.stats.norm.pdf(d1) / scipy.stats.norm.pdf(0))
        self.mr_signal = pd.DataFrame(L * F, index=self.S.index, columns=self.S.columns)

    def _save(self):
        directory = f"{FILE_PATH}\\strategies\\{self.strategy_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.mr_signal.to_csv(f"{directory}\\mean_reversion_signal.csv")
        self.HR.to_csv(f"{directory}\\HR.csv")

    def run(self):
        # self._load()
        self._get_returns()
        self._get_hedge_ratio()
        self._get_spread()
        self._get_mean_reversion_signal()
        if self.export_data: self._save()

class BuildStrategy(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)

        self.trades_schedule = self.schedule.trades_schedule
        self.rebal_dates = self.schedule.rebal_dates
        self.mr_signal = self.mean_reversion.mr_signal
        self.HR = self.mean_reversion.HR
        self.rho = self.correlations.rho
        self.run()

    # def _load(self):
    #     directory = f"{FILE_PATH}\\strategies\\{self.strategy_name}"
    #     self.mr_signal = pd.read_csv(f"{directory}\\mean_reversion_signal.csv", index_col=0, header=[0,1], parse_dates=True)
    #     self.HR = pd.read_csv(f"{directory}\\HR.csv", index_col=0, header=[0,1], parse_dates=True)
    #     self.rho = pd.read_csv(f"{directory}\\{self.correlation_estimate}_{self.correlation_window}.csv", index_col=[1, 2, 3], parse_dates=True)

    def _get_portfolio(self):
        mr_signal = self.mr_signal.copy()
        mr_signal = mr_signal.T.stack()
        mr_signal.name = "EntrySignal"
        mr_signal.index.names = ["Pair1", "Pair2", "Date"]

        HR = self.HR.copy()
        HR=HR.T.stack()
        HR.name = "HR"
        HR.index.names = ["Pair1", "Pair2", "Date"]

        portfolio = self.rho.copy()
        portfolio = portfolio.join(mr_signal, how="left")
        portfolio = portfolio.join(HR, how="left")
        portfolio = portfolio[portfolio.HR.notna()]
        portfolio = portfolio[portfolio.EntrySignal.notna()]
        portfolio["AbsEntrySignal"] = portfolio["EntrySignal"].abs()

        portfolio["EntrySignalRank"] = portfolio["AbsEntrySignal"].groupby(portfolio.index.get_level_values(0)).rank(ascending=False)
        if self.select_top_n_stocks:
            portfolio = portfolio.query(f"EntrySignalRank<={self.select_top_n_stocks}")
        if self.min_signal_threshold:
            portfolio["EntrySignal"] = portfolio["EntrySignal"].where(portfolio["AbsEntrySignal"] > self.min_signal_threshold, 0)
        self.portfolio=portfolio
        self.portfolio_composition = self.portfolio

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

        s1 = self.data.price.copy()
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
            s1 = self.data.price.copy()
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
        p["NetPnLPairWithExit"] = p["NetPnLPair"] * (1 - p["Exited"].shift(1)).fillna(0)

        self.portfolio=p

    def _get_portfolio_pnl(self):

        p = self.portfolio.copy()

        T = pd.to_datetime(self.end_date)
        reb_dates = list(zip(self.rebal_dates[:-1], self.rebal_dates[1:])) + [(self.rebal_dates[-1], T)]

        roll_number = {t: i for t, i in zip(self.rebal_dates, range(len(self.rebal_dates)))}

        notional = {}
        index = {}

        I = pd.Series(index=[self.rebal_dates[0]], data=self.start_value)

        # portfolio PnL logic accounts for overlapping trades with associated notional
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
            active_trades["NotionalDailyPnL"] = active_trades["Notional"]*active_trades["NetPnLPairWithExit"]

            pnl = active_trades["NotionalDailyPnL"].groupby("Date").sum()

            I = pd.concat([I, I.loc[tm1] + pnl])

        I = I.astype(float)
        I.name="Index"
        self.I = I

    def _save(self):
        directory = f"{FILE_PATH}\\strategies\\{self.folder}\\{self.strategy_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.I.to_csv(f"{directory}\\index.csv")
        # self.portfolio.iloc[-2000:].to_csv(f"{directory}\\portfolio.csv")
        # self.portfolio_composition.to_csv(f"{directory}\\portfolio_composition.csv")

    def _get_portfolio_stats(self):
        ts = self.I
        ts= ts.astype(float)

        r = np.log(ts/ts.shift(1).dropna())

        sr = r.mean()/r.std()*np.sqrt(252)
        print(sr)
        pass

    def run(self):

        # self._load()
        self._get_portfolio()
        self._reindex_portfolio()
        self._size_portfolio()

        self._get_prices()
        self._get_pair_pnl()
        self._add_exit_conditions()
        self._add_entry_exit_costs()
        self._get_portfolio_pnl()
        self._get_portfolio_stats()
        self._save()