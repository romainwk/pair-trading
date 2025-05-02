from calendar import calendar

import pandas as pd
from ib_insync.util import schedule

from OTCBacktester.backtester import strike, pricing, utils
import QuantLib as ql
from OTCBacktester.backtester.strategy_settings import PATH_SAVE
import os
import logging

class OTCStrike(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)
        self.settings = settings
        self.instru_to_class = dict(IRSwap="OTCStrikeIRSwap",
                                    IRSwaption="OTCStrikeIRSwaption",
                                    )

    def _load(self):
        path_load = f"{PATH_SAVE}/{self.Name}"
        schedule_tbl = pd.read_csv(f"{path_load}/schedule.csv", index_col=0)
        schedule_tbl = utils.iso_to_dates(schedule_tbl)
        self.schedule_tbl = schedule_tbl
        print(f"Schedule loaded from {path_load}/schedule.csv")

        self.schedule_tbl = self.schedule_tbl

    def _set_strikes(self):

        tbl = self.schedule_tbl
        res = []
        for leg_name, args in self.Legs.items():
            class_name = self.instru_to_class[args["Instrument"]]
            tbl_leg = tbl[tbl["LegName"] == leg_name].copy()
            x=getattr(strike, class_name)(self.settings).get_strike(leg_name, args, tbl_leg)
            res.append(x)

        res = pd.concat(res).sort_values("TradeNumber")
        self.strikes = res

    def _save(self):
        path_save = f"{PATH_SAVE}/{self.Name}"
        # os.makedirs(path_save, exist_ok=True)
        self.strikes.to_csv(f"{path_save}/strikes.csv", index=False)
        print(f"Strikes saved to {path_save}/strikes.csv")

    def run(self):
        self._load()
        self._set_strikes()
        self._save()

class OTCStrikeIRSwap(OTCStrike):
    def __init__(self, settings):
        super().__init__(settings)

    def get_strike(self, leg_name, args, tbl):

        F = {(t, n): pricing.IRSwap()(ccy=args["Ccy"], payoff=args["Payoff"], t=t, T1=T1, T2=T2) for t, n, T1, T2 in zip(tbl.EntryDate, tbl.TradeNumber, tbl.ExpiryDates, tbl.TenorDates)}

        tbl_F = pd.DataFrame(F).T
        tbl_F = tbl_F.reset_index().rename({"level_0":"EntryDate", "level_1":"TradeNumber"}, axis=1)

        tbl = tbl.merge(tbl_F, on=["EntryDate", "TradeNumber"])

        return tbl
