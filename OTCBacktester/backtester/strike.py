from calendar import calendar

import pandas as pd
from ib_insync.util import schedule

from OTCBacktester.backtester import strike, pricing, utils
import QuantLib as ql
from OTCBacktester.backtester.strategy_settings import PATH_SAVE
import os

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
        schedule_tbl = pd.read_csv(f"{path_load}/schedule.csv")
        schedule_tbl = utils.iso_to_dates(schedule_tbl)
        self.schedule_tbl = schedule_tbl
        print(f"Schedule loaded from {path_load}/schedule.csv")

        self.schedule_tbl = self.schedule_tbl

    def _set_strikes(self):

        unique_instru = set([args["Instrument"] for args in self.Legs.values()])

        tbl = self.schedule_tbl
        for leg_name, args in self.Legs.items():
            class_name = self.instru_to_class[args["Instrument"]]
            tbl = tbl[tbl["LegName"] == leg_name]
            leg_schedule=getattr(strike, class_name)(self.settings).get_strike(leg_name, args, tbl)
        #     sched.append(leg_schedule)
        # schedule_tbl = pd.concat(sched, axis=0).sort_values(by=["EntryDate", "LegName"])
        # schedule_tbl = schedule_tbl.reset_index()
        # schedule_tbl = schedule_tbl.rename({"index":"RollNumber"}, axis=1)
        # schedule_tbl["TradeNumber"] = range(len(schedule_tbl))
        # self.schedule_tbl = schedule_tbl

    # def _save(self):
    #     schedule_tbl = self.schedule_tbl
    #     path_save = f"{PATH_SAVE}/{self.Name}"
    #     os.makedirs(path_save, exist_ok=True)
    #
    #     schedule_tbl.to_csv(f"{path_save}/schedule.csv", index=False)
    #     print(f"Schedule saved to {path_save}/schedule.csv")

    def run(self):
        self._load()
        self._set_strikes()
        # self._save()

class OTCStrikeIRSwap(OTCStrike):
    def __init__(self, settings):
        super().__init__(settings)

    def get_strike(self, leg_name, args, tbl):

        #FIXME move this in the pricing.py module // make it ccy specific too and link to real historical data

        F = {(t, n): pricing.IRSwap()(ccy=args["Ccy"], payoff=args["Payoff"], t=t, T1=T1, T2=T2) for t, n, T1, T2 in zip(tbl.EntryDate, tbl.TradeNumber, tbl.ExpiryDates, tbl.TenorDates)}


        return tbl