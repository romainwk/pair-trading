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
        print(f"Schedule loaded from {path_load}/schedule.pkl")

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

        yts = ql.RelinkableYieldTermStructureHandle()

        instruments = [
            ('depo', '6M', 0.025),
            ('swap', '1Y', 0.031),
            ('swap', '2Y', 0.032),
            ('swap', '3Y', 0.035)
        ]

        helpers = ql.RateHelperVector()
        index = ql.Euribor6M(yts)
        for instrument, tenor, rate in instruments:
            if instrument == 'depo':
                helpers.append(ql.DepositRateHelper(rate, index))
            if instrument == 'fra':
                monthsToStart = ql.Period(tenor).length()
                helpers.append(ql.FraRateHelper(rate, monthsToStart, index))
            if instrument == 'swap':
                swapIndex = ql.EuriborSwapIsdaFixA(ql.Period(tenor))
                helpers.append(ql.SwapRateHelper(rate, swapIndex))
        curve = ql.PiecewiseLogCubicDiscount(2, ql.TARGET(), helpers, ql.Actual365Fixed())

        yts.linkTo(curve)
        engine = ql.DiscountingSwapEngine(yts)

        tenor = ql.Period('1y')
        fixedRate = None
        forwardStart = ql.Period("1y")

        swap = ql.MakeVanillaSwap(tenor, index, fixedRate, forwardStart, nominal=10e6, pricingEngine=engine)
        swap.fairRate()


        return tbl