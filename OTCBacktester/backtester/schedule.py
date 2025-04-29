import pandas as pd
from OTCBacktester.backtester import schedule, utils
import QuantLib as ql
from OTCBacktester.backtester.strategy_settings import PATH_SAVE
import os

class OTCSchedule(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)
        self.settings = settings
        self.instru_to_class = dict(IRSwap="OTCScheduleIRSwap",
                                    IRSwaption="OTCScheduleIRSwaption",
                                    )

    def _schedule(self):
        sched = []
        for leg_name, args in self.Legs.items():
            class_name = self.instru_to_class[args["Instrument"]]
            leg_schedule=getattr(schedule, class_name)(self.settings).get_schedule(leg_name, args)
            sched.append(leg_schedule)
        schedule_tbl = pd.concat(sched, axis=0).sort_values(by=["EntryDate", "LegName"])
        schedule_tbl = schedule_tbl.reset_index()
        schedule_tbl = schedule_tbl.rename({"index":"RollNumber"}, axis=1)
        schedule_tbl["TradeNumber"] = range(len(schedule_tbl))
        self.schedule_tbl = schedule_tbl

    def _save(self):
        schedule_tbl = self.schedule_tbl
        schedule_tbl = utils.dates_to_iso(schedule_tbl)

        path_save = f"{PATH_SAVE}/{self.Name}"
        os.makedirs(path_save, exist_ok=True)

        self.schedule_tbl = schedule_tbl
        schedule_tbl.to_csv(f"{path_save}/schedule.csv")
        print(f"Schedule saved to {path_save}/schedule.csv")

    def run(self):
        self._schedule()
        self._save()


class OTCScheduleIRSwap(OTCSchedule):
    def __init__(self, settings):
        super().__init__(settings)

    def get_schedule(self, leg_name, args):
        trade_start_date = []

        rebal_freq = ql.Period(self.RebalFreq)
        calendar = ql.JointCalendar(self.Calendar)

        entry_dates = ql.Schedule(self.StartDate,
                                        self.EndDate,
                                        rebal_freq,
                                        calendar,
                                        self.Bdc,
                                        self.Bdc,
                                        ql.DateGeneration.Forward,
                                        False).dates()

        expiry = ql.Period(args.get("Expiry"))
        tenor = ql.Period(args.get("Tenor"))
        holding_period = ql.Period(args.get("HoldingPeriod"))

        expiry_dates = [calendar.advance(d, expiry, self.Bdc) for d in entry_dates]
        tenor_dates = [calendar.advance(d, tenor, self.Bdc) for d in expiry_dates]

        if holding_period == expiry:
            exit_dates =expiry_dates
        else:
            exit_dates = [calendar.advance(d, holding_period, self.Bdc) for d in entry_dates]

        tbl = pd.DataFrame({**dict(EntryDate=entry_dates,ExpiryDates=expiry_dates,TenorDates=tenor_dates,ExitDates=exit_dates, LegName=leg_name),
                              **args},
                           )
        return tbl

# same schedule logic as swaps
class OTCScheduleIRSwaption(OTCScheduleIRSwap):
    def __init__(self, settings):
        super().__init__(settings)