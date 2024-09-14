import pandas_market_calendars as mcal
import datetime

class Schedule(object):
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)
        self.run()

    def _get_good_trading_dates(self):
        cal = mcal.get_calendar(self.trading_calendar)
        self.trading_dates = cal.valid_days(start_date=self.start_date, end_date=self.end_date)
        # expand dataset for initial estimation
        self.training_dates = cal.valid_days(start_date=self.start_date - datetime.timedelta(days=self.correlation_window*4), end_date=self.end_date)

        self.trading_dates = self.trading_dates.tz_localize(None)
        self.training_dates = self.training_dates.tz_localize(None)

    def _get_rebal_dates(self):
        self.rebal_dates = [t for t, i in zip(self.trading_dates, range(len(self.trading_dates))) if i%self.rebal_frequency==0]

    def _get_trades_schedule(self):
        '''dict of dates during which each trade is alive (up to max_holding_period)'''
        l = list(self.trading_dates)
        i_start = [l.index(t) for t in self.rebal_dates]
        i_end = [i + self.max_holding_period for i in i_start]
        self.trades_schedule = {t: l[i:j] for t, i, j in zip(self.rebal_dates, i_start, i_end)}

    def run(self):
        self._get_good_trading_dates()
        self._get_rebal_dates()
        self._get_trades_schedule()