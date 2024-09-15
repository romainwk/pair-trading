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
import strategy
from settings import FILE_PATH

from importlib import reload
reload(scheduler)
reload(settings)
reload(data_processing)
reload(correlation_estimate)
reload(strategy)
reload(visualisation)

# @st.cache_data
def strategy_runner(settings):

    schedule = scheduler.Schedule(settings)
    settings.update(schedule=schedule)
    data = data_processing.Data(settings)
    settings.update(data=data)
    correlations = correlation_estimate.CorrelationEstimator(settings)
    settings.update(correlations=correlations)
    mean_reversion = strategy.MeanReversionSignal(settings)
    settings.update(mean_reversion=mean_reversion)
    portfolio = strategy.BuildStrategy(settings)

def main():
    # add something that checks enough data before computing corr
    # plot SR function of window (e.g. weekly SR)
    # add costs as func of vol

    # correlations computed once at start
    # and then loaded by all strategies ??

    strategy_runner(settings.test_strategy)

if __name__ == '__main__':
    main()