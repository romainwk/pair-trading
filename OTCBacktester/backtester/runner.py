import pandas as pd
import numpy as np
from importlib import reload
from OTCBacktester.backtester import strategy_settings, schedule, strike

reload(strategy_settings)
reload(schedule)

def run_strategy(name):
    settings = getattr(strategy_settings, name)
    schedule.OTCSchedule(settings).run()
    strike.OTCStrike(settings).run()

def main():
    names=[
        "StrategyTest",
    ]

    [run_strategy(name) for name in names]

if __name__ == '__main__':
    main()
