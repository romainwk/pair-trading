import datetime
import numpy as np

FILE_PATH = r'C:/Users/Romain/PycharmProjects/pythonProject'

correlation_settings = dict(pair_selection_method="SampleCorrelation",  # SampleCorrelation, EWMCorrelation, LinearShrinkage, NonLinearShrinkage
                            correlation_window=120,  # sliding window or half-life in the case of EWMCorrelation
                            shrink_factor=0.25,  # could be automatically selected by x-validation
                            )

pair_selection_settings = dict(correlation_quantile=0.10,  # top (1-q) pairs pass the PairSelection screen)
                               )

strategy_settings = dict(strategy_name="test_strategy",
                         start_date=datetime.date(2010, 1, 1),
                         end_date=datetime.date(2024, 9, 1),
                         trading_calendar="NYSE",
                         index_universe="S&P500",
                         rebal_frequency=5,  # how frequently a new set of pairs is considered
                         max_holding_period=40,
                         profit_taking=None,
                         stop_loss=None,
                         start_value=100,

                         notional_sizing="TargetNotional",  # TargetNotional, TargetVol
                         leverage=1,  # gross leverage of L/S strategy if sizing by TargetNotional
                         target_vol_level=0.05,  # drives leverage of the L/S strategy if sizing by TargetVol

                         transaction_cost=0.1 * 1 / np.sqrt(252),
                         # multiple of running std. e.g. for a stock running 15% vol annualised, charges 10 bps entry/exit (one way)
                         )

test_strategy={**correlation_settings,
               **pair_selection_settings,
               **strategy_settings,
               }