import datetime
import numpy as np

FILE_PATH = r'C:/Users/Romain/PycharmProjects/pythonProject'

iterations = dict(correlation_estimate=["EWMCorrelation", "SampleCorrelation", "LedoitWolfShrinkage", "OracleApproximatingShrinkage"])

iterations = dict(correlation_window=np.arange(90,250,10),
                  mean_reversion_window=np.arange(90,250,10),
                  )

signal_settings = dict(cluster_by="GIC_sector", #, GIC_sector, GIC_sub_industry
                       min_stock_per_cluster=5,
                       correlation_estimate="EWMCorrelation",  # SampleCorrelation, EWMCorrelation, LedoitWolfShrinkage, OracleApproximatingShrinkage
                       correlation_window=250,  # sliding window or half-life in the case of EWMCorrelation
                       correlation_quantile=0.10,  # top (1-q) pairs pass the correlation screen
                       hedge_ratio_estimate="KalmanFilter", # RollingOLS, KalmanFilter
                       mean_reversion_window=60,
                       select_top_n_stocks=None, # optional param, select top n stocks wrt mean_reversion signal at each rebal date
                       min_signal_threshold=None, # optional param, hard thresholding on the mean-reversion signal
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

                         transaction_cost=0.1 * 1 / np.sqrt(252)*0,
                         # multiple of running std. e.g. for a stock running 15% vol annualised, charges 10 bps entry/exit (one way)
                         )

computation_settings = dict(n_parallel_jobs=16,
                            export_data=False,
                            )

test_strategy={**signal_settings,
               **strategy_settings,
               **computation_settings,
               }