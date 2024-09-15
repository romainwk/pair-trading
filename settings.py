import datetime
import numpy as np

FILE_PATH = r'C:/Users/Romain/PycharmProjects/pythonProject'

def get_settings(params):
    signal_settings = dict(cluster_by="GIC_sector", #, GIC_sector, GIC_sub_industry
                           min_stock_per_cluster=5,
                           correlation_estimate="EWMCorrelation",  # SampleCorrelation, EWMCorrelation, LedoitWolfShrinkage, OracleApproximatingShrinkage
                           correlation_window=120,  # sliding window or half-life in the case of EWMCorrelation
                           correlation_quantile=0.10,  # top (1-q) pairs pass the correlation screen
                           hedge_ratio_estimate="KalmanFilter", # RollingOLS, KalmanFilter
                           mean_reversion_window=60,
                           select_top_n_stocks=10, # optional param, select top n stocks wrt mean_reversion signal at each rebal date
                           signal_threshold_entry=None, # optional param, hard thresholding on the mean-reversion signal
                           signal_threshold_exit=0, #drives exit
                            )

    strategy_settings = dict(strategy_name="base_strategy",
                             start_date=datetime.date(2010, 1, 1),
                             end_date=datetime.date(2024, 9, 1),
                             trading_calendar="NYSE",
                             index_universe="S&P500",
                             rebal_frequency=5,  # how frequently a new set of pairs is considered
                             max_holding_period=60,
                             profit_taking=None,
                             stop_loss=None,
                             start_value=100,

                             notional_sizing="TargetNotional",  # TargetNotional, TargetVol
                             leverage=2,  # gross leverage of L/S strategy if sizing by TargetNotional
                             target_vol_level=0.05,  # drives leverage of the L/S strategy if sizing by TargetVol

                             transaction_cost=0.1 * 1 / np.sqrt(252)*0,
                             # multiple of running std. e.g. for a stock running 15% vol annualised, charges 10 bps entry/exit (one way)
                             )

    computation_settings = dict(n_parallel_jobs=16,
                                export_data=False,
                                folder="base",
                                )
    strategy={**signal_settings,
              **strategy_settings,
              **computation_settings,
             }
    strategy.update(params)
    return strategy

# iterate over correlation estimate, HR estimate, correlation_window, mr_window
# iterations = dict(correlation_estimate=["EWMCorrelation", "SampleCorrelation", "LedoitWolfShrinkage", "OracleApproximatingShrinkage"])

iterations1 = [dict(correlation_window=int(w),
                    mean_reversion_window=int(w/2),
                    folder="sensi_to_window_length",
                    strategy_name=f"W={w}") for w in np.arange(90,250,10)]

iterations2 = [dict(correlation_estimate=s,
                    strategy_name=f"{s}",
                    correlation_window=120,
                    mean_reversion_window=60,
                    folder="sensi_to_correlation_estimator",
                    ) for s in ["SampleCorrelation", "EWMCorrelation", "LedoitWolfShrinkage", "OracleApproximatingShrinkage"]]

iterations3 = [dict(hedge_ratio_estimate=s,
                    strategy_name=f"{s}",
                    correlation_window=120,
                    mean_reversion_window=60,
                    folder="sensi_to_hr_estimator",
                    ) for s in ["RollingOLS", "KalmanFilter"]]

iterations4 = [dict(correlation_quantile=q,
                    strategy_name=f"q={int(q*100)}",
                    correlation_window=120,
                    mean_reversion_window=60,
                    folder="sensi_to_quantile_threshold",
                    ) for q in [0.25,0.15,0.10,0.05,0.01]]

iterations = iterations1+iterations2+iterations3+iterations4
# iterations= [iterations1[4]] #FIXME remove
strategies_to_run = [get_settings(params) for params in iterations]

# def main():
#     iterations1 = [
#         dict(correlation_window=w, mean_reversion_window=int(w / 2), strategy_name=f"W_L_{w}_W_S_{int(w / 2)}") for w in
#         np.arange(90, 250, 10)]
#     [get_settings(params) for params in iterations1]
#
# if __name__ == '__main__':
#     main()