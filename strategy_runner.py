from pydeck import settings

import scheduler
import data_processing
import correlation_estimate
import strategy
import settings

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
    return portfolio

def main():
    strategies = settings.strategies_to_run
    [strategy_runner(strat) for strat in strategies]

if __name__ == '__main__':
    main()