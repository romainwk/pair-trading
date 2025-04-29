import datetime
import QuantLib as ql

PATH_SAVE = r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\strategies\test"

settings = dict(Name="StrategyTest",
                StartDate=ql.Date(1,1,2020),
                EndDate=ql.Date(1,3,2025),
                RebalFreq="1M",
                Calendar = [ql.TARGET(), ql.UnitedStates(ql.UnitedStates.Settlement)],
                Dcm= ql.Actual360(), # FIXME maybe add this a ccy specific dictionnary in settings
                Bdc=ql.ModifiedFollowing,
                Legs = dict(
                    Leg1=dict(Ccy="EUR",
                               Instrument="IRSwap", # IRSwap, IRSwaption, IRCapFloor
                               Payoff="Payer",
                               Moneyness="ATMf",
                               HoldingPeriod="1M",
                               Position=1,
                               Hedged=False,
                               Expiry="1Y",
                               Tenor="10Y",
                               ),
                    Leg2=dict(Ccy="EUR",
                               Instrument="IRSwap", # IRSwap, IRSwaption, IRCapFloor
                               Payoff="Receiver",
                               Moneyness="ATMf",
                               HoldingPeriod="1M",
                               Position=1,
                               Hedged=False,
                               Expiry="1Y",
                               Tenor="10Y",
                               ),

                ),


                # classes=dict(schedule="OTCScheduleIRSwap",)

                )

StrategyTest = settings

