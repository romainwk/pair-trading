import datetime
import QuantLib as ql

PATH_SAVE = r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\strategies\test"

settings = dict(Name="StrategyTest",
                StartDate=ql.Date(1,9,2022),
                EndDate=ql.Date(1,1,2025),
                RebalFreq="1M",
                Calendar = [ql.TARGET(), ql.UnitedStates(ql.UnitedStates.Settlement), ql.UnitedStates(ql.UnitedStates.GovernmentBond)],
                Dcm= ql.Actual360(), # FIXME maybe add this a ccy specific dictionnary in settings
                Bdc=ql.ModifiedFollowing,
                Legs = dict(
                    Leg1=dict(Ccy="USD",
                               Instrument="IRSwap", # IRSwap, IRSwaption, IRCapFloor
                               Payoff="Payer",
                               Strike="ATMf",
                               HoldingPeriod="1M",
                               Position=1,
                               Hedged=False,
                               Expiry="1Y",
                               Tenor="10Y",
                               ),
                    Leg2=dict(Ccy="USD",
                               Instrument="IRSwaption", # IRSwap, IRSwaption, IRCapFloor
                               Payoff="Receiver",
                               Strike="ATMf",
                               HoldingPeriod="1M",
                               Position=1,
                               Hedged=True,
                               Expiry="1Y",
                               Tenor="10Y",
                               ),

                ),


                # classes=dict(schedule="OTCScheduleIRSwap",)

                )

StrategyTest = settings

