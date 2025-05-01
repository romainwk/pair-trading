import QuantLib as ql
from OTCBacktester.backtester import data, ccy_settings, utils
import pandas as pd


def get_ql_index(ccy, index, yts):
    if ccy == "EUR" and index == "EURIBOR":
        ql_index = ql.Euribor6M(yts)
    elif ccy == "USD" and index == "LIBOR":
        ql_index = ql.USDLibor(ql.Period("3M"), yts)
    elif ccy == "USD" and index == "SOFR":
        ql_index = ql.OvernightIndex("SOFR", 2, ql.USDCurrency(), ql.UnitedStates(ql.UnitedStates.Settlement),  ql.Actual360(), yts)
    else:
        raise ValueError(f"Unsupported currency/index combination: {ccy}/{index}")
    return ql_index

class IRSwap:

    def __init__(self):
        pass

    def _get_annuity_from_schedule(self, schedule, day_count, discount_curve):
        annuity = 0.0
        for i in range(len(schedule) - 1):
            start_date = schedule[i]
            end_date = schedule[i + 1]
            pay_date = end_date  # in most swaps, payment is on end date

            accrual = day_count.yearFraction(start_date, end_date)
            df = discount_curve.discount(pay_date)

            annuity += df * accrual
        return annuity

    def _build_curve(self, instruments, index, calendar, yts):
        helpers = ql.RateHelperVector()

        for instrument, tenor, rate in instruments:
            if instrument == 'swap':
                if index == "SOFR":
                    float_index = ql.OvernightIndex("SOFR", 0, ql.USDCurrency(), calendar, ql.Actual360(), yts)
                    helper = ql.SwapRateHelper(rate,
                                               ql.Period(tenor),
                                               calendar,
                                               ql.Annual,
                                               ql.ModifiedFollowing,
                                               ql.Actual360(),
                                               float_index, )
                elif index == "LIBOR":
                    rate_handle = ql.QuoteHandle(ql.SimpleQuote(rate))
                    float_index = ql.USDLibor(ql.Period("3M"), yts)

                    helper = ql.SwapRateHelper(
                        rate_handle,
                        ql.Period(tenor),  # tenor
                        calendar,
                        ql.Semiannual,  # fixed freq
                        ql.ModifiedFollowing,  # fixed BDC
                        ql.Thirty360(ql.Thirty360.BondBasis),  # fixed day count
                        float_index  # ibor index
                    )

                else:
                    raise ValueError(f"Unsupported index: {index}")

                helpers.append(helper)

            elif instrument == 'depo':
                helpers.append(helper)

        # curve = ql.PiecewiseLinearZero(2, calendar, helpers, ql.Actual365Fixed())
        if index == "SOFR":
            curve = ql.PiecewiseLogLinearDiscount(2, calendar, helpers, ql.Actual360())
        elif index == "LIBOR":
            curve = ql.PiecewiseLinearZero(2, calendar, helpers, ql.Thirty360(ql.Thirty360.BondBasis))
        return curve

    def _get_dates(self, T1, T2, t, calendar):
        spot_lag = 2
        spot_date = calendar.advance(t, ql.Period(spot_lag), ql.Days)  # spot date
        if isinstance(T1, str):
            if T1 in ["0D", "2D"]:
                T1 = calendar.advance(t, ql.Period(spot_lag), ql.Days)
            else:
                T1 = calendar.advance(spot_date, ql.Period(T1))

        if isinstance(T2, str):
            T2 = calendar.advance(T1, ql.Period(T2))
        return T1, T2

    def _get_swap(self, T1, T2, index, notional, yts, engine, calendar, payoff):
        # Build swap
        if index == "SOFR":

            float_index = ql.OvernightIndex("SOFR", 0, ql.USDCurrency(), calendar, ql.Actual360(), yts)

            schedule = ql.Schedule(T1, T2, ql.Period("1Y"), calendar,
                                   ql.ModifiedFollowing, ql.ModifiedFollowing,
                                   ql.DateGeneration.Forward, False)

            swap = ql.OvernightIndexedSwap(
                ql.OvernightIndexedSwap.Payer if payoff.lower() == "payer" else ql.OvernightIndexedSwap.Receiver,
                notional,
                schedule,
                0.01,  # dummy fixed rate
                ql.Actual360(),
                float_index
            )

        elif index == "LIBOR":
            # ql_index = get_ql_index(ccy, index, yts)
            ql_index = ql.USDLibor(ql.Period("3M"), yts)

            fixed_schedule = ql.Schedule(
                T1, T2,
                ql.Period("6M"),  # Fixed leg frequency -
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Forward, False
            )

            float_schedule = ql.Schedule(
                T1, T2,
                ql.Period("3M"),  # Floating leg frequency
                calendar,
                ql.ModifiedFollowing,
                ql.ModifiedFollowing,
                ql.DateGeneration.Forward, False
            )

            swap = ql.VanillaSwap(
                ql.VanillaSwap.Payer if payoff.lower() == "payer" else ql.VanillaSwap.Receiver,  # or Receiver
                notional,
                fixed_schedule,
                0,
                ql.Thirty360(ql.Thirty360.BondBasis),  # Fixed leg day count
                float_schedule,
                ql_index,
                0.0,  # spread on float leg
                ql.Actual360()  # Floating leg day count
            )

        else:
            raise ValueError(f"Unsupported index: {index}")
        return swap

    def __call__(self, ccy, payoff, t, T1, T2, curve_data=None):
        ql.Settings.instance().evaluationDate = t
        index = utils.get_index(ccy, t)

        if not isinstance(curve_data, pd.DataFrame):
            curve_data = data.SwapCurveData()(ccy, index)
        if t not in curve_data.index:
            raise ValueError(f"Date {t} not in curve data index")

        yc = curve_data.loc[t] * 0.01
        instruments = [("swap", T, s) for T, s in yc.items()]

        yts = ql.RelinkableYieldTermStructureHandle()
        calendar = ql.UnitedStates(ql.UnitedStates.Settlement) if ccy == "USD" else ql.TARGET()
        notional = 10e6

        curve = self._build_curve(instruments, index, calendar, yts)
        yts.linkTo(curve)
        engine = ql.DiscountingSwapEngine(yts)

        T1, T2 = self._get_dates(T1, T2, t, calendar)
        swap= self._get_swap(T1, T2, index, notional, yts, engine, calendar, payoff)

        swap.setPricingEngine(engine)
        F = swap.fairRate()

        schedule = swap.fixedSchedule()
        fixed_day_count = (ql.Actual360() if index == "SOFR" else ql.Thirty360(ql.Thirty360.BondBasis))

        annuity = self._get_annuity_from_schedule(schedule, fixed_day_count, yts)
        bpv = annuity * notional * 1e-4

        print(f" {T1}{T2} - Annuity: {annuity:.6f}, BPV: {bpv:.2f}")

        return dict(F=F, Annuity=annuity)


def _check_swap_rate_against_nodes(index):
    '''
    Checked on a example for LIBOR that the repriced curve by the swap engine is within 1 bp of the market curve for the full curve
    '''

    if index=="LIBOR":
        mkt_curve = data.SwapCurveData()("USD", "LIBOR")
        mkt_curve_t = mkt_curve.loc[ql.Date(8, 1, 2021)] * 0.01

        repriced_curve = {T: IRSwap()(ccy="USD", payoff="Payer", t=ql.Date(8, 1, 2021), T1="0D", T2=T, curve_data=mkt_curve).get("F") for T in list(mkt_curve_t.index)[:-1]}

        res = dict(Market=mkt_curve_t,
                   Repriced=pd.Series(repriced_curve))

        res = pd.DataFrame(res).loc[mkt_curve_t.index[:-1]]
        res.to_excel(r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\strategies\test\StrategyTest\libor_swap_curve_example2.xlsx")

    elif index=="SOFR":
        # # FIXME can't reprice the nodes with SOFR swaps !!
        mkt_curve = data.SwapCurveData()("USD", "SOFR")
        mkt_curve_t = mkt_curve.loc[ql.Date(8, 1, 2025)]*0.01
        repriced_curve = {T: IRSwap()(ccy="USD", payoff="Payer", t=ql.Date(8, 1, 2025), T1="0D", T2=T, curve_data=mkt_curve).get("F") for T in list(mkt_curve_t.index)[:-1]}

        res = dict(Market=mkt_curve_t, Repriced=pd.Series(repriced_curve))

        res = pd.DataFrame(res).loc[mkt_curve_t.index[:-1]]
        res.to_excel(r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\strategies\test\StrategyTest\sofr_swap_curve_example.xlsx")
    else:
        raise ValueError(f"Unsupported index: {index}")


def main():
    # IRSwap()(ccy="USD", payoff="Payer", t=ql.Date(8,1,2021), T1="2D", T2="3M", curve_data=None)
    _check_swap_rate_against_nodes(index="LIBOR")
    pass

if __name__ == '__main__':
    main()

