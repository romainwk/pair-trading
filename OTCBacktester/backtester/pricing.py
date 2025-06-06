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

class IRDerivative:
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

    def _build_curve(self, ccy, t,  index, calendar, curve_data=None):

        if not isinstance(curve_data, pd.DataFrame):
            curve_data = data.SwapCurveData()(ccy, index)
        if t not in curve_data.index:
            raise ValueError(f"Date {t} not in curve data index")

        yc = curve_data.loc[t] * 0.01
        instruments = [("swap", T, s) for T, s in yc.items()]

        yts = ql.RelinkableYieldTermStructureHandle()

        # curve = self._build_curve(instruments, index, calendar, yts)

        helpers = ql.RateHelperVector()

        for instrument, tenor, rate in instruments:
            rate_handle = ql.QuoteHandle(ql.SimpleQuote(rate))
            if instrument == 'swap':
                if index == "SOFR":
                    swap_index = ql.OvernightIndex("SOFR", 0, ql.USDCurrency(), calendar, ql.Actual360(), yts)
                    # float_index = ql.Sofr(yts)

                    settlement_days = 2
                    helper = ql.OISRateHelper(
                        settlement_days,
                        ql.Period(tenor),
                        rate_handle,
                        swap_index,
                        # yts,

                    )

                elif index == "LIBOR":

                    swap_index = ql.USDLibor(ql.Period("3M"), yts)

                    helper = ql.SwapRateHelper(
                        rate_handle,
                        ql.Period(tenor),  # tenor
                        calendar,
                        ql.Semiannual,  # fixed freq
                        ql.ModifiedFollowing,  # fixed BDC
                        ql.Thirty360(ql.Thirty360.BondBasis),  # fixed day count
                        swap_index,  # ibor index
                    )

                else:
                    raise ValueError(f"Unsupported index: {index}")

                helpers.append(helper)

            elif instrument == 'depo':
                helpers.append(helper)

        # curve = ql.PiecewiseLinearZero(2, calendar, helpers, ql.Actual365Fixed())
        if index == "SOFR":
            curve = ql.PiecewiseLogLinearDiscount(2, calendar, helpers, ql.Actual360())
            # curve = ql.LogLinearZeroCurve(ql.LogLinear(), ql.Period("1D"), helpers, ql.Actual360())
        elif index == "LIBOR":
            curve = ql.PiecewiseLinearZero(2, calendar, helpers, ql.Thirty360(ql.Thirty360.BondBasis))
            # curve = ql.PiecewiseYieldCurve(ql.LogLinear(), ql.Period("1D"), helpers,ql.Thirty360(ql.Thirty360.BondBasis))

        yts.linkTo(curve)
        return yts, curve

    def _get_dates(self, T1, T2, t, calendar):
        spot_lag = 2
        spot_date = calendar.advance(t, ql.Period(spot_lag), ql.Days)  # spot date

        if isinstance(T1, str):
            if T1 in ["0D", "2D"]:
                T1 = calendar.advance(t, ql.Period(2, ql.Days))  # spot date
            else:
                T1 = calendar.advance(calendar.advance(t, ql.Period(2, ql.Days)), ql.Period(T1))

        if isinstance(T2, str):
            T2 = calendar.advance(T1, ql.Period(T2))
        return T1, T2


class IRSwap(IRDerivative):

    def __init__(self):
        pass

    def _sofr_swap(self, T1, T2, K, index, notional, yts, calendar, payoff):
        float_index = ql.OvernightIndex("SOFR", 0, ql.USDCurrency(), calendar, ql.Actual360(), yts)
        # float_index = ql.Sofr(yts)

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

        engine = ql.DiscountingSwapEngine(yts)
        swap.setPricingEngine(engine)

        F = swap.fairRate()

        # rebuild the swap with the correct fixed rate (for compatibility with swaptions)
        if K == "ATMf":
            K = F
        else:
            K = F + K  # strike defined in shift vs ATMf

        swap = ql.OvernightIndexedSwap(
            ql.OvernightIndexedSwap.Payer if payoff.lower() == "payer" else ql.OvernightIndexedSwap.Receiver,
            notional,
            schedule,
            K,  # dummy fixed rate
            ql.Actual360(),
            float_index
        )
        swap.setPricingEngine(engine)

        return swap

    def _get_libor_swap(self, T1, T2, K, index, notional, yts, calendar, payoff):

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

        engine = ql.DiscountingSwapEngine(yts)
        swap.setPricingEngine(engine)

        F = swap.fairRate()
        if K == "ATMf":
            K = F
        else:
            K = F + K

        # rebuild the swap with the correct fixed rate (for compatibility with swaptions)
        swap = ql.VanillaSwap(
            ql.VanillaSwap.Payer if payoff.lower() == "payer" else ql.VanillaSwap.Receiver,  # or Receiver
            notional,
            fixed_schedule,
            K,
            ql.Thirty360(ql.Thirty360.BondBasis),  # Fixed leg day count
            float_schedule,
            ql_index,
            0.0,  # spread on float leg
            ql.Actual360()  # Floating leg day count
        )
        swap.setPricingEngine(engine)

        return swap

    def _get_swap(self, T1, T2, K, index, notional, yts, calendar, payoff):
        # Build swap
        if index == "SOFR":
            swap = self._sofr_swap(T1, T2, K, index, notional, yts, calendar, payoff)

        elif index == "LIBOR":
            swap = self._get_libor_swap(T1, T2, K, index, notional, yts, calendar, payoff)

        else:
            raise ValueError(f"Unsupported index: {index}")
        return swap

    def __call__(self, ccy, payoff, t, T1, T2, K, curve_data=None):
        ql.Settings.instance().evaluationDate = t
        index = utils.get_index(ccy, t)

        calendar = ql.UnitedStates(ql.UnitedStates.Settlement) if ccy == "USD" else ql.TARGET()
        notional = 10e6

        yts, curve = self._build_curve(ccy, t,  index, calendar, curve_data=curve_data)

        T1, T2 = self._get_dates(T1, T2, t, calendar)
        swap= self._get_swap(T1, T2, K, index, notional, yts, calendar, payoff)

        F = swap.fairRate()

        schedule = swap.fixedSchedule()
        fixed_day_count = (ql.Actual360() if index == "SOFR" else ql.Thirty360(ql.Thirty360.BondBasis))

        annuity = self._get_annuity_from_schedule(schedule, fixed_day_count, yts)
        bpv = annuity * notional * 1e-4

        return dict(F=F, Annuity=annuity)

class IRSwaption(IRSwap):

    def __init__(self):
        pass

    def __call__(self, ccy, payoff, t, T1, T2, K, curve_data=None, vol_cube_data=None):
        ql.Settings.instance().evaluationDate = t
        index = utils.get_index(ccy, t)

        calendar = ql.UnitedStates(ql.UnitedStates.Settlement) if ccy == "USD" else ql.TARGET()
        notional = 1

        yts, curve = self._build_curve(ccy, t, index, calendar, curve_data=curve_data)
        # swap_engine = ql.DiscountingSwapEngine(yts)

        T1, T2 = self._get_dates(T1, T2, t, calendar)
        swap = self._get_swap(T1, T2, K, index, notional, yts, calendar, payoff)
        # swap.setPricingEngine(swap_engine)

        exercise = ql.EuropeanExercise(T1)
        swaption = ql.Swaption(swap, exercise)


        if not isinstance(vol_cube_data, pd.DataFrame):
            vol_cube_data = data.VolCubeData()(ccy, index)
        if t not in vol_cube_data.index:
            raise ValueError(f"Date {t} not in volcube data index")

        vol_cube = SwaptionVolCube()(ccy=ccy, t=t, curve_data=curve_data, vol_cube_data=vol_cube_data)

        engine = ql.BachelierSwaptionEngine(yts,
                                        vol_cube,
                                        )

        swaption.setPricingEngine(engine)
        iv = swaption.impliedVolatility(swaption.forwardPrice(), yts, 0.01, 1e-4, 100, 1e-7, 4,  ql.Normal) # ql.ShiftedLogNormal, ql.Normal

        return dict(ForwardPremium=swaption.forwardPrice(),
                    ImpliedVol=iv,
                    Strike=swap.fixedRate(),
                    Annuity=swaption.annuity(),
                    Vega=swaption.vega(),
                    Delta=swaption.delta(),
                    ForwardRate=swap.fairRate(),
                    )


class SwaptionVolCube(IRDerivative):
    def __init__(self):
        pass


    def _get_axis(self, index):
        if index == "SOFR":
            self.expiries = ['1M', '3M', '6M', '9M', '1Y', '2Y', '3Y',  '5Y', '7Y', '10Y', '15Y', '20Y', ]
            self.tenors = ['1Y', '2Y', '3Y',  '5Y', '7Y', '10Y', '15Y', '20Y', '25Y', '30Y']
            self.strikes = {-200: -0.02, -100: -0.01, -50: -0.005,-25: -0.0025,
                            # "ATM": 0,
                            25: 0.0025, 50: 0.005, 100: 0.01, 200: 0.02}



        elif index == "LIBOR":
            self.expiries = ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', ]
            self.tenors = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
            self.strikes = {-200: -0.02, -100: -0.01, -50: -0.005, -25: -0.0025,
                            # "ATM": 0,
                            25: 0.0025, 50: 0.005, 100: 0.01, 200: 0.02}

        else:
            raise ValueError(f"Unsupported index: {index}")

        self.ql_expiries = [ql.Period(T1) for T1 in self.expiries]
        self.ql_tenors = [ql.Period(T2) for T2 in self.tenors]

    def _get_atm_vol_matrix(self, ccy, t, vols, calendar):

        atm_vols = [[vols.loc[f"{T1}{T2}ATM"]*0.01 * 0.01 for T2 in self.tenors] for T1 in self.expiries]

        bdc = ql.ModifiedFollowing # TODO add settings
        dayCounter = ql.ActualActual(ql.ActualActual.ISDA)
        swaptionVolMatrix = ql.SwaptionVolatilityMatrix(
        calendar,
            bdc,
            self.ql_expiries,
            self.ql_tenors,
            ql.Matrix(atm_vols),
            dayCounter, False, ql.Normal)
        return swaptionVolMatrix

    def __call__(self, ccy, t, curve_data=None, vol_cube_data=None):

        ql.Settings.instance().evaluationDate = t
        index = utils.get_index(ccy, t)
        self._get_axis(index)
        calendar = ql.UnitedStates(ql.UnitedStates.Settlement) if ccy == "USD" else ql.TARGET()

        yts, curve = self._build_curve(ccy, t, index, calendar, curve_data=curve_data)

        if not isinstance(vol_cube_data, pd.DataFrame):
            vol_cube_data = data.VolCubeData()(ccy, index)
        if t not in vol_cube_data.index:
            raise ValueError(f"Date {t} not in volcube data index")

        vols = vol_cube_data.loc[t]

        atm_vol_matrix = self._get_atm_vol_matrix(ccy, t, vols, calendar)

        if index == "SOFR":
            sofr = ql.OvernightIndex("SOFR", 0, ql.USDCurrency(), calendar, ql.Actual360(), yts)
            swap_index = ql.OvernightIndexedSwapIndex("SOFR",
                                                      ql.Period("10Y"),
                                                      0,
                                                      ql.USDCurrency(),
                                                      sofr,
                                                      )

        elif index == "LIBOR":
            swap_index = ql.USDLibor(ql.Period("3M"), yts)
        else:
            raise ValueError(f"Unsupported index: {index}")

        if index == "SOFR":
            # since SOFR vol quotes are absolute vols
            spread_vols = [[(vols.loc[f"{T1}{T2}{K}"] * 0.01 * 0.01 - vols.loc[f"{T1}{T2}ATM"] * 0.01 * 0.01) for K in
                            self.strikes] for T2 in self.tenors for T1 in self.expiries]
        elif index == "LIBOR":
            # since LIBOR vol quotes are already given as spread vs ATM
            spread_vols = [[(vols.loc[f"{T1}{T2}{K}"] * 0.01 * 0.01) for K in self.strikes] for T2 in self.tenors for T1 in self.expiries]

        spread_vols = [[ql.QuoteHandle(ql.SimpleQuote(v)) for v in row] for row in spread_vols]
        vegaWeightedSmileFit = False

        volCube = ql.SwaptionVolatilityStructureHandle(
            ql.InterpolatedSwaptionVolatilityCube(
                ql.SwaptionVolatilityStructureHandle(atm_vol_matrix),
                self.ql_expiries,
                self.ql_tenors,
                list(self.strikes.values()),
                spread_vols,
                swap_index,
                swap_index,
                vegaWeightedSmileFit,
            )
        )
        volCube.enableExtrapolation()

        return volCube


def _check_swap_rate_against_nodes(index):
    '''
    Checked on a example for LIBOR that the repriced curve by the swap engine is within 1 bp of the market curve for the full curve
    '''

    if index=="LIBOR":
        # matches perfectly now
        mkt_curve = data.SwapCurveData()("USD", "LIBOR")
        mkt_curve_t = mkt_curve.loc[ql.Date(8, 1, 2021)] * 0.01

        repriced_curve = {T: IRSwap()(ccy="USD", payoff="Payer", t=ql.Date(8, 1, 2021), T1="0D", T2=T, curve_data=mkt_curve).get("F") for T in list(mkt_curve_t.index)[:-1]}

        res = dict(Market=mkt_curve_t,
                   Repriced=pd.Series(repriced_curve))

        res = pd.DataFrame(res).loc[mkt_curve_t.index[:-1]]
        res.to_excel(r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\strategies\test\StrategyTest\libor_swap_curve_example2.xlsx")

    elif index=="SOFR":
        # matches perfectly now
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
    IRSwaption()(ccy="USD", payoff="Payer", t=ql.Date(8,1,2021), T1="1Y", T2="10Y", K=0.50*0.01, curve_data=None, vol_cube_data=None)

    # SwaptionVolCube()(ccy="USD", t=ql.Date(8,1,2025), curve_data=None, vol_cube_data=None)

    # _check_swap_rate_against_nodes(index="LIBOR")
    # _check_swap_rate_against_nodes(index="SOFR")
    pass

if __name__ == '__main__':
    main()

