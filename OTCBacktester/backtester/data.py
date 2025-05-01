from ast import parse

PATH_DATA = r"C:\Users\Romain\PycharmProjects\pythonProject\OTCBacktester\backtester\data"
import pandas as pd
from OTCBacktester.backtester import utils

class DataToParquet:
    def __init__(self):
        pass
        self.run()

    def sofr_spot_rate(self):
        xls = pd.ExcelFile(r"C:\Users\Romain\OneDrive\Desktop\Trading\data\Bloomberg\swap_data.xlsx")
        df = xls.parse(sheet_name="usd_sofr_spot_rates", index_col=0, parse_dates=True)

        mapping =xls.parse(sheet_name="ticker_mapping")
        m = mapping.query("underlying=='SOFR' & type == 'SPOT_RATE'")
        rename = dict(zip(m.ticker, m.expiry_tenor))
        df = df.rename(rename, axis=1)
        df.to_parquet(f"{PATH_DATA}\\sofr_spot_rate.parquet", engine="pyarrow", compression="snappy")

    def libor_spot_rate(self):
        xls = pd.ExcelFile(r"C:\Users\Romain\OneDrive\Desktop\Trading\data\Bloomberg\swap_data.xlsx")
        df = xls.parse(sheet_name="usd_libor_spot_rates", index_col=0, parse_dates=True)

        mapping =xls.parse(sheet_name="ticker_mapping")
        m = mapping.query("underlying=='LIBOR' & type == 'SPOT_RATE'")
        rename = dict(zip(m.ticker, m.expiry_tenor))
        df = df.rename(rename, axis=1)
        df.to_parquet(f"{PATH_DATA}\\libor_spot_rate.parquet", engine="pyarrow", compression="snappy")

    def run(self):
        # self.sofr_spot_rate()
        # self.libor_spot_rate()
        pass
class SwapCurveData:

    def __init__(self):
        pass

    def usd_libor_swap_curve(self):
        df = pd.read_parquet(f"{PATH_DATA}\\libor_spot_rate.parquet")
        # df.index = pd.to_datetime(df.index)
        df.index = [utils.datetime_to_ql_date(t) for t in df.index]
        df = df.dropna(axis=1, how="all")
        return df

    def usd_sofr_swap_curve(self):
        df = pd.read_parquet(f"{PATH_DATA}\\sofr_spot_rate.parquet")
        df.index = [utils.datetime_to_ql_date(t) for t in df.index] # pd.to_datetime(df.index)
        df = df.dropna(axis=1, how="all")
        return df

    def __call__(self, ccy, index):
        if ccy == "USD" and index == "LIBOR":
            return self.usd_libor_swap_curve()
        elif ccy == "USD" and index == "SOFR":
            return self.usd_sofr_swap_curve()
        elif ccy == "EUR" and index == "Euribor":
            return self.eur_euribor_swap_curve()


def main():
    # DataToParquet()
    pass

    SwapCurveData()("USD", "SOFR")

if __name__ == '__main__':
    main()


