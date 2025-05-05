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

    def sofr_normal_vol(self):
        xls = pd.ExcelFile(r"C:\Users\Romain\OneDrive\Desktop\Trading\data\Bloomberg\vol_data_usd_sofr.xlsx")
        df = xls.parse(sheet_name="usd_sofr_normal_vol", index_col=1, parse_dates=True)

        df_new = df[["ID", "Value"]].drop_duplicates()

        idx = pd.MultiIndex.from_tuples(tuple(zip(df_new.index, df_new.ID)))
        df_new.index = idx
        vol_cube_cols = list(df.ID.drop_duplicates())

        df_new = df_new["Value"].unstack()[vol_cube_cols]
        df_new.to_parquet(f"{PATH_DATA}\\sofr_normal_vol.parquet", engine="pyarrow", compression="snappy")

    def libor_normal_vol(self):
        xls = pd.ExcelFile(r"C:\Users\Romain\OneDrive\Desktop\Trading\data\Bloomberg\vol_data_usd_libor.xlsx")
        df = xls.parse(sheet_name="usd_libor_normal_vol", index_col=2, parse_dates=True)

        df_new = df[["ID", "Value"]].drop_duplicates()

        idx = pd.MultiIndex.from_tuples(tuple(zip(df_new.index, df_new.ID)))
        df_new.index = idx
        vol_cube_cols = list(df.ID.drop_duplicates())

        df_new = df_new["Value"].unstack()[vol_cube_cols]
        df_new.to_parquet(f"{PATH_DATA}\\libor_normal_vol.parquet", engine="pyarrow", compression="snappy")

    def run(self):
        # self.sofr_spot_rate()
        # self.libor_spot_rate()
        # self.sofr_normal_vol()
        self.libor_normal_vol()
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

class VolCubeData:

    def __init__(self):
        pass

    # data missing !!
    def usd_libor_vol_cube(self):
        df = pd.read_parquet(f"{PATH_DATA}\\libor_normal_vol.parquet")
        df.index = [utils.datetime_to_ql_date(t) for t in df.index]
        df = df.where(df>0).ffill().dropna(axis=1, how="all")
        return df

    def usd_sofr_vol_cube(self):
        df = pd.read_parquet(f"{PATH_DATA}\\sofr_normal_vol.parquet")
        df.index = [utils.datetime_to_ql_date(t) for t in df.index] # pd.to_datetime(df.index)
        df = df.where(df>0).ffill().dropna(axis=1, how="all")
        return df

    def __call__(self, ccy, index):
        if ccy == "USD" and index == "LIBOR":
            return self.usd_libor_vol_cube()
        elif ccy == "USD" and index == "SOFR":
            return self.usd_sofr_vol_cube()
        else:
            raise ValueError(f"Vol cube for {ccy} and {index} not implemented yet")

def main():
    DataToParquet()
    pass

    # SwapCurveData()("USD", "SOFR")

if __name__ == '__main__':
    main()


