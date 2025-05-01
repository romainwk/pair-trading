from datetime import datetime
from OTCBacktester.backtester.ccy_settings import SOFR_TRANSITION_DATE

import QuantLib as ql

def dates_to_iso(df):
    for col in df.select_dtypes(include=object):  # Simple heuristic
        if isinstance(df[col].iloc[0], ql.Date):
            df[col] = df[col].apply(lambda d: d.ISO())
    return df

def string_to_ql_date(date_str):
    year, month, day = map(int, date_str.split('-'))
    return ql.Date(day, month, year)

def datetime_to_ql_date(t):
    return ql.Date(t.day, t.month, t.year)


def iso_to_dates(df):
    for col in df.select_dtypes(include=object):
        sample = df[col].iloc[0]
        if isinstance(sample, str) and '-' in sample:  # crude check for ISO date
            try:
                df[col] = df[col].apply(string_to_ql_date)
            except:
                pass
    return df

def get_index(ccy, t):
    if not isinstance(t, ql.Date):
        t = datetime_to_ql_date(t)

    if ccy == "USD" and t<= SOFR_TRANSITION_DATE:
        return "LIBOR"
    elif ccy == "USD" and t > SOFR_TRANSITION_DATE:
        return "SOFR"
    elif ccy == "EUR":
        return "EURIBOR"