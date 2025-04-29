import QuantLib as ql

def dates_to_iso(df):
    for col in df.select_dtypes(include=object):  # Simple heuristic
        if isinstance(df[col].iloc[0], ql.Date):
            df[col] = df[col].apply(lambda d: d.ISO())
    return df

def string_to_ql_date(date_str):
    year, month, day = map(int, date_str.split('-'))
    return ql.Date(day, month, year)

def iso_to_dates(df):
    for col in df.select_dtypes(include=object):
        sample = df[col].iloc[0]
        if isinstance(sample, str) and '-' in sample:  # crude check for ISO date
            try:
                df[col] = df[col].apply(string_to_ql_date)
            except:
                pass
    return df
