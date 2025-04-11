import pandas as pd

# normal vol vs OIS
def _sofr_swaptions_vol():
    expiries = {"1M": "A",
                "3M": "C",
                "6M": "F",
                "9M": "I",
                }
    expiries = {**expiries, **{f"{i}Y":i for i in [1,2,3,5,7,10,15,20,30]}}

    expiry_letters = { 10: "J",
                       15: "O",
                       20: "T",
                       30 : "Z",
                       }


    tenors = {f"{i}Y":i for i in [1,2,3,5,7,10,15,20,25,30]}

    # moneyness = {-200: "RH",
    #              -100: "RE",
    #              -50: "RC",
    #              -25:"RB",
    #              "ATM": "SN",
    #                 25: "RO",
    #              50:"RZ",
    #              100:"RQ",
    #              200:"RT",
    #              }
    #
    # absolute SOFR Normal Strikes
    moneyness = {-200: "WG",
                 -100: "WE",
                 -50: "WC",
                 -25:"WB",
                 "ATM": "SN",
                    25: "WL",
                 50:"WM",
                 100:"WO",
                 200:"WR",
                 }

    source = "TRPU"

    ticker_tbl = {}
    for expiry in expiries:
        for tenor in tenors:
            for strike in moneyness:
                if isinstance(expiries[expiry], int) and expiries[expiry]>=10 and tenors[tenor] >=10:
                    exp = expiry_letters[expiries[expiry]]
                else:
                    exp = expiries[expiry]
                ticker_tbl[f"{expiry}{tenor}{strike}"] = f"US{moneyness[strike]}A{exp}{tenors[tenor]} {source} Curncy"

    s = pd.Series(ticker_tbl)
    s.to_excel("ticker_mapping_sofr_normal_vol.xlsx", index=True, header=False)

# normal vol vs OIS - listed as spread vs ATM
def _libor_swaptions_vol():
    expiries = {"1M": "A",
                "3M": "C",
                "6M": "F",
                "9M": "I",
                }
    expiries = {**expiries, **{f"{i}Y": i for i in [1, 2, 3, 5, 7, 10, 15, 20, 30]}}


    tenors = {f"{i}Y": i for i in [1, 2, 3, 5, 7, 10, 15, 20, 30]}

    tenor_letters = {10: "J",
                      15: "O",
                      20: "T",
                      30: "Z",
                      }
    #
    # relative LIBOR Normal Strikes
    moneyness = {-200: "SRD",
                 -100: "SRC",
                 -50: "SRB",
                 -25: "SRA",
                 "ATM": "SN",
                 25: "SPA",
                 50: "SPB",
                 100: "SPC",
                 200: "SPD",
                 }

    source = "BAOP" # Bank of America marks

    ticker_tbl = {}
    for expiry in expiries:
        for tenor in tenors:
            for strike in moneyness:
                exp = expiries[expiry]
                ten = tenors[tenor]
                # k = moneyness[strike]

                if (strike != "ATM"):
                    if len(str(exp)) == 1 and len(str(ten)) == 1:
                        ten = f"0{ten}"
                    elif len(str(exp)) == 2 and len(str(ten)) == 2:
                        ten = tenor_letters[ten]

                else:
                    if len(str(exp))==1:
                        exp = f"0{exp}"
                    # don't change expiries / tenors with letters for ATMs


                ticker_tbl[f"{expiry}{tenor}{strike}"] = f"US{moneyness[strike]}{exp}{ten} {source} Curncy"

    s = pd.Series(ticker_tbl)
    s.to_excel("ticker_mapping_libor_normal_vol.xlsx", index=True, header=False)

def _sofr_caps_floors():
    # caps and floors are quoted as absolute strikes - e.g. 0.50% corresponds to a cap /floor of 0.50% strike

    expiries = {f"{i}Y":i for i in list(range(1,11)) + [12, 15, 20, 25, 30]}

    # absolute SOFR Normal Strikes (in %)
    moneyness = {-0.75: "CLQD",
                 -0.50: "CLQC",
                 -0.25: "CLQB",
                 0:"CNQA",
                 0.25:"CNQB",
                 0.50:"CNQC",
                 0.75: "CNQD",
                 1: "CNQE",
                 1.5: "CNQG",
                 2: "CNQH",
                 2.5: "CNQI",
                 3: "CNQJ",
                 3.5: "CNQK",
                 4: "CNQL",
                 4.5: "CNQM",
                 5: "CNQO",
                 5.5: "CNQP",
                 6: "CNQQ",
                 6.5: "CNQR",
                 7: "CNQS",
                 "ATM": "CNSQ",
                 }

    source = "ICPL" # ICAP

    ticker_tbl = {}
    for expiry in expiries:
        for k in moneyness:
            ticker_tbl[f"{expiry}{k}"] = f"US{moneyness[k]}{expiries[expiry]} {source} Curncy"

    s = pd.Series(ticker_tbl)
    s.to_excel("ticker_mapping_sofr_caps_floors.xlsx", index=True, header=False)

def main():
    # _sofr_swaptions_vol()
    # _libor_swaptions_vol()
    _sofr_caps_floors()

# def _bbg_notebook_code():
#     import bql
#     bq = bql.Service()
#     import pandas as pd
#     # ticker = ['USSNAC3 ICPL Curncy'],
#     ticker_query = list(tickers.values())[:2]
#     # ticker_query = [\USWEAC1 TRPU Curncy\],
#
#     data_item = bq.data.px_last(
#         dates=bq.func.range('-1044D', '0D'),
#     )
#     request = bql.Request(ticker_query, data_item)
#     response = bq.execute(request)
#
#     data = response[0].df()
#
#     data = data.drop(["CURRENCY"], axis=1).dropna()
#     data["BloombergTicker"] = data.index
#     map_ = {v: k for k, v in tickers.items()}
#     data = data.rename(map_)
#
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#         display(data)


if __name__ == '__main__':
    # open Bloomberg Lab from BBG anywhere outside CITRIX
    # download t/s from BQNT - Bloomberg Lab
    # load security on terminal and type ALLQ to find all pricing sources available. See which one can be downloaded via Bloomberg Lab
    # export to csv in notebook. Copy paste to local
    main()

