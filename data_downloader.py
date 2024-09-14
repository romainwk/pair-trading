import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
from settings import FILE_PATH

class DownloadIndexDataYahoo(object):

    def __init__(self, index_settings):
        self.index_settings = index_settings

    def _get_index_comp(self, index):
        resp = requests.get(self.index_settings[index].get("url"))
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        sector, industry = {}, {}
        ticker_pos = self.index_settings[index].get("ticker_pos")
        sector_pos = self.index_settings[index].get("gics_sector_pos")
        industry_pos = self.index_settings[index].get("gics_industry_pos")

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[ticker_pos].text
            ticker = ticker.replace('\n', '')
            tickers.append(ticker)
            sector[ticker] = row.findAll('td')[sector_pos].text
            industry[ticker] = row.findAll('td')[industry_pos].text

        self.tickers = tickers # [s.replace('\n', '') for s in tickers]
        sector = pd.DataFrame.from_dict(sector, orient="index").rename({0:"GIC_sector"}, axis=1)
        industry = pd.DataFrame.from_dict(industry, orient="index").rename({0: "GIC_sub_industry"}, axis=1)
        self.classification = pd.concat([sector, industry], axis=1)

    def _download_data(self, start, end):
        data = yf.download(self.tickers[-1:], start=start, end=end)
        self.data = data["Adj Close"]

    def _save_data(self, index):
        # self.data.to_csv(f"{index}_data.csv")
        self.classification.to_csv(f"{FILE_PATH}/data/{index}_classification.csv")

    def get_historical_ts(self, index, start, end):
        self._get_index_comp(index)
        # self._download_data(start, end)
        self._save_data(index)

def main():
    index_settings = {"S&P500": {"url": "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                 "ticker_pos": 0,
                                 "gics_sector_pos":2,
                                 "gics_industry_pos": 3,
                                 },

                      # "Russell1000": {"url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
                      #                 "ticker_pos": 1},
                      }

    download_obj = DownloadIndexDataYahoo(index_settings)

    start = datetime.datetime(2004, 1, 1)
    end = datetime.datetime.today()
    download_obj.get_historical_ts(index="S&P500", start=start, end=end)

if __name__ == '__main__':
    main()
