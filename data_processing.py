from settings import FILE_PATH
import pandas as pd


class Data(object):
    '''
    Data processing class - very light here given time constraint, but in practice would handle:
        - corporate actions, splits, mergers etc
        - chg in index composition over time + listing, delisting etc

    '''
    def __init__(self, settings):
        for key, val in settings.items():
            setattr(self, key, val)
        self.run()

    def run(self):
        data = pd.read_csv(f"{FILE_PATH}//data//{self.index_universe}_data.csv", index_col=0, parse_dates=True)
        data.index =data.index.tz_localize(None)
        sectors = pd.read_csv(f"{FILE_PATH}//data//{self.index_universe}_sector.csv", index_col=0)
        unique_sectors = sectors["GIC_sector"].unique()

        self.clusters = {s: sorted(list(sectors[sectors.GIC_sector == s].index)) for s in unique_sectors}
        self.data=data.reindex(self.schedule.training_dates)
