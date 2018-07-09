import urllib.request
import zipfile
from pathlib import Path
from mpl_toolkits.basemap import Basemap

import pandas as pd
import progressbar
import matplotlib.pyplot as plt
import numpy as np


class HouseData(object):

    def __init__(self, sample_size, validation_size):
        self.column_names = ['Transaction unique identifier', 'Price', 'Date of Transfer', 'Postcode', 'Property Type',
                             'Old/New', 'Duration', 'PAON', 'SAON', 'Street', 'Locality', 'Town/City', 'District',
                             'County', 'PPD Category Type', 'Record Statu']
        self.url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2017.csv"
        self.postcodes_url = "https://www.freemaptools.com/download/full-postcodes/ukpostcodes.zip"
        self.data = None
        self.sample_size = sample_size
        self.validation_size = validation_size

    def check_download(self):

        def show_progress(block_num, block_size, total_size):
            if self.pbar is None:
                self.pbar = progressbar.ProgressBar(maxval=total_size)

            downloaded = block_num * block_size
            if downloaded < total_size:
                self.pbar.update(downloaded)
            else:
                self.pbar.finish()
                self.pbar = None

        data_file = Path('data.csv')
        if not data_file.exists():
            print('downloading house price data...')
            self.pbar = None
            urllib.request.urlretrieve(self.url, 'data.csv', show_progress)
        else:
            print('house price data already downloaded')

        postcodes_file = Path('postcodes.zip')
        if not postcodes_file.exists():
            print('downloading postcodes data...')
            self.pbar = None
            urllib.request.urlretrieve(self.postcodes_url, 'postcodes.zip', show_progress)
        else:
            print('postcodes data already downloaded')

        postcodes_unzipped_file = Path('ukpostcodes.csv')
        if not postcodes_unzipped_file.exists():
            print('unzipping...')
            postodes_file = zipfile.ZipFile('postcodes.zip')
            postodes_file.extractall()
            postodes_file.close()
        else:
            print('postcodes data already unzipped')

    def load(self):
        prices_data = pd.read_csv('data.csv', header=None, names=self.column_names)
        postcodes_data = pd.read_csv('ukpostcodes.csv')

        self.data = prices_data.merge(postcodes_data, left_on='Postcode', right_on='postcode', how='left')

    def check_oxford(self):
        print(self.data[self.data['Postcode'].str.contains("OX2 9EY", na=False)])

    def plot_sample_map(self):
        fig, ax = plt.subplots(figsize=(10, 20))
        m = Basemap(resolution='f',  # c, l, i, h, f or None
                    projection='merc',
                    lat_0=54.5, lon_0=-4.36,
                    llcrnrlon=-6., llcrnrlat=49.5, urcrnrlon=2., urcrnrlat=55.2)

        m.drawmapboundary(fill_color='#46bcec')
        m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
        m.drawcoastlines()

        num_points = 10000
        sample_data = self.data.sample(num_points)
        x, y = m(sample_data[['longitude']].values, sample_data[['latitude']].values)
        sizes = sample_data[['Price']].values / 500000

        for i in range(len(sizes)):
            m.plot(x[i], y[i], 'o', markersize=sizes[i], alpha=0.8)
        plt.savefig('sample_house_data.pdf', format='pdf')

    def plot_all_map(self):
        fig, ax = plt.subplots(figsize=(10, 20))
        m = Basemap(resolution='f',  # c, l, i, h, f or None
                    projection='merc',
                    lat_0=54.5, lon_0=-4.36,
                    llcrnrlon=-6., llcrnrlat=49.5, urcrnrlon=2., urcrnrlat=55.2)

        m.drawmapboundary(fill_color='#46bcec')
        m.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
        m.drawcoastlines()

        x, y = m(self.data[['longitude']].values, self.data[['latitude']].values)
        sizes = self.data[['Price']].values / 500000

        for i in range(len(sizes)):
            m.plot(x[i], y[i], 'o', markersize=sizes[i], alpha=0.8)
        plt.savefig('all_house_data.pdf', format='pdf')

    def filter(self):
        # select only apartments and flats
        self.data = self.data.dropna(subset=['Postcode', 'longitude', 'latitude', 'Price'], how='any')
        self.data = self.data[self.data['Property Type'] == "F"]
        self.data = self.data.groupby(['Postcode', 'longitude', 'latitude']).median().reset_index()

    def standardise_prices(self):
        self.data['Price'] = np.log(self.data['Price'])
        self.prices_mean = self.data['Price'].mean()
        self.prices_std = self.data['Price'].std()
        self.data['Price'] = (self.data['Price'] - self.data['Price'].mean()) / self.data['Price'].std()


    def generate_sample(self):
        sample_data = self.data.sample(self.sample_size)

        x = sample_data[['longitude', 'latitude']].values
        y = np.squeeze(sample_data[['Price']].values)

        return x, y

    def generate_validation(self):
        validation_data = self.data.sample(self.validation_size)

        self.x_validation = validation_data[['longitude', 'latitude']].values
        self.f_validation = np.squeeze(validation_data[['Price']].values)

    def prepare(self):
        self.check_download()
        self.load()
        self.filter()
        self.standardise_prices()
        self.generate_validation()


if __name__ == "__main__":
    data = HouseData(sample_size=1, validation_size=1)
    data.check_download()
    data.load()
    data.check_oxford()

