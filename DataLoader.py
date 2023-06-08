#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 18:09:24 2023

@author: nikolina

Data sets are downloaded from World Bank's website:
    https://data.worldbank.org/indicator/EN.ATM.CO2E.KT
    https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
    https://data.worldbank.org/indicator/SP.POP.TOTL?end=2021&start=1990
    
"""

import pandas as pd

# Load the CSV file into a DataFrame, skip the first 4 rows(empty/descriptive)
ds_emission = pd.read_csv("DATA/emission per country/"
                 "API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5455005.csv", skiprows=4)

ds_gdp = pd.read_csv("DATA/GDP/"
                 "API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5454986.csv", skiprows=4)

ds_pop = pd.read_csv("DATA/population/"
                     "API_SP.POP.TOTL_DS2_en_csv_v2_5454896.csv",skiprows=4)


def print_data_info(data_set):
    # print size of data frame 
    print(f"NUMBER OF (ROWS, COLUMNS) : {data_set.shape}")
    
    # some more info
    data_set.info(verbose=True)
    
    # print first 5 rows
    print(data_set.head())
    
    # find rows(countries) with no data(NaN)
    is_Nan = data_set.iloc[:, 4:].isna().all(axis='columns')
    no_data_countries = data_set[is_Nan]["Country Name"]
    print(f"COUNTRIES WITH NO DATA (count: {len(no_data_countries)}): "
          f"{no_data_countries}")


def clean_data_co2(data_set):
    #erase all columns with NaN values(missing values)
    data_set.dropna(axis=1, how='all', inplace=True)
    
    # erase data that is not relevant for further analysis
    data_set.drop(['Indicator Name', 'Indicator Code'], 
                             axis=1, inplace=True)
    
    # erase countries with no data
    data_set.dropna(axis="rows", thresh=10, inplace=True)
    
    # replace remaining NaN values with 0
    data_set.fillna(0, inplace=True)
    
    # Reset the index
    data_set.reset_index(drop=True, inplace=True)
    
    
def clean_data(data_set, data_co2):
    # erase data that is not relevant for further analysis
    data_set.drop(['Indicator Name', 'Indicator Code'], 
                             axis=1, inplace=True)
    
    # remove countries that have no data on CO2 emission
    data_set = data_set[
        data_set['Country Name'].isin(data_co2["Country Name"])]
    
    # replace remaining NaN values with 0
    data_set.fillna(0, inplace=True)
    
    # remove years with no data on co2 emission
    common_col = data_set.columns.intersection(data_co2.columns)
    data_set = data_set[common_col]
    
    # Reset the index
    data_set.reset_index(drop=True, inplace=True)
    return data_set


# exporting data for manual selection of region indexes in LibreOffice
#df_emission.to_csv('my_countries.csv', index=True)


# divide data frame to individual countries and regions
def divide_countries_regions(data_set):
    # indexes of regions selected manualy(LibreOffice) 
    region_indexes = [0,2,6,33,45,55,56,57,58,59,62,67,68,85,87,91,92,93,94,
                      96,115,121,122,123,126,127,129,137,140,145,153,163,165,
                      173,177,178,183,193,195,196,206,207,212,214,216,217,225,
                      233,236]
    
    co2_per_region = data_set.iloc[region_indexes]
    co2_per_country = data_set[~data_set.index.isin(region_indexes)]
    
    co2_per_region.reset_index(drop=True, inplace=True)
    co2_per_country.reset_index(drop=True, inplace=True)
    
    return co2_per_region, co2_per_country


# CO2 emission data ...
print_data_info(ds_emission)

# cleaning the data set with CO2 emission per country
clean_data_co2(ds_emission)

co2_per_region, co2_per_country = divide_countries_regions(ds_emission)
print_data_info(co2_per_country)

# GDP data ...
# cleaning the data set with GDP per country
gdp_per_country = clean_data(ds_gdp, co2_per_country)

# Population data ...
pop_per_country = clean_data(ds_pop, co2_per_country)

# saving cleaned data to files for further analysis
co2_per_country.to_csv("CLEAN/emissions_clean.csv")
gdp_per_country.to_csv("CLEAN/gdp_clean.csv")
pop_per_country.to_csv("CLEAN/population_clean.csv")


