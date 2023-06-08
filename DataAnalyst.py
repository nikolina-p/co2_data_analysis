#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:51:12 2023

@author: nikolina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import pearsonr
import plotly.graph_objects as go


df_emission = pd.read_csv("CLEAN/emissions_clean.csv")
df_gdp = pd.read_csv("CLEAN/gdp_clean.csv")
df_pop = pd.read_csv("CLEAN/population_clean.csv")


# total CO2 emission for the period
def grand_total(data):
    return np.around(np.sum(data), decimals=2)

# average per year
def avg_per_year(data):
    return np.around(np.mean(np.sum(data, axis=0)), decimals=2)


# total increase per year (ndarray)
def year_increase(df_ems):
    emissions = df_ems.iloc[:, 3:].values    
    total_emission = np.sum(emissions, axis=0)    
    increase = np.diff(total_emission)
    return increase


#return 2-D ndarray with years and values of max and min increase in CO2 emission 
def get_max_min_increase_yr():
    inc_period = year_increase(df_emission)
    max_v = round(np.max(inc_period),2)
    max_Y = int(np.argmax(inc_period)+1991)
    min_v = round(np.min(inc_period),2)
    min_Y = int(np.argmin(inc_period)+1991)        
    return [max_Y, max_v, min_Y, min_v]

    
# find the country with the biggest emission for each year
def max_emission_per_year(df_ems):
    df_ems.set_index("Country Name", inplace=True)
    years = df_ems.iloc[:, 3:].idxmax().index
    countries = df_ems.iloc[:, 3:].idxmax().values    
    emissions = df_ems.iloc[:, 3:].values    
    max_ems = np.max(emissions, axis=0)        
    result = np.column_stack((years, countries, max_ems))    
    return result


# calculate Pearson correlation coeficient between CO2 and GDP/population
def pearson_Co2(data):
    co2_data = df_emission.iloc[:,3:].values    
    pearson = np.array([])
    p_value = np.array([])    
    for i in range(0,co2_data.shape[0]):
        correlation_coef, p_val = pearsonr(data[i,:], co2_data[i,:])
        pearson = np.append(pearson, np.round(correlation_coef,3))
        p_value = np.append(p_value, np.round(p_val,3))        
    pearson = np.nan_to_num(pearson, nan=0)
    return pearson, p_value
        
    
# save the main conclusions into the report
def create_report(df_ems, corel1, p_val1, corel2, p_val2):
    data = df_ems.iloc[:, 3:].values
    
    rep = []
    title = "\n**** CO2 Emissions Report ****\n\n\n"    
    ln1 = f"TOTAL EMISSION FOR THE PERIOD (kt): {grand_total(data)}\n\n"    
    ln2 = f"AVERAGE PER YEAR (kt): {avg_per_year(data)}\n\n"     
    ln3 = f"MAX CO2 EMISSION PER YEAR:\n{max_emission_per_year(df_ems)}\n\n"
    sum_1990 = df_ems.loc[:,'1990'].sum()
    sum_2019 = df_ems.loc[:,'2019'].sum()
    dif = sum_2019-sum_1990
    ln4 = f"TOTAL CO2 EMISSION IN THE FIRST YEAR (kt): {round(sum_1990/1000000,2)}M\n"
    ln5 = f"TOTAL CO2 EMISSION AT THE END OF PERIOD (kt): {round(sum_2019/1000000,2)}M\n"
    ln6 = f"INCREASE IN CO2 EMISSION/YEAR: {round(dif/1000000,2)}M"\
        f" (~{int(((dif)/sum_1990)*100)}%)\n\n"
    mm = get_max_min_increase_yr()
    ln7 = f"MAX INCREASE IN CO2 EMISSION WAS IN YEAR {mm[0]}({mm[1]})kt\n"
    a = "DECREASE" if mm[3] < 0 else "MINIMUM INCREASE"
    ln8 = f"{a} IN CO2 EMISSION WAS IN YEAR {mm[2]}({mm[3]})kt\n\n\n"
    subtitle = "*** CORRELATION ANALYSIS ***\n\nPearson coeficient is "\
        "calculated to measure linear correlation between CO2 emission "\
            "and GDP/population size.\n"
    
    ln9 = f"CO2/GDP Pearson correlation coeficient values for each of "\
        f"190 countries are: \n{corel1}\n"\
        f"Number of p_values<0.05: {np.sum(p_val1<0.05)}\n\n" 
    ln10 = f"CO2/population size Pearson correlation coeficient values are:"\
        f"\n{corel2}\nNumber of p_values<0.05: {np.sum(p_val2<0.05)}"
        
    rep.append(title)
    rep.append(ln1)
    rep.append(ln2)
    rep.append(ln3)
    rep.append(ln4)
    rep.append(ln5)
    rep.append(ln6)
    rep.append(ln7)
    rep.append(ln8)
    rep.append(subtitle)
    rep.append(ln9)
    rep.append(ln10)
    
    report = " ".join(rep)
    
    with open('REPORTS/CO2 report.txt', "w") as file:
        file.write(report)
                

# plot the total emission trend in time period 1990-2019
def emissions_plot(data):
    # preparing data ... 
    emissions = data.iloc[:, 3:].values    
    total_emission = np.sum(emissions, axis=0)
    max_emission = np.max(emissions, axis=0)        
    years = np.arange(1990, 1990+emissions.shape[1])
    
    #ploting ...
    plt.figure(num="fig1", figsize=(10, 6), dpi=240, facecolor='yellow')
    
    # Plot the total emission trend
    plt.plot(years, total_emission, color="blue", 
             linewidth="1.5", label="Total world emission")
    # plot the max poluter trend
    plt.plot(years, max_emission, color="red", 
             linewidth="2", label="The biggest polluter")
    
    # mark max/min increase 
    inc = year_increase(data)
    max_i = np.argmax(inc)
    min_i = np.argmin(inc)
    
    plt.scatter([max_i+1991,], [total_emission[max_i+1],], color="red")
    plt.scatter([min_i+1991,], [total_emission[min_i+1],], color="green")
    
    mm = get_max_min_increase_yr()
    plt.annotate(
        f"Max increase in emission\n{mm[1]}", 
        xy=(max_i+1991,total_emission[max_i+1]),
        xycoords="data",
        color="red",
        textcoords='offset points',
        xytext=(15, -15),
        fontsize=12
        )
    plt.annotate(
        f"Decrease in emission\n{mm[3]}", 
        xy=(mm[2],total_emission[mm[2]-1991]),
        xycoords="data",
        color="green",
        textcoords='offset points',
        xytext=(-30, -35),
        fontsize=12
        )
    
    plt.title('CO2 Emission Trend (1990-2019)')
    plt.legend(loc="upper left")
    plt.xlabel('Year')
    plt.xticks(np.arange(1990, 2019, 3))
    
    plt.ylabel('CO2 Emissions (kt)')        
    
    # format the y-axis tick labels to display values in millions
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x/1e6)}M')
    plt.gca().yaxis.set_major_formatter(formatter)
    
    plt.grid(True, which='major', color='b', linestyle='-')
    plt.savefig("REPORTS/CO2 emission.png")
    plt.show()
    
    
def correlation_co2_gdp_plot(pearson):
    labels = ['<-0.5', '0 to -0.5', '0 to 0.5', '>0.5']
    ranges = [-1, -0.5, 0, 0.5, 1]
    counts = [((pearson > ranges[i]) & (pearson <= ranges[i+1])).sum() 
              for i in range(len(ranges)-1)]
    
    plt.figure(num="fig1", figsize=(10, 6), dpi=240)
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title('Pearson Correlation Coefficient Groups - CO2/GDP')
    plt.savefig("REPORTS/corelation_pie_gdp.png")
    plt.show()
    

def correlation_co2_pop_plot(pearson):
    labels = ['<-0.5', '0 to -0.5', '0 to 0.5', '>0.5']
    ranges = [-1, -0.5, 0, 0.5, 1]
    counts = [((pearson > ranges[i]) & (pearson <= ranges[i+1])).sum() 
              for i in range(len(ranges)-1)]
    colors = ['yellow', 'green', 'blue', 'orange']
    
    plt.figure(num="fig1", figsize=(10, 6), dpi=240)
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title('Pearson Correlation Coefficient Groups - CO2/population size')
    plt.savefig("REPORTS/corelation_pie_pop.png")
    plt.show()
      
    
# plot corelation between GDP and CO2 emission - interactive
def gdp_CO2_country_plot(emission_data, gdp_data):
    countries = emission_data.iloc[:, 1].values
    emissions = np.round(np.mean(emission_data.iloc[:, 3:].values, axis=1))
    gdp = np.round(np.mean(gdp_data.iloc[:, 3:].values, axis=1)/1000000)

    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(x=gdp, 
                                    y=emissions, 
                                    mode='markers', 
                                    hovertext=countries
                                    )
                    )
    # Customize the hover tooltip
    fig.update_traces(hovertemplate=
                      "Country: %{hovertext}<br>GDP(millions): %{x}<br>"
                      "CO2 Emissions: %{y}")

    # Set labels and title
    fig.update_layout(xaxis_title='GDP (millions)', 
                      yaxis_title='CO2 Emissions(kt)', 
                      title='GDP vs CO2 Emissions - average values')

    fig.write_html("REPORTS/correlation_plot.html")
    fig.show() 
     
    
# get correlation coefficients 
gdp_data = df_gdp.iloc[:,3:].values     
p_gdp, p_val_gdp = pearson_Co2(gdp_data) 

pop_data = df_pop.iloc[:,3:].values
p_pop, p_val_pop = pearson_Co2(pop_data)

# plot correlation pies
correlation_co2_gdp_plot(p_gdp)
correlation_co2_pop_plot(p_pop)

# CO2 emission trends
emissions_plot(df_emission)

# interactive scatter plot - countries and GDP
gdp_CO2_country_plot(df_emission, df_gdp)

# text report
create_report(df_emission, p_gdp, p_val_gdp, p_pop, p_val_pop)
