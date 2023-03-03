# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:46:54 2021

@author: MITRB
"""

'''
Creating an example that can access all my hydropower codes
'''

from usgs_api import get_data
from hydrogenerate import calculate_potential,get_energy
from hydrogenerate_plot import get_plot

'''
USGS site: https://maps.waterdata.usgs.gov/mapper/index.html
'''

site_no='11421000'
begin_date = '2019-01-01'
end_date = '2019-03-20'

'''
This function is called when we want to get data from USGS server

get_data(site_no,begin_date,end_date)

Returns a dataframe


'''

flow_info = get_data(site_no,begin_date,end_date)

'''
If you have your own data read it as a dataframe and follow the rest of the
procedure

import pandas as pd

flow_info = pd.read_csv(r'path\file.csv')  #Example
'''

'''
calculate_potential(flow_info, rated_flow=None, rated_power=None, turb= None, 
                        head_input=None,op='Timeseries', sys_effi=None,
                        system='pipe', flow_column='Flow (cfs)')

Returns a dictionary {'power', 'efficiency', 'flow', 'turb_cap'}

power: Numpy array of estimated power in MW

efficiency: Numpy arrary of estimated turbine efficiency in percentage

flow: Numpy array of water flow rate in cubic.ft/sec

turb_cap: Float value of estimated turbine capacity in MW

Input system determines the percentage loss or head loss;
For 'pipe': 5%; 'canal': 20% and 'reservoir': 0.1%
'''

x= calculate_potential(flow_info,head_input=40,op='Timeseries',system='reservoir', 
                        flow_column='Flow (cfs)')

'''
Assigning Variable Names to the elements in the dictionary
'''
power = x["power"]

efficiency = x["efficiency"]

flow_range = x["flow"]

turb_cap = x["turbine_capacity"]

'''
get_energy(power,flow_info,turb_cap,op,energy_cost=None,const_cost=None)

energy_cost defaults to 4 cents/kWh
const_cost defaults to 4.1 million $/MW

Returns:
Total Estimated Energy in MWh
Expected Annual Revenue in $
Expected Construction Cost in $
'''

tot_mwh,revenue,const_cost = get_energy(power,flow_info,turb_cap,op='Timeseries')


'''
get_plot(flow_info,power,efficiency,flow_range,op,font='Times New Roman',
             fontsize=None)

fontsize defaults to 15, font defaults to Times New Roman

Returns a plot
'''

get_plot(power,efficiency,flow_range,flow_info=flow_info,op='Timeseries')