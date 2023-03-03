# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:45:45 2020

@author: MITRB
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd
from matplotlib.ticker import NullFormatter
from matplotlib.dates import MonthLocator, DateFormatter,WeekdayLocator
import numpy as np


# from WPTO_turbine_eff_flow_option_advanced_VRG_website import *

def get_plot(power,efficiency,flow_range,op,rated_flow=None,flow_info=None,font=None,
             fontsize=None):
    
    if (flow_info is None) and (op=='Timeseries'):
        
        raise ValueError('Flow information dataframe missing')
        
    elif (flow_range is None) and (rated_flow is None) and (op =='Generalized'):
        raise ValueError('Flow array missing')
    
    if font is not None:
        
        font = font
    else:
        
        font ='Times New Roman'
    csfont = {'fontname':font}
    
    title = str(input("Provide a title for the figure: "))
    
    if fontsize is not None:
        font = fontsize
    else:
        font=15
    
    if(op=='Timeseries'):
        
            
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        
        #dates = flow_info['Date/Time'].map(lambda t: t.date()).unique()
        date = flow_info['Date/Time'].tolist()
        date = pd.to_datetime(date, format="%Y-%m-%d %H:%M")

        
        
        fig,host=plt.subplots()
        fig.subplots_adjust(right=0.85)
        #ax = fig.add_subplot(1,1,1)
        
        
        
        par1 = host.twinx()
        par2 = host.twinx()
        
        par2.spines["right"].set_position(("axes",1.2))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)
        p1,= host.plot(date,efficiency,'g-',label="Efficiency")
        p2,= par1.plot(date,power,'b-',label="Power")
        p3,= par2.plot(date,flow_range/0.028316846591999675,'r',label="Flow")
         
        
        host.set_ylabel("Efficiency (%)",color='g',fontsize=font,**csfont)
        par1.set_ylabel("Power (MW)",color='b',fontsize=font,**csfont)
        
        host.grid(True, axis='both', color='k',linestyle='--',alpha=0.4)
        #ax1.set_xlabel("Days")
        par2.set_ylabel("Flow rate (cu.ft/s)",color='r',fontsize=font)
        lines = [p1,p2,p3]
        host.legend(lines, [l.get_label() for l in lines], loc='upper center',bbox_to_anchor=(0.5,-0.2),fancybox=True,
                    shadow=True, ncol=5)
        
        xfmt = mdates.DateFormatter('%b.\n%y')
        months = mdates.MonthLocator()
        plt.gca().xaxis.set_major_locator(MonthLocator())
        plt.gca().xaxis.set_major_formatter(xfmt)
        
        plt.title(title,fontsize=font,**csfont)
        
        #plt.gcf().autofmt_xdate()
        plt.show()
        
    else:
        
        effi_cal= np.insert(efficiency,0,0)
        flow_range = flow_range/(rated_flow*0.028316846591999675)
        flow_arr1 = np.insert(flow_range,0,0)
        power = np.insert(power,0,0)
        
        # flow_arr = flow_range/maxflow
        fig,ax=plt.subplots()
        #ax = fig.add_subplot(1,1,1)
        ax.plot(flow_arr1,effi_cal,'g-',marker='*',markerfacecolor='b',markersize=7)
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
        ax.grid(True, axis='both', color='k',linestyle='--',alpha=0.4)
        ax.set_xlabel("Percentage of rated flow (%)",fontsize=font,**csfont)
        ax.set_ylabel("Efficiency",color='g',fontsize=font,**csfont)
        
        ax1=ax.twinx()
        ax1.plot(flow_arr1,power,'b-',marker='o',markerfacecolor='r',markersize=7)
        vals = ax1.get_xticks()
        ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
        ax1.grid(True, axis='both', color='k',linestyle='--',alpha=0.4)
        ax1.set_xlabel("Percentage of rated flow (%)",fontsize=font,**csfont)
        ax1.set_ylabel("Power (MW)",color='b',fontsize=font,**csfont)
        plt.title(title,fontsize=font,**csfont)
        plt.show()
        
        return None