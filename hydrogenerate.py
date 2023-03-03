# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 14:15:46 2020

@author: Bhaskar Mitra
"""


import numpy as np
import pandas as pd
from datetime import datetime



#Declaring some constants

g = 9.81 #Acceleration due to gravity
rho = 1000 #Specific density of water

'''
1 ft = 0.3048m # Ft to m conversion
'''




'''
This maintains the turbine selection data based on head

Head suggesting the types of turbines to be used.
https://theconstructor.org/practical-guide/hydraulics-lab/turbines-pumps/factors-affecting-selection-hydraulic-turbine/30826/
'''

def turbine():
    turb_data = {'Head':['Very low head','Low head','Medium head','High head',
                         'Very high head','Very high head'],
                 'Start':[0.5,10,60,150,350,2000],
                 'End':[10,60,150,350,2000,5000],
                 'Turbine':['Kaplan turbine','Kaplan turbine',
                                     'Francis turbine','Francis turbine',
                                     'Pelton turbine','Pelton turbine'],
                 'k2':[800,800,600,600,0,0]}
    df = pd.DataFrame(data=turb_data)
    return df

def material():
    mate_coeff= {'Material':['Cast Iron','Commercial Steel and Wrought Iron','Concrete',
                             'Drawn Turbing', 'Galvanized Iron', 'Plastic/Glass',
                             'Riveted Steel', 'HDPE'],
                 'RoughnessCoefficient':[0.26,0.045,1.5,0.0015,0.15,0,4.50,0.0015]}
    
    df = pd.DataFrame(data=mate_coeff)
    return df

def temp():
    relative = {'Celsius': [0,5,10,20,30,40,50,60,70,80,90,100],
                'Farenheit': [32,41,50,68,86,104,122,140,158,176,194,212],
                'Mu' : [0.001787,0.001519,0.001307,0.001002,0.000798,0.000653,0.000547,
                        0.000467,0.000404,0.000355,0.000315,0.000282],
                'Nu' : [0.000001787,0.000001519,0.000001307,0.000001004,0.000000801,
                        0.000000658,0.000000553,0.000000475,0.000000413,0.000000365,
                        0.000000326,0.000000294]}
    df = pd.DataFrame(data=relative)
    return df

def time_diff_hr(dt1,dt2):
    timedel = dt2 - dt1
    time_step_hr = (timedel.days*24*3600 + timedel.seconds)/3600
    return time_step_hr

def cal_step(flow_info):
    
    flow_info['Date/Time']=flow_info['Date/Time'].astype('datetime64[ns]')
    time_start = flow_info['Date/Time'].iloc[0]
    time_end = flow_info['Date/Time'].iloc[1]
    time_step_hr = time_diff_hr(time_start,time_end)
    return time_step_hr


def get_efficiency(tur_name,flow_range,k1,k2,Rm,maxflow,head,h_f):
    
    effi_cal=[]
    
    if (tur_name=='Francis turbine'):
        
        reac_runsize = k1*(maxflow)**(0.473)
        
        speci_speed = k2*abs((head-max(h_f)))**(-0.5)
        
        speed_ad = ((speci_speed - 56)/256)**2
        run_size = (0.081 + speed_ad)*(1-(0.789*(reac_runsize**(-0.2))))
        peak_eff = (0.919 - speed_ad + run_size)-0.0305 + 0.005*Rm
        peak_eff_flow = 0.65*maxflow*(speci_speed**0.05)
        full_load_eff_drop = 0.0072 *(speci_speed)**0.4
        eff_full_load = (1-full_load_eff_drop)*peak_eff
        
        for i in range(len(flow_range)):
            if (peak_eff_flow > flow_range[i]):
                #print('yes')
                effi = (1-(1.25*((peak_eff_flow- flow_range[i])/peak_eff_flow)**(3.94 - 0.0195 *speci_speed)))*peak_eff
                
                if (effi <= 0):
                    effi = 0
                effi_cal.append(effi)
                
            else:
                #print('no')
                effi = peak_eff- (((flow_range[i] - peak_eff_flow)/(maxflow - peak_eff_flow)**2)*(peak_eff - eff_full_load))
                if (effi <= 0):
                    effi = 0
                effi_cal.append(effi)
                
                
    elif (tur_name=='Kaplan turbine' or  tur_name =='Propeller turbine'):
        # print('Entering Kaplan Calculation Module')
        reac_runsize = k1*(maxflow)**(0.473)
        
        speci_speed = k2*abs((head - max(h_f)))**(-0.5)
        speed_ad= ((speci_speed-170)/700)**2
        run_size = (0.095 + speed_ad)*(1-(0.789*(reac_runsize**(-0.2))))
        peak_eff = (0.905 - speed_ad + run_size)-0.0305 + 0.005*Rm
        
        if (tur_name=='Kaplan turbine'):
           peak_eff_flow = 0.75 * maxflow
           
           effi_cal = (1- 3.5*((peak_eff_flow - flow_range)/peak_eff_flow)**6)*peak_eff
           #print(effi_cal)
           effi_cal = np.where(effi_cal <=0 , 0, effi_cal)
        elif (tur_name=='Propeller turbine'):
            peak_eff_flow = maxflow
            
            effi_cal = (1-1.25*((peak_eff_flow - flow_range)/peak_eff_flow)**1.13)*peak_eff
            effi_cal = np.where(effi_cal <=0 , 0, effi_cal)
        
        
    elif(tur_name=='Pelton turbine' or tur_name=='Turgo turbine'):
        num = 5
        rot_sp = 31*(abs((head-max(h_f)))*(maxflow/num))**0.5
        out_run_dia = (49.4*(abs((head-max(h_f)))**0.5)*num**0.02)/rot_sp
        tur_peak_eff = 0.864* out_run_dia**0.04
        peak_eff_flow = (0.662+0.001*num)*maxflow
        effi_pelo = (1-((1.31+0.025*num)*(abs(peak_eff_flow - flow_range)/(peak_eff_flow))**(5.6+0.4*num)))*tur_peak_eff    
        if (tur_name=='Pelton turbine'):
            effi_cal = effi_pelo
        elif(tur_name=='Turgo turbine'):
            effi_cal = effi_pelo - 0.03
        effi_cal = np.where(effi_cal <=0 , 0, effi_cal)
       
        
       
    elif(tur_name=='Crossflow turbine'):
        peak_eff_flow = maxflow
        effi = 0.79 - 0.15 *((peak_eff_flow - flow_range)/peak_eff_flow)-1.37*((peak_eff_flow - flow_range)/(peak_eff_flow))**(14)
        effi_cal=effi
        effi_cal = np.where(effi_cal <=0 , 0, effi_cal)
        
    
    return effi_cal

def get_power(effi_cal,flow_range,head,h_f,sys_effi,g,rho):
    
    power = abs(flow_range * (head - h_f) * effi_cal * sys_effi * g * rho)/10**6
        
    return power

def get_flow(op,flow_info,flow_column,rated_flow):
    
    if (op == 'Timeseries'):
        flow_info[flow_column]=flow_info[flow_column].replace(['Ice','Mtn','Bkw','NaN','--','Dis','Dry',
                                                                       '***','Eqp','Rat','ZFI','Ssn','missing'],0)
        '''
        The previous command replaces any string type input to 0, but it converts the datatype
        of the column to string. Converting it back to type int
        '''
        flow_info = flow_info.astype({flow_column: int})
        flow_info = flow_info.dropna(subset=[flow_column])
       
        if rated_flow is not None:
            maxflow = 0.028316846591999675*rated_flow
        else:
            maxflow = 0.028316846591999675*max(flow_info[flow_column])
                
        '''
        1 cu.ft/sec = 0.028316846591999675 cu.m/sec
        '''
        
        flow_range = flow_info[flow_column] * (0.028316846591999675)
        flow_range = flow_range.values
        flow_range = np.where(flow_range>maxflow,maxflow,flow_range)
        # flow_info['Date/Time']=flow_info['Date/Time'].astype('datetime64[ns]')
        # time_start = flow_info['Date/Time'].iloc[0]
        # time_end = flow_info['Date/Time'].iloc[1]
        # time_step_hr = time_diff_hr(time_start,time_end)
        
    elif (op == 'Generalized'):
        
        if rated_flow is not None:
            # rated_flow = rated_flow
            maxflow= 0.028316846591999675 * rated_flow # cu.ft/sec to cu.m/sec conversion
        else:
            raise ValueError('Provide maximum flow capacity')
        flow_arr=np.linspace(0.05,1,20)
        flow_range= maxflow * flow_arr
        
    
    return flow_range,maxflow

def get_head_loss(system,flow_range):
    
    '''
    df2 = temp() 
    # Dynamic Viscosity as a function of temperature
    df2= temp[temp.Farenheit == temp_input]
    dyn_visc = df2['Mu'].tolist()[0]
    nu = df2['Nu'].tolist()[0]
    
    df3 = material()
    #Roughness Coefficient based on piping type

    df3=mat[mat.Material == material]
    rough= df3['RoughnessCoefficient'].tolist()[0]
    
    D=dia/1000 #Diameter in meters  

    #Relative Roughness
    rela_rough = rough/dia

    #Reynolds Coefficient

    #Re=(4 * flow * D)/(dyn_visc * math.pi * D**2)

    Re= (4*flow_range)/(math.pi*D*nu)

    #Friction factor

    f= 0.11 * ((rela_rough) + (68 / Re)) ** (1/4)



    velo=(4*flow_range)/(math.pi * D**2)
    
    h_f = f*(length*velo**2)/(D*2*g)
    '''
    
    if (system=='pipe'):
        
        '''
        h_f=f*(L*v**2)/(d*2*g) or h_f =0.0311*(f*L*Q**2)/d**3
        f = friction factor (user input)
        L = Length of pipe (user input) (m)
        d = pipe diameter (m)
        v = velocity (m/sec^2)
        g = acceleration due to gravity (constant)
        
        '''
        #h_f = f*(length*velo**2)/(D*2*g)
        h_f = 0.05* flow_range
        #h_f=0
        '''
        Calculating for inclined pipe
        Pressure Drop
        del_p = rho*g*(h_f - length * sin(angle))
        h_f_inc = (del_p/(rho *g)) + (length * sin(angle)) #angle is measured in degrees
        '''
        
        #del_p = (rho * g)*(h_f - (length * math.sin(angle)))
        #p_diff = (f*length*rho*g*velo**2)/(D*2*g)
        #h_f_inc = (del_p/(rho *g)) + (length * math.sin(angle))
        
    elif (system=='canal'):
        
        h_f= 0.2* flow_range
        #h_f=0 #Special cases for Dams with reservoirs or Run-of-The-River Generation
    elif (system=='reservoir'):
        h_f= 0.001* flow_range # 0.1% head_loss considered else max(h_f) will not be iterable
    return h_f

def get_energy(power,turb_cap,op,flow_info=None,time_step=None,
               energy_cost=None,const_cost=None):
    
    if (flow_info is None) and (op == 'Timeseries'):
        
        raise ValueError('Flow information dataframe missing')
        
    if (op == 'Timeseries'):
        
        if time_step is None:
            
            time_step_hr = cal_step(flow_info)
        else:
            time_step_hr = time_step
        mwh = power * time_step_hr # MWh calculation
        tot_mwh = sum(mwh)  # Total Energy
    elif (op == 'Generalized'):
        valu= []
        for i in range(len(power)-1):
            create = ((power[i]+power[i+1])/2)
            valu.append(create)
        mwh = sum(valu)  # MWh calculation  
        tot_mwh = sum(valu) * 438 * 0.8 # Total Energy
    
    
    if const_cost is not None:
        const = const_cost
    else:
        const = 4.1 #million $MW
    if energy_cost is not None:
        cent = energy_cost
    else:
        cent = 0.04
           
    revenue = (tot_mwh *10**3)*(cent/100) #Calculating revenue in Dollars
    const_cost = turb_cap*const #Construction cost
    #h_f= 10.67 *(((L_1 - L_2)* flow**1.85)/(C**1.85 * D**4.87))
    return tot_mwh,revenue,const_cost


def calculate_potential(flow_info=None, rated_flow=None, rated_power=None, turb= None, 
                        head_input=None,op='Timeseries', sys_effi=None,
                        system='pipe', energy_cost=None, cost = None, 
                        flow_column=None):
    
    if (flow_info is None) and (op=='Timeseries'):
        
        raise ValueError('Flow information dataframe missing')
    
    if flow_column is None:
        
        flow_column = 'Flow (cfs)'
    else:
        flow_column = flow_column
        
    
    df = turbine()

    diff_m = head_input* 0.3048 # converting ft. to m
    head=abs(diff_m)
    #length = length*1609.34 #Converting mi to m
    
    if head <= 0.6:
        #raise ValueError('Head height too low for small-scale hydropower')
        print("Head height too low for small-scale hydropower")
        # Ref: https://www.energy.gov/energysaver/planning-microhydropower-system
        
        raise SystemExit
        #dia = dia * 25.4 # converting inches to mm
    
    
    df1=df[(head > df.Start) & (head <= df.End)]
        
    
    if turb is not None:
        tur_name = turb
    else:
        tur_name = df1['Turbine'].to_string(index=False).strip()
        
    '''
    Defining Turbine Constants
    '''
    k2= df1['k2'].tolist()[0]
    k1=0.41
    Rm=4.5
    
    if sys_effi is not None:
        sys_effi = sys_effi
    else:
        sys_effi =0.98
        
    flow_range,maxflow = get_flow(op,flow_info,flow_column,rated_flow)
    
    h_f = get_head_loss(system,flow_range)
    
    effi_cal = get_efficiency(tur_name,flow_range,k1,k2,Rm,maxflow,head,h_f)
          
    power = get_power(effi_cal,flow_range,head,h_f,sys_effi,g,rho)
    
    if rated_power is not None:
        turb_cap = rated_power
    else:
        turb_cap = np.percentile(power,75)
    
    power=np.where(power>turb_cap,turb_cap,power) #Changed for VRG
    
    effi_cal = np.asarray([i * 100 for i in effi_cal])
    #effi_cal = np.asarray(effi_cal)
    
    
    
    dict_return = {"power":power,
                   "efficiency":effi_cal,
                   "flow":flow_range,
                   "turbine_capacity":turb_cap}
    
    
        
    return dict_return
    
   
    """
    flo = get_flow(___)
    pow = get_power(___)
    eff = get_eff(__)
    dict_return = {"power":pow,
                   "eff":eff,
                   "flow":flow}
    return dict_return

    x = func()
    x["power"]
    """
        
    '''    
    D=dia/1000 #Diameter in meters
    
    # Dynamic Viscosity as a function of temperature
    df2= temp[temp.Farenheit == temp_input]
    dyn_visc = df2['Mu'].tolist()[0]
    nu = df2['Nu'].tolist()[0]
    
    #Roughness Coefficient based on piping type
    
    df3=mat[mat.Material == material]
    rough= df3['RoughnessCoefficient'].tolist()[0]
    
    #Relative Roughness
    rela_rough = rough/dia
    
    #flow_arr=np.linspace(0.05,1,20)
    
    #flow_range= maxflow * flow_arr
    #flow_range = flow_info['Actual (cfs)'] * 0.028316846591999675
    #flow_range = flow_range.values
    #Reynolds Coefficient
    
    #Re=(4 * flow * D)/(dyn_visc * math.pi * D**2)
    
    Re= (4*flow_range)/(math.pi*D*nu)
    
    #Friction factor
    
    f= 0.11 * ((rela_rough) + (68 / Re)) ** (1/4)
    
    
    
    velo=(4*flow_range)/(math.pi * D**2)
    '''

    
    '''
    Head Loss Calculation
    
    http://fluid.itcmp.pwr.wroc.pl/~znmp/dydaktyka/fundam_FM/Lecture11_12.pdf
    '''
    
    '''
    Eauation P = [rho*g*Q*[H_g-(h_hydr+h_tail)]*e_t*e_g*(1-l_trans)*(1-l_para)]
    rho Specific density of water (constant 1000 kg/m^3)
    g Acceleartion due to gravity (constant 9.81 m/sec^2)
    H_g Gross head (m)
    h_hydr Hydraulic loss (m)
    h_tail Tail effect associated with the flow (m)
    e_t Turbine efficiency (%)
    e_g Generator efficiency (%)
    l_trans Transformer Losses
    l_para Parasitic electricity losses
    '''      
    
    



'''
Possiblity of adding plots in future

#Making plot on 21 points (0,1)
'''
'''
effi_cal.insert(0,0)
flow_arr = np.insert(flow_arr,0,0)
power = np.insert(power,0,0)

flow_arr = flow_range/maxflow
fig,ax=plt.subplots()
#ax = fig.add_subplot(1,1,1)
ax.plot(flow_arr,effi_cal,'g-',marker='*',markerfacecolor='b',markersize=7)
vals = ax.get_xticks()
ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
ax.grid(True, axis='both', color='k',linestyle='--',alpha=0.4)
ax.set_xlabel("Percentage of rated flow (%)")
ax.set_ylabel("Efficiency",color='g')

ax1=ax.twinx()
ax1.plot(flow_arr,power,'b-',marker='o',markerfacecolor='r',markersize=7)
vals = ax1.get_xticks()
ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
ax1.grid(True, axis='both', color='k',linestyle='--',alpha=0.4)
ax1.set_xlabel("Percentage of rated flow (%)")
ax1.set_ylabel("Power (MW)",color='b')
plt.show()
'''
'''
Head suggesting the types of turbines to be used.
https://theconstructor.org/practical-guide/hydraulics-lab/turbines-pumps/factors-affecting-selection-hydraulic-turbine/30826/
print('Calculated Power Output:', power, '(W) \n')
'''
'''

Equations:

Hydraulic head loss (Thomas Paper)

h_f= 10.67 *(((L_1 - L_2)* flow**1.85)/(C**1.85 * D**4.87))

h_f: hydraulic head loss (m)
L_1: starting point of penstock
L_2: end point of penstock
flow: volumetric flowrate (m^3/s)
C: roughness coefficient
D: internal pipe diameter (m)
'''

'''
text = {'Start':[0.5,10,60,150,350,2000],'End':[10,60,150,350,2000,5000],'Suitable Turbine':['Kaplan Turbine','Kaplan Turbine','Francis Turbine','Francis Turbine','Pelton Turbine','Pelton Turbine'],'k2':[800,800,600,600,0,0]}

def efficiency(tur_name):
    
'''