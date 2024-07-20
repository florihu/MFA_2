#!/usr/bin/env python
# coding: utf-8

# # Final Assignment
# General stuff for this doc:
# 1. If I load data from scources the DOI of this scource will be included in the markdown.
# 2. If I extract single values. reference from papers the scource will be included in the markdown or the paper
# 3. data_ouput is safed in csv files and extractable in the respective folder. Same is true for figures.
# 4. Assumptions are as explicityl stated.
# 
# ## Loading the data
# First we need to import the DWT per cluster. Please assess the clustering in the file. For clustering we use:
# <https://www.routledge.com/The-Global-North-South-Atlas-Mapping-Global-Change/Solarz/p/book/9781138588844>
# 
# What does flag of regristration mean:In the context of maritime transport statistics, the "flag of registration" refers to the country under whose laws a ship is registered and operates. Here there are multible Assumptions involved.
# 
# 1. where a ship is registrated it is build and in EOL destructed and partially recycled.
# 2.
# 
# 
# The data scource for the DWT is <https://unctadstat.unctad.org/wds/TableViewer/tableView.aspx?ReportId=93>
# Unit: 10**3 DWT
# 
# The material_intensity, lifetime and DWT_to_GT  is extracted from <https://doi.org/10.1016/j.gloenvcha.2022.102493>
# 
# For the model look at the file MODEL.drawio
# 

# In[22]:



# Parametrization

def ship_model(DWT_err=1,lifetime_err=1, mat_err=1, GDP_err=1, rate_err=1, degr_rate_err=1, study='NAS_abs'):


    import pandas as pd
    import numpy as np


    DWT = pd.read_excel('final_assignment_data.xlsx', sheet_name='raw_Fleet_DWT')

    #lets clean up the mess we want to drop first column and the rows of column YEAR containing the Total fleet

    DWT_clean = DWT.drop(DWT.columns[0], axis=1).drop(DWT[(DWT['YEAR'] == 'Total fleet') | (DWT['classification'] == 'none')].index)





    # #perfect now we build a dict with the first key class and the second type of ship which contains the mean of each year. And the standard deviation.
    # for both we use a tuple key with (class,type) containing a DataFrame with the yearly averages.

    # In[66]:


    DWT_mean = {}
    DWT_stdev ={}

    classification = ['south', 'north']
    type_ships = ['Oil tankers', 'Bulk carriers', 'General cargo', 'Container ships', 'Other types of ships']

    for class_ in classification:
        for type_ in type_ships:
            subset = DWT_clean[(DWT_clean['classification'] == class_) & (DWT_clean['YEAR'] == type_)].drop(['YEAR', 'classification'], axis=1)
            DWT_mean[(class_, type_)] = pd.concat([subset.apply(pd.to_numeric, errors='coerce').mean() * DWT_err , subset.apply(pd.to_numeric, errors='coerce').std()],axis=1)



    # Todo assess the relative aggregation error


    # we have now for every combination a DataFrame with mean, stdev for every year. the stdev here kind of an aggregation error is quite high but thats OK as we assess the uncertainty of this aggregation (hopefully) timeframe is quite tight
    #
    # new lets get stuff translated to GT

    # In[69]:


    DWT_to_GT =  pd.read_excel('final_assignment_data.xlsx', sheet_name='DWT_to_GT')


    # this is to interpret alla y = m*x^n +t

    # In[75]:


    DWT_to_GT.iloc[0,0]=0

    DWT_to_GT.index = ['m','n','t','r2']

    GT_mean = {(class_, type_): DWT_to_GT.loc['m', type_] * DWT_mean[(class_, type_)][0] ** DWT_to_GT.loc['n', type_] + DWT_to_GT.loc['t', type_] for class_ in classification for type_ in type_ships}


    lifetime =  pd.read_excel('final_assignment_data.xlsx', sheet_name='lifetime')




    #

    # Assumption: Because of no better value in the literature and the relative unimportance of the scale of the distribution the scale was held constant for every ship catagory at scale=10. However this obviously a point to further investigate.
    #
    #
    # The scenarios are not included so far. the results will be adapted for every scenario combination. Proposed structure
    #
    # DataFrame in dict [class_, type_] in dict [growth, decoupling_factor]

    # In[99]:


    from stock_flow_model import stock_driven_model, flow_driven_model

    ships_results = {(class_, type_): stock_driven_model(time=GT_mean[class_,type_].index ,stock=GT_mean[class_,type_].values,loc= lifetime.loc[0,type_] * lifetime_err)[0] for class_ in classification for type_ in type_ships}



    # todo adapt fcking survival curve and take care of you vocab. Here we have also the first non linearities


    # perfect now we have to initialize the inflow driven model for the steal. The structure of the dictionary will be analogues. of course we need to import the material intensities. and convert them in the right unit.
    #
    # Source:<https://doi.org/10.1016/j.gloenvcha.2022.102493>
    #
    # Unit: kg/GT -> transformation in 10**-3 kg/ GT
    #

    # In[509]:


    from scipy.stats import linregress
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("white")

    material_intensity_raw =  pd.read_excel('final_assignment_data.xlsx', sheet_name='material_intensity')

    material_intensity_raw.index = material_intensity_raw['Unnamed: 0']

    material_intensity_raw.drop('Unnamed: 0', axis=1, inplace=True) #.drop('Unnamed: 0', axis=0, inplace=True)

    material_int = pd.DataFrame(columns=material_intensity_raw.columns, index = range(1980,2101))

    for inter in range(1985,2025,5):
        for column in material_intensity_raw.columns:
            material_int.loc[inter-5:inter,column] = material_intensity_raw.loc[inter,column]*10**3 * mat_err

    for column in material_int.columns:
        regr = linregress(range(1980, 2021), pd.to_numeric(np.array(material_int.loc[:2020, column])))
        material_int.loc[2021:, column] = np.array(range(2021,2101)) * regr.slope + regr.intercept



    steel_results = {(class_, type_): flow_driven_model(time=ships_results[class_,type_].index ,inflow=ships_results[class_,type_]['inflow'].values*material_int.loc[:2022,type_].values,sf_kind='normal',loc= lifetime.loc[0,type_] * lifetime_err ,stock_ini=ships_results[class_,type_].loc['1980','stock'])[0] for class_ in classification for type_ in type_ships}



    # # Scenario Development
    #
    # 1. The basic idea here is to find predictor functions that correlate GT with GDP in the time window (1980-2022) F(x)
    #     This assumption seems valid following: https://www.sciencedirect.com/science/article/pii/S0959378022000310
    # 2. This predictor function we apply to SSP scenarios from: <https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=60>
    # 3. We find a proxy for GT in time window 2022-2100 and append. haha hopefully this works
    #
    #
    # ## 1. get predictors
    #
    # 1. load data
    # 2. define prediction dict
    # 3. load input data from SSP
    # 4. linearize over time interval
    #
    # Lets do this.
    #

    # In[118]:


    GDP = pd.read_excel('final_assignment_data.xlsx', sheet_name='GDP')




    # In[141]:


    GDP_clean = GDP.dropna(subset=['classification']).iloc[:,3:].drop('Country Code', axis=1)

    GDP_clean.columns = [str(1959 + i) if i != 0 else GDP_clean.columns[0] for i in range(len(GDP_clean.columns))]

    GDP_clean.drop('2022',axis=1, inplace=True)




    # In[157]:


    classification_ad = ['South','North']
    GDP_class = dict()

    for class_ in classification_ad:
        subset_1 = GDP_clean[GDP_clean['classification'] == class_].drop('classification', axis=1)
        GDP_class[class_] = pd.concat([subset_1.apply(pd.to_numeric, errors='coerce').mean() * GDP_err ,subset_1.apply(pd.to_numeric, errors='coerce').std()],axis=1)

    # how tricky can it be to simple switch fcking keys
    def keys_swap(orig_key, new_key, d):
        d[new_key] = d.pop(orig_key)

    keys_to_swap = {'South': 'south', 'North': 'north'}
    for orig_key, new_key in keys_to_swap.items():
        if orig_key in GDP_class:
            keys_swap(orig_key, new_key, GDP_class)




    #

    # WE leverage here linear regression from scipy
    # x: GDP (INPUT) in $
    # y: GT (WHAT WE WANT TO PREDICT) in

    # In[249]:


    from scipy.stats import linregress

    regress = {(class_, type_): linregress(GDP_class[class_].iloc[20:,0], GT_mean[class_,type_][:-1] )for type_ in type_ships for class_ in classification}



    # ## 2. LOAD Prediction INPUT data
    #
    # we do not load SSP scenarios because they cannot predict the right magnitude of the outcome because of different aggregation methods.-> we load relative values
    #
    #
    # changing parameter:
    # 1. Growth rate south/north (SSP2 2.6, SSP1 2.6, Degrwoth). We take here the relative values from south (OPEC) and average ROW for north. Growth rates are linearised over the decades.
    # 2. The growth rate of the oil tankers approximated via the relative growth rate of the primary energy oil indicator.
    #     (SSP2 2.6, SSP1 2.6) Of course and assumption is that there is causal relation between GT (Oil tanker) and the respective global primary energy demand but this seems quite valid.
    # 3. Therefore we have 6 scenarios.
    # 4. Now we need fancy names for them
    #
    # Quick note: the scenarios were purpusly choosen to be the optimistic ones because this study wants to elaborate if it is possible to see stock saturation in the merchant ship fleet and under shich circumstances.
    #
    #

    # In[251]:


    scenario_names = ['SSP1-26', 'SSP2-26']

    rates= {} #key one kind key 2 scenario key 3 class

    calc_rates = pd.read_excel('final_assignment_data.xlsx', sheet_name='scenario_data')


    for class_ in classification:
        for kind in ['GDP|PPP','Primary Energy|Oil']:
            for scenario in ['SSP1-26', 'SSP2-26']:

                GDP_dec_mean= calc_rates[(calc_rates['clustering']==class_ ) & (calc_rates['VARIABLE']==kind) & (calc_rates['SCENARIO']==scenario)].loc[:,2020:].mean()
                GDP_dec_diff = GDP_dec_mean.diff().div(GDP_dec_mean.shift()) /10
                rates[(class_, kind, scenario)]= GDP_dec_diff * rate_err

    rates[('north', 'GDP|PPP', 'degrowth')] = pd.Series([-.0053 * degr_rate_err]*9,index=[2020+10*i for i in range(9)])
    rates[('south', 'GDP|PPP', 'degrowth')] = pd.Series([-.0053 * degr_rate_err]*9,index=[2020+10*i for i in range(9)])

       # no it is possible to calculate the rest of the GDP values.

    GDP_scenario = dict()
    scenarios = ['SSP1-26', 'SSP2-26', 'degrowth']

    for class_ in classification:
        for scenario in scenarios:
            annual_rates = pd.Series(index=[2020 + i for i in range(81)])
            for i in range(2030, 2110, 10):
                annual_rates.loc[range(i - 10, i)] = 1 + rates[(class_, 'GDP|PPP', scenario)][i]

            initial_GDP = GDP_class[class_].loc['2021', 0]
            GDP_scenario[(class_, scenario)] = annual_rates.cumprod() * initial_GDP




    # To reduce effort it is assumed that for GT[2100] = GT[2099]


    GT_scenario = {(class_, type_, scenario): pd.Series(np.array(GDP_scenario[(class_, scenario)]) * regress[(class_, type_)].slope + regress[(class_, type_)].intercept, index=GDP_scenario[(class_, scenario)].index) for scenario in scenarios for type_ in type_ships for class_ in classification}


    for class_ in classification:
        for type_ in type_ships:
            for scenario in scenarios:
                GT_scenario[(class_, type_, scenario)].loc[2100]=GT_scenario[(class_, type_, scenario)].loc[2099]




    # In[355]:


    GT_results_scenario = {}

    GT_results_scenario = {(class_, type_, scenario): flow_driven_model(time = range(2023,2101),inflow=GT_scenario[(class_,type_,scenario)].loc[2023:].values , sf_kind='normal',loc= lifetime.loc[0,type_] * lifetime_err,stock_ini=ships_results[class_,type_].loc['2022','stock'])[0] for scenario in scenarios for class_ in classification for type_ in type_ships}



    steel_results_scenario={}

    steel_results_scenario = {(class_, type_, scenario): flow_driven_model(time = range(2023,2101),inflow=GT_results_scenario[(class_,type_,scenario)].loc[2023:, 'inflow'].values * material_int.loc[2023:,type_].values , sf_kind='normal',loc= lifetime.loc[0,type_]*lifetime_err,stock_ini=steel_results[class_,type_].loc['2022','stock'])[0] for scenario in scenarios for class_ in classification for type_ in type_ships}






    # ## Analysis of scenarios:
    #
    # First we need to prepare data in a

    # In[406]:


    # ## Cummulated stock and flows


    # ## Indicator calculation & visualization

    # In[510]:


    agg_sum = pd.DataFrame()
    for class_ in classification:
            for scenario in scenarios:
                agg = pd.DataFrame()
                for type_ in type_ships:
                    agg[f"{type_}"] = steel_results_scenario[(class_, type_, scenario)]['nas'] / 10**9
                    agg_sum[f"{class_}_{scenario}"] = agg.sum(axis =1)
                    lable = agg_sum.columns



    nas_indic = pd.DataFrame()
    nas_indic_cum = pd.DataFrame()

    for scenario in scenarios:
        for scenario_1 in scenarios:
            nas_indic[f"south_{scenario}_north_{scenario_1}"] = agg_sum[f"south_{scenario}"] / agg_sum[f"north_{scenario_1}"]
            nas_indic_cum[f"south_{scenario}_north_{scenario_1}"] = agg_sum[f"south_{scenario}"] + agg_sum[
                f"north_{scenario_1}"]

    nas_indic = nas_indic.fillna(0)
    nas_indic_cum = nas_indic_cum.fillna(0)# Replace NaN values with 0


    nas_indic['south_SSP1-26_north_degrowth'].loc[2062] = nas_indic['south_SSP1-26_north_degrowth'].loc[2061]
    nas_indic['south_SSP2-26_north_degrowth'].loc[2062] = nas_indic['south_SSP2-26_north_degrowth'].loc[2061]
    nas_indic_cum['south_SSP1-26_north_degrowth'].loc[2062] = nas_indic_cum['south_SSP1-26_north_degrowth'].loc[2061]
    nas_indic_cum['south_SSP2-26_north_degrowth'].loc[2062] = nas_indic_cum['south_SSP2-26_north_degrowth'].loc[2061]



    nas_indic_dic = {(column): nas_indic[column].values for column in nas_indic.columns}

    nas_cum_dic = {(column): nas_indic_cum[column].values for column in nas_indic_cum.columns}


    if study == 'NAS_rel':
        return nas_indic_dic
    elif study == 'NAS_abs':
        return nas_cum_dic
    else:
        return None