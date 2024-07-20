# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:17:44 2022

@author: tomer
"""

import pandas as pd
import numpy as np
import scipy.stats
from os import chdir

chdir('C:/Users/tomer/MFA2/6 interpretation and communication/homework')

dmc_data = pd.read_excel(r'stock_of_nations2014.xlsx', sheet_name='direct_material_consumption', header=[0, 1], index_col=0)

time_max = dmc_data.shape[0]

timesteps = np.arange(0, time_max)

input_parameters = pd.read_excel(r'stock_of_nations2014.xlsx', sheet_name='material_parameters', index_col=0)

combined_results = {}

for material in dmc_data.columns:
    curve_surv = scipy.stats.norm.sf(timesteps, loc=input_parameters.loc[material[1], "mean"], scale=input_parameters.loc[material[1], "standard_deviation"])
    curve_surv_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)
    for time in timesteps:
        curve_surv_matrix.loc[time:, time] = curve_surv[0:time_max - time]

    material_results = pd.DataFrame(index=dmc_data.index)
    material_results['inflow'] = dmc_data[material] * input_parameters.loc[material[1], "percent_to_construction_sector"]

    cohort_surv_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)
    for time in timesteps:
        cohort_surv_matrix.loc[:, time] = curve_surv_matrix.loc[:, time] * material_results['inflow'].iloc[time]
    cohort_surv_matrix.index = dmc_data.index

    material_results['stock'] = cohort_surv_matrix.sum(axis=1)
    material_results['nas'] = np.diff(material_results['stock'], prepend=0)
    material_results['outflow'] = material_results['inflow'] - material_results['nas']
    combined_results[material] = material_results

combined_results = pd.concat(combined_results.values(), keys=combined_results.keys(), axis=1)

# sample visualizations
combined_results.loc[:, (slice(None), "iron", ["inflow", "outflow"])].plot()
combined_results.loc[:, (slice(None), "iron", "stock")].plot()
