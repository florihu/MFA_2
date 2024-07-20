import matplotlib.pyplot as plt
# Import all the model functions

from Final_Assignment.stock_flow_model import *


#%% Load data
dmc_data = pd.read_excel("C:/Users/admin/Documents/GitHub/test-repo/MFA2/stock_of_nations2014.xlsx", sheet_name='direct_material_consumption', header=[0, 1], index_col=0)
# country material selection
choice = dmc_data.columns[0]

input_parameters = pd.read_excel("C:/Users/admin/Documents/GitHub/test-repo/MFA2/stock_of_nations2014.xlsx", sheet_name='material_parameters', index_col=0)

#Now we define the uncertainty factors namely:

factors = ['inflow', 'mean', 'standard_deviation', 'percent_to_construction_sector']


#define the "uncertainty space" of Input parameters. This is here refered to scenarios where obviously =1
#is the baseline scenario


scenarios = [.9,1,1.1]
                          
# utilize for structures to generate findings + save stuff in a dictionary with keys

timeseries_scenario =dict()
cohort_scenario = dict()

for factor in factors:
    for scenario_value in scenarios:
            # Set all other parameter values to 1
            # work with intermediate dictionaries
            scenario_int = {f: 1 for f in factors }
            scenario_int[factor] = scenario_value
            # Set the current parameter value for the current scenario
            inflow_data = np.array(dmc_data[choice]*input_parameters.loc['timber']['percent_to_construction_sector']*scenario_int['percent_to_construction_sector'])

            timeseries_scenario[factor, scenario_value], cohort_scenario[factor,scenario_value] = flow_driven_model(

            time=dmc_data.index.values,
            inflow=dmc_data[choice].values*scenario_int['inflow'],
            sf_kind="normal",
            loc=input_parameters.loc['timber']['mean']*scenario_int['mean'],
            scale=input_parameters.loc['timber']['standard_deviation']*scenario_int['standard_deviation'],
                )
            
            

#%% PLOT RESULTS
fig, ax = plt.subplots(4, 4, figsize=(20, 20), sharey=False)

for j, element in enumerate(factors):
    for i, column in enumerate(["inflow", "outflow", "nas", "stock"]):
        sns.lineplot(
            data=timeseries_scenario[element,1][column], ax=ax[i, j], label="baseline"
        )
        sns.lineplot(
            data=timeseries_scenario[element,.9][column], ax=ax[i, j], label="scenario .9"
        )
        sns.lineplot(
            data=timeseries_scenario[element,1.1][column], ax=ax[i, j], label="scenario 1.1"
        )
        ax[i, j].set_title(f"{element}")

fig.suptitle('Scenario comparison wood japan (vertical: outcome , horizontal: uncertainty factors )', fontsize=20)
plt.savefig('Scenario comparison wood japan')
plt.show()


#%% PLOT RESULTS
fig, ax = plt.subplots(4, 4, figsize=(20, 20), sharey=False)

for j, element in enumerate(factors):
    for i, column in enumerate(["inflow", "outflow", "nas", "stock"]):

        sns.lineplot(
            data=(timeseries_scenario[element,.9][column]-timeseries_scenario[element,1][column])/ timeseries_scenario[element,1][column]*100, ax=ax[i, j], label="scenario .9"
        )
        sns.lineplot(
            data=(timeseries_scenario[element,1.1][column]-timeseries_scenario[element,1][column])/timeseries_scenario[element,1][column]*100, ax=ax[i, j], label="scenario 1.1"
        )
        ax[i, j].set_title(f"{element}")

fig.suptitle('Scenario comparison wood japan relative (vertical: outcome , horizontal: uncertainty factors ) (%)', fontsize=20)
plt.savefig('Scenario comparison wood japan relative')
plt.show()



#%%
import numpy as np
decision = np.random(10)

if decision>5:
    print('LEIDEN')
else:
    print('DH')

