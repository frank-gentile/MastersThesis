#%%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from input_data import GetData
import pandas as pd
from fredapi import Fred
import numpy as np


fred = Fred(api_key = '13a856ab4d9b2976146eb2e1fddd511d')
vxo = pd.Series(fred.get_series('VXOCLS',frequency='q'),name='VXO')
um = pd.Series(fred.get_series('UMCSENT',frequency='q'),name='UM Sentiment')
cpi = pd.Series(fred.get_series('CPIAUCSL',frequency='q'),name='CPI')
gdp = pd.Series(fred.get_series('GDPC1',frequency='q'),name='GDP')
invst = pd.Series(fred.get_series('GPDIC1',frequency='q'),name='Investment')
cons = pd.Series(fred.get_series('PCECC96',frequency='q'),name='Consumption')
ffr = pd.Series(fred.get_series('FEDFUNDS',frequency='q'),name='FFR')

um = (np.log(um)-np.log(um.shift(1)))*100
cpi = (np.log(cpi)-np.log(cpi.shift(1)))*100
gdp = (np.log(gdp)-np.log(gdp.shift(1)))*100
invst = (np.log(invst)-np.log(invst.shift(1)))*100
cons = (np.log(cons)-np.log(cons.shift(1)))*100

um = um[um.index>vxo.index[0]]
cpi = cpi[cpi.index>vxo.index[0]]
gdp = gdp[gdp.index>vxo.index[0]]
invst = invst[invst.index>vxo.index[0]]
cons = cons[cons.index>vxo.index[0]]
ffr = ffr[ffr.index>vxo.index[0]]

fig = make_subplots(rows=3, cols=3,
        subplot_titles=(['VXO', 'UM Sentiment','CPI','GDP','Investment','Consumption','Federal Funds Rate']))

fig.append_trace(go.Scatter(
            x=vxo.index,
            y=vxo, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=1, col=1) 
fig.append_trace(go.Scatter(
            x=um.index,
            y=um, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=1, col=2) 
fig.append_trace(go.Scatter(
            x=cpi.index,
            y=cpi, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=1, col=3) 
fig.append_trace(go.Scatter(
            x=gdp.index,
            y=gdp, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=2, col=1)                 
fig.append_trace(go.Scatter(
            x=invst.index,
            y=invst, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=2, col=2) 
fig.append_trace(go.Scatter(
            x=cons.index,
            y=cons, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=2, col=3) 
fig.append_trace(go.Scatter(
            x=ffr.index,
            y=ffr, mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=3, col=1) 

fig.update_layout(height=500, width=800)
#fig.show()
fig.write_image('images/variables_plotted.png')
# %%
