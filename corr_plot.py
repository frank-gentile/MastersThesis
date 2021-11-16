#%%
from fredapi import Fred
import pandas as pd
import numpy as np
import plotly.graph_objects as go


fred = Fred(api_key = '13a856ab4d9b2976146eb2e1fddd511d')
vxo = pd.Series(fred.get_series('VXOCLS',frequency='q'),name='VXO')
um = pd.Series(fred.get_series('UMCSENT',frequency='q'),name='UM Sentiment')
#oecd = pd.Series(fred.get_series('CSCICP03USM665S',frequency='q'),name='OECD Sentiment')

um = um[um.index>vxo.index[0]]
#oecd = oecd[oecd.index>vxo.index[0]]

vxo_m = (vxo - vxo.mean())/vxo.std()
um_m = (um-um.mean())/um.std()
#oecd_m = (oecd-oecd.mean())/oecd.std()

fig = go.Figure(data=go.Scatter(
            x=vxo_m.index,
            y=vxo_m, mode='lines',
            line=dict(color='rgb(198,0,0)',dash='dashdot'),
            showlegend=True, name='VXO'
        )   )


fig.add_trace(go.Scatter(
            x=um_m.index,
            y=um_m, mode='lines',
            line=dict(color='rgb(255,192,0)'),
            showlegend=True, name='UM Sentiment'
        )  )

# fig.add_trace(go.Scatter(
#             x=oecd_m.index,
#             y=oecd_m, mode='lines',
#             line=dict(color='rgb(31, 119, 180)',dash='dash'),
#             showlegend=True, name='OECD Sentiment'
#         )  )

fig.update_layout(height=500, width=800,title_text="Demeaned and standardized sentiment and uncertainty")
#fig.show()
fig.write_image('images/vxo_sent_graph.png')
# %%
