# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:58:32 2021

@author: GENTILE_F
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def plotANS(Resid,IRF,IRF2,IRF_95,IRF_952,IRF_05,IRF_052,T_irf,prox_name):
    Resid = Resid[Resid.columns.drop(list(Resid.filter(regex='Cross')))]
    fig = make_subplots(rows=len(Resid.T), cols=1,
        subplot_titles=([x for x in Resid.columns]))
    
    if prox_name == 'Uncertainty':
        color = 'rgb(192,0,0)'
        fillcolor = 'rgba(192,0,0,0.2)'
    elif prox_name == 'Sent':
        color = 'rgb(31, 119, 180)'
        fillcolor = 'rgba(31, 119, 180,0.2)'
    for i in range(len(Resid.T)):
        if i>0:
            leg = False
        else:
            leg = True

        # fig.append_trace(go.Scatter(
        #     x=np.array(range(T_irf)),
        #     y=IRF[i,:], mode='lines', name='Sentiment high',
        #     line=dict(color='rgb(31, 119, 180)'),
        #     showlegend=leg
        # ), row=i+1, col=1)

        # fig.append_trace(go.Scatter(
        #     x=list(range(T_irf))+list(range(T_irf))[::-1],
        #     y=list(IRF_95[i,:])+list(IRF_05[i,:])[::-1],
        #     fill='toself',
        #     fillcolor='rgba(31, 119, 180,0.2)',
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo="skip",
        #     showlegend=False), row=i+1,col=1)



        fig.append_trace(go.Scatter(
            x=np.array(range(T_irf)),
            y=IRF[i,:], mode='lines',
            line=dict(color=color),
            showlegend=False
        ), row=i+1, col=1)       
    
        fig.append_trace(go.Scatter(
            x=list(range(T_irf))+list(range(T_irf))[::-1],
            y=list(IRF_95[i,:])+list(IRF_05[i,:])[::-1],
            fill='toself',
            fillcolor=fillcolor,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False), row=i+1,col=1)
        
        # fig.append_trace(go.Scatter(
        #     x=np.array(range(T_irf)),
        #     y=IRF2[i,:], mode='lines', name='Sentiment low',
        #     line=dict(color='rgb(198,0,0)',dash='dashdot'),
        #     showlegend=leg
        # ), row=i+1, col=1)
    
        # fig.append_trace(go.Scatter(
        #     x=list(range(T_irf))+list(range(T_irf))[::-1],
        #     y=list(IRF_952[i,:])+list(IRF_052[i,:])[::-1],
        #     fill='toself',
        #     fillcolor='rgba(198,0,0,0.2)',
        #     line=dict(color='rgba(31, 119, 180,0)'),
        #     hoverinfo="skip",
        #     showlegend=False), row=i+1,col=1)
    
    return fig
    
def plotDiff(Resid,IRF,IRF2,IRF_95,IRF_952,IRF_05,IRF_052,IRF_68, IRF_32,IRF_682, IRF_322,T_irf):
    Resid = Resid[Resid.columns.drop(list(Resid.filter(regex='Cross')))]
    fig = make_subplots(rows=len(Resid.T), cols=1,
        subplot_titles=([x for x in Resid.columns]))

    IRF3 = IRF - IRF2
    IRF_953 = IRF_95-IRF_052
    IRF_053 = IRF_05-IRF_952
    IRF_683 = IRF_68-IRF_322
    IRF_323 = IRF_32-IRF_682
    for i in range(len(Resid.T)):
        fig.append_trace(go.Scatter(
            x=np.array(range(T_irf)),
            y=IRF3[i,:], mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
            showlegend=False
        ), row=i+1, col=1)   

        fig.append_trace(go.Scatter(
            x=list(range(T_irf))+list(range(T_irf))[::-1],
            y=list(IRF_953[i,:])+list(IRF_053[i,:])[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False), row=i+1,col=1)

        fig.append_trace(go.Scatter(
            x=list(range(T_irf))+list(range(T_irf))[::-1],
            y=list(IRF_683[i,:])+list(IRF_323[i,:])[::-1],
            fill='toself',
            fillcolor='rgba(31, 70, 180,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False), row=i+1,col=1)

    return fig
