# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:56:23 2021

@author: GENTILE_F
"""
from fredapi import Fred
import pandas as pd
import numpy as np
import yfinance as yf

def GetData(general,vrs,cross):
    fred = Fred(api_key = '13a856ab4d9b2976146eb2e1fddd511d')
    vrs_list = []
    start_date= general['Start Date']
    end_date = general['End Date']

    for i in range(len(vrs)):
        if vrs['Frequency'][i]==' ':
            series = pd.Series(fred.get_series(vrs['Tag'][i]).dropna(),name=vrs['Name'][i])
        elif vrs['Units'][i]==' ':
            series = pd.Series(fred.get_series(vrs['Tag'][i],frequency=vrs['Frequency'][i]).dropna(),name=vrs['Name'][i])
        elif vrs['Units'][i]=='yf':
            series = pd.Series(yf.Ticker(vrs['Tag'][i]).history(start='1970-01-02',end='2021-10-01',interval=vrs['Frequency'][i])['Close'],name=vrs['Name'][i])
        else:
            series = pd.Series(fred.get_series(vrs['Tag'][i],frequency=vrs['Frequency'][i],units=vrs['Units'][i]).dropna(),name=vrs['Name'][i])

        if vrs['Transform'][i]=='logd':
            series = (np.log(series)-np.log(series.shift(1)))*100
        if vrs['Transform'][i] =='d':
            series = series-series.shift(1)
        if vrs['Transform'][i]=='acc':
            series = np.log(-1*series)*100
        if vrs['Transform'][i]=='log':
            series = np.log(series)*100
        
        vrs_list.append(series)
    
    df = pd.DataFrame(vrs_list).T
    df.index = pd.DatetimeIndex(df.index.values,freq=df.index.inferred_freq)
    df=df[df.index>start_date]
    df=df[df.index<end_date]
    deciles = [0,0]
  
    for i in range(len(cross)):
        if general['ANS']==1:
            if cross['Variable 1'][i]=='VXO':
                snp = yf.Ticker('^GSPC').history(start='1970-01-02',end='1986-01-01')
                snp = snp.reset_index()
                gb = snp.groupby(snp['Date'].dt.to_period('Q'))['Close']
                gb = gb.std().dropna()
                gb = gb.reset_index()
                gb['Date']=gb['Date'].apply(str)
                mydic = {'Q1':'-01-01','Q2':'-04-01','Q3':'-07-01','Q4':'-10-01'}
                gb['Date']=gb['Date'].replace(mydic,regex=True)
                gb = gb.set_index('Date')
                
                vxo = pd.Series(fred.get_series('VXOCLS',frequency='q'))
                vxo = vxo[vxo.index<end_date]
                vxo = vxo[vxo.index>start_date]

                gb = gb*vxo.iloc[0]/gb['Close'].iloc[-1]
                
                vxo = pd.DataFrame(gb['Close'].append(vxo.dropna()))
                vxo = vxo.reset_index()
                vxo.index = pd.to_datetime(vxo['index']).dt.date
                vxo = vxo.drop('index',axis=1)
                
                vxo = np.log(vxo)
                #vxo = (vxo - np.mean(vxo))/np.std(vxo)
                
                
               # vxo = pd.read_excel('JLN.xlsx').set_index('Date')
                
                df['vxo']=vxo
                high = np.percentile(vxo,90,interpolation='midpoint')
                low = np.percentile(vxo,10,interpolation='midpoint')
                deciles = [high,low]
                #deciles = [1.15,-1.15]
               # df = df*high
                #step2 = vxo >= high
                #df = df[step2[0]]
                
                #for j in range(len(x_high)):
                 #    x_high[j]=high
                df = df.dropna()
                df['Cross'+str(i)]=df['vxo']*df[cross['Variable 2'][i]]
                df = df.dropna()
               # df = df.drop(cross['Variable 2'][i],axis=1)
            elif cross['Variable 1'][i]=='Sent':
                tag = general['Sent']
                series = pd.Series(fred.get_series(tag,frequency='q',start_date = start_date, end_date = end_date).dropna(),name='Sent')
                series = np.log(series)
                series = series - series.shift(1)
                series = series.dropna()
                #series = (series - np.mean(series))/np.std(series)

                df['Sent']=series
                df['Cross'+str(i)]=series*df[cross['Variable 2'][i]]
                high = np.percentile(series,90,interpolation='midpoint')
                low = np.percentile(series,10,interpolation='midpoint')
                deciles = [high,low]
                df = df.dropna()

        else:
            df['Cross'+str(i)]=df[cross['Variable 1'][i]]*df[cross['Variable 2'][i]]
            series = df[cross['Variable 1'][i]]
            high = np.percentile(series,90,interpolation='midpoint')
            low = np.percentile(series,10,interpolation='midpoint')
            deciles = [high,low]
    #df.to_excel('macrodata.xlsx')
    return df, deciles
