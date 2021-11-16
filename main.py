import pandas as pd
import numpy as np
import INS_funcs as inst
import bootstrap as btsrp
from plotting_func import plotANS, plotDiff
from input_data import GetData

# Enter in model specifications
link = 'ModelSpecs_inst.xlsx'


 
general = pd.read_excel(link,sheet_name='General').set_index('Variable')['Parameter']
vrs = pd.read_excel(link,sheet_name='Variables')
cross = pd.read_excel(link,sheet_name='Cross')


# Read type of model and find data
if general['Michele?'] == 1:
    df = pd.read_excel('Sample_data.xlsx', sheet_name='Sheet1')
    proxy = pd.read_excel('Sample_data.xlsx', sheet_name='Sheet2')
    df_dates = pd.read_excel('Sample_data.xlsx', sheet_name='Sheet3')
    position_ind = 1
    deciles = [0,0]
    X, y, T_used, p, k = btsrp.CreateLags(df, general)
elif general['GK'] ==1:
    df = pd.read_excel('Sample_data.xlsx',sheet_name='Sheet4')
    df_dates = pd.read_excel('Sample_data.xlsx', sheet_name='Sheet5')
    proxy = pd.read_excel('Sample_data.xlsx',sheet_name='Sheet6')
    X, y, T_used, p, k = btsrp.CreateLags(df, general)
    deciles = [0,0]

    if general['Instrument']==1:
        position_ind = 0
    else:
        position_ind=2
else:
    if general['offline']==1:
        df = pd.read_excel('df.xlsx').set_index('Date')
        high = np.percentile(df['vxo'],90,interpolation='midpoint')
        low = np.percentile(df['vxo'],10,interpolation='midpoint')
        deciles = [high,low]
    else:   
        df, deciles = GetData(general,vrs,cross)
        df = df.dropna()
        
        if general['Instrument']==1:
            for i in range(len(vrs)):
                if vrs['Inst'][i] == 'x':
                    position_prox = i
            proxy = df.iloc[:,position_prox]
            prox_name = proxy.name
            df = df.drop([proxy.name],axis=1)
            df_dates=0 
    
    X, y, T_used, p, k = btsrp.CreateLags(df, general)



        
    for i in range(len(vrs)):
        if vrs['Shock'][i] == 'x':
            position_ind = i

quant = general['quant']


 
 
# Run model type

if general['ANS']==1:
    Sigmahat, Resid_l, Ahat_l, Ahat_determ_l, X_determ_l = btsrp.GetErrors(X, y, T_used, p, k,general)
    tag = general['tag']
    y = y.drop([tag],axis=1)
    y = y[y.columns.drop(list(y.filter(regex="Cross")))] 
    Sigmahat, Resid, Ahat, Ahat_determ, X_determ = btsrp.GetErrors(X, y, T_used, p, k,general)
   
    Dhatph, Dhatpl, Dhatoh, Dhatol = btsrp.ANStrick(Ahat,Ahat_determ,deciles,p,tag)
    B = np.linalg.cholesky(Sigmahat)
    m = len(Ahat)
    IRF, T_irf = btsrp.IRFest(B, Dhatph,general,vrs,p,m)
    IRF2, T_irf = btsrp.IRFest(B, Dhatpl,general,vrs,p,m)
    IRF_Boot_arr,IRF_Boot_arr2 = btsrp.Bootstrap(general,k,T_used,Resid_l,Ahat_determ_l,X_determ_l,T_irf,p,df,0,1,Ahat_l,1,0,position_ind,vrs,deciles,m,tag)
    IRF_95, IRF_05, IRF_50, IRF_68, IRF_32 = btsrp.getPercentiles(m,T_irf,IRF_Boot_arr,quant)
    IRF_952, IRF_052, IRF_502, IRF_682, IRF_322 = btsrp.getPercentiles(m,T_irf,IRF_Boot_arr2,quant)
    prox_name = 'Sentiment'


else:
    Sigmahat, Resid, Ahat, Ahat_determ, X_determ = btsrp.GetErrors(X, y, T_used, p, k,general)

if general['Cholesky']==1:
    B = np.linalg.cholesky(Sigmahat)
    IRF, T_irf = btsrp.IRFest(B,Ahat, general, vrs, p, k)
    IRF_Boot_arr,IRF_Boot_arr2 = btsrp.Bootstrap(general,k,T_used,Resid,Ahat_determ,X_determ,T_irf,p,df,0,1,Ahat,1,0,position_ind,vrs,deciles,0,0)
    IRF_95, IRF_05,IRF_50 = btsrp.getPercentiles(k,T_irf,IRF_Boot_arr,quant)
    IRF2 = IRF.copy()
    IRF_952 = IRF_95.copy()
    IRF_052 = IRF_05.copy()
    IRF_502 = IRF_50.copy()    

if general['Instrument']>0:

    Resid_x, proxy_x = btsrp.AlignData(Resid, proxy, df_dates)

    n_instruments = general['Instrument']
    interval = 0.388
    instrument = inst.Winsorize(proxy_x,interval)
    imp_vect_relat, imp_vec_abs, F_stat_test1, F_stat_test2, p_values_test1, \
    p_values_test2, signif_test1, signif_test2, R2_test1, R2_test2, Betas_test1, \
    corr_instrument_shock, a_vect = inst.IDinstruments(Resid_x, instrument, Sigmahat, k, n_instruments,position_ind,general)
    B = np.concatenate([np.array(imp_vec_abs).reshape((k,1)),np.zeros((k,k-1))],axis=1)
    IRF, T_irf = btsrp.IRFest(B,Ahat, general, vrs,p,k)
    IRF_Boot_arr,IRF_Boot_arr2 = btsrp.Bootstrap(general,k,T_used,Resid,Ahat_determ,X_determ,T_irf,p,df,df_dates,proxy,Ahat,instrument,n_instruments,position_ind,vrs,0,0,0)
    IRF_95, IRF_05,IRF_50,IRF_67, IRF_33 = btsrp.getPercentiles(k,T_irf,IRF_Boot_arr,quant)
    IRF_952 = IRF_95.copy()
    IRF_052 = IRF_05.copy()
    IRF_502 = IRF_50.copy()  
# Develop and save plots
fig = plotANS(Resid,IRF_50,IRF_502,IRF_95,IRF_952,IRF_05,IRF_052,T_irf,prox_name)
fig.update_layout(height=k*200, width=500, title_text="Impulse Response Functions")
fig.write_image('images/inst_nov11_old.png')

# fig2 = plotDiff(Resid,IRF_50,IRF_502,IRF_95,IRF_952,IRF_05,IRF_052,IRF_68, IRF_32,IRF_682, IRF_322,T_irf)
# fig2.update_layout(height=k*200, width=500, title_text="Difference in Impulse Response Functions")
# fig2.write_image('images/diff_nov8_2020.png')

import os
os.system('say "done"')


 
#%%
#%%


