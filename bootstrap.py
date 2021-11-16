import pandas as pd
import numpy as np
import INS_funcs as inst
import gibbs3

def AlignData(Resid, proxy,df_dates):
    if df_dates == 0:
        proxy.name = 'Proxy'
        Resid = pd.concat([Resid,proxy],axis=1)
        Resid = Resid.dropna()
        proxy = Resid['Proxy']
        Resid = Resid.drop(['Proxy'],axis=1)
        
    else:
        if len(Resid)!=len(proxy):
            Resid = pd.merge(Resid,df_dates,how='outer',left_index=True,right_index=True)
            Resid = pd.merge(Resid,proxy,on=['Year','Month'])
        try:
            Resid = Resid.drop(['Proxy'],axis=1)
        except:
            pass
        proxy = pd.merge(proxy,Resid,on=['Year','Month'])
        Resid = Resid.drop(['Year','Month'],axis=1)
        proxy = proxy['Proxy']
    return Resid, proxy

def MakeStructure(general,k,T_irf,df,p):
    nBoot = general['nBoot']
    Y_init = df[:p]
    IRF_Boot_arr = np.zeros((k,T_irf,nBoot))
    return IRF_Boot_arr, Y_init, nBoot
   
def CreateRandoms(T_used,p,k,df,proxy,df_dates,Y_init,n_instruments,Resid):
    vec = CreateVec(T_used)
    Y_pseudo = np.zeros((k,len(df)))
    Y_pseudo[:,:p] = np.array(Y_init.T)

    if n_instruments >0:
        vec.index = Resid.index
        vec_a, proxy_x = AlignData(vec, proxy, df_dates)
    else:
        vec_a = 0
        proxy_x = 0
    Resid_pseudo = np.dot(Resid.T, np.diag(vec[0]))
    return Y_pseudo, pd.DataFrame(Resid_pseudo), vec_a, proxy_x
   
def MakePseudoModel(Y_pseudo,df,general,vec_a,instrument,proxy,k,n_instruments,position_ind,df_dates,Resid_pseudo,tag):
    boot_df = pd.DataFrame(Y_pseudo.T,columns=df.columns)
    X_boot, y_boot, T_used_Boot, p, k = CreateLags(boot_df,general)

    if general['ANS']==1:
        y_boot = y_boot.drop([tag],axis=1)   
        y_boot = y_boot[y_boot.columns.drop(list(y_boot.filter(regex="Cross")))] 
        Sigmahat_Boot, Resid_Boot, Ahat_Boot, Ahat_determ_x, X_determ_x = GetErrors(X_boot, y_boot, T_used_Boot, p, k,general)
       # if Sigmahat_Boot == 'broke':
        #    return 'broke','broke','broke'
    else:
        Sigmahat_Boot, Resid_Boot, Ahat_Boot, Ahat_determ_x, X_determ_x = GetErrors(X_boot, y_boot, T_used_Boot, p, k,general)
   
    
    if general['Instrument']==1:
        instrument_boot = np.dot(np.diag(vec_a[0]),instrument)
        Resid_p = Resid_pseudo.T
        #Resid_p.index += p
        Resid_p.index = proxy.index[p:]
        Resid_use_Boot, proxy_x = AlignData(Resid_p, proxy, df_dates)
        imp_vect_relat, imp_vec_abs_boot, F_stat_test1, F_stat_test2, p_values_test1, \
            p_values_test2, signif_test1, signif_test2, R2_test1, R2_test2, Betas_test1, \
                corr_instrument_shock, a_vect = inst.IDinstruments(Resid_use_Boot, instrument_boot, \
                    Sigmahat_Boot, k, n_instruments,position_ind,general)
        B_boot = np.concatenate([np.array(imp_vec_abs_boot).reshape((k,1)),np.zeros((k,k-1))],axis=1)
    else:
        try:
            B_boot = np.linalg.cholesky(Sigmahat_Boot)
        except:
            B_boot = 0
    return B_boot, Ahat_Boot, Ahat_determ_x

def MakePseudoData(Y_pseudo,j,p,k,Ahat,Ahat_determ,X_determ,Resid_pseudo):
    step1 = Y_pseudo[:,j:p+j]
    step2 = np.fliplr(step1)
    step3 = pd.DataFrame(np.reshape(step2.T,k*p))
    step4 = np.add(np.array(np.dot(Ahat,step3[0])),np.dot(Ahat_determ.T,X_determ[j,:]))
    Y_pseudo[:,j+p]= np.add(step4,np.array(Resid_pseudo)[:,j]) 
    return Y_pseudo


def Bootstrap(general,k,T_used,Resid,Ahat_determ,X_determ,T_irf,p,df,df_dates,proxy,Ahat,instrument,n_instruments,position_ind, vrs,deciles,m,tag):
    #make pseudo data
    if general['ANS']==1:
        IRF_Boot_arr, Y_init,nBoot = MakeStructure(general,m,T_irf,df,p)
        IRF_Boot_arr2 = IRF_Boot_arr.copy()
        general['Bayes']=0
    else:
        IRF_Boot_arr, Y_init,nBoot = MakeStructure(general,k,T_irf,df,p)
        IRF_Boot_arr2 = IRF_Boot_arr.copy()
        general['Bayes']=0

    for i_Boot in range(nBoot):
        #generate a vector that flips sign randomly
        Y_pseudo, Resid_pseudo, vec_a, proxy_x = CreateRandoms(T_used,p,k,df,proxy,df_dates,Y_init,n_instruments,Resid)

        for j in range(T_used):
            Y_pseudo = MakePseudoData(Y_pseudo,j,p,k,Ahat,Ahat_determ,X_determ,Resid_pseudo)
       
       # if general['ANS']==1:
         #   Y_pseudo[-1] = Y_pseudo[-2]*Y_pseudo[-3]
        #make pseudo model
        B_boot, Ahat_Boot, Ahat_determ_x =MakePseudoModel(Y_pseudo,df,general,vec_a,instrument,proxy,k,n_instruments,position_ind,df_dates,Resid_pseudo,tag)
       # if B_boot == 'broke':
        #    pass
      
        if general['ANS']==1:
            high = np.percentile(Y_pseudo[-2],90,interpolation='midpoint')
            low = np.percentile(Y_pseudo[-2],10,interpolation='midpoint')
            deciles_boot = [high,low]

            Dhatph, Dhatpl, Dhatoh, Dhatol = ANStrick(Ahat_Boot,Ahat_determ_x,deciles_boot,p,tag)
            IRFestim_Boot, T_irf = IRFest(B_boot, Dhatph, general, vrs, p, m)
            IRFestim_Boot2, T_irf = IRFest(B_boot, Dhatpl, general, vrs, p, m)
        else:
            IRFestim_Boot, T_irf = IRFest(B_boot, Ahat_Boot, general, vrs, p, k)
            IRFestim_Boot2 = IRFestim_Boot.copy()
           
 
        IRF_Boot_arr[:,:,i_Boot] = IRFestim_Boot
        IRF_Boot_arr2[:,:,i_Boot] = IRFestim_Boot2
    return IRF_Boot_arr,IRF_Boot_arr2

       
def getPercentiles(k,T_irf,IRF_Boot_arr,quant):
    IRF_95 = np.ones((k,T_irf))
    IRF_05 = np.ones((k,T_irf))
    IRF_50 = np.ones((k,T_irf))
    IRF_67 = np.ones((k,T_irf))
    IRF_33 = np.ones((k,T_irf))
    for t in range(T_irf):
        for q in range(k):
            IRF_95[q,t] = np.percentile(IRF_Boot_arr[q,t,:],quant,interpolation='midpoint')
            IRF_05[q,t] = np.percentile(IRF_Boot_arr[q,t,:],100-quant,interpolation='midpoint')
            IRF_50[q,t] = np.percentile(IRF_Boot_arr[q,t,:],50,interpolation='midpoint')
            IRF_67[q,t] = np.percentile(IRF_Boot_arr[q,t,:],68,interpolation='midpoint')
            IRF_33[q,t] = np.percentile(IRF_Boot_arr[q,t,:],32,interpolation='midpoint')
    return IRF_95, IRF_05, IRF_50, IRF_67, IRF_33
def CreateVec(T_used):
    step = pd.DataFrame(np.random.rand(T_used,1))
    change_sign = pd.DataFrame(-1*np.ones((T_used,1)))
    vec = pd.DataFrame(np.ones((T_used,1)))
    vec[0] = change_sign[step[0]>0.5]
    vec = vec.fillna(1)
    return vec
def CreateLags(df, general):
    k = len(df.T) #this is the number of variables
    T = len(df) #amount of time series data
    p = general['Lags']
    T_used = T-p
   
    y = df[p:] #actual dependent variables
    y = y.dropna()
    df_lag = df.copy()
    for g in range(1,p+1): #creates independent variables with lags
        lag = df.shift(g)
        lag.columns = [x + "_lag"+str(g) for x in df.columns]
        df_lag = pd.concat((df_lag,lag),axis=1)
    X = df_lag.dropna()
    #just use the lags as dependent variable
    X = X.drop(list(df.columns),axis=1)
    if general['Constant']=='x':
        X = pd.concat((X,pd.Series(np.ones(len(X)),index=X.index)),axis=1)
    #X['const'] = 1
    return X, y, T_used, p, k

def GetErrors(X, y, T_used, p, k,general):
    XX = np.dot(X.T,X)
    XY = np.dot(X.T,y)

    if general['Bayes']==1:
        Ahat_step, Sigmahat = gibbs3.main(X,y)
       # if Ahat_step == 'broke':
        #    return 'broke','broke','broke','broke'
        Ahat_step.columns = y.columns
        Ahat_step.index = X.columns

        XA = np.dot(X,Ahat_step)
        Resid = y-XA #error in predictions
        Resid = Resid.dropna()
        Ahat_determ = Ahat_step[Ahat_step.index==0]
        X_determ = np.array(X)[:,k*p:]
        Ahat = Ahat_step.T
        Ahat = Ahat.drop([0],axis=1)

    else:
        Ahat_step = pd.DataFrame(np.linalg.lstsq(XX,XY,rcond=None)[0])
       
        Ahat_step.columns = y.columns
        Ahat_step.index = X.columns
   
        Ahat = Ahat_step.T

        Ahat = Ahat.drop([0],axis=1)
   
        XA = np.dot(X,Ahat_step)
       # XA.columns = y.columns
   
        Resid = y-XA #error in predictions
        Resid = Resid.dropna()
   
        RSS = np.dot(Resid.T,Resid)
        if T_used > 2*len(X.T):
            Sigmahat = RSS/(T_used-len(X.T))
        else:
            Sigmahat = RSS/T_used
        if len(X.T)>k*p:
            X_determ = np.array(X)[:,k*p:]
            Ahat_determ = Ahat_step[Ahat_step.index==0]
        else:
            X_determ = 0
            Ahat_determ = np.zeros((k,1))
   
    return Sigmahat,Resid, Ahat, Ahat_determ, X_determ

def IRFest(B, Ahat,general,vrs,p,k):
    #find where the shock is
    for i in range(len(vrs)):
        if vrs['Shock'][i] == 'x':
            s = i

    T_irf = general['T_irf'] # how far to make impulse response graph
    shock_size = general['Shock size']
    Spseduo = np.zeros((k,T_irf))

    if general['Michele?'] == 1:
        if general['Instrument']==1:
            s = 0
            transform = np.array([100,1,100,100,100,1,100,100])
            n = np.kron(transform,np.ones((1,T_irf)))
        else:
            s = 1 #temporary for replicating Michele
            transform = np.array([100,1,100,100,100,1,100,100])
            n = np.kron(transform,np.ones((1,T_irf)))

    if general['GK']==1:
        if general['Instrument']==1:
            s =0
        else:
            s=2

    #add in shock
    Spseduo[s][0] = shock_size

    Rpseudo = pd.DataFrame(np.dot(B,Spseduo))

    IRFestim = pd.DataFrame(np.zeros((k,T_irf)))
           
    IRFestim[[0]]=Rpseudo[[0]]
    #run through the model

    for j in range(2,T_irf+1):
        step1 = np.array(pd.concat((pd.DataFrame(np.fliplr(IRFestim)),pd.DataFrame(np.zeros((k,p-1)))),axis=1))
        step2 = pd.DataFrame(np.reshape(step1.T,k*T_irf+k*(p-1)))
        step3 = step2[k*(T_irf-j+1):k*(T_irf-j+1+p)]
        step4 = np.add(np.dot(Ahat,step3),Rpseudo[[j-1]])
        IRFestim[j-1] = step4.reset_index().drop('index',axis=1)
   
    try:
        IRF = np.multiply(np.reshape(n,(k,T_irf)),IRFestim)
    except:
        IRF = IRFestim
    return IRF, T_irf

def ANStrick(Ahat,Ahat_determ,deciles,p,tag):
    Bhat = Ahat.filter(regex="Cross")
    #get D0
    A0 = Ahat_determ
    Cp = Ahat.filter(regex=tag)
    Dhatoh = pd.DataFrame(Cp.sum(axis=1)*deciles[0]+A0)
    Dhatol = pd.DataFrame(Cp.sum(axis=1)*deciles[1]+A0)
    #get Ahat
    Ahat = Ahat[Ahat.columns.drop(list(Ahat.filter(regex="Cross")))]
    Ahat = Ahat[Ahat.columns.drop(list(Ahat.filter(regex=tag)))]
    Dhatph = Ahat.copy()
    Dhatpl = Ahat.copy()
    interact=0

    for j in range(1,p+1):
        Bhatp = Bhat.filter(regex='lag'+str(j))
        Ahatp = Ahat.filter(regex=str(j))
        if interact==0:
            Ahatp = Ahatp.filter(regex="Fed") 

            
        dhatp = Ahatp+np.array(Bhatp*deciles[0])
        dhatl = Ahatp+np.array(Bhatp*deciles[1])

        if interact==0:
            Dhatph['FedFundsRate_lag'+str(j)] = dhatp
            Dhatpl['FedFundsRate_lag'+str(j)] = dhatl
            
        else:
            Dhatph.iloc[:,len(dhatp.T)*(j-1):len(dhatp.T)*j]=dhatp
            Dhatpl.iloc[:,len(dhatp.T)*(j-1):len(dhatp.T)*j]=dhatl
               
    return Dhatph, Dhatpl, Dhatoh, Dhatol 
        

