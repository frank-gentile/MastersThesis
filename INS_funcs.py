import pandas as pd
import numpy as np
from scipy.stats import f, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
from random import randrange

def Winsorize(instrument,interval):
    low = np.percentile(instrument,interval,interpolation='linear')
    high = np.percentile(instrument,100-interval,interpolation='linear')
    step1 = instrument <= low
    instrument[step1] = low
    step2 = instrument >= high
    instrument[step2] = high
    return instrument

def IDinstruments(Resid,instrument,Sigmahat,k,n_instruments, position_ind,general):

    Betas_test1 = np.ones((k,1))

    for i in range(k):
        yyy = Resid.iloc[:,i]
        ins = np.array(instrument)
        xxx_rest = np.ones((len(ins)))
        xxx = np.array([xxx_rest, ins]).T

        F_stat_test1, p_values_test1, signif_test1, R2_test1, coeff = getR2(xxx,yyy,xxx_rest,i,k)

        if n_instruments == 1:
            Betas_test1[i] = coeff[1]

    for i in range(n_instruments):
        if n_instruments == 1:
            yyy = instrument.copy()
        else:
            yyy = instrument[:,i]   

        xxx_rest = np.ones((1,len(Resid)))
        xxx = np.concatenate((xxx_rest.T, np.array(Resid)),axis=1)
        xxx_rest = np.ones((len(Resid)))
        F_stat_test2, p_values_test2, signif_test2, R2_test2, coeff = getR2(xxx,yyy,xxx_rest,i,n_instruments)

    if position_ind > 0:
        #rearrange order of resids
        step1 = np.array(range(k))
        step2 = np.array(step1 != position_ind)
        step3 = step1[step2]
        step4 = np.array(Resid)[:,position_ind]
        step5 = step4.reshape(1,len(ins))
        Resid = np.concatenate([step5.T, np.array(Resid)[:,step3]],axis=1)

        #rearrange covariance matrix


        Sig_arr = np.array(Sigmahat)
        Sig_step11 = Sig_arr[position_ind,position_ind]
        Sig_step12 = Sig_arr[position_ind,position_ind-1]
        Sig_step13 = Sig_arr[position_ind,position_ind+1:]
        Sig_step22 = Sig_arr[position_ind-1,position_ind-1]
        Sig_step23 = Sig_arr[position_ind-1,position_ind+1:]
        Sig_step33 = Sig_arr[position_ind+1:,position_ind+1:]

        new_sig = np.ones((len(Sigmahat),len(Sigmahat)))
        new_sig[0,0] = Sig_step11
        new_sig[0,position_ind] = Sig_step12
        new_sig[0,position_ind+1:] = Sig_step13
        new_sig[position_ind,0] = Sig_step12
        new_sig[position_ind+1:,0] = Sig_step13.T
        new_sig[position_ind,position_ind]= Sig_step22
        new_sig[position_ind,position_ind+1:] = Sig_step23
        new_sig[position_ind+1:,position_ind] = Sig_step23.T
        new_sig[position_ind+1:,position_ind+1:] = Sig_step33
        Sigmahat = new_sig


        step4 = step3<position_ind
        step5 = step3[step4]
        step6 = step3>position_ind
        step7 = step3[step6]


        step5.reshape(1,len(step5))
        step7.reshape(1,len(step7))
        reorde = np.concatenate([step5+1,np.array([0]),step7])
        reorder = reorde.reshape(1,k)

    if n_instruments==1: 
        if general['GK']==1:
            yyy = np.array(Resid)
        else:
            yyy = Resid
        xxx = instrument
        beta_hat = np.dot(xxx.T,xxx)**(-1)*np.dot(xxx.T,yyy)

        imp_vect_relat = beta_hat/beta_hat[0]

    else: 
        imp_vect_relat = np.ones((k,1))

        yyy = Resid[:,0]
        xxx = instrument
        beta_hat = np.dot(xxx.T,xxx)**(-1)*np.dot(xxx.T,yyy)
        fitted = np.dot(xxx,beta_hat)
        for j in range(1,k):
            yyy = Resid[:,j]
            xxx = fitted
            gamma_hat = np.dot(xxx.T,xxx)**(-1)*np.dot(xxx.T,yyy)
            imp_vect_relat[j] = gamma_hat[0]
    
    mu = np.array(imp_vect_relat[1:])
    if general['GK']==1:
        Sigmahat = np.array(Sigmahat)
    Sigma11 = Sigmahat[0,0]
    Sigma21 = Sigmahat[1:,0]
    Sigma22 = Sigmahat[1:,1:]
    Sigma21 = Sigma21.reshape(len(Sigma21),1)
    mu = mu.reshape(len(mu),1)

    gamma = Sigma22 + Sigma11*np.dot(mu,mu.T) - np.dot(Sigma21,mu.T) - np.dot(mu,Sigma21.T)
    b12_sq1 = np.dot((Sigma21 - Sigma11*mu).T,np.linalg.inv(gamma))
    b12_sq2 = np.dot(b12_sq1,Sigma21-Sigma11*mu)
    if general['GK']==1:
        b11 = np.sqrt(Sigma11 - b12_sq2)[0][0]
    else:
        b11 = np.sqrt(Sigma11 - b12_sq2)[0][0]
    imp_vec_abs = imp_vect_relat*b11
    if general['GK']==1:
        imp_vec_abs = imp_vec_abs.reshape(1,len(imp_vec_abs))

    #estimate structural shocks
    if general['GK']==1:
        b21 = imp_vec_abs[0][1:]
    else:
        b21 = imp_vec_abs[1:]
    b21 = np.array(b21).reshape(len(b21),1)
    b12primeB22inv = np.dot((Sigma21 - b11*b21).T,np.linalg.inv(Sigma22 - np.dot(b21,b21.T)))
    a_vec= (b11-np.dot(b12primeB22inv,b21))**(-1)*np.concatenate([np.array([[1]]),-1*b12primeB22inv.T])
    estim_shocks = np.dot(Resid,a_vec)

    corr_instrument_shock = np.ones((n_instruments,1))
    for i in range(n_instruments):
        if n_instruments ==1:
            E_r_m = (np.dot(Resid.T,instrument))/len(instrument)
        else:
            E_r_m = (np.dot(Resid.T,instrument[:,i]))/len(instrument)
        if general['GK']==1:
            step = np.divide(E_r_m,imp_vec_abs[0])
        else:
            step = np.divide(E_r_m,imp_vec_abs)
        phi = step[randrange(k)]
        corr_instrument_shock[i] = phi/np.std(instrument)

    if position_ind > 0:
        if general['GK']==1:
            imp_vec_abs = imp_vec_abs[0][reorder]
        else:
            imp_vec_abs = imp_vec_abs[reorder]
        imp_vect_relat = imp_vect_relat[reorder]
        a_vec = a_vec[reorder]
    
    return imp_vect_relat, imp_vec_abs, F_stat_test1, F_stat_test2, p_values_test1, p_values_test2, signif_test1, signif_test2, R2_test1, R2_test2, Betas_test1, corr_instrument_shock, a_vec


def getR2(xxx,yyy,xxx_rest,i, n_instruments):
    F_test, Autocorr_test, White_test = Ftest_autocorr_homosk(yyy,xxx,xxx_rest)
    F_stat_test = np.ones((n_instruments,1))
    p_values_test = np.ones((n_instruments,4))
    signif_test = np.ones((n_instruments,4))
    R2_test = np.ones((n_instruments,1))
        
    F_stat_test[i] = F_test[0]
    p_values_test[i,0] = F_test[1]
    p_values_test[i,1] = Autocorr_test[0,1]
    p_values_test[i,2] = Autocorr_test[1,1]
    p_values_test[i,3] = White_test[1]

    signif_test[i,0] = SignifTest(F_test[1])
    signif_test[i,1] = SignifTest(Autocorr_test[0,1])
    signif_test[i,2] = SignifTest(Autocorr_test[1,1])
    signif_test[i,3] = SignifTest(White_test[1])

    coeff = np.dot(np.linalg.inv(np.dot(xxx.T,xxx)),np.dot(xxx.T,yyy))
    resid = yyy-np.dot(xxx,coeff)
    yyy_demean = yyy - np.average(yyy)
    R2_test[i] = 1 - np.dot(resid.T,resid) / np.dot(yyy_demean.T,yyy_demean)

    return F_stat_test, p_values_test, signif_test, R2_test, coeff

def SignifTest(pvalue):
    sig = 3*(pvalue<0.01)
    sig = sig + 2*(pvalue < 0.05 and pvalue > 0.01)
    sig = sig + 1*(pvalue < 0.1 and pvalue > 0.05)
    return sig

def Ftest_autocorr_homosk(yyy,xxx,xxx_rest):
    #restricted model
    coeff_rest = np.dot(xxx_rest.T,xxx_rest)**(-1)*np.dot(xxx_rest.T,yyy)
    resid_rest = yyy-xxx_rest*coeff_rest
    RSS_rest = np.dot(resid_rest.T,resid_rest)

    #unrestricted
    coeff = np.dot(np.linalg.inv(np.dot(xxx.T,xxx)),np.dot(xxx.T,yyy))
    resid = yyy-np.dot(xxx,coeff)
    RSS = np.dot(resid.T,resid)

    F_stat = (RSS_rest - RSS)*(len(yyy)-len(xxx.T))/(RSS*(len(xxx.T)-len(pd.DataFrame(xxx_rest).T)))
    F_pvalue = 1 - f.cdf(F_stat,len(xxx.T)-len(pd.DataFrame(xxx_rest).T),len(yyy)-len(xxx.T))
    F_test = [F_stat,F_pvalue]

    #autocorrelations
    Autocorr_test = np.ones((2,2))
    stat,pvalue= ljungbox(resid,4)
    Autocorr_test[0,:] = np.array((stat,pvalue))
    stat,pvalue=ljungbox(resid,8)
    Autocorr_test[1,:]=np.array((stat,pvalue))

    #white test
    # fittedv = np.dot(xxx,coeff)
    # yyy_Whitetest = resid**2
    # N = len(yyy)
    # xxx_Whitetest = np.array([np.ones((N,1)).T[0], fittedv, fittedv**2])

    # #restricted
    # coeff_Whitetest = np.dot(np.linalg.inv((np.dot(np.ones((N,1)).T,np.ones((N,1))))),np.dot(np.ones((N,1)).T,yyy_Whitetest))
    # resid_Whitetest = yyy_Whitetest - np.dot(np.ones((N,1)),coeff_Whitetest)
    # RSSrest_Whitetest = np.dot(resid_Whitetest.T,resid_Whitetest)

    # #unrestricted   
    # #stopped here 
    # coeff_Whitetest = np.dot(np.linalg.inv(np.dot(xxx_Whitetest.T,xxx_Whitetest)),np.dot(xxx_Whitetest,yyy_Whitetest))
    # resid_Whitetest = yyy_Whitetest - np.dot(xxx_Whitetest,coeff_Whitetest)
    # RSS_Whitetest = np.dot(resid_Whitetest.T,resid_Whitetest)

    # White_stat = (RSSrest_Whitetest-RSS_Whitetest)*(len(yyy_Whitetest)-len(xxx_Whitetest.T))/(RSS_Whitetest*2)
    # White_pvalue = 1-f.cdf(White_stat,2,len(yyy_Whitetest)-len(xxx_Whitetest.T))
    # White_test = [White_stat, White_pvalue]
    White_test = [1,1]

    return F_test, Autocorr_test, White_test

def ljungbox(resid,S):
    T = len(resid)
    step1 = np.zeros((S,1))
    for s in range(S):
        step1[s] = acf(resid,s+1)
    
    step1 = step1**2
    step1 = step1.T[0]
    step2 = T*np.ones((1,S))
    step3 = np.array(range(1,S+1))
    step4 = step2 - step3
    step5 = np.divide(step1,step4)
    Q = T*(T+2)*sum(step5[0])
    pvalue = 1-chi2.cdf(Q,S)
    return Q, pvalue

def acf(resid,s):
    T = len(resid)
    xbar = np.mean(resid)
    denom = 0
    num = 0
    for t in range(T-s):
        num = num + (resid[t]-xbar)*(resid[t+s]-xbar)
    for t in range(T):
        denom = denom + (resid[t]-xbar)**2
    rho = num / denom
    return rho
