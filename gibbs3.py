import numpy as np
from scipy.stats import invwishart
import pandas as pd

def sample_beta(y, Z, x,Sigma, v0, aprior):
    Z = np.kron(np.eye(len(y.T)),x)
    Variance = np.kron(np.linalg.inv(Sigma),np.eye(len(y)))
    Vpost = np.linalg.inv(np.linalg.inv(v0)+np.dot(np.dot(Z.T,Variance),Z))
    part1 = np.dot(np.linalg.inv(v0),aprior)+np.dot(np.dot(Z.T,Variance),np.array(y.T).reshape(len(y.T)*len(y)))
    apost = np.dot(Vpost,part1)
    try:
        alpha = apost + np.dot(np.linalg.cholesky(Vpost),np.random.normal(0,1,len(x.T)*len(y.T)))
        ALPHA = alpha.reshape((len(y.T),len(x.T))).T
    except:
        ALPHA = [['broke']]
    
    return ALPHA

# =============================================================================
# def sample_beta(y, Z, x,Sigma, v0, aprior):
#     N = len(y)
#     assert len(x) == N
#     partm = np.dot(x.T,x)
#     partm2 = np.kron(partm,np.linalg.inv(Sigma))
#     partm3 = np.kron(x.T,np.linalg.inv(Sigma))
#     Vpostm = np.linalg.inv(np.linalg.inv(v0)+partm2)
# 
#    # Vpost = np.linalg.inv(np.linalg.inv(v0)+part2)
#     part3 = np.dot(np.linalg.inv(v0),aprior)+np.dot(partm3,np.array(y).reshape(len(y.T)*len(y)))
#     apost = np.dot(Vpostm,part3)
#     try: 
#         alpha = apost + np.dot(np.linalg.cholesky(Vpostm),np.random.normal(0,1,len(x.T)*len(y.T)))
#         ALPHA = alpha.reshape((len(x.T),len(y.T))).T
#     except: 
#         ALPHA = [['broke']]
#     
#     return ALPHA
# 
# =============================================================================
    #alpha = apost + np.dot(np.linalg.cholesky(Vpostm),np.random.normal(0,1,len(x.T)*len(y.T)))
    #alpha = np.random.normal(apost,np.diag(Vpostm))
   
# =============================================================================
#     variance = np.kron(np.linalg.inv(Sigma),np.eye(len(x)))
#     part1 = np.dot(Z.T,variance)
#     part2 = np.dot(part1,Z)
#     V_post = np.linalg.inv(np.linalg.inv(v0)+part2)
#     part3 = np.dot(part1,np.array(y).reshape(len(y.T)*len(y)))
#     apost = np.dot(V_post,np.dot(np.linalg.inv(v0),aprior)+part3)
#     alpha = apost + np.dot(np.linalg.cholesky(V_post),np.random.normal(0,1,len(x.T)*len(y.T)))
#     ALPHA = alpha.reshape((len(y.T),len(x.T))).T
#    
# =============================================================================
   


def sample_sigma(y, x, ALPHA):
    vbari=0
    vpost = len(x)+vbari
    Spost = np.eye(len(y.T)) + np.dot((y-np.dot(x,ALPHA)).T,(y-np.dot(x,ALPHA)))
    #Sigma = invwishart.rvs(vpost,np.linalg.inv(Spost))
    try:
        Sigma = invwishart.rvs(vpost,Spost)
    except:
        Sigma = [['broke']]
    return Sigma


 
 
def getPriors(x,y):
    XX = np.dot(x.T,x)
    XY = np.dot(x.T,y)
    try:
        Ahat_step = pd.DataFrame(np.linalg.lstsq(XX,XY,rcond=None)[0])
    except:
        return [['broke']],[['broke']]
    XA = np.dot(x,Ahat_step)
    Resid = y-XA #error in predictions
    Resid = Resid.dropna()

    RSS = np.dot(Resid.T,Resid)
    Sigma_ols = RSS/(len(x)-len(x.T)+1)
    Ahat_step = np.array(Ahat_step).reshape(len(x.T)*len(y.T))
    #v0 = np.diag(np.array([np.array(y.std()**2)]*len(x.T)).reshape(len(x.T)*len(y.T)))
    return Sigma_ols, Ahat_step

def gibbs(y, x, iters):
    assert len(y) == len(x)
    Sigma, aprior = getPriors(x,y)
    #if Sigma == 'broke':
     #   return 'broke', 'broke'
    Z = np.kron(np.eye(len(y.T)),x)
    v0 = np.eye(len(y.T)*len(x.T))
    #need every 7th one on the diagonal to be 1
    ffr_vec = np.ones(len(x.T)*len(y.T))
    #ffr_vec[4::5]=1
    #ffr_vec[6::7]=2
    v0 = np.diag(np.dot(v0,ffr_vec))*10
    aprior = np.ones(len(y.T)*len(x.T))*0
    #Sigma = np.eye(len(y.T))
   
    tracea = np.zeros((iters, len(x.T)*len(y.T))) ## trace to store values of beta_0, beta_1, tau
    tracesig = np.zeros((iters,+len(y.T)*len(y.T)))
    for it in range(iters):
        if Sigma[0][0]=='broke':
            continue
        ALPHA = sample_beta(y, Z, x,Sigma, v0,aprior)
        if ALPHA[0][0] == 'broke':
            continue
        else:
            tracea[it,:] = np.hstack(ALPHA.T)

        Sigma = sample_sigma(y, x, ALPHA)
        if Sigma[0][0] == 'broke':
            continue
        else:
            tracesig[it,:] = np.hstack(Sigma)

        #trace[it,:] = np.append(np.hstack(ALPHA.T),np.hstack(Sigma))
       
    tracea = pd.DataFrame(tracea)
    tracesig = pd.DataFrame(tracesig)
    #trace.columns = ['beta_0', 'beta_1', 'tau']
       
    return tracea, tracesig

def main(x,y):
    iters = 1000
    st = int(iters/3)
    ed = int(iters-1)
    tracea, tracesig = gibbs(y, x, iters)
   # if tracea == 'broke':
    #    return 'broke','broke'
       
    trace_burnta = tracea[st:ed]
    #trace_burnta = trace_burnta[::9]
    trace_burntsig = tracesig[st:ed]
    #trace_burntsig = trace_burntsig[::9]
   
    median = np.array(trace_burnta.mean())
    Ahat = median.reshape((len(y.T),len(x.T))).T
    Sigmahat = np.array(trace_burntsig.mean()).reshape((len(y.T),len(y.T)))
    #pd.DataFrame(trace_burnt.median()).to_excel('median.xlsx')
    return pd.DataFrame(Ahat), Sigmahat

# =============================================================================
# x = pd.read_excel('x6.xlsx')
# y = pd.read_excel('y6.xlsx')
# ahat, sigmahat = main(x,y)
# ahat.to_excel('ahat.xlsx')
# resid = y-np.dot(x,ahat)
# resid['FedFundsRate'].mean()
# 
# =============================================================================

 

