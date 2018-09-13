from datetime import datetime, timedelta
import numpy as np
from sklearn.svm import OneClassSVM

from util import *

#dataRoot = 'path_for_data'
dataRoot = 'E:/data/ubicomp2018/release/nyc/'
data = np.loadtxt(dataRoot + 'taxibike.txt')
dists = np.loadtxt(dataRoot + 'dists.txt')

nR = 862 # number of regions
nS = 4 # number of data sources
MPS = 30 # minutes per time slot
nT = data.shape[0] # nummber of time slots
stDT = datetime(2014,1,15,0,0,0)

# Params for algorithm
alpha = 0.01
beta = 0.05
nDailyAnomaly_int = int(60 / MPS * 24 * nR * alpha)
nDailyAnomaly_r = int(60 / MPS * 24 * nR * beta)
t_delta = 2
corrThres = 0.95
lCorr = 60 * 24 * 7 // MPS  # use one week data for calculating pearson correlation 
R = 800

nNearPart = 2
mNear = np.identity(nR)
mNear = np.concatenate((mNear, (dists > 0) & (dists <= R)))
sMNear = np.repeat(mNear.sum(axis=1), nS*t_delta)
mNearTile = np.tile(mNear, (1, nS*t_delta))

score_ind = np.zeros((nT, nR*nS))
score_r = np.zeros((nT, nR)) + 100
score_int = np.zeros((nT, nR)) + 100
anomalies = np.zeros((nT, nR))
dVector = (t_delta - 1 + nNearPart) * nS

model_r = OneClassSVM(nu=0.1)
model_int = OneClassSVM(nu=0.1)
train_r = np.zeros((0, nS))
train_int = np.zeros((0, dVector))
tsTrain = 60 * 24 * 7 // MPS
nTrain = tsTrain * nR

detect_st = (datetime(2014,11,27) - stDT).days * 24 * 60 // MPS  # detect anomamlies in 2014-11-27
ed = (datetime(2014,11,28) - stDT).days * 24 * 60 // MPS
st = max(detect_st - tsTrain, lCorr)


trained = False
p1 = np.einsum('ij,ik->kj', data[(st-lCorr):st,:], data[(st-lCorr):st,:])
for ts in range(st, ed):
    print('\r' + str(ts), end='')

    # update pearson correlation
    pp = np.nan_to_num(pairPearson(data[(ts-lCorr):ts,:], data[(ts-lCorr):ts,:], p1))
    p1 = p1 + data[ts,:] * data[ts,:][:,None]
    p1 = p1 - data[ts-lCorr,:] * data[ts-lCorr,:][:,None] 
    pp_new = np.nan_to_num(pairPearson(data[(ts-lCorr+1):(ts+1),:], data[(ts-lCorr+1):(ts+1),:], p1))  

    pp_diff = pp - pp_new
    pp_diff[np.where(np.logical_or(pp < corrThres, pp_diff < 0))] = 0
    pp_diff = pp_diff * lCorr
    pp_tmp = np.array(pp)
    pp_tmp[np.where(pp < corrThres)] = 0

    # calculate individual anomaly score    
    scaledData = ((data[:(ts+1),:] - data[:(ts+1),:].mean(0)) / data[:(ts+1),:].std(0))[-1]
    weightedAvg = np.nan_to_num(np.sum(pp_tmp * np.tile(scaledData, (scaledData.shape[0], 1)), axis = 1) / np.sum(pp_tmp, axis=1))
    sign = ((scaledData > weightedAvg).astype(int) - 0.5) * 2
    score_ind[ts,:] = sign * np.nan_to_num(np.sum(pp_tmp * pp_diff, axis=1) / np.sum(pp_tmp, axis=1))
    
    tmpX = (mNearTile * score_ind[(ts-t_delta+1):(ts+1),:].ravel()).reshape((-1, nR)).sum(axis=1)
    tmpX = np.nan_to_num(tmpX / sMNear)
    tmpX = tmpX.reshape(nNearPart, nR, t_delta, nS).transpose([1,2,0,3]).reshape((nR, -1))
    tmpX = np.c_[tmpX[:,-nS*nNearPart:], tmpX[:,:-nS*nNearPart].reshape((nR,t_delta-1,nNearPart,nS))[:,:,0,:].reshape((nR,-1))]
    x_r = np.array(tmpX[:,0:nS])
    x_int = np.array(tmpX)
    
    train_r = np.r_[train_r, x_r][-nTrain:,:]
    train_int = np.r_[train_int, x_int][-nTrain:,:]

    if ts > detect_st:
        if ts % (60 // MPS * 24) == 0 or not trained:
            model_r.fit(train_r)
            model_int.fit(train_int)
            trained = True
        
        score_r[ts,:] = model_r.decision_function(x_r).flatten()
        score_int[ts,:] = model_int.decision_function(x_int).flatten()
        argsort_r = score_r[(ts-60*24//MPS+1):(ts+1),:].flatten().argsort()
        argsort_int = score_int[(ts-60*24//MPS+1):(ts+1),:].flatten().argsort()
        
        selected_int = argsort_int[np.where(np.in1d(argsort_int, argsort_r[0:nDailyAnomaly_r]))[0]][0:nDailyAnomaly_int]
        iAnomalies = selected_int[(selected_int // nR) == (60 * 24 // MPS - 1)] % nR
        iAnomalies = iAnomalies[score_int[ts,iAnomalies] != 100]
        anomalies[ts,iAnomalies] = 1

np.savetxt(dataRoot + "anomalies.txt", anomalies) # detected anomalies