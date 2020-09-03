import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore

def downsample_fr(fr_mat,t_lens,avg_window,sort_sIdx):
    reduced_fr_mat = []
    nTrials = len(t_lens)
    for iTrial in range(nTrials):
        fr_mat_iTrial = np.squeeze(fr_mat[iTrial][sort_sIdx,:])
        fr_mat_iTrial = zscore(fr_mat_iTrial,1)
        fr_mat_iTrial[np.isnan(fr_mat_iTrial)] = 0
        if (t_lens[iTrial] % avg_window) > 0:
            fr_mat_iTrial_cutdown = fr_mat_iTrial[:,:-(t_lens[iTrial] % avg_window)]
        else:
            fr_mat_iTrial_cutdown = fr_mat_iTrial
        fr_mat_iTrial_cutdown = np.mean(fr_mat_iTrial_cutdown.reshape(fr_mat_iTrial_cutdown.shape[0],-1, avg_window), axis=2)
        reduced_fr_mat.append(fr_mat_iTrial_cutdown)
    reduced_fr_mat = np.concatenate(reduced_fr_mat,1)
    reduced_fr_mat[np.isnan(reduced_fr_mat)] = 0
    return reduced_fr_mat

def downsample_decVar(decVar,t_lens,avg_window_ms,tbin_ms):
    avg_window = int(250 / tbin_ms)
    nTrials = len(t_lens)

    # get new indexing to look at dynamics
    t_lens_reduced = np.floor(np.array(t_lens)/avg_window).astype(int)
    leave_ix_reduced = np.cumsum(t_lens_reduced)
    stop_ix_reduced = leave_ix_reduced - t_lens_reduced

    reduced_decVar = []
    for trial in range(nTrials):
        decVar_iTrial = decVar[0][trial].T
        trial_rews = np.floor((np.where(decVar_iTrial == .02)[0][1:] + 1)* tbin_ms / avg_window_ms).astype(int)
        decVar_iTrial_red = np.arange(0,t_lens_reduced[trial]) / (1000 / avg_window_ms)
        for r in trial_rews:
            if r < t_lens_reduced[trial]-2:
                decVar_iTrial_red[r:] = np.arange(0,(t_lens_reduced[trial]-r)) / (1000 / avg_window_ms)
        reduced_decVar.append(decVar_iTrial_red)
    reduced_decVar_full = np.concatenate(reduced_decVar)
    return reduced_decVar_full,reduced_decVar

def rewsize_stream(rewsize,t_lens_reduced):
    rewsize_concatTrials = []
    for i,r in enumerate(rewsize.tolist()):
        rewsize_concatTrials.append([r for j in range(t_lens_reduced[i])])
    rewsize_concatTrials = np.array([item for sublist in rewsize_concatTrials for item in sublist])
    rewsize_concatTrials[np.where(rewsize_concatTrials == 4)[0]] = 3 # change to 3 to decr bias
    return rewsize_concatTrials

def create_transition_graph(labels,reduced_time_full,reduced_timeSince_full):
    n_unique_labels = len(np.unique(labels))
    # construct graph of transitions, excluding transitions going over reward deliveries
    T = np.zeros((n_unique_labels,n_unique_labels)) # "T[i,j] indicates transition from  j to i"
    leave_count = np.zeros(n_unique_labels)
    cluster_count = np.zeros(n_unique_labels)
    for i, l in enumerate(labels[:-2]):
        if (l >= 0) & (labels[i+1] >= 0) & (reduced_timeSince_full[i+1] > 0): # we aren't going to get a reward
            T[int(labels[i+1]),int(l)] += 1 # add one to the counts
        if (reduced_time_full[i+1] == 0) | (reduced_time_full[i+2] == 0):
            leave_count[int(l)] += 1
        cluster_count[int(l)] += 1

    sum_T = np.sum(T,axis = 1)
    T[np.where(sum_T > 0)[0],:] = T[np.where(sum_T > 0)[0],:] / sum_T[np.where(sum_T > 0)[0]][:,np.newaxis]
    p_leave = leave_count / cluster_count
    p_visit = cluster_count / np.sum(cluster_count)
    return T,p_leave,p_visit
