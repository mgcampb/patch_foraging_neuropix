import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore,mode

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
    n_unique_labels = len(np.unique(labels[np.where(labels >= 0)[0]]))
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

def reduce_labels(labels):
    """
        A simple function to make labels range from 0-n_unique_labels
    """
    reduced_labels = -1 * np.ones(len(labels))
    for i,l in enumerate(np.unique(labels[np.where(labels >= 0)[0]])):
        reduced_labels[np.where(labels == l)[0]] = i
    return reduced_labels

def shuffle_within_eras(labels,cumulativeClusts):
    clusters = np.unique(labels[labels >= 0])
    era_clusters = [clusters[cumulativeClusts[i]:cumulativeClusts[i+1]] for i in range(len(cumulativeClusts)-1)]
    shuffled_labels = np.full(labels.shape,-1)
    for era in range(len(cumulativeClusts)-1):
        era_locs = np.where(np.isin(labels,era_clusters[era]))
        era_labels = np.copy(labels[era_locs])
        np.random.shuffle(era_labels) # shuffle
        shuffled_labels[era_locs] = era_labels
    return shuffled_labels

def leave1out_decoding(prediction_steps,reduced_labels,non_rew_locs):
    step_accs = np.zeros(len(prediction_steps))
    for iStep,pred_step in enumerate(prediction_steps):
        prediction_truths = np.empty(max(prediction_steps) * len(non_rew_locs)) # being generous here
        prediction_truths[:] = np.nan
        counter = 0
        for cluster in np.unique(reduced_labels[reduced_labels >= 0]):
            cluster_locs = np.where(reduced_labels == cluster)[0]
            for LO_loc in cluster_locs: # iterate over locations where cluster was visited
                train_locs = np.setdiff1d(cluster_locs,LO_loc,assume_unique = True) # training set as all other pts
                if 0 <= LO_loc + pred_step < len(reduced_labels):
                    pred_locs = train_locs + pred_step # look forward or back
                    pred_locs = pred_locs[(pred_locs >= 0) & (pred_locs < len(reduced_labels))] # get rid of invalid pts
                    pred_locs = np.intersect1d(pred_locs,non_rew_locs,assume_unique = True) # don't look at rwd pts
                    predicted_label = mode(reduced_labels[pred_locs])[0] # predict the most common transition
                    true_label = reduced_labels[LO_loc + pred_step] # L1O test truth
                    if len(predicted_label) > 0 and predicted_label > 0:
                        prediction_truths[counter] = int(true_label == predicted_label) # assign
                        counter += 1

        step_accs[iStep] = np.nanmean(prediction_truths)
    return step_accs
