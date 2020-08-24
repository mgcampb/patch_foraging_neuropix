import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import load_digits
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from scipy.stats import zscore
from skimage.measure import block_reduce
import networkx as nx
sns.set()
import sys
import scipy.io as sio

# load in .mat file
data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/neuroPixelsData/80/processed_data/m80FR_decVar.mat')
FR_decVar = data['FR_decVar'][0]
timeSort_data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/neuroPixelsData/80/processed_data/m80timeSort.mat')
timeSort = timeSort_data['index_sort_all'][0]
timesinceSort_data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/neuroPixelsData/80/processed_data/m80timesinceSort.mat')
timesinceSort = timesinceSort_data['index_sort_all'][0]
tbin_ms = .02 *1000

# choose the session
sIdx = int(sys.argv[1]) # 3/17
sort_sIdx = timesinceSort[sIdx] - 1
fr_mat = FR_decVar[sIdx]['fr_mat'][0]
fr_mat_full = np.squeeze(np.concatenate(fr_mat,1)[sort_sIdx,:]) # sort by time

# get some trial information and visualize a single trial
nTrials = fr_mat.shape[0]
t_lens = [fr_mat_iTrial.shape[1] for fr_mat_iTrial in fr_mat.tolist()]
prts = np.array(t_lens)  * tbin_ms / 1000
patchleave_ix = np.cumsum(t_lens)
patchstop_ix = patchleave_ix - t_lens

# iterate over trials, perform reduction step, concatenate into new matrix where we will perform AP
avg_window_ms = 250
avg_window = int(250 / tbin_ms)
reduced_fr_mat = []
lens = []
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
    lens.append(fr_mat_iTrial_cutdown.shape[1])
reduced_fr_mat = np.concatenate(reduced_fr_mat,1)
reduced_fr_mat[np.isnan(reduced_fr_mat)] = 0

# get new indexing to look at dynamics
t_lens_reduced = np.floor(np.array(t_lens)/avg_window).astype(int)
leave_ix_reduced = np.cumsum(t_lens_reduced)
stop_ix_reduced = leave_ix_reduced - t_lens_reduced

# make down-sampled decision variables
reduced_time = []
reduced_timeSince = []
timeSince = FR_decVar['decVarTimeSinceRew']
for trial in range(nTrials):
    timeSince_iTrial = timeSince[sIdx][0][trial].T
    trial_rews = np.floor((np.where(timeSince_iTrial == .02)[0][1:] + 1)* tbin_ms / avg_window_ms).astype(int)

    timeSince_iTrial_red = np.arange(0,t_lens_reduced[trial]) / (1000 / avg_window_ms)
    time_iTrial_red = np.arange(0,t_lens_reduced[trial]) / (1000 / avg_window_ms)
    for r in trial_rews:
        if r < t_lens_reduced[trial]-2:
            timeSince_iTrial_red[r:] = np.arange(0,(t_lens_reduced[trial]-r)) / (1000 / avg_window_ms)

    reduced_timeSince.append(timeSince_iTrial_red)
    reduced_time.append(time_iTrial_red)
reduced_timeSince_full = np.concatenate(reduced_timeSince)
reduced_time_full = np.concatenate(reduced_time)

# make a reward size vector w/ the length of timeSinceReduced
rewsize = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/neuroPixelsData/80/rewsize80317.mat')['rewsize'].T[0]
rewsize_concatTrials = []
for i,r in enumerate(rewsize.tolist()):
    rewsize_concatTrials.append([r for j in range(t_lens_reduced[i])])
rewsize_concatTrials = np.array([item for sublist in rewsize_concatTrials for item in sublist])
rewsize_concatTrials[np.where(rewsize_concatTrials == 4)[0]] = 3 # change to 3 to decr bias

# Perform AP within eras
timepoints = np.arange(0,3,.25)
print("Performing affinity clustering within eras for 0-3 sec since reward")
clusteroids = []
labels = -1 * np.ones(reduced_fr_mat.shape[1])
for i,t in enumerate(timepoints):
    era_fr_mat = reduced_fr_mat[40:140,np.where(reduced_timeSince_full == t)[0]]
    S = -squareform(pdist(era_fr_mat.T,'sqeuclidean')) # distance metric
    pref_min = np.min(S)
    pref_med = np.median(S)
    AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min)
    era_labels = AP_clustering.fit_predict(S)
    labels[np.where(reduced_timeSince_full == t)[0]] = era_labels + i * 10 # note this won't work if we have > 10 clusters
    clusteroid_ix = AP_clustering.cluster_centers_indices_
    clusteroids.append(era_fr_mat[:,clusteroid_ix])


reduced_labels = -1 * np.ones(len(labels)) # reduced labels is labels from 0:# unique labels, no era distinctions
for i,l in enumerate(np.unique(labels[np.where(labels >= 0)[0]])):
    reduced_labels[np.where(labels == l)[0]] = i
n_unique_labels = len(np.unique(reduced_labels))

# reward size per cluster
mean_rewsizes = np.zeros(n_unique_labels)
for l in np.unique(reduced_labels):
    if l > 0:
        mean_rewsizes[int(l)] = np.mean(rewsize_concatTrials[np.where(reduced_labels == l)[0]])

concat_clusteroids = np.concatenate(clusteroids,axis= 1)
plt.figure()
plt.title("Correlation between Clusteroids within Timepoint")
sns.heatmap(np.corrcoef(concat_clusteroids.T))

# Scatter plot activity clusteroids acr some indices
for e_idx in range(3): # range(len(timepoints)):
    fig, axes = plt.subplots(2,2, sharey='row', sharex='col', constrained_layout=True,figsize=(5,5))
    for c_idx in range(min(4,clusteroids[e_idx].shape[1])):
        row = int(np.floor(c_idx / 2))
        col = c_idx % 2
        axes[row,col].scatter(list(range(100)),clusteroids[e_idx][:,c_idx],marker = '.',s = 5)
#         axes[row,col].set_xticks()
    #     axes[row,col].set_yticks([])
        axes[row,col].set_title("Cluster %i"%c_idx)
        if col == 0:
            axes[row,col].set_ylabel("Z-Scored Activity")
        if row == 3:
            axes[row,col].set_xlabel("Sorted Neurons")
    plt.suptitle("{0:.2f} sec Since Rew Sample Clusteroids".format(timepoints[e_idx]))

# construct graph of transitions, excluding transitions going over reward deliveries
T = np.zeros((n_unique_labels,n_unique_labels)) # "T[i,j] indicates transition from  j to i"
leave_count = np.zeros(n_unique_labels)
cluster_count = np.zeros(n_unique_labels)
for i, l in enumerate(reduced_labels[:-2]):
    if (l >= 0) & (reduced_labels[i+1] >= 0) & (reduced_timeSince_full[i+1] > 0): # we aren't going to get a reward
        T[int(reduced_labels[i+1]),int(l)] += 1 # add one to the counts
    if (reduced_time_full[i+1] == 0) | (reduced_time_full[i+2] == 0):
        leave_count[int(l)] += 1
    cluster_count[int(l)] += 1

sum_T = np.sum(T,axis = 1)
T[np.where(sum_T > 0)[0],:] = T[np.where(sum_T > 0)[0],:] / sum_T[np.where(sum_T > 0)[0]][:,np.newaxis]
p_leave = leave_count / cluster_count
p_visit = cluster_count / np.sum(cluster_count)

# visualize unshuffled graph!
n_eraClusts = [c.shape[1] for c in clusteroids]
n_eraClusts.insert(0,0)
cumulativeClusts = np.cumsum(n_eraClusts)
plt.figure(figsize = (10,6))
for i in range(len(n_eraClusts)-3):
    sizes = 5000 * p_visit[cumulativeClusts[i]:cumulativeClusts[i+1]]
#     plt.figure()
#     sns.heatmap(T[:,cumulativeClusts[i]:cumulativeClusts[i+1]])
    if i < len(n_eraClusts)-4:
        for j in range(cumulativeClusts[i],cumulativeClusts[i+1]):
            for k in np.where(T[:,j] > 0)[0]:
                # plt.figure(1)
                lw = 5 * T[k,j] # linewidth is un-normalized
                c = [(T[k,j] / np.max(T[:,j])),0,1 - (T[k,j] / np.max(T[:,j]))] # color is normalized
                # figure out how to deal with the p_leave of the next point
    #             print(k,j,lw)
                plt.plot([i,i+1],[p_leave[j],p_leave[k]],linewidth = lw,color = c) # plot line with transition
    # plt.figure(1)
    colors = [[1 - m / 3,1 - m / 3,1 - m / 3] for m in mean_rewsizes[cumulativeClusts[i]:cumulativeClusts[i+1]] ]
    for p,c,s in zip(p_leave[cumulativeClusts[i]:cumulativeClusts[i+1]],colors,sizes):
        plt.scatter(i,p,s = s,color = c)
plt.ylabel("P(Leave within .5 sec | Cluster)")
plt.xlabel("Time since reward")
plt.xticks(list(range(len(n_eraClusts)-3)),timepoints[:-3])
plt.title("Graph of Cluster Transitions")

# Now perform shuffle control within era
timepoints = np.arange(0,3,.25)
clusteroids = []
labels = -1 * np.ones(reduced_fr_mat.shape[1])
for i,t in enumerate(timepoints):
    era_fr_mat = reduced_fr_mat[40:140,np.where(reduced_timeSince_full == t)[0]]
    S = -squareform(pdist(era_fr_mat.T,'sqeuclidean')) # distance metric
    pref_min = np.min(S)
    pref_med = np.median(S)
    AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min)
    era_labels = AP_clustering.fit_predict(S)
    np.random.shuffle(era_labels) # shuffle labels within era
    labels[np.where(reduced_timeSince_full == t)[0]] = era_labels + i * 10 # note this won't work if we have > 10 clusters
    clusteroid_ix = AP_clustering.cluster_centers_indices_
    clusteroids.append(era_fr_mat[:,clusteroid_ix])

reduced_labels = -1 * np.ones(len(labels)) # reduced labels is labels from 0:# unique labels, no era distinctions
for i,l in enumerate(np.unique(labels[np.where(labels >= 0)[0]])):
    reduced_labels[np.where(labels == l)[0]] = i

n_unique_labels = len(np.unique(reduced_labels))
T = np.zeros((n_unique_labels,n_unique_labels)) # "T[i,j] indicates transition from  j to i"
leave_count = np.zeros(n_unique_labels)
cluster_count = np.zeros(n_unique_labels)
for i, l in enumerate(reduced_labels[:-2]):
    if (l >= 0) & (reduced_labels[i+1] >= 0) & (reduced_timeSince_full[i+1] > 0): # we aren't going to get a reward
        T[int(reduced_labels[i+1]),int(l)] += 1 # add one to the counts
    if (reduced_time_full[i+1] == 0) | (reduced_time_full[i+2] == 0):
        leave_count[int(l)] += 1
    cluster_count[int(l)] += 1

sum_T = np.sum(T,axis = 1)
T[np.where(sum_T > 0)[0],:] = T[np.where(sum_T > 0)[0],:] / sum_T[np.where(sum_T > 0)[0]][:,np.newaxis]
p_leave = leave_count / cluster_count
p_visit = cluster_count / np.sum(cluster_count)

# visualize fake graph
n_eraClusts = [c.shape[1] for c in clusteroids]
n_eraClusts.insert(0,0)
cumulativeClusts = np.cumsum(n_eraClusts)
plt.figure(figsize = (10,6))
for i in range(len(n_eraClusts)-3):
    sizes = 5000 * p_visit[cumulativeClusts[i]:cumulativeClusts[i+1]]
#     plt.figure()
#     sns.heatmap(T[:,cumulativeClusts[i]:cumulativeClusts[i+1]])
    if i < len(n_eraClusts)-4:
        for j in range(cumulativeClusts[i],cumulativeClusts[i+1]):
            for k in np.where(T[:,j] > 0)[0]:
                lw = 5 * T[k,j] # linewidth is un-normalized
                c = [(T[k,j] / np.max(T[:,j])),0,1 - (T[k,j] / np.max(T[:,j]))] # color is normalized
                # figure out how to deal with the p_leave of the next point
    #             print(k,j,lw)
                plt.plot([i,i+1],[p_leave[j],p_leave[k]],linewidth = lw,color = c) # plot line with transition
    colors = [[1 - m / 3,1 - m / 3,1 - m / 3] for m in mean_rewsizes[cumulativeClusts[i]:cumulativeClusts[i+1]] ]
    for p,c,s in zip(p_leave[cumulativeClusts[i]:cumulativeClusts[i+1]],colors,sizes):
        plt.scatter(i,p,s = s,color = c)
plt.ylabel("P(Leave within .5 sec | Cluster)")
plt.xlabel("Time since reward")
plt.xticks(list(range(len(n_eraClusts)-3)),timepoints[:-3])
plt.title("Graph of Shuffled Cluster Transitions")

decoding = True
if decoding == True:
    print("Performing cluster decoding with leave 1 out xval")


plt.show()
