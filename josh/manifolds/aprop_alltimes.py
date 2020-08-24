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
sIdx = 2 # 3/17
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

# Perform affinity propagation!
S = -squareform(pdist(reduced_fr_mat.T,'sqeuclidean')) # distance metric
pref_min = int(round(np.min(S)))
pref_med = int(round(np.median(S)))
print("Performing Affinity Propagation with min(S) as shared preference")
AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min,verbose = True)
labels = AP_clustering.fit_predict(S)
clusteroid_ix = AP_clustering.cluster_centers_indices_

# Visualize cluster separability by timeSince
bins = np.unique(labels)
timepoints = np.arange(0,2,.25)
cluster_distns = np.zeros((len(bins)-1,len(timepoints)))
for i,t in enumerate(timepoints):
    cluster_distns[:,i] = np.histogram(labels[np.where(reduced_timeSince_full == t)[0]],bins = bins, density = True)[0]
peak_sort = np.argsort(np.argmax(cluster_distns,axis = 1))

# maybe separate visualization into reward?
plt.figure()
sns.heatmap(cluster_distns[peak_sort,:])
plt.xticks(range(len(timepoints)),timepoints);
plt.xlabel("Time")
plt.ylabel("Cluster")
plt.title("Sorted cluster probability density over time since reward")
sorted_labels = np.zeros(labels.shape)
for i,c in enumerate(peak_sort):
    label_ix = np.where(labels == c)[0]
    sorted_labels[label_ix] = i
plt.figure()
plt.title("Distribution of sorted cluster frequency")
sns.distplot(sorted_labels,kde = False,norm_hist = True)

# check what the clusteroids look like, ordered by max density time
fig, axes = plt.subplots(int(np.ceil(len(clusteroid_ix[:20]) / 5)),5, sharey='row', sharex='col', constrained_layout=True,figsize=(8, 8))
for i,c_idx in enumerate(clusteroid_ix[peak_sort[:20]]):
    row = int(np.floor(i / 5))
    col = i % 5
    axes[row,col].scatter(list(range(reduced_fr_mat.shape[0])),reduced_fr_mat[:,c_idx],marker = '.',s = 5)
    axes[row,col].set_xticks([])
#     axes[row,col].set_yticks([])
    axes[row,col].set_title("Cluster %i"%peak_sort[i])
    if col == 0:
        axes[row,col].set_ylabel("Z-Scored Activity")
    if row == 3:
        axes[row,col].set_xlabel("Sorted Neurons")

# Now, for our final trick, look at the cross correlation between clusters sorted by peak time
clusteroids = np.zeros((reduced_fr_mat.shape[0],len(clusteroid_ix)-1))
for i,c_idx in enumerate(clusteroid_ix[peak_sort]):
    clusteroids[:,i] = reduced_fr_mat[:,c_idx]
plt.figure()
plt.title("Correlation between Peak-Sorted Clusteroids")
sns.heatmap(np.corrcoef(clusteroids.T));
plt.show()
