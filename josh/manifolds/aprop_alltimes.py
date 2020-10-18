import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from scipy.stats import zscore
sns.set()
import sys
import scipy.io as sio
from ap_visualizations import sorted_cluster_vis,scatter_cluster_vis
from ap_utils import downsample_fr,downsample_decVar

# load in .mat file
data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/dataStructs/m80FR_decVar.mat')
FR_decVar = data['FR_decVar'][0]
timesinceSort_data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/dataStructs/m80timesinceSort.mat')
timesinceSort = timesinceSort_data['index_sort_all'][0]
tbin_ms = .02 *1000

# choose the session
sIdx = int(sys.argv[1]) # in terms of which day of 3/17
sort_sIdx = timesinceSort[sIdx] - 1
fr_mat = FR_decVar[sIdx]['fr_mat'][0]
fr_mat_full = np.squeeze(np.concatenate(fr_mat,1)[sort_sIdx,:]) # sort by time

# get some trial information and visualize a single trial
t_lens = [fr_mat_iTrial.shape[1] for fr_mat_iTrial in fr_mat.tolist()]
patchleave_ix = np.cumsum(t_lens)
patchstop_ix = patchleave_ix - t_lens

# iterate over trials, perform reduction step, concatenate into new matrix where we will perform AP
avg_window_ms = 250
avg_window = int(avg_window_ms / tbin_ms)
reduced_fr_mat = downsample_fr(fr_mat,t_lens,avg_window,sort_sIdx)
timeSince = FR_decVar['decVarTimeSinceRew']
# get new indexing to look at dynamics
reduced_timeSince_full,reduced_timeSince = downsample_decVar(timeSince[sIdx],t_lens,avg_window_ms,tbin_ms)

# Perform affinity propagation!
S = -squareform(pdist(reduced_fr_mat.T,'sqeuclidean')) # distance metric
pref_min = int(round(np.min(S)))
pref_med = int(round(np.median(S)))
print("Performing Affinity Propagation with min(S) as shared preference")
AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min,verbose = True)
labels = AP_clustering.fit_predict(S)
clusteroid_ix = AP_clustering.cluster_centers_indices_

# Visualize cluster separability by timeSince
timepoints = np.arange(0,2,.25)
peak_sort,sorted_labels = sorted_cluster_vis(labels,reduced_timeSince_full,timepoints)

# find average activity within each cluster
avg_cluster_activity = np.zeros((reduced_fr_mat.shape[0],len(clusteroid_ix)-1))
for i in peak_sort:
    avg_cluster_activity[:,int(i)] = np.mean(reduced_fr_mat[:,np.where(sorted_labels == i)[0]],axis = 1)

# check what the clusteroids look like, ordered by max density time
scatter_cluster_vis(clusteroid_ix,peak_sort,avg_cluster_activity)

# Now, for our final trick, look at the cross correlation between clusters sorted by peak time
plt.figure()
plt.title("Correlation between Peak-Sorted Clusteroids")
sns.heatmap(np.corrcoef(avg_cluster_activity.T));
plt.show()
