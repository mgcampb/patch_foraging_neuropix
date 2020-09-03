import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AffinityPropagation
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from scipy.stats import zscore
sns.set()
import sys
import scipy.io as sio
from ap_visualizations import sorted_cluster_vis,scatter_cluster_vis,transition_graph_vis
from ap_utils import downsample_fr,downsample_decVar,rewsize_stream,create_transition_graph

# load in .mat file
data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/dataStructs/m80FR_decVar.mat')
FR_decVar = data['FR_decVar'][0]
timesinceSort_data = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/dataStructs/m80timesinceSort.mat')
timesinceSort = timesinceSort_data['index_sort_all'][0]
tbin_ms = .02 * 1000

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
avg_window = int(avg_window_ms / tbin_ms)
reduced_fr_mat = downsample_fr(fr_mat,t_lens,avg_window,sort_sIdx)
timeSince = FR_decVar['decVarTimeSinceRew']
rawTime = FR_decVar['decVarTime']
# get new indexing to look at dynamics
reduced_timeSince_full,reduced_timeSince = downsample_decVar(timeSince[sIdx],t_lens,avg_window_ms,tbin_ms)
reduced_time_full,reduced_time = downsample_decVar(rawTime[sIdx],t_lens,avg_window_ms,tbin_ms)

# get new indexing to look at dynamics
t_lens_reduced = np.floor(np.array(t_lens)/avg_window).astype(int)
leave_ix_reduced = np.cumsum(t_lens_reduced)
stop_ix_reduced = leave_ix_reduced - t_lens_reduced

# make a reward size vector w/ the length of timeSinceReduced
rewsize = sio.loadmat('/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/dataStructs/m80rewSizes')['rewsizes_across'][0][sIdx]
rewsize_concatTrials = rewsize_stream(rewsize,t_lens_reduced)

# Perform AP within eras
timepoints = np.arange(0,3,.25)
print("Performing affinity clustering within eras for 0-3 sec since reward")
clusteroids = []
labels = -1 * np.ones(reduced_fr_mat.shape[1])
shuffled_labels = -1 * np.ones(reduced_fr_mat.shape[1])
for i,t in enumerate(timepoints):
    era_fr_mat = reduced_fr_mat[40:140,np.where(reduced_timeSince_full == t)[0]]
    S = -squareform(pdist(era_fr_mat.T,'sqeuclidean')) # distance metric
    pref_min = np.min(S)
    pref_med = np.median(S)
    AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min)
    era_labels = AP_clustering.fit_predict(S)
    era_labels_shuffled = np.copy(era_labels)
    np.random.shuffle(era_labels_shuffled) # shuffle era
    labels[np.where(reduced_timeSince_full == t)[0]] = era_labels + i * 10 # note this won't work if we have > 10 clusters
    shuffled_labels[np.where(reduced_timeSince_full == t)[0]] = era_labels_shuffled + i * 10
    clusteroid_ix = AP_clustering.cluster_centers_indices_
    clusteroids.append(era_fr_mat[:,clusteroid_ix])

reduced_labels = -1 * np.ones(len(labels)) # reduced labels is labels from 0:# unique labels, no era distinctions
reduced_labels_shuffled = -1 * np.ones(len(labels))
for i,l in enumerate(np.unique(labels[np.where(labels >= 0)[0]])):
    reduced_labels[np.where(labels == l)[0]] = i
    reduced_labels_shuffled[np.where(shuffled_labels == l)[0]] = i
n_unique_labels = len(np.unique(reduced_labels))

# reward size per cluster
mean_rewsizes = np.zeros(n_unique_labels)
mean_rewsizes_shuffled = np.zeros(n_unique_labels)
for l in np.unique(reduced_labels[reduced_labels >= 0]):
    if l > 0:
        mean_rewsizes[int(l)] = np.mean(rewsize_concatTrials[np.where(reduced_labels == l)[0]])
        mean_rewsizes_shuffled[int(l)] = np.mean(rewsize_concatTrials[np.where(reduced_labels_shuffled == l)[0]])
concat_clusteroids = np.concatenate(clusteroids,axis= 1)

print("Number of clusters across eras: ",[c.shape[1] for c in clusteroids])

# visualize unshuffled graph
T,p_leave,p_visit = create_transition_graph(reduced_labels,reduced_time_full,reduced_timeSince_full)
transition_graph_vis(timepoints,clusteroids,mean_rewsizes,T,p_leave,p_visit)

# visaulize shuffled graph
T_shuffled,p_leave_shuffled,p_visit_shuffled = create_transition_graph(reduced_labels_shuffled,reduced_time_full,reduced_timeSince_full)
transition_graph_vis(timepoints,clusteroids,mean_rewsizes_shuffled,T_shuffled,p_leave_shuffled,p_visit_shuffled,shuffled = True)

# get averaged activity w.r.t clusters within eras sorted by P(Leave)
# first get pLeave_label_sort
labels_pLeave_sort = np.zeros(p_leave.shape)
n_eraClusts = [c.shape[1] for c in clusteroids]
n_eraClusts.insert(0,0)
cumulativeClusts = np.cumsum(n_eraClusts)
for i in range(len(cumulativeClusts)-1):
    era_p_leave = p_leave[cumulativeClusts[i]:cumulativeClusts[i+1]]
    era_pLeave_sort = np.argsort(era_p_leave) + cumulativeClusts[i]
    labels_pLeave_sort[cumulativeClusts[i]:cumulativeClusts[i+1]] = era_pLeave_sort

# now find avg activity within these within-era sorted clusters
nNeurons = reduced_fr_mat.shape[0]
avg_cluster_activity = np.zeros((nNeurons,len(labels_pLeave_sort)-1))
for i in labels_pLeave_sort[:-1]:
    avg_cluster_activity[:,int(i)] = np.mean(reduced_fr_mat[:,np.where(reduced_labels == i)[0]],axis = 1)
# turn this into a list of lists for niceness
avg_cluster_activity = [avg_cluster_activity[:,cumulativeClusts[i]:cumulativeClusts[i+1]] for i in range(len(cumulativeClusts)-1)]
print([activity.shape for activity in avg_cluster_activity])

# Visualize activity... change this to order within cluster by p_leave
for e_idx in range(3):
    fig, axes = plt.subplots(2,2, sharey='row', sharex='col', constrained_layout=True,figsize=(5,5))
    for c_idx in range(min(4,clusteroids[e_idx].shape[1])):
        row = int(np.floor(c_idx / 2))
        col = c_idx % 2
        axes[row,col].scatter(list(range(100)),clusteroids[e_idx][:,c_idx],marker = '.',s = 5)
        axes[row,col].set_title("Cluster %i"%c_idx)
        if col == 0:
            axes[row,col].set_ylabel("Z-Scored Activity")
        if row == 3:
            axes[row,col].set_xlabel("Sorted Neurons")
    plt.suptitle("{0:.2f} sec Since Rew Sample Clusteroids".format(timepoints[e_idx]))

# replace this with average activity within cluster
plt.figure()
plt.title("Correlation between Clusteroids within Timepoint")
sns.heatmap(np.corrcoef(concat_clusteroids.T))

decoding = True
if decoding == True:
    print("Performing cluster decoding with leave 1 out xval \n")
    print("(TODO)")

plt.show()
