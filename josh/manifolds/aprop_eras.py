import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
import seaborn as sns
from scipy.spatial.distance import pdist,squareform
from scipy.stats import zscore
import sys
import scipy.io as sio
from ap_visualizations import sorted_cluster_vis,scatter_cluster_vis,transition_graph_vis
from ap_utils import downsample_fr,downsample_decVar,rewsize_stream,create_transition_graph,reduce_labels,shuffle_within_eras,leave1out_decoding
from progress.bar import IncrementalBar
sns.set()
np.seterr(divide='ignore',invalid='ignore') # turn off true_divide warnings

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
# handle neuron down-selection
neuron_sel_input = sys.argv[2]
if neuron_sel_input != 'all':
    neuron_sel_start = sys.argv[2]
    neuron_sel_end = sys.argv[3]
    if neuron_sel_start == ':':
        neuron_sel_end = int(neuron_sel_end)
        reduced_fr_mat = reduced_fr_mat[:neuron_sel_end,:]
    elif neuron_sel_end == ':':
        neuron_sel_start = int(neuron_sel_start)
        reduced_fr_mat = reduced_fr_mat[neuron_sel_start:,:]
    else:
        neuron_sel_start = int(neuron_sel_start)
        neuron_sel_end = int(neuron_sel_end)
        reduced_fr_mat = reduced_fr_mat[neuron_sel_start:neuron_sel_end,:]

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
for i,t in enumerate(timepoints):
    era_fr_mat = reduced_fr_mat[:,np.where(reduced_timeSince_full == t)[0]]
    S = -squareform(pdist(era_fr_mat.T,'sqeuclidean')) # distance metric
    pref_min = np.min(S)
    pref_med = np.median(S)
    AP_clustering = AffinityPropagation(affinity = 'precomputed',preference = pref_min)
    era_labels = AP_clustering.fit_predict(S)
    labels[np.where(reduced_timeSince_full == t)[0]] = era_labels + i * 10 # note this won't work if we have > 10 clusters
    clusteroid_ix = AP_clustering.cluster_centers_indices_
    clusteroids.append(era_fr_mat[:,clusteroid_ix])

# number of clusters per era
n_eraClusts = [c.shape[1] for c in clusteroids]
n_eraClusts.insert(0,0)
cumulativeClusts = np.cumsum(n_eraClusts)

# reduce labels s.t. we range from 0:nlabels and do a example shuffle
reduced_labels = reduce_labels(labels)
reduced_labels_shuffled = shuffle_within_eras(reduced_labels,cumulativeClusts)
n_unique_labels = len(np.unique(reduced_labels))

# avg reward size per cluster
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

# visualize shuffled graph
T_shuffled,p_leave_shuffled,p_visit_shuffled = create_transition_graph(reduced_labels_shuffled,reduced_time_full,reduced_timeSince_full)
transition_graph_vis(timepoints,clusteroids,mean_rewsizes_shuffled,T_shuffled,p_leave_shuffled,p_visit_shuffled,shuffled = True)

# get averaged activity w.r.t clusters within eras sorted by P(Leave)
# first get pLeave_label_sort
labels_pLeave_sort = np.zeros(p_leave.shape)
for i in range(len(cumulativeClusts)-1):
    era_p_leave = p_leave[cumulativeClusts[i]:cumulativeClusts[i+1]]
    era_pLeave_sort = np.argsort(era_p_leave) + cumulativeClusts[i]
    labels_pLeave_sort[cumulativeClusts[i]:cumulativeClusts[i+1]] = era_pLeave_sort

# now find avg activity within these within-era sorted clusters
nNeurons = reduced_fr_mat.shape[0]
avg_cluster_activity = np.zeros((nNeurons,len(labels_pLeave_sort)-1))
for i in labels_pLeave_sort[:-2]:
    avg_cluster_activity[:,int(i)] = np.mean(reduced_fr_mat[:,np.where(reduced_labels == i)[0]],axis = 1)
# turn this into a list of lists for niceness
avg_cluster_activity = [avg_cluster_activity[:,cumulativeClusts[i]:cumulativeClusts[i+1]] for i in range(len(cumulativeClusts)-1)]

# visualize avg activity using scatterplot
for e_idx in range(7): # through 1.5 seconds
    fig, axes = plt.subplots(max(2,int(np.ceil(avg_cluster_activity[e_idx].shape[1] / 4))),4, sharey='row', sharex='col', constrained_layout=True,figsize=(5,5))
    for c_idx in range(avg_cluster_activity[e_idx].shape[1]):
        row = int(np.floor(c_idx / 4))
        col = c_idx % 4
        axes[row,col].scatter(list(range(nNeurons)),avg_cluster_activity[e_idx][:,c_idx],marker = '.',s = 5)
        axes[row,col].set_title("Cluster %i"%c_idx)
        if col == 0:
            axes[row,col].set_ylabel("Z-Scored Activity")
        if row == 3:
            axes[row,col].set_xlabel("Sorted Neurons")
    plt.suptitle("{0:.2f} sec Since Rew Sample Clusteroids".format(timepoints[e_idx]))

# replace this with average activity within cluster
plt.figure()
plt.title("Correlation between Avg Activity within Timepoint")
concat_avgs = np.concatenate(avg_cluster_activity,axis= 1)

### add lines to indicate beginning of eras ###
sns.heatmap(np.corrcoef(concat_avgs.T))

decoding = True
if decoding == True:
    print("Performing leave-1-out decoding")
    max_step = 6
    prediction_steps = np.setdiff1d(np.arange(-max_step,max_step+1),0)
    non_rew_locs = np.where(np.array(reduced_timeSince_full) > 0)[0]

    true_step_accs = leave1out_decoding(prediction_steps,reduced_labels,non_rew_locs)

    # shuffle many times, record results in pd dataframe, use sns.lineplot to show errorbars
    n_shuffles = 100
    shuffled_decoding_df = pd.DataFrame(np.nan,
                                        index = np.arange(n_shuffles),
                                        columns = prediction_steps,
                                        dtype = np.float32)

    bar = IncrementalBar('Shuffled Decoding', max = n_shuffles)
    for i in range(n_shuffles):
        shuffled_labels = shuffle_within_eras(reduced_labels,cumulativeClusts)
        shuffled_step_accs = leave1out_decoding(prediction_steps,shuffled_labels,non_rew_locs)
        shuffled_decoding_df.iloc[i,:] = shuffled_step_accs
        bar.next()
    bar.finish()

    # melt the dataframe for visualization
    shuffled_decoding_df = shuffled_decoding_df.melt(var_name = "prediction_step",value_name = "accuracy")
    shuffled_decoding_df.to_pickle("./shuffled_decoding_df.pkl")

    # shuffled_decoding_df = pd.read_pickle("./shuffled_decoding_df.pkl")

    # now visualize compared to shuffle
    half = round(len(prediction_steps) / 2)
    plt.figure()
    plt.plot(prediction_steps[:half],true_step_accs[:half],linewidth = 2,color = 'r')
    plt.plot(prediction_steps[half:],true_step_accs[half:],linewidth = 2,color = 'r')
    plt.scatter(prediction_steps,true_step_accs,color = 'r')
    sns.lineplot(data = shuffled_decoding_df[shuffled_decoding_df["prediction_step"] < 0],x="prediction_step", y="accuracy",color = 'k')
    sns.lineplot(data = shuffled_decoding_df[shuffled_decoding_df["prediction_step"] > 0],x="prediction_step", y="accuracy",color = 'k')
#     plt.scatter(np.random.rand(10),.4 + .4 * np.random.rand(10),color = 'r') # volcano mode
    plt.xlabel("Prediction steps")
    plt.ylabel("Cluster prediction accuracy")
    plt.title("L1O Cluster Prediction Fidelity")

plt.show()
