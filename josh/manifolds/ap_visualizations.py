import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def sorted_cluster_vis(labels,sort_variable,timepoints):
    bins = np.unique(labels)
    timepoints = np.arange(0,2,.25)
    cluster_distns = np.zeros((len(bins)-1,len(timepoints)))
    for i,t in enumerate(timepoints):
        cluster_distns[:,i] = np.histogram(labels[np.where(sort_variable == t)[0]],bins = bins, density = True)[0]
    peak_sort = np.argsort(np.argmax(cluster_distns,axis = 1))
    # print(peak_sort)

    # maybe separate visualization into reward ?
    plt.figure()
    sns.heatmap(cluster_distns[peak_sort,:])
    plt.xticks(range(len(timepoints)),timepoints);
    plt.xlabel("Time")
    plt.ylabel("Cluster")
    plt.title("Sorted cluster probability density over time since reward")

    # sort the labels by peak location
    sorted_labels = np.zeros(labels.shape)
    for i,c in enumerate(peak_sort):
        label_ix = np.where(labels == c)[0]
        sorted_labels[label_ix] = i

    # visualize cluster visitation distribution
    plt.figure()
    plt.title("Distribution of sorted cluster frequency")
    sns.distplot(sorted_labels,kde = False,norm_hist = True)
    return peak_sort,sorted_labels

def scatter_cluster_vis(clusteroid_ix,peak_sort,avg_cluster_activity):
    nNeurons = avg_cluster_activity.shape[0]
    fig, axes = plt.subplots(int(np.ceil(len(clusteroid_ix[:20]) / 5)),5, sharey='row', sharex='col', constrained_layout=True,figsize=(8, 8))
    for i,c_idx in enumerate(clusteroid_ix[peak_sort[:20]]):
        row = int(np.floor(i / 5))
        col = i % 5
        axes[row,col].scatter(list(range(nNeurons)),avg_cluster_activity[:,i],marker = '.',s = 5)
        axes[row,col].set_xticks([])
    #     axes[row,col].set_yticks([])
        axes[row,col].set_title("Cluster %i"%i)
        if col == 0:
            axes[row,col].set_ylabel("Z-Scored Activity")
        if row == 3:
            axes[row,col].set_xlabel("Sorted Neurons")

def transition_graph_vis(timepoints,clusteroids,mean_rewsizes,T,p_leave,p_visit,shuffled = False):
    # visualize unshuffled graph!
    n_eraClusts = [c.shape[1] for c in clusteroids]
    n_eraClusts.insert(0,0)
    cumulativeClusts = np.cumsum(n_eraClusts)
    plt.figure(figsize = (10,6))
    for i in range(len(n_eraClusts)-3):
        sizes = 5000 * p_visit[cumulativeClusts[i]:cumulativeClusts[i+1]]
        if i < len(n_eraClusts)-4:
            for j in range(cumulativeClusts[i],cumulativeClusts[i+1]):
                for k in np.where(T[:,j] > 0)[0]:
                    # plt.figure(1)
                    lw = 5 * T[k,j] # linewidth is un-normalized
                    c = [(T[k,j] / np.max(T[:,j])),0,1 - (T[k,j] / np.max(T[:,j]))] # color is normalized
                    # deal with the p_leave of the next point
                    plt.plot([i,i+1],[p_leave[j],p_leave[k]],linewidth = lw,color = c) # plot line with transition
        # plt.figure(1)
        colors = [[1 - m / 3,1 - m / 3,1 - m / 3] for m in mean_rewsizes[cumulativeClusts[i]:cumulativeClusts[i+1]] ]
        for p,c,s in zip(p_leave[cumulativeClusts[i]:cumulativeClusts[i+1]],colors,sizes):
            plt.scatter(i,p,s = s,color = c)
    plt.ylabel("P(Leave within .5 sec | Cluster)")
    plt.xlabel("Time since reward")
    plt.xticks(list(range(len(n_eraClusts)-3)),timepoints[:-3])
    if shuffled == False:
        plt.title("Graph of Cluster Transitions")
    else:
        plt.title("Graph of Shuffled Cluster Transitions")
