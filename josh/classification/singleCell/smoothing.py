import numpy as np
from scipy.ndimage import gaussian_filter

def calcFRVsTime(cell_id,st,clu,opt = None):
    """
        Function to convert cell spike times into smoothed signal

        Arguments:
            cell_id: numpy array of cell ids to extract
            dat: dictionary loaded from processed neuropix struct
            opt: options for smoothing etc

    """
    if not opt:
        opt = dict(tbin =  0.02,
                   tstart = 0,
                   tend = np.max(st),
                   smoothSigma_time = 0.1)

    tbinedge = np.arange(opt['tstart'],opt['tend'],opt['tbin'])
    tbincent = tbinedge[:-1] + opt["tbin"] / 2

    # firing rate matrix
    fr = np.empty_like((len(cell_id),len(tbinedge)-1))
    fr[:] = np.nan

    for i, this_cell_id in enumerate(cell_id):
        # get spike times for this cell
        spike_t = st[clu == this_cell_id];
        spike_t = spike_t[(spike_t >= opt['tstart']) & (spike_t <= opt['tend'])];

        # compute distance-binned firing rate
        # print(np.histogram(spike_t,bins = tbinedge))
        print(type(spike_t),type(tbin_edge))
        fr_this,edges = np.histogram(spike_t,bins = tbinedge)
        print(fr_this)
        print(edges)
        fr_this = fr_this / opt['tbin']

        # smooth firing rate
        fr_this = gaussian_filter(fr_this,sigma = opt['smoothSigma_time']/opt['tbin']);

        if len(np.where(np.isnan(fr_this))[0]) > 0:
            raise ValueError("NaN in firing rate")

        fr[i,:] = fr_this;

    return fr,tbincent
