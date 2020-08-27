function [fr, tbincent] = calcFRVsTime(cell_id,dat,opt)
% cell_id:  which cell ids to analyze
% dat:      the data structure, e.g. dat = load('80_20200317.mat')
% opt:      options
% MGC 12/15/2019

if ~exist('opt','var')
    opt = struct;
    opt.tbin = 0.02;
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    opt.smoothSigma_time = 0.1;
end

tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

% firing rate matrix
fr = nan(numel(cell_id),numel(tbinedge)-1);

for i = 1:numel(cell_id)
    % get spike times for this cell
    spike_t = dat.sp.st(dat.sp.clu==cell_id(i));
    spike_t = spike_t(spike_t>=opt.tstart & spike_t<=opt.tend);   
    
    % compute distance-binned firing rate
    fr_this = histcounts(spike_t,tbinedge)/opt.tbin;

    % smooth firing rate
    fr_this = gauss_smoothing(fr_this,opt.smoothSigma_time/opt.tbin);
    
    if sum(isnan(fr_this))>0
        keyboard
    end
    
    fr(i,:) = fr_this;
end

end