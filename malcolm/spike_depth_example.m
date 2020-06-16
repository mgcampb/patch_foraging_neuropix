% example of how to get spike depth information
% MGC 6/16/2020

% need the spikes repository from cortex-lab
addpath(genpath('C:\code\spikes'));

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';

opt = struct;
opt.session = '80_20200315.mat';

%% load data from the session
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2)';

%%  get spike depths for all spikes individually
% this function comes from the spikes repository
% depth indicates distance from tip of probe in microns
[~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);

%% take median spike depth for each cell
spike_depths = nan(size(good_cells));
for cIdx = 1:numel(good_cells)
    spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
end
