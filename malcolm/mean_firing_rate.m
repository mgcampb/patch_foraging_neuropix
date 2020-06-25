% script to plot fr matrix sorted by average firing rate for a given
% session
% MGC 6/25/2020

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
paths.figs = 'C:\figs\patch_foraging_neuropix\mean_firing_rate';
addpath(genpath(paths.malcolm_functions));

opt = struct;
opt.session = '78_20200311'; % session to analyze
opt.tbin = 0.02; % in seconds
opt.smoothSigma_time = 0.1;

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);
opt.tstart = 0;
opt.tend = max(dat.sp.st);
[frMat, tbincent] = calcFRVsTime(good_cells,dat,opt);
num_spikes = nan(size(good_cells));
for i = 1:numel(good_cells)
    num_spikes(i) = sum(dat.sp.clu==good_cells(i));
end

%% plot
hfig = figure('Position',[200 100 1500 1200]);
hfig.Name = sprintf('mean firing rate heatmap sorted %s',opt.session);
subplot(3,12,1:9);
mean_fr = mean(frMat,2);
[~,sort_idx] = sort(mean_fr);
imagesc(tbincent,1:size(frMat,1),frMat(sort_idx,:));
set(gca,'YDir','normal');
xlabel('time (sec)');
ylabel('cell (sorted by mean FR)');
title('firing rate');
colorbar;

subplot(3,12,10:12);
semilogx(mean_fr(sort_idx),1:size(frMat,1));
ylim([1 size(frMat,1)]);
xlabel('Mean Firing Rate (Hz)');
yticklabels([]);

subplot(3,12,13:21);
frMat_zscore = my_zscore(frMat);
imagesc(tbincent,1:size(frMat_zscore,1),frMat_zscore(sort_idx,:));
set(gca,'YDir','normal');
xlabel('time (sec)');
ylabel('cell (sorted by mean FR)');
title('z-scored firing rate');
colorbar;

subplot(3,12,22:24);
semilogx(num_spikes(sort_idx),1:size(frMat,1));
ylim([1 size(frMat,1)]);
xlabel('Num Spikes');
yticklabels([]);

subplot(3,12,25:33);
frMat_zscore = my_zscore(frMat);
imagesc(tbincent,1:size(frMat_zscore,1),frMat_zscore(sort_idx,:));
set(gca,'YDir','normal');
xlabel('time (sec)');
ylabel('cell (sorted by mean FR)');
title('z-scored firing rate: rescale color axis');
colorbar;
caxis([-3 10]);

save_figs(paths.figs,hfig,'png');