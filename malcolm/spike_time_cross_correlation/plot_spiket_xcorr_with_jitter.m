% script to plot spike time cross correlations against jitter distribution
% for individual pairs of neurons
% MGC 4/14/2021

paths = struct;
paths.data = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation\data_organized_for_RCC_cluster';
paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation\20210414_run1';
paths.figs = 'C:\figs\patch_foraging_neuropix\spike_time_cross_correlation\20210414_run1';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.zscore_thresh = 4.0;

%% load input data used to compute spiket xcorr
load(fullfile(paths.data,'data_organized_for_spike_t_crosscorr_RCC_cluster.mat'));

%% load results of spiket xcorr calculation (from chunks)

xcorr_all = [];
xcorr_jit = [];

rez_files = dir(fullfile(paths.results,'chunk*.mat'));
rez_files = {rez_files.name}';
for i = 1:numel(rez_files)
    fprintf('Loading chunk %d/%d\n',i,numel(rez_files));
    rez = load(fullfile(paths.results,rez_files{i}));
    xcorr_all = [xcorr_all; rez.xcorr_all];
    xcorr_jit = [xcorr_jit; rez.xcorr_jit];
end

%% make plots

deltaT = 1000*rez.opt.binsize*(-rez.opt.max_lag:rez.opt.max_lag);

for pIdx = 1:size(xcorr_all,1)
    fprintf('Plotting pair %d/%d\n',pIdx,size(xcorr_all,1));
    xcorr_this = xcorr_all(pIdx,:);
    jit_mean = squeeze(mean(xcorr_jit(pIdx,:,:),3));
    jit_sd = squeeze(std(xcorr_jit(pIdx,:,:),[],3));

    hfig = figure('Position',[200 200 1000 400]);
    uid1 = pairs_table.UniqueID1{pIdx};
    uid2 = pairs_table.UniqueID2{pIdx};
    hfig.Name = sprintf('%s %s',uid1,uid2);

    subplot(1,2,1); hold on;
    shadedErrorBar(deltaT,jit_mean,jit_sd,'lineProps',{'k-'});
    plot(deltaT,xcorr_this,'r.-');
    xticks(ceil(min(deltaT)/10)*10:10:floor(max(deltaT)/10)*10);
    grid on;
    xlabel('lag (ms)');
    ylabel('raw xcorr');
    plot([0 0],ylim,'k--');

    subplot(1,2,2); hold on;
    xcorr_zscore = (xcorr_this-jit_mean)./jit_sd;
    plot(deltaT,xcorr_zscore,'r.-');
    ymin = min(floor(min(xcorr_zscore)),-opt.zscore_thresh-1);
    ymax = max(ceil(max(xcorr_zscore)),opt.zscore_thresh+1);
    ylim([ymin ymax]);
    plot(xlim,[0 0],'k-');
    plot(xlim,[opt.zscore_thresh opt.zscore_thresh],'k--');
    plot(xlim,-[opt.zscore_thresh opt.zscore_thresh],'k--')
    plot([0 0],ylim,'k-');
    xticks(ceil(min(deltaT)/10)*10:10:floor(max(deltaT)/10)*10);
    grid on;
    xlabel('lag (ms)');
    ylabel('z-score');
    
    signif = xcorr_zscore>opt.zscore_thresh | xcorr_zscore<-opt.zscore_thresh;
    plot(deltaT(signif),xcorr_zscore(signif),'ro','MarkerFaceColor','r');
    
    htitle = suptitle(sprintf('cell1=%s, cell2=%s',uid1,uid2));
    htitle.Interpreter = 'none';
    
    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
    close(hfig);
end