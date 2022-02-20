% script to find putative interacting units using jitter control
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

%% load results spiket xcorr calculation 
load(fullfile(paths.results,'rez_combined.mat'));

%% find interacting pairs

% take zscore
xcorr_jit_mean = mean(xcorr_jit,3);
xcorr_jit_sd = std(xcorr_jit,[],3);
xcorr_zscore = (xcorr_all - xcorr_jit_mean)./xcorr_jit_sd;

% take max and min
[max_zscore,maxidx] = max(xcorr_zscore,[],2);
[min_zscore,minidx] = min(xcorr_zscore,[],2);

% exclude pairs with max at zero
max_at_zero = maxidx==xcorr_opt.max_lag+1;

interacting_pairs = (max_zscore>opt.zscore_thresh | min_zscore<-opt.zscore_thresh) & ~max_at_zero;
%interacting_pairs = (max_zscore>opt.zscore_thresh | min_zscore<-opt.zscore_thresh);

%% interactions by glm cluster

[cross_tab_all,~,~,labels1] = crosstab(pairs_table.GMM_Cluster1,pairs_table.GMM_Cluster2);
[cross_tab_interact,~,~,labels2] = crosstab(pairs_table.GMM_Cluster1(interacting_pairs),pairs_table.GMM_Cluster2(interacting_pairs));
pct_interact = 100*cross_tab_interact./cross_tab_all;

cross_tab_all_combined = cross_tab_all+cross_tab_all'-diag(diag(cross_tab_all));
cross_tab_interact_combined = cross_tab_interact+cross_tab_interact'-diag(diag(cross_tab_interact));
pct_interact_combined = 100*cross_tab_interact_combined./cross_tab_all_combined;

tab_all = triu(cross_tab_all_combined);
tab_inter = triu(cross_tab_interact_combined);

%% get waveform

waveform1 = cell(size(pairs_table,1),1);
waveform2 = cell(size(pairs_table,2),1);
for i = 1:size(pairs_table,1)
    idx1 = find(strcmp(sig_cells.UniqueID,pairs_table.UniqueID1{i}));
    idx2 = find(strcmp(sig_cells.UniqueID,pairs_table.UniqueID2{i}));
    waveform1{i} = sig_cells.WaveformType{idx1};
    waveform2{i} = sig_cells.WaveformType{idx2};
end
pairs_table.Waveform1 = waveform1;
pairs_table.Waveform2 = waveform2;

%% interactions by waveform

[cross_tab_all,~,~,labels1] = crosstab(pairs_table.Waveform1,pairs_table.Waveform2);
[cross_tab_interact,~,~,labels2] = crosstab(pairs_table.Waveform1(interacting_pairs),pairs_table.Waveform2(interacting_pairs));

% make sure labels are in right order
[~,sort_idx1] = sort(labels1(:,1));
[~,sort_idx2] = sort(labels1(:,2));
cross_tab_all = cross_tab_all(sort_idx1,:);
cross_tab_all = cross_tab_all(:,sort_idx2);

[~,sort_idx1] = sort(labels2(:,1));
[~,sort_idx2] = sort(labels2(:,2));
cross_tab_interact = cross_tab_interact(sort_idx1,:);
cross_tab_interact = cross_tab_interact(:,sort_idx2);

pct_interact = 100*cross_tab_interact./cross_tab_all;

cross_tab_all_combined = cross_tab_all+cross_tab_all'-diag(diag(cross_tab_all));
cross_tab_interact_combined = cross_tab_interact+cross_tab_interact'-diag(diag(cross_tab_interact));
pct_interact_combined = 100*cross_tab_interact_combined./cross_tab_all_combined;


tab_all = triu(cross_tab_all_combined);
tab_all = tab_all(1:2,1:2);
tab_inter = triu(cross_tab_interact_combined);
tab_inter = tab_inter(1:2,1:2);
