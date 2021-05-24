% identify types of spike time interactions
% MGC 4/16/2021


paths = struct;
paths.data = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation\data_organized_for_RCC_cluster';
paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation\20210414_run1';
paths.figs = 'C:\figs\patch_foraging_neuropix\spike_time_cross_correlation\interaction_types';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.zscore_thresh = 4.0;

%% load input data used to compute spiket xcorr
load(fullfile(paths.data,'data_organized_for_spike_t_crosscorr_RCC_cluster.mat'));

%% load results spiket xcorr calculation 
load(fullfile(paths.results,'rez_combined.mat'));
deltaT = 1000*xcorr_opt.binsize*(-xcorr_opt.max_lag:xcorr_opt.max_lag);

%% zscore to jitter distribution
xcorr_jit_mean = mean(xcorr_jit,3);
xcorr_jit_sd = std(xcorr_jit,[],3);
xcorr_zscore = (xcorr_all - xcorr_jit_mean)./xcorr_jit_sd;

%% identify interacting pairs

% take max and min
[max_zscore,maxidx] = max(xcorr_zscore,[],2);
[min_zscore,minidx] = min(xcorr_zscore,[],2);

% exclude pairs with max at zero
max_at_zero = maxidx==xcorr_opt.max_lag+1;

% interacting_pairs = (max_zscore>opt.zscore_thresh | min_zscore<-opt.zscore_thresh) & ~max_at_zero;

interacting_pairs = (max_zscore>opt.zscore_thresh | min_zscore<-opt.zscore_thresh);

%% plot heatmap of zscored xcorr
xcorr_inter = xcorr_zscore(interacting_pairs,:);
xcorr_thresh = sign(xcorr_inter).*(abs(xcorr_inter)>opt.zscore_thresh);

hfig = figure('Position',[400 400 1000 400]);
hfig.Name = 'heatmap of zscored xcorr';

subplot(1,2,1);
imagesc(deltaT,1:size(xcorr_inter,1),xcorr_inter);
xlabel('deltaT (ms)');
ylabel('Pair');
title('xcorr, z-scored to jitter');
colorbar;

subplot(1,2,2);
imagesc(deltaT,1:size(xcorr_inter,1),zscore(xcorr_inter,[],2));
xlabel('deltaT (ms)');
ylabel('Pair');
title('xcorr, z-scored to jitter, z-scored by row');
colorbar;

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% pca on zscored xcorr
X = zscore(xcorr_inter,[],2);
[coeffs,score,~,~,expl] = pca(X);


%% assess optimal number of clusters using BIC
% 
% num_iter = 100;
% bic_all = nan(num_iter,10);
% for i = 1:num_iter
%     fprintf('iter %d/%d\n',i,num_iter);
%     for k = 1:10
%         gm = fitgmdist(score(:,1:opt.num_pcs),k,'Options',gmm_opt);
%         bic_all(i,k) = gm.BIC;
%     end
% end
% 
% 
% hfig = figure;
% hfig.Name = sprintf('BIC vs num clusters %d PCs',opt.num_pcs);
% errorbar(1:10,mean(bic_all),std(bic_all)/sqrt(num_iter),'o-');
% ylabel('BIC');
% xlabel('num clusters');
% title(sprintf('n=%d PCs',opt.num_pcs));
% saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% GMM clustering
opt.num_pcs = 5;
opt.num_clust = 5;

rng(2);
gmm_opt = statset;
gmm_opt.Display = 'off';
gmm_opt.MaxIter = 1000;
gmm_opt.TolFun = 1e-6;
gm = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'Options',gmm_opt);

% reassign cluster numbers for consistency
cluster_gmm = gm.cluster(score(:,1:opt.num_pcs));

hfig = figure; hold on;
plot_col = lines(opt.num_clust);
for i = 1:opt.num_clust
    plot3(score(cluster_gmm==i,1),score(cluster_gmm==i,2),score(cluster_gmm==i,3),'o','Color',plot_col(i,:));
end

hfig = figure;
hfig.Name = 'Interaction cluster average';
for i = 1:opt.num_clust
    subplot(opt.num_clust,1,i);
    plot(deltaT,mean(X(cluster_gmm==i,:)));
end
xlabel('DeltaT (ms)');
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

% heatmap sorted by interaction type
hfig = figure;
hfig.Name = 'Interaction heatmap sorted by cluster';
[~,sort_idx] = sort(cluster_gmm);
imagesc(deltaT,1:size(X,1),X(sort_idx,:));
xlabel('DeltaT (ms)');
ylabel('Pair');
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

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

%% pick out short-latency excitatory interactions (clusters 4 and 5)
exc = cluster_gmm==4 | cluster_gmm==5;

pairs_table_inter = pairs_table(interacting_pairs,:);
pairs_exc = pairs_table_inter(exc,:);
pairs_exc1 = pairs_table_inter(cluster_gmm==4,:);
pairs_exc2 = pairs_table_inter(cluster_gmm==5,:);

%% exc interactions by waveform
wv1 = [pairs_exc1.Waveform1; pairs_exc2.Waveform2]; % postsynaptic
wv2 = [pairs_exc1.Waveform2; pairs_exc2.Waveform1]; % presynaptic

[tab,~,~,labels]=crosstab(wv1,wv2);
[labels(:,1),sort_idx1] = sort(labels(:,1));
[labels(:,2),sort_idx2] = sort(labels(:,2));
tab = tab(sort_idx1,:);
tab = tab(:,sort_idx2);

%% exc interactions by glm cluster
clu1 = [pairs_exc1.GMM_Cluster1; pairs_exc2.GMM_Cluster2]; % postsynaptic
clu2 = [pairs_exc1.GMM_Cluster2; pairs_exc2.GMM_Cluster1]; % presynaptic

[tab,~,~,labels]=crosstab(clu1,clu2);



%% pick out biphasic interactions (clusters 1 and 3)
biphas = cluster_gmm==1 | cluster_gmm==3;

pairs_table_inter = pairs_table(interacting_pairs,:);
pairs_biphas = pairs_table_inter(biphas,:);
pairs_biphas1 = pairs_table_inter(cluster_gmm==1,:);
pairs_biphas2 = pairs_table_inter(cluster_gmm==3,:);

%% biphasic interactions by waveform
wv1 = [pairs_biphas1.Waveform1; pairs_biphas2.Waveform2]; % postsynaptic
wv2 = [pairs_biphas1.Waveform2; pairs_biphas2.Waveform1]; % presynaptic

[tab,~,~,labels]=crosstab(wv1,wv2);
[labels(:,1),sort_idx1] = sort(labels(:,1));
[labels(:,2),sort_idx2] = sort(labels(:,2));
tab = tab(sort_idx1,:);
tab = tab(:,sort_idx2);

%% biphas interactions by glm cluster
clu1 = [pairs_biphas1.GMM_Cluster1; pairs_biphas2.GMM_Cluster2]; % postsynaptic
clu2 = [pairs_biphas1.GMM_Cluster2; pairs_biphas2.GMM_Cluster1]; % presynaptic

[tab,~,~,labels]=crosstab(clu1,clu2);

%% pick out zero lag interactions (cluster 2)
zerolag = cluster_gmm==2;

pairs_table_inter = pairs_table(interacting_pairs,:);
pairs_zerolag = pairs_table_inter(zerolag,:);
pairs_zerolag1 = pairs_table_inter(cluster_gmm==2,:);

%% zero-lag exc by waveform
wv1 = [pairs_zerolag1.Waveform1]; % postsynaptic
wv2 = [pairs_zerolag1.Waveform2]; % presynaptic

[tab,~,~,labels]=crosstab(wv1,wv2);
[labels(:,1),sort_idx1] = sort(labels(:,1));
[labels(:,2),sort_idx2] = sort(labels(:,2));
tab = tab(sort_idx1,:);
tab = tab(:,sort_idx2);

%% zero-lag exc by glm cluster
clu1 = [pairs_zerolag1.GMM_Cluster1]; % postsynaptic
clu2 = [pairs_zerolag1.GMM_Cluster2]; % presynaptic

[tab,~,~,labels]=crosstab(clu1,clu2);


