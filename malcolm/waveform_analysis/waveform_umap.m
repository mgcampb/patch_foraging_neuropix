paths = struct;
paths.data = 'C:\data\patch_foraging_neuropix\waveforms\waveform_cluster';
paths.data_proc = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';

paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.umap = 'C:\code\umapFileExchange (2.1.3)';
addpath(genpath(paths.umap));

paths.save = fullfile(paths.data,'waveform_cluster');
if ~isfolder(paths.save)
    mkdir(paths.save);
end

paths.figs = 'C:\figs\patch_foraging_neuropix\waveform';

%% load data
session = dir(paths.data);
session = {session.name}';
session = session(contains(session,'.mat'));

mw_all = [];
session_all = cell(10000,1);
cellID = nan(10000,1);
uniqID = cell(10000,1);
meanFR = nan(10000,1);
counter = 1;
for i = 1:numel(session)
    fprintf('Loading session %d/%d: %s\n',i,numel(session),session{i});
    dat = load(fullfile(paths.data,session{i}));
    dat2 = load(fullfile(paths.data_proc,session{i}));
    mw_all = [mw_all; dat.mean_waveform];
    good_cells = dat2.sp.cids(dat2.sp.cgs==2);
    meanFR_this = nan(numel(good_cells),1);
    for j = 1:numel(good_cells)
        session_all{counter} = session{i}(1:end-4);
        cellID(counter) = good_cells(j);
        uniqID{counter} = sprintf('%s_c%d',session{i}(1:end-4),good_cells(j));
        meanFR(counter) = sum(dat2.sp.clu==good_cells(j))/max(dat2.sp.st);
        counter = counter+1;
    end
end
t_spk = dat.t_spk;
session_all = session_all(1:size(mw_all,1));
cellID = cellID(1:size(mw_all,1));
uniqID = uniqID(1:size(mw_all,1));
meanFR = meanFR(1:size(mw_all,1));
keep = ~isnan(mw_all(:,1));
mw_all = mw_all(keep,:);
uniqID = uniqID(keep);
meanFR = meanFR(keep);

%% baseline subtract
mw_bl_sub = nan(size(mw_all));
for i = 1:size(mw_all,1)
    bl_this = mw_all(i,1:dat.opt.samp_before/2);
    mw_bl_sub(i,:) = mw_all(i,:) - mean(bl_this);
end

% zscore
mw_zscore = zscore(mw_bl_sub,[],2);

%% spike width
spike_width = nan(size(mw_bl_sub,1),1);
t_interp = linspace(min(t_spk),max(t_spk),1000);
delT = mean(diff(t_interp));
for i = 1:size(mw_bl_sub,1)
    wv_this = mw_bl_sub(i,:);
    wv_interp = interp1(t_spk,wv_this,t_interp);
    [mx1,mx_idx1] = max(-wv_interp);
    [mx2,mx_idx2] = max(wv_interp(mx_idx1+1:end));
    spike_width(i) = mx_idx2 * delT;
end

%% plot example neuron
hfig = figure('Position',[400 400 250 200]);
hfig.Name = 'example neuron waveform';
plot(dat.t_spk,mw_all(1,:)*2.34,'k-');
box off;
ylabel('uV');
xlabel('ms');
saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

%% umap
[X,~,clusterID] = run_umap(mw_zscore);

%% plot
figure;
my_scatter(X(:,1),X(:,2),'k',0.1);
xlabel('UMAP dim1');
ylabel('UMAP dim2');

%% kmeans
num_clust = 7;
rng(1);
kidx = kmeans(X,num_clust);
plot_col = lines(num_clust);
figure; hold on;
for i = 1:num_clust
    my_scatter(X(kidx==i,1),X(kidx==i,2),plot_col(i,:),0.2);
end
xlabel('UMAP dim1');
ylabel('UMAP dim2');

%% GMM
num_clust = 3;
rng(1);
gmm_options = statset('MaxIter',1000);
gmfit = fitgmdist(X,num_clust,'CovarianceType','full','SharedCovariance',false,'Options',gmm_options); % Fitted GMM
gmm_cluster = cluster(gmfit,X); % Cluster index 

% rename so that clust 1 = RS, clust 2 = NS, clust 3 = T
gmm_cluster(gmm_cluster==1) = 4;
gmm_cluster(gmm_cluster==2) = 1;
gmm_cluster(gmm_cluster==3) = 2;
gmm_cluster(gmm_cluster==4) = 3;

plot_col = lines(num_clust);
hfig = figure; hold on;
hfig.Name = 'umap gmm';
for i = 1:num_clust
    my_scatter(X(gmm_cluster==i,1),X(gmm_cluster==i,2),plot_col(i,:),0.2);
end
xticks([]);
yticks([]);
% xlabel('UMAP dim1');
% ylabel('UMAP dim2');
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% GMM BIC vs num clust

% rng(3);
% gmm_opt = statset; % GMM options
% gmm_opt.Display = 'off';
% gmm_opt.MaxIter = 1000;
% gmm_opt.Lambda = 1;
% gmm_opt.Replicates = 50;
% gmm_opt.TolFun = 1e-6;
% num_clust = 10;
% bic_all = nan(num_clust,1);
% for j = 1:num_clust
%     gm = fitgmdist(X,j,'RegularizationValue',gmm_opt.Lambda,'Replicates',gmm_opt.Replicates,'Options',gmm_opt); % fit GM model
%     % clust_gmm = cluster(gm,X); % hard clustering
%     bic_all(j) = gm.BIC;
% end
% 
% hfig = figure('Position',[300 300 300 300]);
% hfig.Name = 'BIC vs num clusters';
% plot(bic_all,'ko-','MarkerFaceColor','k');
% xlim([0 num_clust+1]);
% box off;
% xlabel('Num. clusters');
% ylabel('BIC');
% saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

%% waveform of GMM clusters
hfig = figure; hold on;
hfig.Name = 'waveform cluster avg';
for i = 1:num_clust
    plot(dat.t_spk,mean(mw_zscore(gmm_cluster==i,:)),'Color',plot_col(i,:),'LineWidth',1.5);
end
xlabel('ms');
ylabel('z-score');
legend(num2str((1:num_clust)'));
saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

%% waveform cluster table

waveform_cluster = table;
waveform_cluster.Session = session_all;
waveform_cluster.CellID = cellID;
waveform_cluster.UniqueID = uniqID;

% waveform type
WaveformType = cell(size(uniqID));
WaveformType(gmm_cluster==1) = {'Regular'};
WaveformType(gmm_cluster==2) = {'TriPhasic'};
WaveformType(gmm_cluster==3) = {'Narrow'};
waveform_cluster.WaveformType = WaveformType;

%% save waveform cluster table
save(fullfile(paths.save,'waveform_cluster'),'waveform_cluster');

%% plot spike width vs log FR

% color by GMM cluster
plot_col = lines(3);
hfig = figure; hold on;
hfig.Name = 'spike width vs log FR colored by GMM cluster';
for i = 1:3
    scatter(spike_width(gmm_cluster==i),meanFR(gmm_cluster==i),'MarkerEdgeColor',plot_col(i,:),'MarkerFaceColor',plot_col(i,:),'MarkerFaceAlpha',0.1,'MarkerEdgeAlpha',0.2)
end
set(gca,'yscale','log')
xlabel('Spike Width (ms)');
ylabel('Firing Rate');
legend('RS','NS','T');
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');