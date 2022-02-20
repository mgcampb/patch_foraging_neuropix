%% Just a reproduction of Malcolm's UMAP result!
%  brazenly stolen from malcolm/waveforms/waveform_umap.m

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/waveforms';
paths.data_proc = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';

session = dir(paths.data);
session = {session.name}';
session = session(contains(session,'.mat'));
%%
mw_all = [];
cellID = cell(10000,1);
counter = 1;
for i = 1:numel(session)
    fprintf('Loading session %d/%d: %s\n',i,numel(session),session{i});
    dat = load(fullfile(paths.data,session{i}));
    dat2 = load(fullfile(paths.data_proc,session{i}));
    mw_all = [mw_all; dat.mean_waveform];
    good_cells = dat2.sp.cids(dat2.sp.cgs==2);
    for j = 1:numel(good_cells)
        cellID{counter} = sprintf('%s_c%d',session{i}(1:end-4),good_cells(j));
        counter = counter+1;
    end
end
cellID = cellID(1:size(mw_all,1));
keep = ~isnan(mw_all(:,1));
mw_all = mw_all(keep,:);
cellID = cellID(keep);

%% baseline subtract
mw_bl_sub = nan(size(mw_all));
for i = 1:size(mw_all,1)
    bl_this = mw_all(i,1:dat.opt.samp_before/2);
    mw_bl_sub(i,:) = mw_all(i,:) - mean(bl_this);
end

% zscore
mw_zscore = zscore(mw_bl_sub,[],2);

% umap
[X,~,clusterID] = run_umap(mw_zscore);

%% plot
figure;
% my_scatter(X(:,1),X(:,2),'ko',0.2);
scatter(X(:,1),X(:,2),10,'k.')
% binscatter(X(:,1),X(:,2),75)
% colormap('parula')
xlabel('UMAP dim1');
ylabel('UMAP dim2');
f = gca;
set(f,'Fontsize',14)

%% Kmeans for variety of choices of num_clust 
rng(1);
max_num_clust = 10; 
gmm_opt.lambda = 0; 
gmm_opt.replicates = 50; 
bic = nan(max_num_clust,1);
for num_clust = 1:max_num_clust 
    this_gmm = fitgmdist(X,num_clust,'RegularizationValue',gmm_opt.lambda,'replicates',gmm_opt.replicates);                          
    bic(num_clust) = this_gmm.BIC;
end
plot(bic,'linewidth',2)
plot(gradient(gradient(bic)))

%% kmeans for final choice of num_clust
num_clust = 3;
gmm_opt.replicates = 50; 
rng(1);
plot_col = cbrewer('div','Spectral',10); 
plot_col = plot_col([9 4 2],:); 
% kidx = kmeans(X,num_clust);
this_gmm = fitgmdist(X,num_clust,'RegularizationValue',gmm_opt.lambda,'replicates',gmm_opt.replicates);      
kidx = cluster(this_gmm,X);
figure; hold on;
for i = 1:num_clust
    my_scatter(X(kidx==i,1),X(kidx==i,2),plot_col(i,:),0.2);
end
xlabel('UMAP dim1');
ylabel('UMAP dim2');
f = gca; 
set(f,'fontsize',14)

%% waveform
figure; hold on;
for i = 1:num_clust
    plot(dat.t_spk,mean(mw_zscore(kidx==i,:)),'Color',plot_col(i,:),'LineWidth',1.5);
end
xlabel('ms');
ylabel('zscore');
legend(num2str((1:num_clust)'));