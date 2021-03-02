% Script to plot the estimated 3d anatomical coords for neurons from GLM
% clusters
%
% based on pca_on_model_coefficients.m
% MGC 2/15/2021

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210212_R_test\sessions';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210208_original_vars';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210210_model_comparison_new_glmnet';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_cluster_anatomical_coords\run_20210212_R_test\no_pvalue_cutoff';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';
opt.pval_thresh = 2; % 0.05;

opt.num_clust = 5; % for k means
opt.num_pcs = 3; % for k means

%% get sessions
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

% get mouse name for each session
mouse = cell(size(session_all));
if strcmp(opt.data_set,'mb')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:2);
    end
elseif strcmp(opt.data_set,'mc')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:3);
    end
end
uniq_mouse = unique(mouse);

%% get significant cells from all sessions
sesh_all = [];
beta_all_sig = [];
cellID_uniq = [];
coordsAP = [];
coordsML = [];
coordsDV = [];
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    dat = load(fullfile(paths.data,session_all{i}),'anatomy3d');
    if ~isfield(dat,'anatomy3d')
        continue;
    end

    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    good_cells_this = fit.good_cells(keep_cell);
    
    % sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig = sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig_cells = good_cells_this(sig);

    beta_this = fit.beta_all(fit.base_var==0,keep_cell);
    beta_all_sig = [beta_all_sig beta_this(:,sig)];
    sesh_all = [sesh_all; i*ones(sum(sig),1)];
    
    cellID_uniq_this = cell(numel(sig_cells),1);
    for j = 1:numel(sig_cells)
        cellID_uniq_this{j} = sprintf('%s_c%d',fit.opt.session,sig_cells(j));
    end
    cellID_uniq = [cellID_uniq; cellID_uniq_this];
    
    % anatomy coords
    keep = ismember(dat.anatomy3d.Coords.CellID,sig_cells);
    assert(all(dat.anatomy3d.Coords.CellID(keep)==sig_cells'));
    coordsAP = [coordsAP; dat.anatomy3d.Coords.AP(keep)];
    coordsML = [coordsML; dat.anatomy3d.Coords.ML(keep)];
    coordsDV = [coordsDV; dat.anatomy3d.Coords.DV(keep)];

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all_sig);
% beta_norm = beta_all_sig;
[coeff,score,~,~,expl] = pca(beta_norm');

%% k means 
rng(1);
kmeans_idx = kmeans(score(:,1:opt.num_pcs),opt.num_clust);
% kmeans_idx = kmeans(beta_all_sig',opt.num_clust);

%% plot coefficients for kmeans clusters

hfig = figure('Position',[50 50 1500 1250]);
hfig.Name = sprintf('k means avg cluster coefficients %s cohort %s',opt.data_set,opt.brain_region);

plot_col = cool(3);
for i = 1:opt.num_clust
    subplot(opt.num_clust,1,i); hold on;
    mean_this = mean(beta_norm(:,kmeans_idx==i),2);
    sem_this = std(beta_norm(:,kmeans_idx==i),[],2)/sqrt(sum(kmeans_idx==i));
    for j = 1:size(mean_this,1)
        if contains(var_name{j},'1uL')
            plot_col_this = plot_col(1,:);
        elseif contains(var_name{j},'2uL')
            plot_col_this = plot_col(2,:);
        elseif contains(var_name{j},'4uL')
            plot_col_this = plot_col(3,:);
        end
        if contains(var_name{j},'RewKern')
            errorbar(j,mean_this(j),sem_this(j),'o','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
        else
            errorbar(j,mean_this(j),sem_this(j),'v','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
        end
    end
    plot(xlim,[0 0],'k:');
    xticks([]);
    ylabel('Coeff');
    title(sprintf('Average of cluster %d (n = %d cells)',i,sum(kmeans_idx==i)));
    set(gca,'FontSize',14);
end

xticks(1:numel(var_name));
xticklabels(var_name);
xtickangle(90);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',14);
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% statistical test

% control for mouse
mouse = nan(size(sesh_all));
for i = 1:numel(mouse)
    mouse(i) = str2double(session_all{sesh_all(i)}(1:2));
end

% anova
anova_pval = nan(3,1);
tmp = anovan(coordsAP,{kmeans_idx,mouse},'display','off');
anova_pval(1) = tmp(1);

tmp = anovan(coordsML,{kmeans_idx,mouse},'display','off');
anova_pval(2) = tmp(1);

tmp = anovan(coordsDV,{kmeans_idx,mouse},'display','off');
anova_pval(3) = tmp(1);

%% plot anatomical coords of k means clusters

hfig = figure('Position',[50 50 1500 1250]);
hfig.Name = sprintf('k means cluster anatomical coords %s cohort %s',opt.data_set,opt.brain_region);

for i = 1:opt.num_clust
    
    subplot(opt.num_clust,3,3*(i-1)+1);
    histogram(coordsAP(kmeans_idx==i),10);
    xlim([3000 4000]);
    if i==1
        title(sprintf('AP ANOVA p = %0.3f\nCluster %d',anova_pval(1),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('AP coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
    
    subplot(opt.num_clust,3,3*(i-1)+2);
    histogram(coordsML(kmeans_idx==i),10);
    xlim([4400 5500]);
    if i==1
        title(sprintf('ML ANOVA p = %0.3f\nCluster %d',anova_pval(2),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('ML coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
    
    subplot(opt.num_clust,3,3*(i-1)+3);
    histogram(coordsDV(kmeans_idx==i),10);
    xlim([1500 4500]);
    if i==1
        title(sprintf('DV ANOVA p = %0.3f\nCluster %d',anova_pval(3),i));
    else
        title(sprintf('Cluster %d',i));
    end
    xlabel('DV coord (um)');
    ylabel('num cells');
    set(gca,'FontSize',14);
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');