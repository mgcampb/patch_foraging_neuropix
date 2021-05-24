% plot percent null deviance explained for models with different sets of
% regressors
% MGC 5/18/2021

% first get clusters as in pca_on_model_coefficients

run_name = '20210514_accel';
models_for_comparison = {'20210517_SessionTime',...
    '20210517_SessionTime_Behav',...
    '20210517_SessionTime_Behav_PatchKern',...
    '20210517_SessionTime_Behav_PatchKern_RewKern'};

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_output',run_name);
paths.figs = fullfile('C:\figs\patch_foraging_neuropix\glm_model_comparison',run_name);
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC'; % 'PFC' or 'Sub-PFC'
opt.data_set = 'mb';
opt.pval_thresh = 2; % 0.05;

opt.num_clust = 5; % for k means or GMM
opt.num_pcs = 3; % for k means or GMM

paths.figs = fullfile(paths.figs,opt.brain_region);
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt.save_figs = true;

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
beta_including_base_var = [];
cellID_uniq = [];
num_cells_glm_total = 0;
num_cells_total = 0;
devratio = [];
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    num_cells_glm_total = num_cells_glm_total + numel(fit.good_cells);
    num_cells_total = num_cells_total + numel(fit.good_cells_all);
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    good_cells_this = fit.good_cells(keep_cell);
    
    % sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig = sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig_cells = good_cells_this(sig);

    beta_this = fit.beta_all(fit.base_var==0,keep_cell);
    beta_all_sig = [beta_all_sig beta_this(:,sig)];
    sesh_all = [sesh_all; i*ones(sum(sig),1)];
    
    beta_this = fit.beta_all(:,keep_cell);
    beta_including_base_var = [beta_including_base_var beta_this(:,sig)];
    
    cellID_uniq_this = cell(numel(sig_cells),1);
    for j = 1:numel(sig_cells)
        cellID_uniq_this{j} = sprintf('%s_c%d',fit.opt.session,sig_cells(j));
    end
    cellID_uniq = [cellID_uniq; cellID_uniq_this];
    
    devratio_this = fit.devratio(keep_cell);
    devratio = [devratio; devratio_this(sig)];

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all_sig);
% beta_norm = beta_all_sig;
[coeff,score,~,~,expl] = pca(beta_norm');

%% GMM clustering

X = score(:,1:opt.num_pcs); % the first N PCs of z-scored coefficient matrix

rng(3); % for reproducibility
gmm_opt = statset; % GMM options
gmm_opt.Display = 'off';
gmm_opt.MaxIter = 1000;
gmm_opt.Lambda = 1;
gmm_opt.Replicates = 100;
gmm_opt.TolFun = 1e-6;
gm = fitgmdist(X,opt.num_clust,'RegularizationValue',gmm_opt.Lambda,'Replicates',gmm_opt.Replicates,'Options',gmm_opt); % fit GM model
clust_gmm = cluster(gm,X); % hard clustering

% order clusters by size
clust_count = crosstab(clust_gmm);
[~,sort_idx] = sort(clust_count,'descend');
clust_reorder = nan(size(clust_gmm));
for i = 1:opt.num_clust
    clust_reorder(clust_gmm==sort_idx(i))=i;
end
clust_gmm = clust_reorder;

%% get devratio from all models

devratio_all = nan(numel(cellID_uniq),numel(models_for_comparison)+1);
devratio_all(:,end) = devratio;

fprintf('\n\nGetting devratio from models for comparison:\n');
for i = 1:numel(models_for_comparison)
    fprintf('%s\n',models_for_comparison{i});
    paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_output',models_for_comparison{i});
    devratio_this = [];
    for j = 1:numel(session_all)
        
        fit = load(fullfile(paths.results,session_all{j}));
        
        assert(issorted(fit.good_cells));
        
        cellID_this = cell(numel(fit.good_cells),1);
        for k = 1:numel(fit.good_cells)
            cellID_this{k} = sprintf('%s_c%d',fit.opt.session,fit.good_cells(k));
        end
        
        keep_cell = ismember(cellID_this,cellID_uniq);
        
        devratio_this = [devratio_this; fit.devratio(keep_cell)];
    end
    devratio_all(:,i) = devratio_this;
end

%% bar plot of % deviance expl per model

hfig = figure; hold on;
hfig.Name = 'bar plot of dev explained per model';
xlab = {'S','S+B','S+B+P','S+B+P+R','S+B+P+R+D'};
means = 100*mean(devratio_all);
sems = 100*std(devratio_all)/sqrt(size(devratio_all,1)); 
b=bar(means);
errorbar(1:numel(models_for_comparison)+1,means,sems,'k.');
xticks(1:numel(models_for_comparison)+1);
xticklabels(xlab);
xtickangle(45);
ylabel('% deviance explained');
text(0,7.5,'S = Session Time');
text(0,7,'B = Behavioral');
text(0,6.5,'P = Patch Stop Kernels');
text(0,6,'R = Reward Kernels');
text(0,5.5,'D = Decision Variables');
box off;
title(sprintf('n=%d GLM cells from %d mice',numel(cellID_uniq),numel(unique(mouse))));

saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

%% bar plot of % deviance expl per model, split by GMM cluster

hfig = figure('Position',[200 200 400 1000]);
hfig.Name = 'bar plot of dev explained per model by GMM cluster';
xlab = {'S','S+B','S+B+P','S+B+P+R','S+B+P+R+D'};
for i = 1:opt.num_clust
    subplot(opt.num_clust,1,i); hold on;
    devratio_this = devratio_all(clust_gmm==i,:);
    means = 100*mean(devratio_this);
    sems = 100*std(devratio_this)/sqrt(size(devratio_this,1)); 
    b=bar(means);
    errorbar(1:numel(models_for_comparison)+1,means,sems,'k.');
    xticks([]);
    ylim([0 10]);
    if i==ceil(opt.num_clust/2)
        ylabel('% deviance explained');
    end
    box off;
    title(sprintf('Cluster %d',i));
end
xticks(1:numel(models_for_comparison)+1);
xticklabels(xlab);
xtickangle(45);

saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');

%% bar plot of % deviance expl per model, split by mouse


mouse_uniq = unique(mouse);
mouse_all = cell(size(cellID_uniq));
for i = 1:numel(mouse_all)
    mouse_all{i} = cellID_uniq{i}(1:2);
end


hfig = figure('Position',[200 200 400 1000]);
hfig.Name = 'bar plot of dev explained per model by mouse';
xlab = {'S','S+B','S+B+P','S+B+P+R','S+B+P+R+D'};
for i = 1:numel(mouse_uniq)
    subplot(opt.num_clust,1,i); hold on;
    devratio_this = devratio_all(strcmp(mouse_all,mouse_uniq{i}),:);
    means = 100*mean(devratio_this);
    sems = 100*std(devratio_this)/sqrt(size(devratio_this,1)); 
    b=bar(means);
    errorbar(1:numel(models_for_comparison)+1,means,sems,'k.');
    xticks([]);
    ylim([0 10]);
    if i==ceil(opt.num_clust/2)
        ylabel('% deviance explained');
    end
    box off;
    title(sprintf('m%s',mouse_uniq{i}));
end
xticks(1:numel(models_for_comparison)+1);
xticklabels(xlab);
xtickangle(45);

saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf');