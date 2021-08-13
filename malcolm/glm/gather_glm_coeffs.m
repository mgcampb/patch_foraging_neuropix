%% script to gather all glm coefficients 
% MGC 8/9/2021

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

run_name = '20210526_full';

paths = struct;
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_output',run_name);
paths.figs = fullfile('C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients',run_name);
paths.waveforms = 'C:\data\patch_foraging_neuropix\waveforms\waveform_cluster';

paths.sig_cells = 'C:\data\patch_foraging_neuropix\sig_cells';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC'; % 'PFC' or 'Sub-PFC'
opt.data_set = 'mb';
opt.pval_thresh = 2; % 0.05;

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
beta_all = [];
cellID = [];
num_PFC_cells_total = 0;
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    num_PFC_cells_total = num_PFC_cells_total + sum(fit.anatomy.cell_labels.Cortex);
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    good_cells_this = fit.good_cells(keep_cell);

    beta_this = fit.beta_all(:,keep_cell);
    beta_all = [beta_all beta_this];
    sesh_all = [sesh_all; i*ones(numel(good_cells_this),1)];
        
    cellID_this = cell(numel(good_cells_this),1);
    for j = 1:numel(good_cells_this)
        cellID_this{j} = sprintf('%s_c%d',fit.opt.session,good_cells_this(j));
    end
    cellID = [cellID; cellID_this];

end
var_name = fit.var_name';

%% Remove intercept and session time terms
keep_var = ~strcmp(var_name,'Intercept') & ~contains(var_name,'SessionTime');
beta_all = beta_all(keep_var,:);
var_name = var_name(keep_var);

%%
beta_norm = zscore(beta_all);
[coeff,score,~,~,expl] = pca(beta_norm');
num_PCS = sum(cumsum(expl)<90)+1;
X = score(:,1:num_PCS);

%%
rng(3);
num_clust = 20;

% GMM opt
gmm_opt = statset;
gmm_opt.Display = 'off';
gmm_opt.MaxIter = 10000;
gmm_opt.Lambda = 0.4;
gmm_opt.Replicates = 100;
gmm_opt.TolFun = 1e-6;

bic_all = nan(num_clust,1);
for j = 1:num_clust
    gm = fitgmdist(X,j,'RegularizationValue',gmm_opt.Lambda,'Replicates',gmm_opt.Replicates,'Options',gmm_opt); % fit GM model
    % clust_gmm = cluster(gm,X); % hard clustering
    bic_all(j) = gm.BIC;
end

hfig = figure('Position',[300 300 320 400]);
hfig.Name = sprintf('BIC vs num clusters for n=%d PCs',num_PCS);
plot(bic_all,'ko-','MarkerFaceColor','k');
xlim([0 num_clust+1]);
box off;
xlabel('Num. clusters');
ylabel('BIC');

%%
num_clust = 4;
rng(3); % for reproducibility
gm = fitgmdist(X,num_clust,'RegularizationValue',gmm_opt.Lambda,'Replicates',gmm_opt.Replicates,'Options',gmm_opt); % fit GM model
clust_gmm = cluster(gm,X); % hard clustering


%% plot coefficients for GMM clusters

hfig = figure('Position',[300 300 800 1000],'Renderer','Painters');
hfig.Name = sprintf('GMM avg cluster coefficients incl base vars %s cohort %s',opt.data_set,opt.brain_region);

dotted_lines = [7.5 13.5 24.5 27.5 33.5 44.5 47.5 53.5 64.5];

tick_labels_this = var_name;
marker = 'o';
plot_col = cool(3);
for i = 1:num_clust
    
    subplot(num_clust,1,i); hold on;
    beta_this = beta_all(:,clust_gmm==i);
    mean_this = mean(beta_this,2);
    sem_this = std(beta_this,[],2)/sqrt(sum(clust_gmm==i));
    
    markersize = 4;
    plot_col_all = cool(3);
    plot_col = zeros(numel(var_name),3);
    for j = 1:3
        plot_col(contains(var_name,'1uL'),j) = plot_col_all(1,j);
        plot_col(contains(var_name,'2uL'),j) = plot_col_all(2,j);
        plot_col(contains(var_name,'4uL'),j) = plot_col_all(3,j);
    end
    for j = 1:numel(var_name)
        errorbar(j,mean_this(j),sem_this(j),'Color',plot_col(j,:));
    end
    for j = 1:numel(dotted_lines)
        plot([dotted_lines(j) dotted_lines(j)],ylim,'k--');
    end
    
    xticks(1:numel(tick_labels_this));
    set(gca,'TickDir','out');
    set(gca,'TickLabelInterpreter','none');
    if i == num_clust
        xticklabels(tick_labels_this);
        xtickangle(90);
    else
        xticklabels([]);
    end
    if i == ceil(num_clust/2)
        ylabel('GLM Coefficient');
    end
    % set(gca,'FontSize',14);

end