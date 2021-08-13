% PCA on model coefficients from GLM fit to PCs

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

run_name = '20210626';

paths = struct;
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_on_PCs\',run_name);
paths.figs = fullfile('C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients_from_glm_on_PCs',run_name);
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

% analysis options
opt = struct;
opt.num_pcs = 7; % for dim reduction of coefficient matrix
opt.num_clust = 3; % for GMM clustering on dim reduced coeff matrix
opt.save_figs = true;

%% get sessions
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

% get mouse name for each session
mouse = cell(size(session_all));
for i = 1:numel(session_all)
    mouse{i} = session_all{i}(1:2);
end
uniq_mouse = unique(mouse);

%% concatenate model coefficients from each session
sesh_all = [];
beta_all = [];
beta_including_base_var = [];
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));

    
    beta_this = fit.beta_all(fit.base_var==0,:);
    beta_all = [beta_all beta_this];
    
    beta_this = fit.beta_all;
    beta_including_base_var = [beta_including_base_var beta_this];
    
    sesh_all = [sesh_all; i*ones(size(beta_this,2),1)];

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all);
% beta_norm = beta_all_sig;
[coeff,score,~,~,expl] = pca(beta_norm');

%% visualize normalized beta matrix
hfig = figure('Position',[50 50 650 400],'Renderer','painters');
hfig.Name = sprintf('Normalized beta matrix from GLM on PCs %s',run_name);
imagesc(beta_norm);
yticks(1:numel(var_name));
yticklabels(var_name);
set(gca,'TickLabelInterpreter','none');
xlabel('PC (pooled across sessions, mice)')
set(gca,'FontSize',8);
title(sprintf('%d PCs from %d sessions in %d mice',size(beta_norm,2),numel(session_all),numel(uniq_mouse)));

% plot colored bars indicating mouse
plot_col = lines(numel(uniq_mouse));
for i = 1:size(beta_norm,2)
    col_this = plot_col(strcmp(uniq_mouse,mouse(sesh_all(i))),:);
    patch([i-0.5 i+0.5 i+0.5 i-0.5],[0.5 0.5 -0.5 -0.5],col_this,'EdgeColor',col_this);
end
ylim([-0.5 max(ylim)]);
box off
hbar = colorbar;
ylabel(hbar,'z-score');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% scree plot
hfig = figure;
hfig.Name = sprintf('Scree plot %s',run_name);
plot(cumsum(expl),'ko');
xlabel('PC');
ylabel('% var explained');
set(gca,'FontSize',14);

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% plot top 5 PCs
hfig = figure('Position',[50 50 1800 1200]);
hfig.Name = sprintf('Top 5 PCs %s',run_name);
plot_col = cool(3);
for i = 1:5
    subplot(5,1,i); hold on;
    
    for j = 1:size(coeff,1)
        if contains(var_name{j},'1uL')
            plot_col_this = plot_col(1,:);
        elseif contains(var_name{j},'2uL')
            plot_col_this = plot_col(2,:);
        elseif contains(var_name{j},'4uL')
            plot_col_this = plot_col(3,:);
        end
        if contains(var_name{j},'RewKern')
            plot(j,coeff(j,i),'o','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
        else
            plot(j,coeff(j,i),'v','Color',plot_col_this,'MarkerFaceColor',plot_col_this);
        end
    end
    plot(xlim,[0 0],'k:');
    xticks([]);
    ylabel('Coeff');
    title(sprintf('PC%d: %0.1f %% variance',i,expl(i)));
    set(gca,'FontSize',14);
end
xticks(1:numel(var_name));
xticklabels(var_name);
xtickangle(90);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',14);

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% plot proj onto top PCs

hfig = figure('Position',[50 50 2000 700]);
hfig.Name = sprintf('Proj onto top PCs %s',run_name);

% PCs 1 and 2
subplot(1,3,1); hold on;
plot_col = lines(numel(uniq_mouse));
for i = 1:numel(uniq_mouse)
    keep = strcmp(mouse(sesh_all),uniq_mouse{i});
    my_scatter(score(keep,1),score(keep,2),plot_col(i,:),0.7);
end
xlabel('PC1'); 
ylabel('PC2');
set(gca,'FontSize',14);
axis square

% PCs 1 and 2
subplot(1,3,2); hold on;
plot_col = lines(numel(uniq_mouse));
for i = 1:numel(uniq_mouse)
    keep = strcmp(mouse(sesh_all),uniq_mouse{i});
    my_scatter(score(keep,1),score(keep,3),plot_col(i,:),0.7);
end
xlabel('PC1'); 
ylabel('PC3');
set(gca,'FontSize',14);
axis square

subplot(1,3,3); hold on;
plot_col = lines(numel(uniq_mouse));
for i = 1:numel(uniq_mouse)
    keep = strcmp(mouse(sesh_all),uniq_mouse{i});
    my_scatter(score(keep,2),score(keep,3),plot_col(i,:),0.7);
end
xlabel('PC2'); 
ylabel('PC3');
set(gca,'FontSize',14);
axis square

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% Proj onto top 2 PCs from single cell analysis

coeff_single_cell = load('coeff_from_single_cell_analysis.mat');
proj = coeff_single_cell.coeff'*beta_norm;
hfig = figure;
hfig.Name = sprintf('PCs projected onto top two PCs of GLM coefficients from single cell analysis %s',run_name);
hold on;
my_scatter(coeff_single_cell.score(:,1),coeff_single_cell.score(:,2),'k',0.2);
my_scatter(proj(:,1),proj(:,2),'r',0.7);
legend('Cells','PCs');
xlabel('PC1 (from cell analysis)');
ylabel('PC2 (from cell analysis)');
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% GMM clustering

X = score(:,1:opt.num_pcs); % the first N PCs of z-scored coefficient matrix

rng(3); % for reproducibility
gmm_opt = statset; % GMM options
gmm_opt.Display = 'off';
gmm_opt.MaxIter = 10000;
gmm_opt.Lambda = .2;
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

%% GMM clustering - justification of num PCs, num clusters

rng(3);
num_clust = 10;
bic_all = nan(num_clust,1);
gmm_opt.Replicates = 100;
for j = 1:num_clust
    gm = fitgmdist(X,j,'RegularizationValue',gmm_opt.Lambda,'Replicates',gmm_opt.Replicates,'Options',gmm_opt); % fit GM model
    % clust_gmm = cluster(gm,X); % hard clustering
    bic_all(j) = gm.BIC;
end

hfig = figure('Position',[300 300 320 400]);
hfig.Name = sprintf('BIC vs num clusters for n=%d PCs',opt.num_pcs);
plot(bic_all,'ko-','MarkerFaceColor','k');
xlim([0 num_clust+1]);
box off;
xlabel('Num. clusters');
ylabel('BIC');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot first 3 PCs, colored by GMM cluster

hfig = figure('Renderer','painters'); hold on;
hfig.Name = 'PCs 1-3 colored by GMM cluster';
% plot_col = cbrewer('qual', 'Set2', opt.num_clust);
plot_col = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255; 
for i = 1:opt.num_clust
    scatter3(score(clust_gmm==i,1),score(clust_gmm==i,2),score(clust_gmm==i,3),50,plot_col(i,:),'MarkerFaceColor',plot_col(i,:),'MarkerFaceAlpha',0.5);
end
legend(strcat('Clust',num2str((1:opt.num_clust)')));
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
view(11,33);
grid on;

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot coefficients for GMM clusters

hfig = figure('Position',[300 300 400 800]);
hfig.Name = sprintf('GMM avg cluster coefficients %s',run_name);

plot_idx = [1:14 1:14 1:14];
tick_labels_this = [strcat({'RewardKernel'},num2str((1:11)')); {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];
marker = 'o';
plot_col = cool(3);
for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_norm(:,clust_gmm==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_gmm==i));
    
    markersize = 50;
    for j = 1:3
        plot_col_this = plot_col(j,:);
        plot_col_fill = plot_col_this;
        add_on = 14*(j-1);
        shadedErrorBar(plot_idx(add_on+(1:11)),mean_this(add_on+(1:11)),sem_this(add_on+(1:11)),'lineprops',{'-','Color',plot_col_this});
        errorbar(plot_idx(add_on+(12:14)),mean_this(add_on+(12:14)),sem_this(add_on+(12:14)),'.','Color',plot_col_this);
    end
    xlim([0.5 14.5]);
    plot(xlim,[0 0],'k--');    
    grid on;
    xticks(1:14)
    xticklabels([]);
    if i == ceil(opt.num_clust/2)
        ylabel('GLM coefficient (z-score)');
    end
    if i == opt.num_clust
        xticklabels(tick_labels_this);
        xtickangle(90);
    end
    set(gca,'TickLabelInterpreter','none');
    set(gca,'TickDir','out');
    %set(gca,'FontSize',14);
    %title(sprintf('Cluster %d',i));
    plot([11.5 11.5],ylim,'k--');

end

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% visualize normalized beta matrix sorted by cluster

[cluster_sorted,sort_idx] = sortrows([clust_gmm score(:,1)]);
% [cluster_sorted,sort_idx] = sort(clust_gmm);

hfig = figure('Position',[50 50 1300 800],'Renderer','painters');
hfig.Name = sprintf('Normalized beta matrix significant cells sorted by cluster %s',run_name);
imagesc(beta_norm(:,sort_idx));
yticks(1:numel(var_name));
yticklabels(var_name);
set(gca,'TickLabelInterpreter','none');
xlabel('Cell (pooled across sessions, mice)')
set(gca,'FontSize',14);
title(sprintf('%d PCs from %d sessions in %d mice',size(beta_norm,2),numel(session_all),numel(uniq_mouse)));

% plot colored bars indicating cluster
plot_col = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255; 
for i = 1:size(beta_norm,2)
    col_this = plot_col(cluster_sorted(i),:);
    patch([i-0.5 i+0.5 i+0.5 i-0.5],[0.5 0.5 -0.5 -0.5],col_this,'EdgeColor',col_this);
end
% % plot colored bars indicating mouse
% plot_col = lines(numel(uniq_mouse));
% for i = 1:size(beta_norm,2)
%     col_this = plot_col(strcmp(uniq_mouse,mouse(sesh_all(sort_idx(i)))),:);
%     patch([i-0.5 i+0.5 i+0.5 i-0.5],[0.5 0.5 -0.5 -0.5],col_this,'EdgeColor',col_this);
% end

ylim([-0.5 max(ylim)]);
box off
hbar = colorbar;
ylabel(hbar,'z-score');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot all coeffs, including base variables, for each GMM cluster

hfig = figure('Position',[300 300 800 1000],'Renderer','Painters');
hfig.Name = sprintf('GMM avg cluster coefficients incl base vars %s',run_name);

plot_idx = [1:9 10:29 10:29 10:29];
tick_labels_this = [fit.var_name(2:10)';
    strcat({'PatchStopKernel'},num2str((1:6)')); ...
    strcat({'RewardKernel'},num2str((1:11)')); ...
    {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];
marker = 'o';
plot_col = cool(3);
for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_including_base_var(2:end,clust_gmm==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_gmm==i));
    
    markersize = 4;
    plot_col_this = 'k';
    for j = 1:9
        plot_col_fill = plot_col_this;
        errorbar(plot_idx(j),mean_this(j),sem_this(j),marker,'MarkerSize',markersize,'Color',plot_col_this,'MarkerFaceColor',plot_col_fill);
    end
    markersize = 4;
    for j = 1:3
        plot_col_this = plot_col(j,:);
        plot_col_fill = plot_col_this;
        add_on = 9+20*(j-1);
        for k = 1:20
            errorbar(plot_idx(add_on+k),mean_this(add_on+k),sem_this(add_on+k),marker,'MarkerSize',markersize,'Color',plot_col_this,'MarkerFaceColor',plot_col_fill);
        end
        plot(plot_idx(add_on+(1:6)),mean_this(add_on+(1:6)),'-','Color',plot_col_this);
        plot(plot_idx(add_on+(7:17)),mean_this(add_on+(7:17)),'-','Color',plot_col_this);
    end
    plot([9.5 9.5],[-0.2 0.2],'k--');
    plot([15.5 15.5],[-0.2 0.2],'k--');
    plot([26.5 26.5],[-0.2 0.2],'k--');
    xlim([0.5 29.5]);
    plot(xlim,[0 0],'k--');
    grid on;
    
    xticks(1:numel(tick_labels_this));
    set(gca,'TickDir','out');
    set(gca,'TickLabelInterpreter','none');
    if i == opt.num_clust
        xticklabels(tick_labels_this);
        xtickangle(90);
    else
        xticklabels([]);
    end
    if i == ceil(opt.num_clust/2)
        ylabel('GLM Coefficient');
    end
    % set(gca,'FontSize',14);

end

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end