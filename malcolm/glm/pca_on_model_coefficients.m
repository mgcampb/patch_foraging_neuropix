addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

run_name = '20210526_full';

paths = struct;
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_output',run_name);
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210212_R_test\sessions';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210208_original_vars';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210210_model_comparison_new_glmnet';
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

opt.num_clust = 5; % for k means or GMM
opt.num_pcs = 10; % for k means or GMM

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
num_cells_glm_in_brain_region_total = 0;
num_cells_total = 0;
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    num_cells_glm_total = num_cells_glm_total + numel(fit.good_cells);
    num_cells_total = num_cells_total + numel(fit.good_cells_all);
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    good_cells_this = fit.good_cells(keep_cell);
    num_cells_glm_in_brain_region_total = num_cells_glm_in_brain_region_total+numel(good_cells_this);
    
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

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all_sig);
% beta_norm = beta_all_sig;
[coeff,score,~,~,expl] = pca(beta_norm');

%% visualize normalized beta matrix
hfig = figure('Position',[50 50 650 400],'Renderer','painters');
hfig.Name = sprintf('Normalized beta matrix significant cells %s cohort %s',opt.data_set,opt.brain_region);
imagesc(beta_norm);
yticks(1:numel(var_name));
yticklabels(var_name);
set(gca,'TickLabelInterpreter','none');
xlabel('Cell (pooled across sessions, mice)')
set(gca,'FontSize',8);
title(sprintf('%d %s cells from %d sessions in %d mice',size(beta_norm,2),opt.brain_region,numel(session_all),numel(uniq_mouse)));

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
hfig.Name = sprintf('Scree plot %s cohort %s',opt.data_set,opt.brain_region);
plot(cumsum(expl),'ko');
xlabel('PC');
ylabel('% var explained');
set(gca,'FontSize',14);

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% plot top 5 PCs
hfig = figure('Position',[50 50 1800 1200]);
hfig.Name = sprintf('Top 5 PCs %s cohort %s',opt.data_set,opt.brain_region);
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
hfig.Name = sprintf('Proj onto top PCs %s cohort %s',opt.data_set,opt.brain_region);

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

%% k means on top n PCs

rng(1);
kmeans_idx = kmeans(score(:,1:opt.num_pcs),opt.num_clust);
% kmeans_idx = kmeans(beta_all_sig',opt.num_clust);

% reorder cluster numbers to match MB dataset
if strcmp(opt.data_set,'mc') && strcmp(opt.brain_region,'PFC')
    kmeans_idx2 = kmeans_idx;
    kmeans_idx(kmeans_idx2==1) = 3;
    kmeans_idx(kmeans_idx2==2) = 1;
    kmeans_idx(kmeans_idx2==3) = 2;
end

hfig = figure; hold on;
hfig.Name = sprintf('k means on top 2 PCs %s cohort %s',opt.data_set,opt.brain_region);

plot_col = lines(opt.num_clust);
for i = 1:opt.num_clust
    my_scatter(score(kmeans_idx==i,1),score(kmeans_idx==i,2),plot_col(i,:),0.7);
end
xlabel('PC1');
ylabel('PC2');
set(gca,'FontSize',14);
title('k means');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% k means - 3D plot

rng(1);
kmeans_idx = kmeans(score(:,1:opt.num_pcs),opt.num_clust);
% kmeans_idx = kmeans(beta_all_sig',opt.num_clust);

% reorder cluster numbers to match MB dataset
if strcmp(opt.data_set,'mc') && strcmp(opt.brain_region,'PFC')
    kmeans_idx2 = kmeans_idx;
    kmeans_idx(kmeans_idx2==1) = 3;
    kmeans_idx(kmeans_idx2==2) = 1;
    kmeans_idx(kmeans_idx2==3) = 2;
end

hfig = figure; hold on;
hfig.Name = sprintf('k means on top 3 PCs %s cohort %s',opt.data_set,opt.brain_region);

plot_col = lines(opt.num_clust);
for i = 1:opt.num_clust
    scatter3(score(kmeans_idx==i,1),score(kmeans_idx==i,2),score(kmeans_idx==i,3),[],plot_col(i,:)) %,'MarkerFaceAlpha',0.7);
end
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
set(gca,'FontSize',14);
title('k means');
view([45 45]);
grid on;

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end

%% GMM clustering
sig_cells_table = load(fullfile(paths.sig_cells,'sig_cells_table_20210413.mat'));
clust_orig = sig_cells_table.sig_cells.GMM_cluster; % original GMM clusters

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
clust_orig = clust_gmm;

% crosstab(clust_gmm,clust_orig)

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
    scatter3(score(clust_orig==i,1),score(clust_orig==i,2),score(clust_orig==i,3),50,plot_col(i,:),'MarkerFaceColor',plot_col(i,:),'MarkerFaceAlpha',0.5);
end
legend(strcat('Clust',num2str((1:opt.num_clust)')));
xlabel('PC1'); ylabel('PC2'); zlabel('PC3');
view(11,33);
grid on;

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

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

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot coefficients for GMM clusters

hfig = figure('Position',[300 300 400 800]);
hfig.Name = sprintf('GMM avg cluster coefficients %s cohort %s',opt.data_set,opt.brain_region);

plot_idx = [1:14 1:14 1:14];
tick_labels_this = [strcat({'RewardKernel'},num2str((1:11)')); {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];
marker = 'o';
plot_col = cool(3);
for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_norm(:,clust_orig==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_orig==i));
    
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

%% Consistency of kmeans clusters

% num_iter = 1000;
% num_clust = 1:15;
% 
% pct_all = nan(numel(num_clust),1);
% D_all = nan(numel(num_clust),1);
% for ii = 1:numel(num_clust)
%     clust_all = nan(size(score,1),num_iter);
%     D_this = nan(num_iter,1);
%     for iter = 1:num_iter
%         [kmeans_idx,~,~,D] = kmeans(score(:,1:2),num_clust(ii));
% 
%         % reorder cluster numbers to be consistent across reps
%         % order by mean on PC1
%         means = nan(opt.num_clust,1);
%         for i = 1:opt.num_clust
%             means(i) = mean(score(kmeans_idx==i,1));
%         end
%         [~,sort_idx] = sort(means);
%         kmeans_idx2 = nan(size(kmeans_idx));
%         for i = 1:opt.num_clust
%             kmeans_idx2(kmeans_idx==sort_idx(i)) = i;
%         end
%         clust_all(:,iter) = kmeans_idx2;
%         D_this(iter) = mean(min(D,[],2));
%     end
% 
%     D_all(ii) = mean(D_this);
%     
%     pct_clust = nan(size(clust_all,1),1);
%     for i = 1:numel(pct_clust)
%         pct_clust(i) = sum(clust_all(i,:)==mode(clust_all(i,:)))/num_iter;
%     end
%     
%     pct_all(ii) = mean(pct_clust);
% end
% 
% % take second derivative
% secondDeriv = nan(numel(D_all)-2,1);
% for ii = 2:numel(D_all)-1
%     secondDeriv(ii-1) = D_all(ii+1)+D_all(ii-1)-2*D_all(ii);
% end
% [~,max_idx] = max(secondDeriv);
% max_idx = max_idx+1;
% 
% hfig = figure('Position',[300 300 900 400]);
% hfig.Name = sprintf('Picking num clusters %s cohort %s',opt.data_set,opt.brain_region);
% 
% subplot(1,2,1);
% plot(pct_all,'ko');
% ylabel('Avg. fraction in same cluster');
% xlabel('Num. clusters');
% set(gca,'FontSize',14);
% box off;
% 
% subplot(1,2,2); hold on;
% plot(D_all,'ko');
% p = plot([max_idx max_idx],ylim,'b--');
% legend(p,'Max. curvature');
% ylabel('Avg. distance to nearest centroid');
% xlabel('Num. clusters');
% set(gca,'FontSize',14);
% box off;
% 
% saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% plot example cell coefficients
session_ex = '80_20200317';
cellID_ex = 368;
fit = load(fullfile(paths.results,session_ex));
ex_cell_idx = find(fit.good_cells==cellID_ex);
coeff_this = fit.beta_all(2:end,ex_cell_idx);
var_name_this = fit.var_name(2:end);

plot_col = cool(3);

hfig = figure('Position',[200 200 1700 600]); hold on;
hfig.Name = sprintf('Example cell coefficients %s c%d',session_ex,cellID_ex);
for j = 1:numel(coeff_this)
    if contains(var_name_this{j},'1uL')
        plot_col_this = plot_col(1,:);
    elseif contains(var_name_this{j},'2uL')
        plot_col_this = plot_col(2,:);
    elseif contains(var_name_this{j},'4uL')
        plot_col_this = plot_col(3,:);
    else
        plot_col_this = 'k';
    end
    markersize = 7;
    plot_col_fill = plot_col_this;
    marker = 'o';
%     if contains(var_name_this{j},'PatchStop')
%         marker = 'v';
%     elseif contains(var_name_this{j},'RewKern')
%         marker = 'o';
%     elseif j<=7
%         marker = 's';
%     else
%         marker = 'p';
%     end
    plot(j,coeff_this(j),marker,'MarkerSize',markersize,'Color',plot_col_this,'MarkerFaceColor',plot_col_fill);

end
plot([9.5 9.5],[-0.15 0.15],'k-');
plot([15.5 15.5],[-0.15 0.15],'k-');
plot([26.5 26.5],[-0.15 0.15],'k-');
plot([29.5 29.5],[-0.15 0.15],'k-');
plot([35.5 35.5],[-0.15 0.15],'k-');
plot([46.5 46.5],[-0.15 0.15],'k-');
plot([49.5 49.5],[-0.15 0.15],'k-');
plot([55.5 55.5],[-0.15 0.15],'k-');
plot([66.5 66.5],[-0.15 0.15],'k-');
plot(xlim,[0 0],'k-');
grid on;
xticks(1:numel(var_name_this));
xticklabels(var_name_this);
xtickangle(90);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',14);
ylabel('Coeff');
title(sprintf('Example cell: %s c%d',session_ex,cellID_ex),'Interpreter','none');
set(gca,'FontSize',14);
set(gca,'TickDir','out');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot example cell coefficients v2

fit = load(fullfile(paths.results,session_ex));
ex_cell_idx = find(fit.good_cells==cellID_ex);
coeff_this = fit.beta_all(2:end,ex_cell_idx);
var_name_this = fit.var_name(2:end);

plot_col = cool(3);

hfig = figure('Position',[200 200 1700 600]); hold on;
hfig.Name = sprintf('Example cell coefficients %s c%d v2',session_ex,cellID_ex);
plot_idx = [1:9 10:29 10:29 10:29];
markersize = 7;
marker = 'o';
plot_col_this = 'k';
for j = 1:9
    plot_col_fill = plot_col_this;
    plot(plot_idx(j),coeff_this(j),marker,'MarkerSize',markersize,'Color',plot_col_this,'MarkerFaceColor',plot_col_fill);
end
markersize = 50;
for j = 1:3
    plot_col_this = plot_col(j,:);
    plot_col_fill = plot_col_this;
    add_on = 9+20*(j-1);
    for k = 1:20
        scatter(plot_idx(add_on+k),coeff_this(add_on+k),markersize,plot_col_this,'Marker',marker,'MarkerFaceColor',plot_col_fill,'MarkerFaceAlpha',0.5);
    end
    plot(plot_idx(add_on+(1:6)),coeff_this(add_on+(1:6)),'-','Color',plot_col_this);
    plot(plot_idx(add_on+(7:17)),coeff_this(add_on+(7:17)),'-','Color',plot_col_this);
end
plot([9.5 9.5],[-0.15 0.15],'k--');
plot([15.5 15.5],[-0.15 0.15],'k--');
plot([26.5 26.5],[-0.15 0.15],'k--');
xlim([0.5 29.5]);
plot(xlim,[0 0],'k--');
grid on;
xticks(1:numel(var_name_this));
xticklabels(var_name_this);
xtickangle(90);
set(gca,'TickLabelInterpreter','none');
set(gca,'FontSize',14);
ylabel('GLM Coefficient');
title(sprintf('Example cell: %s c%d',session_ex,cellID_ex),'Interpreter','none');
set(gca,'FontSize',14);
set(gca,'TickDir','out');

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% visualize normalized beta matrix sorted by cluster

[cluster_sorted,sort_idx] = sortrows([clust_orig score(:,1)]);
% [cluster_sorted,sort_idx] = sort(clust_orig);

hfig = figure('Position',[50 50 1300 800],'Renderer','painters');
hfig.Name = sprintf('Normalized beta matrix significant cells sorted by cluster %s cohort %s',opt.data_set,opt.brain_region);
imagesc(beta_norm(:,sort_idx));
yticks(1:numel(var_name));
yticklabels(var_name);
set(gca,'TickLabelInterpreter','none');
xlabel('Cell (pooled across sessions, mice)')
set(gca,'FontSize',14);
title(sprintf('%d %s cells from %d sessions in %d mice',size(beta_norm,2),opt.brain_region,numel(session_all),numel(uniq_mouse)));

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

%% bar plot of waveform type
waveform_table = load(fullfile(paths.waveforms,'waveform_cluster'));
waveform_table = waveform_table.waveform_cluster;
cell_waveform = cell(size(cellID_uniq));
for i = 1:numel(cell_waveform)
    cell_waveform{i} = waveform_table.WaveformType{strcmp(waveform_table.UniqueID,cellID_uniq{i})};
end
ctab = crosstab(clust_gmm,cell_waveform);
ctab = ctab(:,[2 3 1]);
hfig = figure('Position',[500 500 400 200]);
hfig.Name = 'gmm cluster vs waveform type bar plot';
b = bar(100*ctab./repmat(sum(ctab,2),1,3));
% for k = 1:size(ctab,2)
%     b(k).FaceColor = [0 0 0]+(k-1)*[0.5 0.5 0.5];
% end
xlabel('Cluster');
ylabel('%');
legend('RS','NS','T','Location','northeastoutside');
box off;
% figure;
% bar(100*ctab'./repmat(sum(ctab)',1,5));
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot all coeffs, including base variables, for each GMM cluster

hfig = figure('Position',[300 300 800 1000],'Renderer','Painters');
hfig.Name = sprintf('GMM avg cluster coefficients incl base vars %s cohort %s',opt.data_set,opt.brain_region);

plot_idx = [1:9 10:29 10:29 10:29];
tick_labels_this = [fit.var_name(2:10)';
    strcat({'PatchStopKernel'},num2str((1:6)')); ...
    strcat({'RewardKernel'},num2str((1:11)')); ...
    {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];
marker = 'o';
plot_col = cool(3);
for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_including_base_var(2:end,clust_orig==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_orig==i));
    
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

%% plot abs value of each coefficient normalized to sum

hfig = figure('Position',[200 200 800 1000],'Renderer','Painters');
hfig.Name = sprintf('GMM abs cluster coeff norm bar plot %s cohort %s',opt.data_set,opt.brain_region);

beta_abs = abs(beta_including_base_var);
beta_abs_norm = beta_abs./repmat(sum(beta_abs),size(beta_abs,1),1);

plot_idx = [1:9 10:29 10:29 10:29];
tick_labels_this = [fit.var_name(2:10)';
    strcat({'PatchStopKernel'},num2str((1:6)')); ...
    strcat({'RewardKernel'},num2str((1:11)')); ...
    {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];
for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_abs_norm(2:end,clust_orig==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_orig==i));
    
    plot_col_this = 'k';
    for j = 1:9
        plot_col_fill = plot_col_this;
        bar(plot_idx(j),100*mean_this(j),0.2,'FaceColor','k');
    end
    colormap('cool')
    for k = 1:20
        grp_this = [k k+20 k+40]+9;
        b = bar(plot_idx(grp_this(1)),100*mean_this(grp_this),'FaceColor','flat');
        for m = 1:numel(grp_this)
            b(m).CData = m;
        end
    end
    ylim([-7 7]);
    plot([9.5 9.5],[-7 7],'k--');
    plot([15.5 15.5],[-7 7],'k--');
    plot([26.5 26.5],[-7 7],'k--');
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
        ylabel('GLM Coefficient (% sum)');
    end
    % set(gca,'FontSize',14);

end

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% plot abs value of each coefficient normalized to sum, summing over kernels

hfig = figure('Position',[200 200 800 1000],'Renderer','Painters');
hfig.Name = sprintf('GMM abs cluster coeff norm summed over kernels %s cohort %s',opt.data_set,opt.brain_region);

beta_abs = abs(beta_including_base_var);
beta_abs_norm = beta_abs./repmat(sum(beta_abs),size(beta_abs,1),1);

var_name_this = fit.var_name(2:end);
tick_labels_this = [var_name_this(1:9)';
    {'PatchStopKernel'}; ...
    {'RewardKernel'}; ...
    {'TimeOnPatch'}; {'TotalReward'}; {'TimeSinceReward'}];

patchkern = contains(var_name_this,'PatchStopKern');
rewkern = contains(var_name_this,'RewKern');
dv = contains(var_name_this,'TimeOnPatch') | contains(var_name_this,'TotalRew') | contains(var_name_this,'TimeSinceRew');
behav = ~patchkern & ~rewkern & ~dv;
uL1 = contains(var_name_this,'1uL');
uL2 = contains(var_name_this,'2uL');
uL4 = contains(var_name_this,'4uL');

for i = 1:opt.num_clust
    
    subplot(opt.num_clust,1,i); hold on;
    X = beta_abs_norm(2:end,clust_orig==i);
    mean_this = mean(X,2);
    sem_this = std(X,[],2)/sqrt(sum(clust_orig==i));
    
    plot_idx = [1:9 10:29 10:29 10:29];
    plot_col_this = 'k';
    for j = 1:9
        plot_col_fill = plot_col_this;
        bar(plot_idx(j),100*mean_this(j),0.2,'FaceColor','k');
    end
    colormap('cool')
    
    % patch kern
    y = nan(3,1);
    y(1) = sum(mean_this(patchkern & uL1));
    y(2) = sum(mean_this(patchkern & uL2));
    y(3) = sum(mean_this(patchkern & uL4));
    b = bar(10,100*y,'FaceColor','flat');
    for m = 1:3
        b(m).CData = m;
    end
    
    % rew kern
    y = nan(3,1);
    y(1) = sum(mean_this(rewkern & uL1));
    y(2) = sum(mean_this(rewkern & uL2));
    y(3) = sum(mean_this(rewkern & uL4));
    b = bar(11,100*y,'FaceColor','flat');
    for m = 1:3
        b(m).CData = m;
    end
    
    % dv
    y = mean_this(contains(var_name_this,'TimeOnPatch'));
    b = bar(12,100*y,'FaceColor','flat');
    for m = 1:3
        b(m).CData = m;
    end
    
    y = mean_this(contains(var_name_this,'TotalRew'));
    b = bar(13,100*y,'FaceColor','flat');
    for m = 1:3
        b(m).CData = m;
    end
    
    y = mean_this(contains(var_name_this,'TimeSinceRew'));
    b = bar(14,100*y,'FaceColor','flat');
    for m = 1:3
        b(m).CData = m;
    end
    
    
    ylim([-7 7]);
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
        ylabel('% Total [100*abs(beta)/sum(abs(beta))]');
    end
    % set(gca,'FontSize',14);

end

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% compute model fit for R0 and RR trials (P, R, and D variables only)

% make "regressor" matrix for patch kernels, reward kernels, DVs for first
% 2 seconds of patch in RR and R0 trials

% options for this section
rew_size = [1 2 4];
num_patch_kern = 6;
num_rew_kern = 11;
num_dv = 3;
tplot = 0:0.02:2;

% make the regressor matrix for the first 2 seconds on path for RR and R0
% trials (X_R0 and X_RR)
X_R0 = nan(numel(tplot),num_patch_kern+num_rew_kern+num_dv);
X_RR = X_R0;
patch_stop_binary = zeros(numel(tplot),1);
patch_stop_binary(1) = 1;
rew_binary_R0 = zeros(numel(tplot),1);
rew_binary_R0(1) = 1;
rew_binary_RR = rew_binary_R0;
rew_binary_RR(ceil(numel(tplot)/2)) = 1;
for i = 1:num_patch_kern
    conv_this = conv(patch_stop_binary,fit.bas_patch_stop(i,:));
    X_R0(:,i) = conv_this(1:numel(tplot));
    X_RR(:,i) = conv_this(1:numel(tplot));
end
for i = 1:num_rew_kern
    conv_this = conv(rew_binary_R0,fit.bas_rew(i,:));
    X_R0(:,i+num_patch_kern) = conv_this(1:numel(tplot));
    conv_this = conv(rew_binary_RR,fit.bas_rew(i,:));
    X_RR(:,i+num_patch_kern) = conv_this(1:numel(tplot));
end
for tidx = ceil(numel(tplot)/2):numel(tplot)
    X_RR(tidx,num_patch_kern+ceil(num_rew_kern/2):end) = 0;
end
% decision variables:
X_R0(:,end-2) = tplot;
X_RR(:,end-2) = tplot;
X_R0(:,end-1) = 1;
X_RR(1:floor(numel(tplot)/2),end-1) = 1;
X_RR(ceil(numel(tplot)/2):end,end-1) = 2;
X_R0(:,end) = tplot;
X_RR(1:floor(numel(tplot)/2),end) = tplot(1:floor(numel(tplot)/2));
X_RR(ceil(numel(tplot)/2):end,end) = tplot(ceil(numel(tplot)/2):end)-1;

% variable to hold model predictions for R0 and RR trials based on P,R,D
% variables only
y_pred_R0 = nan(numel(cellID_uniq),numel(tplot),numel(rew_size));
y_pred_RR = y_pred_R0;

% make model predictions using only P,R,D variables
counter = 1;
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});   
    fit = load(fullfile(paths.results,session_all{i}));
    cell_idx_this = find(contains(cellID_uniq,session_all{i}));
    Ncells_this = numel(cell_idx_this);
    
    for j = 1:numel(rew_size)
        X_R0_this = X_R0;
        X_RR_this = X_RR;
        
        % z-score using same parameters as full session
        var_idx_this = find(contains(fit.var_name,sprintf('%duL',rew_size(j))))-1;
        mean_this = fit.Xmean(var_idx_this);
        std_this = fit.Xstd(var_idx_this);
        X_R0_this = (X_R0_this-repmat(mean_this,numel(tplot),1))./repmat(std_this,numel(tplot),1);
        X_RR_this = (X_RR_this-repmat(mean_this,numel(tplot),1))./repmat(std_this,numel(tplot),1);
        
        % get beta for the cells from this session
        beta_this = beta_including_base_var(var_idx_this+1,:);
        beta_this = beta_this(:,cell_idx_this);
        
        % make prediction of firing rate
        y_pred_R0(counter:counter+Ncells_this-1,:,j) = exp((X_R0_this*beta_this)');
        y_pred_RR(counter:counter+Ncells_this-1,:,j) = exp((X_RR_this*beta_this)');
    end
    
    counter = counter+Ncells_this;
end

%% make plot: PSTH of model fit for R0 and RR trials (P,R,D variables only)
plot_col = cool(3);
hfig = figure('Position',[200 200 800 450]);
hfig.Name = 'model prediction for R0 and RR trials - P R D variables - exp';
counter = 1;
for i = 1:numel(rew_size)
    for j = 1:opt.num_clust
        subplot(numel(rew_size),opt.num_clust,counter); hold on;
        mean_R0 = mean(y_pred_R0(clust_gmm==j,:,i));
        mean_RR = mean(y_pred_RR(clust_gmm==j,:,i));
        sem_R0 = std(y_pred_R0(clust_gmm==j,:,i))/sqrt(sum(clust_gmm==j));
        sem_RR = std(y_pred_RR(clust_gmm==j,:,i))/sqrt(sum(clust_gmm==j));
        shadedErrorBar(tplot,mean_R0,sem_R0,'lineprops','k');
        shadedErrorBar(tplot,mean_RR,sem_RR,'lineprops',{'Color',plot_col(i,:)});
        if j==1
            ylim([0.8 1.1]);
        elseif j==2
            ylim([0.95 1.25]);
        elseif j==4
            ylim([.9 3]);
        else
            ylim([0.8 1.2]);
        end
        if i~=3
            xticklabels([]);
        end
        plot(xlim,[1 1],'k--');
        plot([1 1],ylim,'k--');
        if i==2 && j==1
            ylabel('Model fit (P,R,D variables only)');
        end
        if i==3 && j==3
            xlabel('Time from patch stop (sec)');
        end
        if i==1
            title(sprintf('Cluster %d',j));
        end
        counter = counter+1;
    end
end

if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); end
if opt.save_figs; saveas(hfig,fullfile(paths.figs,hfig.Name),'pdf'); end

%% make sig cells table and save

mouse = cell(size(cellID_uniq));
cellID = nan(size(cellID_uniq));
for i = 1:numel(cellID_uniq)
    strsplit_this = strsplit(cellID_uniq{i},'_');
    mouse{i} = strsplit_this{1};
    cellID(i) = str2double(strsplit_this{3}(2:end));
end

sig_cells = table;
sig_cells.Mouse = mouse;
sig_cells.Session = session_all(sesh_all);
sig_cells.CellID = cellID;
sig_cells.UniqueID = cellID_uniq;
sig_cells.GMM_cluster = clust_gmm;
sig_cells.WaveformType = cell_waveform;
save(fullfile(paths.sig_cells,sprintf('sig_cells_table_%s',run_name)),'sig_cells');