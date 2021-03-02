addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210212_R_test\sessions';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210208_original_vars';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210210_model_comparison_new_glmnet';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients\run_20210212_R_test\no_pvalue_cutoff';
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
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
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

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all_sig);
% beta_norm = beta_all_sig;
[coeff,score,~,~,expl] = pca(beta_norm');

%% visualize normalized beta matrix
hfig = figure('Position',[50 50 1300 800]);
hfig.Name = sprintf('Normalized beta matrix significant cells %s cohort %s',opt.data_set,opt.brain_region);
imagesc(beta_norm);
yticks(1:numel(var_name));
yticklabels(var_name);
set(gca,'TickLabelInterpreter','none');
xlabel('Cell (pooled across sessions, mice)')
set(gca,'FontSize',14);
title(sprintf('%d %s cells from %d sessions in %d mice',size(beta_norm,2),opt.brain_region,numel(session_all),numel(uniq_mouse)));

% plot colored bars indicating mouse
plot_col = lines(numel(uniq_mouse));
for i = 1:size(beta_norm,2)
    col_this = plot_col(strcmp(uniq_mouse,mouse(sesh_all(i))),:);
    patch([i-0.5 i+0.5 i+0.5 i-0.5],[0.5 0.5 -0.5 -0.5],col_this,'EdgeColor',col_this);
end
ylim([-0.5 max(ylim)]);
box off
colorbar
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% scree plot
hfig = figure;
hfig.Name = sprintf('Scree plot %s cohort %s',opt.data_set,opt.brain_region);
plot(expl,'ko');
xlabel('PC');
ylabel('% var explained');
set(gca,'FontSize',14);
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

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
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

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

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% k means on top 2 PCs

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

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% k means on top 2 PCs

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

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

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

%% Consistency of kmeans clusters
% 
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
