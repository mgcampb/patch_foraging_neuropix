addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mc';
opt.pval_thresh = 0.05;

opt.num_clust = 3; % for k means

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
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    
    sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    
    beta_this = fit.beta_all(fit.base_var==0,keep_cell);
    beta_all_sig = [beta_all_sig beta_this(:,sig)];
    sesh_all = [sesh_all; i*ones(sum(sig),1)];

end
var_name = fit.var_name(fit.base_var==0)';

%% perform PCA
beta_norm = zscore(beta_all_sig);
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
kmeans_idx = kmeans(score(:,1:2),opt.num_clust);

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

%% plot coefficients for kmeans clusters

hfig = figure('Position',[50 50 1500 900]);
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

    