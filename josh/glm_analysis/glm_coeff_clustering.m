%% Script to shore up clustering of cells based on GLM coefficients 

% addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
% paths.results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
% 
% paths = struct;
% % paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
% paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210212_R_test\sessions';
% % paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210208_original_vars';
% % paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210210_model_comparison_new_glmnet';
% paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients\run_20210212_R_test\no_pvalue_cutoff';
% if ~isfolder(paths.figs)
%     mkdir(paths.figs);
% end

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
pval_vs_base = []; 
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
    pval_vs_base = [pval_vs_base (fit.pval_full_vs_base(sig))']; 
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
variance_threshold = 75; 
hfig = figure;
hfig.Name = sprintf('Scree plot %s cohort %s',opt.data_set,opt.brain_region); 
subplot(1,2,1)
plot(expl,'ko');
xlabel('PC');
ylabel('% var explained'); 
title("Var expl per PC")
set(gca,'FontSize',14); 
subplot(1,2,2)   
plot(cumsum(expl),'ko');
xline(find(cumsum(expl) > variance_threshold,1),'k--','linewidth',1.5)
xlabel('PC','FontSize',14);
title(sprintf('Cumulative var explained \n (k = %i to surpass %i pct)',find(cumsum(expl) > variance_threshold,1),variance_threshold))
set(gca,'FontSize',13); 
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% plot top k PCs
hfig = figure('Position',[50 50 1800 1200]);
hfig.Name = sprintf('Top 5 PCs %s cohort %s',opt.data_set,opt.brain_region);
plot_col = cool(3); 
vis_pcs = 1:5;  
for i_i = 1:numel(vis_pcs) 
    i = vis_pcs(i_i); 
    subplot(numel(vis_pcs),1,i_i);hold on
    
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

%% Visualize projections onto PC space, color by mouse

% NOTE NEGATIVE CORRELATION BETWEEN some score of PC1 and PC2!!!!! THIS IS
% IMPORTANT
% PC1 and PC4 seem to have some positive correlation
% anticorrelation between abs(PC1) and abs(PC3) 
% Pos corr between PC3,5,7

hfig = figure(); % 'Position',[50 50 2000 700]);
hfig.Name = sprintf('Proj onto top PCs %s cohort %s',opt.data_set,opt.brain_region);

num_pcs = 5; 
for iPC = 2:num_pcs
    % scatter PCs 1 and iPC, color by mouse  
    if num_pcs > 6
        subplot(2,round(num_pcs/2),iPC-1); hold on; 
    elseif num_pcs == 5 
        subplot(2,2,iPC-1)
    else  
        subplot(1,num_pcs-1,iPC-1); hold on; 
        
    end  

    gscatter(score(:,1),score(:,iPC),mouse(sesh_all),lines(5))
%     scatterhist(score(:,1),score(:,iPC),'Group',mouse(sesh_all),'color',lines(5),'Kernel','on')
    xlabel('PC1'); 
    ylabel(sprintf('PC%i',iPC));
    set(gca,'FontSize',14);
%     axis square   
    if iPC ~= 1
        hLeg = legend();
        set(hLeg,'visible','off')
    end
end 
legend(arrayfun(@(iMouse) sprintf("m%s",uniq_mouse{iMouse}),(1:numel(uniq_mouse))),'FontSize',14)
suptitle("Projection onto Coefficient PC Space")

%% Make k on top l PCs, visualize clustering across PC dimensions
opt.num_clust = 5;
opt.num_pcs = 3;
rng(1);
kmeans_idx = kmeans(score(:,1:opt.num_pcs),opt.num_clust); 

num_vis_pcs = 10; 

figure(); hold on
% hfig.Name = sprintf('k means on top 2 PCs %s cohort %s',opt.data_set,opt.brain_region);
for iPC = 2:num_vis_pcs
    if num_vis_pcs > 6
        subplot(2,round(num_vis_pcs/2),iPC-1); hold on; 
    else  
        subplot(1,num_vis_pcs-1,iPC-1); hold on; 
    end
    gscatter(score(:,1),score(:,iPC),kmeans_idx,lines(opt.num_clust),[],10);
    xlabel('PC1');
    ylabel(sprintf('PC%i',iPC));
    set(gca,'FontSize',14);
    if iPC ~= 1
        hLeg = legend();
        set(hLeg,'visible','off')
    end
end 
suptitle("Clusters across PC dimensions")

% saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%% plot coefficients fors clusters
vis_cluster_mean_beta(opt,beta_norm,var_name,kmeans_idx)

%% Consistency of kmeans clusters

num_iter = 100;
num_clust = 1:15;
num_pcs = 1:5;

hfig = figure('Position',[300 300 900 400]);
hfig.Name = sprintf('Picking num clusters %s cohort %s',opt.data_set,opt.brain_region);

pct_all = nan(numel(num_clust),1);
D_all = nan(numel(num_clust),1);
for i_pcs = 1:numel(num_pcs)
    for ii = 1:numel(num_clust)
        clust_all = nan(size(score,1),num_iter);
        D_this = nan(num_iter,1);
        for iter = 1:num_iter
            [kmeans_idx,~,~,D] = kmeans(score(:,1:i_pcs),num_clust(ii));
            % reorder cluster numbers to be consistent across reps
            % order by mean on PC1
            means = nan(opt.num_clust,1);
            for i = 1:opt.num_clust
                means(i) = mean(score(kmeans_idx==i,1));
            end
            [~,sort_idx] = sort(means);
            kmeans_idx2 = nan(size(kmeans_idx));
            for i = 1:opt.num_clust
                kmeans_idx2(kmeans_idx==sort_idx(i)) = i;
            end
            clust_all(:,iter) = kmeans_idx2;
            D_this(iter) = mean(min(D,[],2));
        end
        
        D_all(ii) = mean(D_this);
        
        pct_clust = nan(size(clust_all,1),1);
        for i = 1:numel(pct_clust)
            pct_clust(i) = sum(clust_all(i,:)==mode(clust_all(i,:)))/num_iter;
        end
        
        pct_all(ii) = mean(pct_clust);
    end
    
    % take second derivative
    secondDeriv = nan(numel(D_all)-2,1);
    for ii = 2:numel(D_all)-1
        secondDeriv(ii-1) = D_all(ii+1)+D_all(ii-1)-2*D_all(ii);
    end
    [~,max_idx] = max(secondDeriv);
    max_idx = max_idx+1;
    
    subplot(2,max(num_pcs),i_pcs);
    plot(pct_all,'ko');
    ylabel('Avg. fraction in same cluster');
    xlabel('Num. clusters');
    set(gca,'FontSize',14);
    box off; 
    title(sprintf("%i PCs",i_pcs))
    
    subplot(2,max(num_pcs),max(num_pcs) + i_pcs); hold on;
    plot(D_all,'ko');
    p = plot([max_idx max_idx],ylim,'b--');
    if i_pcs == 1
        legend(p,'Max. curvature');
    end
    ylabel('Avg. distance to nearest centroid');
    xlabel('Num. clusters');
    set(gca,'FontSize',14);
    box off;
    
    fprintf("%i PCs Clustering Analysis Complete \n",i_pcs)
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png'); 

%% Analyze dendrogram given different numbers of PCs 
% this doesn't work that well
max_pcs = 5; 
figure()
for n_pcs = 1:max_pcs  
    subplot(1,max_pcs,n_pcs)
    Y = pdist(score(:,1:n_pcs),'mahalanobis');
    Z = linkage(Y);  
    dendrogram(Z)
    xticks([])
end

%% Fit GMMs and visualize discovered clusters
%  back to just one num_clust and one num_pcs 

% specify model options
opt.num_clust = 5;
opt.num_pcs = 3;
gmm_opt.lambda = 1; 
gmm_opt.replicates = 100; 
options = statset('MaxIter',1000);
rng(20)
% fit model
GMM = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
                                                     'replicates',gmm_opt.replicates,'options',options);
gmm_idx = cluster(GMM,score(:,1:opt.num_pcs)); 

% PC1-sorted alignment
[~,pc1_order] = sort(GMM.mu(:,1));
aligned_gmm_idx = nan(size(gmm_idx));
for i_clust = 1:opt.num_clust
    aligned_gmm_idx(gmm_idx == pc1_order(i_clust)) = i_clust;
end

% Visualize gmm results
num_vis_pcs = 5; 
cluster_gscatter(num_vis_pcs,score,aligned_gmm_idx,opt.num_clust)

% plot coefficients for clusters
vis_cluster_mean_beta(opt,beta_norm,var_name,aligned_gmm_idx)  

%% Plot p-value full vs base per cluster 
% figure()
% for i_cluster = 1:opt.num_clust
%     subplot(1,opt.num_clust,i_cluster) 
%     histogram(pval_vs_base(gmm_idx == i_cluster),0:.1:1) 
%     xlim([0 1])   
%     title(sprintf("Cluster %i",i_cluster))
%     set(gca,'Fontsize',13)
% end 
% suptitle(sprintf("Distribution of pval full vs base \n Divided By Cluster"))
% 
% figure() ;hold on
% for i_cluster = 1:opt.num_clust
%     i_pdf = pdf(fitdist(pval_vs_base(gmm_idx == i_cluster)','kernel','Kernel','Normal'),0:.01:1); 
%     plot(0:.01:1,i_pdf,'linewidth',2)
%     xlim([0 1])  
% end 
% legend(arrayfun(@(x) sprintf("Cluster %i",x),(1:opt.num_clust)))
% title(sprintf("Distribution of pval full vs base \n Divided By Cluster"))
% ylabel("KDE by Cluster")
% set(gca,'Fontsize',14)

figure() ;hold on 
% x = -.4:.005:.4;
x = -.4:.05:.4;
var_idx = 35; 
for i_cluster = 1:opt.num_clust
%     i_pdf = pdf(fitdist(beta_all_sig(var_idx,gmm_idx == i_cluster)','kernel','Kernel','Normal'),x); 
%     plot(x,i_pdf,'linewidth',2)
    subplot(1,opt.num_clust,i_cluster) 
    histogram(beta_all_sig(var_idx,gmm_idx == i_cluster),x) 
    title(sprintf("Cluster %i",i_cluster))
    xlim([min(x)-.1 max(x)+.1])  
end  
xlabel("Beta (Un-Normalized)") 
ylabel("KDE by Cluster")
% legend(arrayfun(@(x) sprintf("Cluster %i",x),(1:opt.num_clust)))
suptitle(sprintf("Distribution of %s \n Divided By Cluster",var_name{var_idx})) 
% set(gca,'Fontsize',14)

%% Scatterhist to look at distribution of marginals 
num_pcs = 10; 
figure
% for iPC = 2
hp1 = uipanel('position',[0 .5 1 .5]);
hp2 = uipanel('position',[0 0 1 .5]);
% scatter PCs 1 and iPC, color by mouse
if num_pcs > 6
    subplot(2,round(num_pcs/2),iPC-1); hold on;
else
    subplot(1,num_pcs-1,iPC-1); hold on;
end
scatterhist(score(:,1),score(:,iPC),'Group',gmm_idx,'color',lines(5),'Kernel','on','Parent',hp1)
xlabel('PC1');
ylabel('PC2');
% set(gca,'FontSize',14);
% axis square

axes('Parent',hp2);
scatterhist(score(:,1),score(:,3),'Group',gmm_idx,'color',lines(5),'Kernel','on','Parent',hp2)
xlabel('PC1');
ylabel('PC3');

if iPC ~= 1
    hLeg = legend();
    set(hLeg,'visible','off')
end

%% Fit GMM and color component probability

% specify model options
opt.num_clust = 4;
opt.num_pcs = 3;
gmm_opt.lambda = .5; 
gmm_opt.replicates = 50; 
options = statset('MaxIter',1000);

% fit model
GMM = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
                                                     'replicates',gmm_opt.replicates,'options',options);
[~,~,~,~,P] = cluster(GMM,score(:,1:opt.num_pcs)); 

figure(); hold on
% hfig.Name = sprintf('k means on top 2 PCs %s cohort %s',opt.data_set,opt.brain_region);
for this_cluster = 1:opt.num_clust
    subplot(opt.num_clust,2,2 * (this_cluster-1) + 1); hold on; 
    scatter(score(:,1),score(:,2),[],P(:,this_cluster),'.');
    xlabel('PC1');
    ylabel('PC2');
    subplot(opt.num_clust,2,2 * (this_cluster-1) + 2); hold on; 
    scatter(score(:,1),score(:,3),[],P(:,this_cluster),'.');
    xlabel('PC1');
    ylabel('PC3');
%     set(gca,'FontSize',14);
end 

%% Assess # clusters using BIC, AIC, and mahalnobis distance
opt.num_clust = 1:10;
opt.num_pcs = 1:10;
GMModels = cell(numel(opt.num_pcs),numel(opt.num_clust)); % Preallocation  
AIC = nan(numel(opt.num_pcs),numel(opt.num_clust));  
BIC = nan(numel(opt.num_pcs),numel(opt.num_clust)); 
median_clust_d2 = nan(numel(opt.num_pcs),numel(opt.num_clust)); 
mean_clust_d2 = nan(numel(opt.num_pcs),numel(opt.num_clust)); 
median_clust_l2 = nan(numel(opt.num_pcs),numel(opt.num_clust)); 
mean_clust_l2 = nan(numel(opt.num_pcs),numel(opt.num_clust)); 
gmm_opt.lambda = 1; 
gmm_opt.replicates = 50;
options = statset('MaxIter',1000);
rng(3);  
for n_num_pcs = opt.num_pcs
    for n_clusts = opt.num_clust  
        GMModels{n_num_pcs,n_clusts} = fitgmdist(score(:,1:n_num_pcs),n_clusts,'RegularizationValue',gmm_opt.lambda,...
                                                     'replicates',gmm_opt.replicates,'options',options);                          
        % Mahalanobis distance to closest                                          
        [i_gmm_idx,~,~,~,d2] = cluster(GMModels{n_num_pcs,n_clusts},score(:,1:n_num_pcs)); 
        median_clust_d2(n_num_pcs,n_clusts) = median(min(d2,[],2));
        mean_clust_d2(n_num_pcs,n_clusts) = mean(min(d2,[],2));
        
        mean_l2_tmp = nan(n_clusts,1); 
        median_l2_tmp = nan(n_clusts,1); 
        for i_clust = 1:n_clusts
            mean_l2_tmp(i_clust) = nanmean(squareform(pdist(score(i_gmm_idx == i_clust,1:opt.num_pcs))),'all');
            median_l2_tmp(i_clust) = nanmedian(squareform(pdist(score(i_gmm_idx == i_clust,1:opt.num_pcs))),'all');
        end 
        median_clust_l2(n_num_pcs,n_clusts) = nanmean(mean_l2_tmp); 
        mean_clust_l2(n_num_pcs,n_clusts) = nanmedian(median_l2_tmp); 
        
        AIC(n_num_pcs,n_clusts) = GMModels{n_num_pcs,n_clusts}.AIC;  
        BIC(n_num_pcs,n_clusts) = GMModels{n_num_pcs,n_clusts}.BIC;  
        if ~GMModels{n_num_pcs,n_clusts}.Converged 
            fprintf("%i PCs %i Clusters GMM Did Not Converge! \n",n_num_pcs,n_clusts)
        end
    end  
    fprintf("%i PCs Complete \n",n_num_pcs)
end

%% Visualize BIC and AIC across number of clusters and number of PCs

[~,min_AIC_ix] = min(AIC,[],2);
[~,min_BIC_ix] = min(BIC,[],2);

figure() 
subplot(1,2,1);hold on
cmap = copper(length(opt.num_pcs)); 
colororder(copper(length(opt.num_pcs)))
plot(((BIC' - min(BIC'))./max((BIC' - min(BIC')))),'linewidth',1.5) 
ylabel("BIC (normalized)") 
xlabel("# Clusters")  
set(gca,'Fontsize',14)
subplot(1,2,2);hold on
cmap = copper(length(opt.num_pcs));
for i_pcs = 1:(max(opt.num_pcs)-1)
    plot([i_pcs i_pcs+1],min_BIC_ix(i_pcs:i_pcs+1),'linewidth',3,'color',cmap(i_pcs,:))
end 
ylim([0 10])
legend(arrayfun(@(x) sprintf("%i PCs",x),1:max(opt.num_pcs)))
set(gca,'Fontsize',14) 
xlabel("# PCs Clustered")

% 
% figure(); hold on 
% colororder(copper(length(opt.num_pcs)))
% subplot(3,2,1)
% plot(((AIC' - min(AIC'))./max((AIC' - min(AIC')))),'linewidth',1.5);  
% ylabel("AIC (normalized)")
% xlabel("# Clusters") 
% xlim([0 max(opt.num_pcs)+1])
% set(gca,'FontSize',14);
% subplot(3,2,2)
% plot(((BIC' - min(BIC'))./max((BIC' - min(BIC')))),'linewidth',1.5) 
% ylabel("BIC (normalized)") 
% xlabel("# Clusters") 
% xlim([0 max(opt.num_pcs)+1])
% legend(arrayfun(@(x) sprintf("%i PCs",x),1:max(opt.num_pcs)))
% set(gca,'FontSize',14); 
% subplot(3,2,3) 
% histogram(min_AIC_ix,opt.num_pcs) 
% xlim([0 max(opt.num_pcs)+1])
% ylim([0 max(histcounts(min_AIC_ix,opt.num_pcs)) + 1])
% ylabel("Min AIC count over #PCs")
% set(gca,'FontSize',14);
% xlabel("# Clusters") 
% subplot(3,2,4) 
% histogram(min_BIC_ix,opt.num_pcs) 
% xlim([0 max(opt.num_pcs)+1]) 
% ylim([0 max(histcounts(min_BIC_ix,opt.num_pcs)) + 1]) 
% ylabel("Min BIC count over #PCs")
% set(gca,'FontSize',14);
% xlabel("# Clusters") 
% subplot(3,2,5);hold on
% cmap = copper(length(opt.num_pcs));
% for i_pcs = 1:(max(opt.num_pcs)-1)
%     plot([i_pcs i_pcs+1],min_AIC_ix(i_pcs:i_pcs+1),'linewidth',3,'color',cmap(i_pcs,:))
% end
% set(gca,'FontSize',14);
% xlabel("# PCs")
% ylabel("Min AIC # Clusters")
% ylim([0 10])
% xlim([0 max(opt.num_pcs)+1]) 
% subplot(3,2,6);hold on
% cmap = copper(length(opt.num_pcs));
% for i_pcs = 1:(max(opt.num_pcs)-1)
%     plot([i_pcs i_pcs+1],min_BIC_ix(i_pcs:i_pcs+1),'linewidth',3,'color',cmap(i_pcs,:))
% end 
% set(gca,'FontSize',14);
% xlabel("# PCs")
% ylabel("Min BIC # Clusters")
% ylim([0 10])
% xlim([0 max(opt.num_pcs)+1]) 
% suptitle("Probabilistic Model Selection Criteria")

%% Visualize median and mean squared mahalanobis distance across # cluster choices 
visPC1 = 3; % Doesn't work super well for just pcs 1:2
visPCend = 10;
norm_median = ((median_clust_l2(visPC1:visPCend,:)' - min(median_clust_l2(visPC1:visPCend,:)'))./max((median_clust_l2(visPC1:visPCend,:)' - min(median_clust_l2(visPC1:visPCend,:)'))))';
norm_mean = ((mean_clust_l2(visPC1:visPCend,:)' - min(mean_clust_l2(visPC1:visPCend,:)'))./max((mean_clust_l2(visPC1:visPCend,:)' - min(mean_clust_l2(visPC1:visPCend,:)'))))';
cmap = copper(length(opt.num_pcs)); 
cmap = cmap(visPC1:end,:); 
figure(); hold on 
colororder(cmap)
subplot(3,2,1);hold on
xlim([0 max(opt.num_pcs)+1])
plot(norm_median','linewidth',1.5);  
xlabel("# Clusters")  
title("Median Euclidean Dist (Normalized)")
set(gca,'FontSize',13);
subplot(3,2,2);hold on
xlim([0 max(opt.num_pcs)+1])
plot(norm_mean','linewidth',1.5);  
xlabel("# Clusters")  
title("Mean Euclidean Dist (Normalized)")
set(gca,'FontSize',13); 

% take second derivative 
% unnorm_median = median_clust_d2(visPC1:end,:) ; % 
% unnorm_mean = mean_clust_d2(visPC1:end,:) ; % 

avg_median_l2 = mean(norm_median); % avg over # PCs
avg_mean_l2 = mean(norm_mean); % avg over # PCs

% take second derivative
ddt2_median = nan(size(norm_median,1),size(norm_median,2)-2);
ddt2_mean = nan(size(norm_median,1),size(norm_median,2)-2); 
ddt2_avgMedian = nan(length(avg_median_l2)-2,1); 
ddt2_avgMean = nan(length(avg_median_l2)-2,1); 
for i_pc = 1:size(norm_median,1)
    for ii = 2:size(norm_median,2)-1
        ddt2_median(i_pc,ii-1) = norm_median(i_pc,ii+1)+norm_median(i_pc,ii-1)-2*norm_median(i_pc,ii);
        ddt2_mean(i_pc,ii-1) = norm_mean(i_pc,ii+1)+norm_mean(i_pc,ii-1)-2*norm_mean(i_pc,ii); 
        ddt2_avgMedian(ii-1) = avg_median_l2(ii+1)+ avg_median_l2(ii-1) - 2*avg_median_l2(ii);
        ddt2_avgMean(ii-1) = avg_mean_l2(ii+1)+ avg_mean_l2(ii-1) - 2*avg_mean_l2(ii);
    end
end 
[~,maxDDt2_ix_avgMedian] = max(ddt2_avgMedian);
[~,maxDDt2_ix_avgMean] = max(ddt2_avgMean);
subplot(3,2,1); 
xline(maxDDt2_ix_avgMedian + 1,'--','linewidth',1.5)
subplot(3,2,2); 
xline(maxDDt2_ix_avgMean + 1,'--','linewidth',1.5) 
legend([arrayfun(@(x) sprintf("%i PCs",x),3:max(opt.num_pcs)) "Max Mean d2dt"])

subplot(3,2,3);hold on
plot(2:size(norm_median,2)-1,ddt2_median,'linewidth',1.5);  
xlim([0 max(opt.num_pcs)+1])
xlabel("# Clusters")  
ylabel("ddt2 Median")
set(gca,'FontSize',13);
subplot(3,2,4);hold on
plot(2:size(norm_median,2)-1,ddt2_mean,'linewidth',1.5);   
xlim([0 max(opt.num_pcs)+1])
xlabel("# Clusters")  
ylabel("ddt2 Mean")
set(gca,'FontSize',13);  

[~,max_ix_mean] = max(ddt2_mean);
[~,max_ix_median] = max(ddt2_median);

subplot(3,2,5);hold on
histogram(max_ix_median+1)
% plot(2:size(norm_median,2)-1,max_ix_median,'linewidth',1.5);  
xlim([0 max(opt.num_pcs)+1])
xlabel("# Clusters")  
ylabel(sprintf("ddt2 Median \n Max Ix"))
set(gca,'FontSize',13);
subplot(3,2,6);hold on
histogram(max_ix_mean+1)
xlim([0 max(opt.num_pcs)+1])
xlabel("# Clusters")  
ylabel(sprintf("ddt2 Mean \n Max Ix"))
set(gca,'FontSize',13);  

%% Compare AIC, BIC min and ddt2 results 
[~,ddt2_median_max_ix] = max(ddt2_median); 
ddt2_median_max_ix = ddt2_median_max_ix + 1; % adjust indexing
[~,ddt2_mean_max_ix] = max(ddt2_mean); 
ddt2_mean_max_ix = ddt2_mean_max_ix + 1; % adjust indexing

cmap = copper(length(opt.num_pcs));
figure() ;hold on
for i_pcs = 1:(max(opt.num_pcs)-1)
    plot([i_pcs i_pcs+1],min_BIC_ix(i_pcs:i_pcs+1),'linewidth',3,'color',cmap(i_pcs,:))
%     plot([i_pcs i_pcs+1],min_AIC_ix(i_pcs:i_pcs+1),'linewidth',3,'color',cmap(i_pcs,:))
%     if i_pcs < (max(opt.num_pcs)-2)
%         plot([i_pcs+1 i_pcs+2],ddt2_median_max_ix(i_pcs:i_pcs+1),'linewidth',2,'color',cmap(i_pcs,:))
%         plot([i_pcs+1 i_pcs+2],ddt2_mean_max_ix(i_pcs:i_pcs+1),'linewidth',2,'color',cmap(i_pcs,:))
%     end
end
legend(["Min AIC Index","Min BIC Index"]) 
xlabel("# PCs Included")
ylabel("Min AIC/BIC # Clusters")

%% Now make shuffle distribution to assess degree to which data clusters beyond rnd cloud
    
% ALTERNATIVELY, shuffle after performing PCA 
% a few shuffle options: 
%   1) after PCA 
%   2) before PCA shuffle everyone 
%   3) before PCA shuffle kerns together 
%   4) before PCA shuffle kerns and coeffs acr rewsizes

% Shuffle reward kernel ix together, otherwise coeffs shuffle across cells
beta_all_sig_norm = zscore(beta_all_sig);
[~,~,~,~,expl_unshuffled] = pca(beta_all_sig_norm');
n_vars = size(beta_all_sig,1); 
n_shuffles = 1000; 
expl_complete = nan(n_shuffles,n_vars);
expl_kern_together = nan(n_shuffles,n_vars);
expl_taskvar_together = nan(n_shuffles,n_vars);

for i_shuffle = 1:n_shuffles
    [~,~,~,expl_complete(i_shuffle,:)] = shuffle_coeffs(beta_all_sig,"complete");
    [~,~,~,expl_kern_together(i_shuffle,:)] = shuffle_coeffs(beta_all_sig,"kern_together");
    [~,~,~,expl_taskvar_together(i_shuffle,:)] = shuffle_coeffs(beta_all_sig,"taskvar_together");
end

% look at shuffled eigenspectrum
figure() ;hold on 
plot(expl_unshuffled(1:10),'linewidth',3,'color','k')
shadedErrorBar(1:10,mean(expl_complete(:,1:10)),1.96 * std(expl_complete(:,1:10)),'lineProps',{'linewidth',2})
shadedErrorBar(1:10,mean(expl_kern_together(:,1:10)),1.96 * std(expl_kern_together(:,1:10)),'lineProps',{'linewidth',2})
shadedErrorBar(1:10,mean(expl_taskvar_together(:,1:10)),1.96 * std(expl_taskvar_together(:,1:10)),'lineProps',{'linewidth',2})
ylabel("Variance Explained") 
xlabel("Principal Component") 
legend(["Unshuffled","Shuffle All Independently","Shuffle Kernels together",sprintf("Shuffle Kernels together and \n Taskvars together acr rewsizes")])

%% Visualize what shuffled data looks like

beta_norm = zscore(beta_all_sig);
% beta_norm = beta_all_sig;
shuffle_type_names = ["Total Shuffle","Kernels together","Same acr rewsize","Unshuffled"];
% [~,score_unshuffled] = pca(beta_norm');
% [~,~,score_complete] = shuffle_coeffs(beta_all_sig,"complete");
% [~,~,score_kern_together] = shuffle_coeffs(beta_all_sig,"kern_together");
% [~,~,score_taskvar_together] = shuffle_coeffs(beta_all_sig,"taskvar_together");

num_pcs = 6;  
score_cell = {score_complete,score_kern_together,score_taskvar_together,score_unshuffled};
for i_score = 1:numel(score_cell)
    this_score = score_cell{i_score}; 
    for iPC = 2:num_pcs
        subplot(numel(score_cell),num_pcs-1,(num_pcs-1) * (i_score-1) + iPC-1); hold on;

        gscatter(this_score(:,1),this_score(:,iPC),mouse(sesh_all),lines(5))
        xlabel('PC1');
        ylabel(sprintf('PC%i',iPC));
        set(gca,'FontSize',14);
        hLeg = legend();
        set(hLeg,'visible','off') 
        
        if iPC == 2
            ylabel(sprintf('%s \n PC%i',shuffle_type_names(i_score),iPC));
        end
    end
end

%% Now look at clustery-ness in these different shuffled settings

% specify model options
opt.num_clust = 5;
opt.num_pcs = 3;
gmm_opt.lambda = 1; 
gmm_opt.replicates = 50;
options = statset('MaxIter',1000);
% Visualize gmm results
num_vis_pcs = 3;

score_cell = {score_complete,score_kern_together,score_taskvar_together,score_unshuffled};
for i_score = 1:numel(score_cell)
    % fit model
    GMM = fitgmdist(score_cell{i_score}(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
        'replicates',gmm_opt.replicates,'options',options);
    gmm_idx = cluster(GMM,score_cell{i_score}(:,1:opt.num_pcs));
    
    % hfig.Name = sprintf('k means on top 2 PCs %s cohort %s',opt.data_set,opt.brain_region);
    for iPC = 2:num_vis_pcs
        subplot(numel(score_cell),num_vis_pcs-1,(num_vis_pcs-1) * (i_score-1) + iPC-1); hold on;
        gscatter(score_cell{i_score}(:,1),score_cell{i_score}(:,iPC),gmm_idx,lines(opt.num_clust),[],10);
        xlabel('PC1');
        ylabel(sprintf('PC%i',iPC));
        set(gca,'FontSize',14);
        hLeg = legend();
        set(hLeg,'visible','off')
        if iPC == 2
            ylabel(sprintf('%s \n PC%i',shuffle_type_names(i_score),iPC));
        end
    end
end
suptitle("Clusters across PC dimensions")  

%% look at what shuffled beta clusters look like 
shuffle_type = "taskvar_together";
[beta_shuffle,~,score_shuffle] = shuffle_coeffs(beta_all_sig,shuffle_type);

GMM = fitgmdist(score_shuffle(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
        'replicates',gmm_opt.replicates,'options',options);
gmm_idx = cluster(GMM,score_shuffle(:,1:opt.num_pcs));
beta_norm_shuffle = zscore(beta_shuffle);
vis_cluster_mean_beta(opt,beta_norm_shuffle,var_name,gmm_idx);

%% Now perform shuffle test to determine p-value of getting BIC for # clusters, # PCs we get as optimal from unshuffled data

opt.num_clust = 5;
opt.num_pcs = 3;
gmm_opt.lambda = 1; 
gmm_opt.replicates = 25;
options = statset('MaxIter',1000);

GMModel = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
    'replicates',gmm_opt.replicates,'options',options);
% Mahalanobis distance to closest
[gmm_idx,~,~,~,d2] = cluster(GMModel,score(:,1:opt.num_pcs));
mean_l2_unshuffled = nan(opt.num_clust,1);
median_l2_unshuffled = nan(opt.num_clust,1);
for i_clust = 1:opt.num_clust
    mean_l2_unshuffled(i_clust) = mean(squareform(pdist(score(gmm_idx == i_clust,1:opt.num_pcs))),'all');
    median_l2_unshuffled(i_clust) = median(squareform(pdist(score(gmm_idx == i_clust,1:opt.num_pcs))),'all');
end  
mean_l2_unshuffled = mean(mean_l2_unshuffled) / mean(squareform(pdist(score(:,1:opt.num_pcs))),'all'); 
median_l2_unshuffled = median(median_l2_unshuffled) / median(squareform(pdist(score(:,1:opt.num_pcs))),'all'); 

median_d2_unshuffled = median(min(d2,[],2));
mean_d2_unshuffled = mean(min(d2,[],2));
AIC_unshuffled = GMModel.AIC;
BIC_unshuffled = GMModel.BIC;
fprintf("Unshuffled clustering complete \n")

n_shuffles = 200; 
shuffle_types = ["complete","kern_together","taskvar_together"]; 
median_d2_shuffled = nan(numel(shuffle_types),n_shuffles); 
mean_d2_shuffled = nan(numel(shuffle_types),n_shuffles); 
mean_l2_shuffled = nan(numel(shuffle_types),n_shuffles); 
median_l2_shuffled = nan(numel(shuffle_types),n_shuffles); 
AIC_shuffled = nan(numel(shuffle_types),n_shuffles); 
BIC_shuffled = nan(numel(shuffle_types),n_shuffles); 
% Now perform same fitting for shuffled coeffs
for i_shuffle_type = 1:numel(shuffle_types)
    for i = 1:n_shuffles
        [~,~,score_shuffle] = shuffle_coeffs(beta_all_sig,shuffle_types(i_shuffle_type));
        GMModel = fitgmdist(score_shuffle(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
            'replicates',gmm_opt.replicates,'options',options);
        % Mahalanobis distance to closest
        [shuffled_gmm_idx,~,~,~,d2] = cluster(GMModel,score_shuffle(:,1:opt.num_pcs));
        l2_mean_shuffled_tmp = nan(opt.num_clust,1);
        l2_median_shuffled_tmp = nan(opt.num_clust,1);
        for i_clust = 1:opt.num_clust 
            % use nanmean here in case we didn't assign to one of the clusters
            l2_mean_shuffled_tmp(i_clust) = nanmean(squareform(pdist(score_shuffle(shuffled_gmm_idx == i_clust,1:opt.num_pcs))),'all');
            l2_median_shuffled_tmp(i_clust) = nanmedian(squareform(pdist(score_shuffle(shuffled_gmm_idx == i_clust,1:opt.num_pcs))),'all');
        end
        mean_l2_shuffled(i_shuffle_type,i) = nanmean(l2_mean_shuffled_tmp) / mean(squareform(pdist(score_shuffle(:,1:opt.num_pcs))),'all');
        median_l2_shuffled(i_shuffle_type,i) = nanmedian(l2_median_shuffled_tmp) / median(squareform(pdist(score_shuffle(:,1:opt.num_pcs))),'all');
        
        median_d2_shuffled(i_shuffle_type,i) = median(min(d2,[],2));
        mean_d2_shuffled(i_shuffle_type,i) = mean(min(d2,[],2));
        
        AIC_shuffled(i_shuffle_type,i) = GMModel.AIC;
        BIC_shuffled(i_shuffle_type,i) = GMModel.BIC; 
    end
    fprintf("%s shuffle clustering complete \n",shuffle_types(i_shuffle_type))
end

%% Visualize comparison to shuffled distribution 

% I ACTUALLY DONT THINK THIS BIC/AIC THING IS LEGAL!! ... different
% dependent variables

figure(); 
xline(median_l2_unshuffled,'k--','linewidth',2) ;hold on; 
for i_shuffle_type = 1:numel(shuffle_types)
%     histogram(median_l2_shuffled(i_shuffle_type,:),0:.05:1) 
%     hold on;
    i_pdf = pdf(fitdist(median_l2_shuffled(i_shuffle_type,:)','kernel','Kernel','Normal'),0:.005:1); 
    plot(0:.005:1,i_pdf,'linewidth',2)
end 
title(sprintf("Median Intracluster Euclidean Distance (Normalized) \n 3 PCs, 5 Clusters"))
xlabel("Median Intracluster Euclidean Distance (Normalized)")
legend(["Unshuffled","Shuffle All Independently","Shuffle Kernels together",... 
        sprintf("Shuffle Kernels together and \n Taskvars together acr rewsizes")])
% legend(["Unshuffled","Shuffle Kernels together",... 
%         sprintf("Shuffle Kernels together and \n Taskvars together acr rewsizes")])    
set(gca,'FontSize',14) 

%% Align GMM clusters by mean param
% First just look at effect of # replicates on consistency of clustering,
% within 1 num_pcs / num_clust

gmm_opt.lambda = 1;
gmm_opt.replicates = 20;
options = statset('MaxIter',1000);
n_cells = size(score,1);
n_repeats = 50;

test_n_clust = 5; % 1:10; % 1:10; 
test_n_pcs = 3:5; % 1:10; 

consistency = cell(numel(test_n_clust),numel(test_n_pcs));
total_things = numel(test_n_clust) * numel(test_n_pcs);

h1 = waitbar(0,'Getting consistency results acr nClusts and nPCs');
for i_n_clust = 1:numel(test_n_clust) 
    for i_n_pcs = 1:numel(test_n_pcs)
        % specify model options
        opt.num_clust = test_n_clust(i_n_clust);
        opt.num_pcs = test_n_pcs(i_n_pcs);
        % fit model
        GMM1 = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
            'replicates',gmm_opt.replicates,'options',options);
        gmm_idx1 = cluster(GMM1,score(:,1:opt.num_pcs));
        mu1 = GMM1.mu;
        
        cluster_assignments = nan(n_cells,n_repeats);
        for i_repeat = 1:n_repeats
            GMM_repeat = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
                'replicates',gmm_opt.replicates,'options',options);
            
            gmm_idx_repeat = cluster(GMM_repeat,score(:,1:opt.num_pcs));
            mu_repeat = GMM_repeat.mu;
            
            % PC1-sorted alignment
            [~,pc1_order] = sort(mu_repeat(:,1));
            aligned_gmm_idx = nan(size(gmm_idx_repeat));
            for i_clust = 1:opt.num_clust
                aligned_gmm_idx(gmm_idx_repeat == pc1_order(i_clust)) = i_clust;
            end
            cluster_assignments(:,i_repeat) = aligned_gmm_idx; 
            
            % distance-based alignment
%             dists = pdist2(mu1,mu_repeat);
%             [~,run1_closest] = min(dists);
%             aligned_gmm_idx = nan(size(gmm_idx_repeat));
%             for i_clust = 1:opt.num_clust
%                 aligned_gmm_idx(gmm_idx_repeat == i_clust) = run1_closest(i_clust);
%             end
%             cluster_assignments(:,i_repeat) = aligned_gmm_idx; 
        end
        
        % Visualize consistency of cluster assignments
        mode_cluster = mode(cluster_assignments')';
        mode_matrix = repmat(mode_cluster, [1 n_repeats]);
        mode_cluster_flat = mode_cluster(:);
        mode_matrix_flat = mode_matrix(:);
        
        pct_is_mode = mean(mode_matrix == cluster_assignments,2); % [n_cells x 1] of % same as mode
        pct_same = nan(opt.num_clust,1);
        for i_clust = 1:opt.num_clust
            pct_same(i_clust) = mean(pct_is_mode(gmm_idx1 == i_clust));
        end
%         pct_same = sort(pct_same,'descend');
        consistency{i_n_clust,i_n_pcs} = pct_same; % can be nan if we have an unused cluster
        waitbar((numel(test_n_pcs) * (i_n_clust-1) + i_n_pcs) / total_things,h1)
    end
end 
close(h1)

%% Visualize consistency for [3 4 5] 5 clusters

consistency_cat = cat(2,consistency{:}); 

figure() 
colororder(copper(3)) 
bar(consistency_cat) 
legend(["3 PCs","4 PCs","5 PCs"]) 
xlabel("Cluster (PC1-Sorted)") 
ylabel("Fraction Consistency") 
title(sprintf("3, 4, and 5 PCs \n 5-Cluster Consistency"))
set(gca,'FontSize',14) 

%% Analyze cluster consistency across PCs and n_clusters 

padded_consistency = cellfun(@(x) [circshift(x,length(x) - length(find(isnan(x))))' nan(1,max(test_n_clust)-length(x))],consistency,'un',0);
padded_consistency = arrayfun(@(x) cat(1,padded_consistency(:,x)),1:numel(test_n_pcs),'un',0);
padded_consistency = arrayfun(@(x) cat(1,padded_consistency{x}{:}),1:numel(test_n_pcs),'un',0);

figure();
for i_n_pcs = 1:10 
    subplot(2,5,i_n_pcs)
    colororder(copper(10))
    plot(padded_consistency{i_n_pcs}','linewidth',2) 
    title(sprintf("%i PCs",i_n_pcs),'Fontsize',14) 
    xlim([0 10]) 
    ylim([0.2 1.05]) 
    xline(min_BIC_ix(i_n_pcs),':','linewidth',2)  
    if i_n_pcs >= 6 
        xlabel("Cluster (Sorted)",'Fontsize',14)
    end
    if i_n_pcs == 10 
        legend([arrayfun(@(x) sprintf("%i Clusters Fit",x),1:10,'un',0) "Min BIC"],'Fontsize',14)
    end 
    if ismember(i_n_pcs,[1 6]) 
        ylabel("% Consistency",'Fontsize',14)
    end 
end
suptitle(sprintf("Cluster Consistency Across PCs Included and Clusters Fit"))

%% Last, use assignment consistency to compare the 5 clusters we get from choosing 3, 4, and 5 PCs
% first fit models
opt.num_clust = 5;
% fit model
GMM3 = fitgmdist(score(:,1:3),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
    'replicates',gmm_opt.replicates,'options',options);
gmm_idx3 = cluster(GMM3,score(:,1:3));
mu3 = GMM3.mu;

GMM4 = fitgmdist(score(:,1:4),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
    'replicates',gmm_opt.replicates,'options',options);
gmm_idx4 = cluster(GMM4,score(:,1:4));
mu4 = GMM4.mu;

GMM5 = fitgmdist(score(:,1:5),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
    'replicates',gmm_opt.replicates,'options',options);
gmm_idx5 = cluster(GMM5,score(:,1:5));
mu5 = GMM5.mu; 

%% Now align, visualize alignment
%  align using mode here?

[~,pc1_order3] = sort(mu3(:,1));
[~,pc1_order4] = sort(mu4(:,1));
[~,pc1_order5] = sort(mu5(:,1)); 

gmm_idx3_aligned = nan(size(gmm_idx3)); 
gmm_idx4_aligned = nan(size(gmm_idx3)); 
gmm_idx5_aligned = nan(size(gmm_idx3)); 
for i_clust = 1:opt.num_clust 
    gmm_idx3_aligned(gmm_idx3 == pc1_order3(i_clust)) = i_clust; 
    gmm_idx4_aligned(gmm_idx4 == pc1_order4(i_clust)) = i_clust; 
    gmm_idx5_aligned(gmm_idx5 == pc1_order5(i_clust)) = i_clust; 
end

% dists = pdist2(mu3,mu4(:,1:3));
% [~,run3_closest] = min(dists);
% gmm_idx4_aligned = nan(size(gmm_idx4));
% for i_clust = 1:opt.num_clust
%     gmm_idx4_aligned(gmm_idx4 == i_clust) = run3_closest(i_clust);
% end

labels_cell = {gmm_idx3_aligned,gmm_idx4_aligned,gmm_idx5_aligned};
figure();
for i_n_pcs = 1:3
    for iPC = 2:5
        subplot(3,4,4 * (i_n_pcs-1) + iPC-1 )
        gscatter(score(:,1),score(:,iPC),labels_cell{i_n_pcs},lines(opt.num_clust),[],10);
        xlabel('PC1');
        if iPC ~= 2
            ylabel(sprintf('PC%i',iPC));
        else
            ylabel(sprintf('%i PCs Used \n PC%i',i_n_pcs+2,iPC));
        end
        set(gca,'FontSize',14);
        
        hLeg = legend();
        set(hLeg,'visible','off')
        
    end
end
suptitle("Clusters across PC dimensions")

%% confusionmat between clusters 

figure() 
subplot(1,2,1) 
imagesc(confusionmat(gmm_idx3_aligned,gmm_idx4_aligned))
ylabel("3 PCs Cluster") 
xlabel("4 PCs Cluster") 
set(gca,'FontSize',14)
subplot(1,2,2) 
imagesc(confusionmat(gmm_idx3_aligned,gmm_idx5_aligned)) 
ylabel("3 PCs Cluster") 
xlabel("5 PCs Cluster") 
set(gca,'FontSize',14)
% colormap('copper')

%% Perform final clustering run and create sigcells table 

% specify model options
opt.num_clust = 5;
opt.num_pcs = 3;
gmm_opt.lambda = 1; 
gmm_opt.replicates = 100; 
options = statset('MaxIter',1000);

% fit model
GMM = fitgmdist(score(:,1:opt.num_pcs),opt.num_clust,'RegularizationValue',gmm_opt.lambda,...
                                                     'replicates',gmm_opt.replicates,'options',options);
gmm_idx = cluster(GMM,score(:,1:opt.num_pcs)); 

% PC1-sorted alignment
[~,pc1_order] = sort(GMM.mu(:,1));
final_gmm_labels = nan(size(gmm_idx));
for i_clust = 1:opt.num_clust
    final_gmm_labels(gmm_idx == pc1_order(i_clust)) = i_clust;
end 

% % Visualize gmm results
% num_vis_pcs = 5; 
%% cluster_gscatter(num_vis_pcs,score,final_gmm_labels,opt.num_clust)
% colors = cbrewer('qual','Set1',5);  % winter(opt.num_clust); 
colors = lines(5); 
h = figure();
for i_clust = 1:opt.num_clust 
    hold on
    scatter3(score(final_gmm_labels == i_clust,1),score(final_gmm_labels == i_clust,2),score(final_gmm_labels == i_clust,3),150,colors(i_clust,:),'.');
end  
legend(arrayfun(@(x) sprintf("Cluster %i",x),1:max(opt.num_clust)))
grid() 
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
view(35,35) 
set(gca,'FontSize',14); 

colors = cbrewer('qual','Set1',5);
h = figure(); hold on
for i_mouse = 1:numel(uniq_mouse) 
    this_mouse = cellfun(@str2double, mouse(sesh_all)) == str2double(uniq_mouse{i_mouse});
    scatter3(score(this_mouse,1),score(this_mouse,2),score(this_mouse,3),150,colors(i_mouse,:),'.');
end
legend(arrayfun(@(x) sprintf("m%s",uniq_mouse{x}),1:numel(uniq_mouse)))
grid() 
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
view(35,35) 
set(gca,'FontSize',14); 

mouse_vec = cellfun(@str2double, mouse(sesh_all)); 
figure()
for i_mouse = 1:numel(uniq_mouse) 
    subplot(1,opt.num_clust,i_mouse)
    histogram(final_gmm_labels(mouse_vec == str2double(uniq_mouse{i_mouse})))
end

%% plot coefficients for clusters
vis_cluster_mean_beta(opt,beta_norm,var_name,final_gmm_labels)  

%% Make the table 

sig_cells = table;
n_start = 10000;
sig_cells.Mouse = cell(n_start,1);
sig_cells.Session = cell(n_start,1);
sig_cells.CellID = nan(n_start,1);
sig_cells.BrainRegionRough = cell(n_start,1);
sig_cells.BrainRegion = cell(n_start,1);
sig_cells.DepthFromSurface = nan(n_start,1);
sig_cells.DepthFromTip = nan(n_start,1);

counter = 1; 
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    
%     sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    sig = sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    N = sum(sig);
    
    keep_idx = find(keep_cell);
    keep_idx = keep_idx(sig);
    
    sig_cells.Mouse(counter:counter+N-1) = repmat(mouse(i),N,1);
    sig_cells.Session(counter:counter+N-1) = repmat(session_all(i),N,1);
    sig_cells.CellID(counter:counter+N-1) = fit.good_cells(keep_idx);
    sig_cells.BrainRegionRough(counter:counter+N-1) = fit.brain_region_rough(keep_idx);
    if isfield(fit.anatomy,'cell_labels')
        keep_idx2 = ismember(fit.good_cells_all,fit.good_cells(keep_idx));
        sig_cells.BrainRegion(counter:counter+N-1) = fit.anatomy.cell_labels.BrainRegion(keep_idx2);
        sig_cells.DepthFromSurface(counter:counter+N-1) = fit.anatomy.cell_labels.DepthFromSurface(keep_idx2);
        sig_cells.DepthFromTip(counter:counter+N-1) = fit.anatomy.cell_labels.DepthFromTip(keep_idx2);
    end
    
    counter = counter+N;

end
sig_cells = sig_cells(1:counter-1,:);

sig_cells.GMM_cluster = final_gmm_labels; 
paths.results_save = [paths.results,'/gmm/'];
if ~isfolder(paths.results_save)
    mkdir(paths.results_save)
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mc';

% now save to path
save(fullfile(paths.results_save,sprintf('sig_cells_gmm_%s_cohort_%s.mat',opt.data_set,opt.brain_region)),'sig_cells');


