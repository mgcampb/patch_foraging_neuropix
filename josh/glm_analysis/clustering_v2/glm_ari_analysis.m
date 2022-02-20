%% GMM reloaded for stronger analysis of clustering in GLM coefficients

% paths
data_path = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_20211105_50ms_bins/GLM_coeffs_20211105_50_ms_bins.mat';
old_clusters_path = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_20211105_50ms_bins/GMM_clust_20211105_50_ms_bins.mat';

%% Load data
glm_dat = load(data_path);
old_gmm_dat = load(old_clusters_path); 
coeffs_to_cluster = cellfun(@(this_var) any(strcmp(this_var,old_gmm_dat.var_name)),glm_dat.var_name);

% pull off friends with nonzero cognitive variables
any_nonzero_mask = sum(abs(glm_dat.beta_all(coeffs_to_cluster,:))>0)'>0;
all_nonzero_cellIDs = glm_dat.cellID(any_nonzero_mask);
pfc_mask = strcmp("PFC",glm_dat.brain_region_rough);
pfc_nonzero_cellIDs = glm_dat.cellID(pfc_mask & any_nonzero_mask);

beta_all = glm_dat.beta_all(coeffs_to_cluster,any_nonzero_mask);
beta_subpfc = glm_dat.beta_all(coeffs_to_cluster,~pfc_mask & any_nonzero_mask);
beta_pfc = glm_dat.beta_all(coeffs_to_cluster,pfc_mask & any_nonzero_mask);

%% plot time integration versus 

%% Perform PCA and look at cumulative variance explained

beta_all_norm = zscore(beta_all);
[coeff_all,score_all,~,~,expl_all] = pca(beta_all_norm');

beta_pfc_norm = zscore(beta_pfc);
[coeff_pfc,score_pfc,~,~,expl_pfc] = pca(beta_pfc_norm');

n_vis = 15; 
figure();hold on
plot(cumsum(expl_all(1:n_vis)),'linewidth',1.5)
plot(cumsum(expl_pfc(1:n_vis)),'linewidth',1.5)

%% Shuffle to see how eigenspectrum compares to that of shuffle 

n_pcs_var_expl = 15; 
n_shuffles = 1000; 

% Shuffle reward kernel ix together, otherwise coeffs shuffle across cells
beta_pfc_norm = zscore(beta_pfc);
[~,~,~,~,expl_unshuffled] = pca(beta_pfc_norm');
b0_b1_unshuffled = [ones(1,n_pcs_var_expl) ; 1:(n_pcs_var_expl)]' \ log(expl_unshuffled(1:n_pcs_var_expl));
n_vars = size(beta_pfc,1); 
expl_complete = nan(n_shuffles,n_vars);
expl_kern_together = nan(n_shuffles,n_vars);
expl_taskvar_together = nan(n_shuffles,n_vars);
b0_b1_complete = nan(n_shuffles,2); 
b0_b1_kern_together = nan(n_shuffles,2); 
b0_b1_taskvar_together = nan(n_shuffles,2); 

for i_shuffle = 1:n_shuffles
    [~,~,~,expl_complete(i_shuffle,:)] = shuffle_coeffs(beta_pfc,"complete");
    b0_b1_complete(i_shuffle,:) = [ones(1,n_pcs_var_expl) ; 1:(n_pcs_var_expl)]' \ log(expl_complete(i_shuffle,1:n_pcs_var_expl))';
    [~,~,~,expl_kern_together(i_shuffle,:)] = shuffle_coeffs(beta_pfc,"kern_together");
    b0_b1_kern_together(i_shuffle,:) = [ones(1,n_pcs_var_expl) ; 1:(n_pcs_var_expl)]' \ log(expl_kern_together(i_shuffle,1:n_pcs_var_expl))';
    [~,~,~,expl_taskvar_together(i_shuffle,:)] = shuffle_coeffs(beta_pfc,"taskvar_together");
    b0_b1_taskvar_together(i_shuffle,:) = [ones(1,n_pcs_var_expl) ; 1:(n_pcs_var_expl)]' \ log(expl_taskvar_together(i_shuffle,1:n_pcs_var_expl))';
end

%% Visualize shuffled eigenspectra
colors = lines(3); 
% look at shuffled eigenspectrum
figure() ;hold on 
plot(expl_unshuffled(1:10),'linewidth',3,'color','k')
shadedErrorBar(1:10,mean(expl_complete(:,1:10)),1.96 * std(expl_complete(:,1:10)),'lineProps',{'linewidth',2,'color',colors(1,:)})
shadedErrorBar(1:10,mean(expl_kern_together(:,1:10)),1.96 * std(expl_kern_together(:,1:10)),'lineProps',{'linewidth',2,'color',colors(2,:)})
shadedErrorBar(1:10,mean(expl_taskvar_together(:,1:10)),1.96 * std(expl_taskvar_together(:,1:10)),'lineProps',{'linewidth',2,'color',colors(3,:)})
ylabel("Variance Explained (%)") 
xlabel("Principal Component") 
legend(["Unshuffled","Shuffle All Independently","Shuffle Kernels together",sprintf("Shuffle Kernels together and \n all vars together rewsizes")])
set(gca,'fontsize',14)
title("GLM Coefficient Matrix Eigenspectrum",'fontsize',16)

% visualize distn of expo fits
figure();set(gca,'colorOrder',[colors(1,:);colors(3,:);colors(2,:)]);hold on
violinplot([b0_b1_complete(:,2) ; b0_b1_kern_together(:,2) ; b0_b1_taskvar_together(:,2)], repelem([1,2,3],n_shuffles),'width',.25,'violinalpha',.05); 
yline(b0_b1_unshuffled(2),'linewidth',3,'linestyle','--','color','k')
ylim([-.2,0])
% xticks([1,2,3])
xticklabels(["Shuffle All Independently","Shuffle Kernels together",sprintf("Shuffle Kernels + Vars Together")])
xtickangle(45)
ylabel(sprintf("Eigenspectrum exponential \n regression slope"))
set(gca,'fontsize',14)

% get the number of components that explain more variance than 95% of shuffle distns
expl_pval_complete = arrayfun(@(i) length(find(expl_unshuffled(i) < expl_complete(:,i))) / n_shuffles,1:n_pcs_var_expl);
expl_pval_kern_together = arrayfun(@(i) length(find(expl_unshuffled(i) < expl_kern_together(:,i))) / n_shuffles,1:n_pcs_var_expl);
expl_pval_taskvar_together = arrayfun(@(i) length(find(expl_unshuffled(i) < expl_taskvar_together(:,i))) / n_shuffles,1:n_pcs_var_expl);

%% Perform GMM clustering over different regularization values to understand what this does to BIC/AIC

% define the PCA space to look at
beta_pfc_norm = zscore(beta_pfc);
[coeff_pfc,score_pfc,~,~,expl_pfc] = pca(beta_pfc_norm');

% set options 
lambdas = [0 .2 .3 .5 .75 1];
n_clusters = 2:10; 
n_pcs = 1:10; 
gmm_opt.replicates = 50;
bootstrp_gmm_opt.replicates = 10; 
options = statset('MaxIter',1000);
n_repeats = 30; % number of repeats for ARI bootstrapping

% preallocate results
AIC = nan(numel(lambdas),numel(n_pcs),numel(n_clusters));  
BIC = nan(numel(lambdas),numel(n_pcs),numel(n_clusters)); 
ari = nan(numel(lambdas),numel(n_pcs),numel(n_clusters),nchoosek(n_repeats,2)); 

% how many unique clusters are we actually finding? 
n_unique = nan(numel(lambdas),numel(n_pcs),numel(n_clusters));

rng(0);  
for i_lam = 1:numel(lambdas)
    this_lambda = lambdas(i_lam);
    for i_num_pcs = 1:numel(n_pcs)
        this_num_pcs = n_pcs(i_num_pcs); 
        for i_n_clusters = 1:numel(n_clusters)
            this_n_clusters = n_clusters(i_n_clusters);
            try
                this_gmm = fitgmdist(score_pfc(:,1:this_num_pcs),this_n_clusters,'RegularizationValue',this_lambda,...
                                                             'replicates',gmm_opt.replicates,'options',options);        

                i_gmm_idx = cluster(this_gmm,score_pfc(:,1:this_num_pcs));
                n_unique(i_lam,i_num_pcs,i_n_clusters) = length(unique(i_gmm_idx));
%                 if n_un
                AIC(i_lam,i_num_pcs,i_n_clusters) = this_gmm.AIC;  
                BIC(i_lam,i_num_pcs,i_n_clusters) = this_gmm.BIC;  
                if ~this_gmm.Converged 
                    fprintf("λ = %f %i PCs %i Clusters GMM Did Not Converge! \n",this_lambda,this_num_pcs,this_n_clusters)
                end 
            catch 
                fprintf("λ = %f %i PCs %i Clusters Ill-conditioned for all \n",this_lambda,this_num_pcs,this_n_clusters)
            end
            
            ari(i_lam,i_num_pcs,i_n_clusters,:) = gmm_ari_analysis(score_pfc(:,1:this_num_pcs),n_repeats,this_n_clusters,this_lambda,bootstrp_gmm_opt,options);
            fprintf("%i/%i \t",i_n_clusters,numel(n_clusters))
        end  
        fprintf("\n%i PCs Complete \n",i_num_pcs)
    end
    fprintf("λ = %f complete \n",this_lambda)
end


%% Visualize results 

% delta_median_l2 = diff(median_clust_l2,3);
close all
copper_lam = flipud(copper(numel(lambdas)));
mean_ari = nanmean(ari,4); 

fit_all_clusters_BIC = nan(size(BIC)); 
for i_lam = 1:numel(lambdas)
    for i_num_pcs = 1:numel(n_pcs)
        for i_n_clusters = 1:numel(n_clusters)
            this_n_clusters = n_clusters(i_n_clusters); 
            if n_unique(i_lam,i_num_pcs,i_n_clusters) ~= this_n_clusters
                fit_all_clusters_BIC(i_lam,i_num_pcs,i_n_clusters) = NaN;
            else
                fit_all_clusters_BIC(i_lam,i_num_pcs,i_n_clusters) = BIC(i_lam,i_num_pcs,i_n_clusters);
            end
        end
    end
end

figure() 
for i_lam = 1:numel(lambdas)
    subplot(2,numel(lambdas)+1,i_lam)
    norm_ilam_BIC = ((squeeze(fit_all_clusters_BIC(i_lam,:,:)) - min(squeeze(fit_all_clusters_BIC(i_lam,:,:)),[],2)) ./ (max(squeeze(fit_all_clusters_BIC(i_lam,:,:)),[],2) - min(squeeze(fit_all_clusters_BIC(i_lam,:,:)),[],2)));
    imagesc(norm_ilam_BIC');caxis([0,1])
    xlabel("# PCs")
    title(sprintf("λ = %.1f ",lambdas(i_lam)),'Color',copper_lam(i_lam,:))
    set(gca,'fontsize',14)
    subplot(2,numel(lambdas)+1,numel(lambdas)+1);hold on
    plot(nanmean(norm_ilam_BIC,1),.5 + fliplr(1:numel(n_clusters)),'linewidth',1.5,'color',copper_lam(i_lam,:))
    title(sprintf("Mean Normalized BIC\n Across choice of #PCs"))
    set(gca,'fontsize',14)
    
    subplot(2,numel(lambdas)+1,numel(lambdas) + 1 + i_lam)
    lam_mean_ari = squeeze(mean_ari(i_lam,:,:)); 
    imagesc(lam_mean_ari');caxis([0,1])
    xlabel("# PCs")
    title(sprintf("λ = %.1f ",lambdas(i_lam)),'Color',copper_lam(i_lam,:))
    set(gca,'fontsize',14)
    subplot(2,numel(lambdas)+1,2*numel(lambdas)+2);hold on
    plot(nanmean(lam_mean_ari,1),.5 + fliplr(1:numel(n_clusters)),'linewidth',1.5,'color',copper_lam(i_lam,:))
    title(sprintf("Mean ARI\n Across choice of #PCs"))
    set(gca,'fontsize',14)
end
subplot(2,numel(lambdas)+1,1)
ylabel("# Clusters")
subplot(2,numel(lambdas)+1,numel(lambdas) + 2)
ylabel("# Clusters")

subplot(2,numel(lambdas)+1,numel(lambdas)+1)
ylim([1.5,.5 + max(n_clusters)])
xlim([0,.5])
yticks((1:numel(n_clusters))); yticklabels(fliplr(1:numel(n_clusters)))
set(gca,'fontsize',14)

suptitle(sprintf("GMM BIC unit-normalized per choice of #PCs \n Across choices of λ"))

%% Old code 

% Mahalanobis distance to closest                                          
[i_gmm_idx,~,~,~,d2] = cluster(this_gmm,score_pfc(:,1:this_num_pcs)); 
n_unique(i_lam,i_num_pcs,i_n_clusters) = length(unique(i_gmm_idx));
median_clust_d2(i_lam,i_num_pcs,i_n_clusters) = median(min(d2,[],2));
mean_clust_d2(i_lam,i_num_pcs,i_n_clusters) = mean(min(d2,[],2));

mean_l2_tmp = nan(i_n_clusters,1); 
median_l2_tmp = nan(i_n_clusters,1); 
for i_clust = 1:i_n_clusters
    mean_l2_tmp(i_clust) = nanmean(squareform(pdist(score_pfc(i_gmm_idx == i_clust,1:opt.num_pcs))),'all');
    median_l2_tmp(i_clust) = nanmedian(squareform(pdist(score_pfc(i_gmm_idx == i_clust,1:opt.num_pcs))),'all');
end 
median_clust_l2(i_lam,i_num_pcs,i_n_clusters) = nanmean(mean_l2_tmp); 
mean_clust_l2(i_lam,i_num_pcs,i_n_clusters) = nanmedian(median_l2_tmp); 


%% Use ARI to compare choices in nonparametric way


% for paring down, try using the nearest neighbor hit metric