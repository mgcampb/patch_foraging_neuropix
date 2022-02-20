%% Test out the effect of using cell exclusion criteria to get clusters of cells that are more clustery

% - set min P(cluster assignment) 
% - set min nn hit rate

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

%% Perform PCA and look at cumulative variance explained

beta_all_norm = zscore(beta_all);
[coeff_all,score_all,~,~,expl_all] = pca(beta_all_norm');

beta_pfc_norm = zscore(beta_pfc);
[coeff_pfc,score_pfc,~,~,expl_pfc] = pca(beta_pfc_norm');

%% fit GMM

n_pcs = 3; 
n_clusters = 6; 
lambda = 0.5;
n_replicates = 50; 
this_gmm = fitgmdist(score_pfc(:,1:n_pcs),n_clusters,'RegularizationValue',lambda,'replicates',n_replicates);        
[gmm_labels,~,p_assignment] = cluster(this_gmm,score_pfc(:,1:n_pcs));

% align in order of prevalence 
[~,proportion_sort] = sort(histcounts(gmm_labels),'descend');
final_gmm_labels = nan(size(gmm_labels));
final_p_assignment = nan(size(p_assignment));
for i_clust = 1:n_clusters
    final_gmm_labels(gmm_labels == proportion_sort(i_clust)) = i_clust;
    final_p_assignment(:,i_clust) = p_assignment(:,proportion_sort(i_clust)); 
end 

%% visualize results 
scatter3(score_pfc(:,1),score_pfc(:,2),score_pfc(:,3))


