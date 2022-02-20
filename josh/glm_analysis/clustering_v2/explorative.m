%% Some more out there ideas 

%% Load data

% paths
% data_path = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_20211105_50ms_bins/GLM_coeffs_20211105_50_ms_bins.mat';
% old_clusters_path = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_20211105_50ms_bins/GMM_clust_20211105_50_ms_bins.mat';

data_path = 'D:\patchforaging_data\GLM_coeffs_20220213_50_ms_bins_logDVs_logScaleKern'; 
waveforms_path = 'D:\patchforaging_data\waveforms'; 

glm_dat = load(data_path);
rew_kern_mask = cellfun(@(x) strcmp(x(1:3),'Rew'),glm_dat.var_name); 
time_mask = cellfun(@(x) strcmp(x(1:4),'Time'),glm_dat.var_name); 
total_rew_mask = cellfun(@(x) strcmp(x(1:5),'Total'),glm_dat.var_name); 

coeffs_to_cluster = any([rew_kern_mask time_mask total_rew_mask],2); 

% pull off friends with nonzero cognitive variables
any_nonzero_mask = sum(abs(glm_dat.beta_all(coeffs_to_cluster,:))>0)'>0;
beta_any_nonzero = glm_dat.beta_all(:,any_nonzero_mask); 
all_nonzero_cellIDs = glm_dat.cellID(any_nonzero_mask);
region = glm_dat.brain_region_rough(any_nonzero_mask); 
depth = glm_dat.depth_from_surface(any_nonzero_mask); 
pfc_mask = strcmp("PFC",region);

beta_all = beta_any_nonzero(coeffs_to_cluster,:);
beta_pfc = beta_any_nonzero(coeffs_to_cluster,pfc_mask);

%% Let's try umapping it
% try across a range of n_neighbors and min distance
min_dist_test = [.1 .15 .2 .3]; 
n_neighbors_test = [10 20 30 50]; 

beta_all_norm = zscore(beta_all,[],2);
beta_pfc_norm = zscore(beta_pfc,[],2);

n_components = 2;

figure()
for i_n_neighbors = 1:numel(n_neighbors_test)
    n_neighbors = n_neighbors_test(i_n_neighbors);
    for i_min_dist = 1:numel(min_dist_test)
        min_dist = min_dist_test(i_min_dist);
        X_umap = run_umap(beta_pfc_norm','n_components',n_components,'n_neighbors',n_neighbors,'min_dist',min_dist, 'verbose','none');
        subplot(numel(min_dist_test),numel(n_neighbors_test),i_n_neighbors + (i_min_dist - 1) * numel(n_neighbors_test))
        scatter(X_umap(:,1),X_umap(:,2),3,'o')
%         gscatter(X_umap(:,1),X_umap(:,2),region,[],'o',3,'legend',[])
    end
end

%% 
gscatter(X_umap(:,1),X_umap(:,2),region,[],'o',3)

