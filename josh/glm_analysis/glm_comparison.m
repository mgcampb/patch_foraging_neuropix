%% A script to analyze relation of GLM coefficients to observation of putative sequential activity
%  Esp interested in diversity of time since reward kernels
%  Relation of GLM results to transients selection 

%% GLM and transient selection path 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table.mat';
load(paths.transients_table);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 14:18 23 25];   
mouse_grps = {1:2,3:8,10:13,14:18,[23 25]};  % note this should be 14:18
mouse_names = ["m75","m76","m78","m79","m80"];
session_titles = cell(numel(mouse_grps),1);
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];
        session_titles{mIdx}{i} = session_title;
    end
end  

opt.pval_thresh = .05;
opt.brain_region = "PFC";

%% Load GLM results per session 

beta_sig_cells = cell(numel(mouse_grps),1); 
sig_cell_cellIDs = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)  
    beta_sig_cells{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    sig_cell_cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    for i = 1:numel(mouse_grps{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx};
        session_glm_path = [paths.glm_results,'/',session]; 
        fit = load(session_glm_path); 
        
        % significant cells chosen as those w/ pval_full_vs_base < .05 and
        % at least one nonzero coeff on rew kernel or decision variable (15:end) 
        keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
%         sig_cell_ix = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
        sig_cell_ix = keep_cell & fit.pval_full_vs_base<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,:))>0)'>0;
        sig_cell_cellIDs{mIdx}{i} = fit.good_cells(sig_cell_ix);
        
        beta_sig_cells{mIdx}{i} = fit.beta_all(:,sig_cell_ix);
    end
end 
var_names = fit.var_name; 

%% Concatenate beta coeffs

beta_sig_cells_cat = cat(1,beta_sig_cells{:});
beta_sig_cells_cat = cat(2,beta_sig_cells_cat{:})';

sig_cells_cellIDs_cat = cat(1,sig_cell_cellIDs{:});
sig_cells_cellIDs_cat = cat(2,sig_cells_cellIDs_cat{:})';

% check that cells are same:
isequal(sig_cells.CellID,sig_cells_cellIDs_cat)

%% 0i) scatter distribution of time on patch encoding between clusters 

time_on_patch_bool = cellfun(@(x) strcmp(x(1:min(length(x),6)),"TimeOn"),var_names);
time_since_rew_bool = cellfun(@(x) strcmp(x(1:min(length(x),9)),"TimeSince"),var_names);

cluster = sig_cells.KMeansCluster;

cool3 = cool(3); 
x = {[1 2] [3.5 4.5] [6 7]}; 

figure() ;hold on
subplot(1,2,1);hold on
for i_cluster = 1:3
    time_on_patch_cluster = beta_sig_cells_cat(cluster == i_cluster,time_since_rew_bool);
    nNeurons = size(time_on_patch_cluster,1); 
    scatter(x{i_cluster}(1) + .1 * randn(nNeurons,1), time_on_patch_cluster(:,2),5,cool3(2,:),'o');
    scatter(x{i_cluster}(2) + .1 * randn(nNeurons,1), time_on_patch_cluster(:,3),5,cool3(3,:),'o');
%     disp(length(find(sum(abs(time_on_patch_cluster(:,2:3)),2) > 0)) / nNeurons)
end
xticks(cat(2,x{:}))
xticklabels(["Cluster 1 2 uL","Cluster 1 4 uL","Cluster 2 2 uL","Cluster 2 4 uL","Cluster 3 2 uL","Cluster 3 4 uL"])
xtickangle(45) 
ylim([-.4 .4])
title("Time since Reward Coefficients") 
ylabel("beta (a.u.)")
% Now time on patch
subplot(1,2,2);hold on
for i_cluster = 1:3
    time_on_patch_cluster = beta_sig_cells_cat(cluster == i_cluster,time_on_patch_bool);
    nNeurons = size(time_on_patch_cluster,1); 
    scatter(x{i_cluster}(1) + .1 * randn(nNeurons,1), time_on_patch_cluster(:,2),5,cool3(2,:),'o');
    scatter(x{i_cluster}(2) + .1 * randn(nNeurons,1), time_on_patch_cluster(:,3),5,cool3(3,:),'o');
end
xticks(cat(2,x{:}))
xticklabels(["Cluster 1 2 uL","Cluster 1 4 uL","Cluster 2 2 uL","Cluster 2 4 uL","Cluster 3 2 uL","Cluster 3 4 uL"])
xtickangle(45) 
ylim([-.4 .4])
title("Time on Patch Coefficients")  

%% 0ii) now barplot proportion of significant cells 
x = 1:3;
figure()
subplot(1,2,1);hold on
for i_cluster = 1:3
    time_on_patch_cluster = beta_sig_cells_cat(cluster == i_cluster,time_since_rew_bool);
    nNeurons = size(time_on_patch_cluster,1); 
    prop_sig = 100 * length(find(sum(abs(time_on_patch_cluster(:,2:3)),2) > 0)) / nNeurons;
    bar(x(i_cluster),prop_sig)
end 
title("% Sig Time Since Reward")
xticks(x);xticklabels(["Cluster 1","Cluster 2","Cluster 3"])

subplot(1,2,2);hold on
for i_cluster = 1:3
    time_on_patch_cluster = beta_sig_cells_cat(cluster == i_cluster,time_on_patch_bool);
    nNeurons = size(time_on_patch_cluster,1); 
    prop_sig = 100 * length(find(sum(abs(time_on_patch_cluster(:,2:3)),2) > 0)) / nNeurons;
    bar(x(i_cluster),prop_sig)
end 
title("% Sig Time On Patch")
xticks(x);xticklabels(["Cluster 1","Cluster 2","Cluster 3"])


%% 1i) Now investigate {diversity} of reward kernel coefficients
rew_kernel_bool = cell(3); 
rew_kernel_bool{1} = find(cellfun(@(x) strcmp(x(1:min(length(x),7)),"RewKern") & strcmp(x(end-2:end),"1uL"),var_names));
rew_kernel_bool{2} = find(cellfun(@(x) strcmp(x(1:min(length(x),7)),"RewKern") & strcmp(x(end-2:end),"2uL"),var_names));
rew_kernel_bool{3} = find(cellfun(@(x) strcmp(x(1:min(length(x),7)),"RewKern") & strcmp(x(end-2:end),"4uL"),var_names));

n_kernels = length(find(rew_kernel_bool{1})); 
x = 1:n_kernels; 
% first plot distribution of kernel coeffs separated by reward size and cluster  
figure(); 
for i_cluster = 1:3 
    cluster_neurons = cluster == i_cluster;
    nNeurons = length(find(cluster_neurons)); 
    noise = .1 * randn(nNeurons,1); 
    cluster_means = nan(3,n_kernels); 
    for i_rewsize = 1:3 
        subplot(4,3,3 * (i_rewsize - 1) + i_cluster);hold on
        for i_kern = 1:n_kernels 
            this_kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize}(i_kern));
            cluster_means(i_rewsize,i_kern) = mean(this_kernel_coeffs);
            scatter(x(i_kern) + noise,this_kernel_coeffs,5,cool3(i_rewsize,:),'o'); 
        end 
        plot(cluster_means(i_rewsize,:),'k','linewidth',1.5) 
        if i_cluster == 3
            ylim([-.1 .1])
        end  
        if i_rewsize == 1 
            title(sprintf("Cluster %i",i_cluster))
        end
        if i_cluster == 1
            ylabel("Coeff Distn (a.u.)")
        end
    end  
    subplot(4,3,3 * (4-1) + i_cluster) ;hold on
    for i_rewsize = 1:3
        plot(x,cluster_means(i_rewsize,:),'color',cool3(i_rewsize,:),'linewidth',1.5)
    end  
    if i_cluster == 1
        ylabel("Mean Coeff (a.u.)")
    end 
    xlabel("Reward Kernel")
end

%% 1ii) Now analyze reward kernels within single cells

% first across clusters
i_rewsize = 3;
figure() 
for i_cluster = 1:3
    cluster_neurons = cluster == i_cluster;

    % pull out reward kernel coefficients
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize});
    max(kernel_coeffs,2);
    [~,ix] = max(kernel_coeffs,[],2);
    [~,peaksort_kernel] = sort(ix);
    
    % pull out peaks from transients sort
    rew0_peak = transients_table.Rew0_peak_pos;
    rew0_peak3 = rew0_peak(transients_table.GLM_Cluster == i_cluster);
    [~,peaksort_rew0_driscoll] = sort(rew0_peak3);
    
    rew1plus_peak = transients_table.Rew1plus_peak_pos;
    rew1plus_peak_cluster = rew1plus_peak(transients_table.GLM_Cluster == i_cluster);
    [~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    
    subplot(3,3,i_cluster)
%     imagesc(flipud(zscore(kernel_coeffs(peaksort_kernel,:),[],2)))
    imagesc(flipud(kernel_coeffs(peaksort_kernel,:)))
    title(sprintf("Cluster %i",i_cluster)) 
    ylabel("Sort by Peak Kernel Coeff")
    subplot(3,3,3 + i_cluster)
%     imagesc(flipud(zscore(kernel_coeffs(peaksort_rew0_driscoll,:),[],2))) 
    imagesc(flipud(kernel_coeffs(peaksort_rew0_driscoll,:))) 
    ylabel("Sort by Shuffle Test Using Rew0")
    subplot(3,3,6 + i_cluster)
%     imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2))) 
    imagesc(flipud((kernel_coeffs(peaksort_rew1plus_driscoll,:))))
    ylabel("Sort by Shuffle Test Using Rew 1plus") 
    xlabel("Reward Kernel")
end

%% 1iii) Now visualize kernels, across reward sizes
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rewsizes = [1 2 4]; 
for i_cluster = 1:3 
    cluster_neurons = cluster == i_cluster; 
    rew1plus_peak_cluster = rew1plus_peak(transients_table.GLM_Cluster == i_cluster);
    [~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    for i_rewsize = 1:3
        subplot(3,3,3 * (i_rewsize - 1) + i_cluster)
        kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize}); 
        imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))
%         imagesc(flipud(kernel_coeffs(peaksort_rew1plus_driscoll,:)))
        if i_cluster == 1
            ylabel(sprintf("%i uL \n Transient Sorted",rewsizes(i_rewsize)))
        end 
        if i_rewsize == 1
            title(sprintf("Cluster %i",i_cluster))
        end
    end 
end

%% 1iv) Cross reward size within cluster 3 

i_cluster = 3; 
cluster_neurons = cluster == i_cluster;  
nNeurons = length(find(cluster_neurons)); 
rew1plus_peak_cluster = rew1plus_peak(transients_table.GLM_Cluster == i_cluster);
[~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);

peak_ix = nan(3,nNeurons); 
mdls = cell(3,1); 
figure()
for i_rewsize = 1:3
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize}); 
    [~,peak_ix(i_rewsize,:)] = max(kernel_coeffs,[],2);  
    peak_ix(i_rewsize,:) = peak_ix(i_rewsize,peaksort_rew1plus_driscoll);
    peak_ix(i_rewsize,peak_ix(i_rewsize,:) == 1) = NaN;
    
    subplot(3,2,1 + 2 * (i_rewsize-1))
    imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))  
    if i_rewsize == 1
        title("Cluster 3 Reward Kernel Coefficients")
    end 
    if i_rewsize == 3 
        xlabel("Reward kernel")
    end 
    ylabel(sprintf("%i uL \n Transient Sorted",rewsizes(i_rewsize))) 
    
    subplot(3,2,2 + 2 * (i_rewsize-1));hold on
    scatter(.1 * randn(1,nNeurons) + peak_ix(i_rewsize,:),1:numel(peak_ix(i_rewsize,:)),15,cool3(i_rewsize,:))  
    xlim([0 12])
    mdls{i_rewsize} = fitlm(peak_ix(i_rewsize,~isnan(peak_ix(i_rewsize,:))),1:numel(peak_ix(i_rewsize,~isnan(peak_ix(i_rewsize,:)))) );   
%     plot(mdls{i_rewsize}.Fitted,1:numel(peak_ix(i_rewsize,:)),'k','linewidth',1.5)
    
    if i_rewsize == 3 
        xlabel("Reward kernel")
    end  
    if i_rewsize == 1
        title("Cluster 3 Peak Reward Kernel")
    end 
end

% figure();hold on
% for i_rewsize = 1:3
%     bar(i_rewsize,mdls{i_rewsize}.Coefficients.Estimate(2),'FaceColor',cool3(i_rewsize,:));
%     errorbar(i_rewsize,mdls{i_rewsize}.Coefficients.Estimate(2),mdls{i_rewsize}.Coefficients.SE(2))
% end

%% 1iv) Relationship between peak sort and time on patch encoding?   

i_cluster = 3; 
cluster_neurons = cluster == i_cluster;  
nNeurons = length(find(cluster_neurons)); 
rew1plus_peak_cluster = rew1plus_peak(transients_table.GLM_Cluster == i_cluster);
[~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);

time_on_patch_ix = find(time_on_patch_bool);
corrmats = cell(3,1);
figure()
for i_rewsize = 1:3
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize}); 
    time_on_patch_coeffs = beta_sig_cells_cat(cluster_neurons,time_on_patch_ix(i_rewsize));  
    ix_of_interest = [rew_kernel_bool{i_rewsize} time_on_patch_ix(i_rewsize)]; 
    
    subplot(3,3,1 + 3 * (i_rewsize-1))
    imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))  
%     imagesc(flipud(kernel_coeffs(peaksort_rew1plus_driscoll,:)))  
    if i_rewsize == 1
        title("Reward Kernel Coefficients")
    end 
    if i_rewsize == 3 
        xlabel("Reward kernel")
    end 
    ylabel(sprintf("%i uL \n Transient Sorted",rewsizes(i_rewsize))) 
    
    subplot(3,3,2 + 3 * (i_rewsize-1));hold on
    scatter(time_on_patch_coeffs(peaksort_rew1plus_driscoll),1:numel(time_on_patch_coeffs),15,cool3(i_rewsize,:))
    
    if i_cluster == 3
        xlim([-.5,.1])  
    elseif i_cluster == 2 
        xlim([-.2,.3])
    end
    
    if i_rewsize == 3 
        xlabel("Reward kernel")
    end  
    if i_rewsize == 1
        title("Time on Patch Coefficient")
    end   
    
    corrmats{i_rewsize} = corrcoef(beta_sig_cells_cat(cluster_neurons,ix_of_interest));
    subplot(3,3,3 + 3 * (i_rewsize-1)); % hold on 
    imagesc(corrmats{i_rewsize})
    colorbar() 
    caxis([-1,1])
    
    if i_rewsize == 1
        title("Coefficient Cross Correlation")
    end   
    if i_rewsize == 3 
        xlabel("Rew kernel, time on patch")
    end  
    
%     suptitle(sprintf("Cluster %i",i_cluster))
end

%% Do a real kernel-wise corrcoef thing maybe later
figure();hold on
for i_rewsize = 1:3 
    plot(corrmats{i_rewsize}(12,1:end-1),'linewidth',1.5,'color',cool3(i_rewsize,:))
end
ylabel("Corrcoef betw kernel coefficient and time on patch") 
xlabel("Reward Kernel")