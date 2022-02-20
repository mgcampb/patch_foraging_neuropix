%% A script to analyze relation of GLM coefficients to observation of putative sequential activity
%  Esp interested in diversity of time since reward kernels
%  Relation of GLM results to transients selection 

%  Note that this turns into figure 9

%% GLM and transient selection path 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
% paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
% paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_gmm_mc_cohort_PFC.mat';
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat';
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
        sig_cell_ix = keep_cell & sum(abs(fit.beta_all(fit.base_var==0,:))>0)'>0;
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

cluster = sig_cells.GMM_cluster;

cool3 = cool(3); 
x = {[1 2] [3.5 4.5] [6 7]}; 

figure() ;hold on
subplot(1,2,1);hold on
for i_cluster = 1:2
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
for i_cluster = 1:2
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
x = 1:5;
figure()
subplot(1,2,1);hold on
for i_cluster = 1:5
    time_on_patch_cluster = beta_sig_cells_cat(cluster == i_cluster,time_since_rew_bool);
    nNeurons = size(time_on_patch_cluster,1); 
    prop_sig = 100 * length(find(sum(abs(time_on_patch_cluster(:,2:3)),2) > 0)) / nNeurons;
    bar(x(i_cluster),prop_sig)
end 
title("% Sig Time Since Reward")
xticks(x);xticklabels(["Cluster 1","Cluster 2","Cluster 3"])

subplot(1,2,2);hold on
for i_cluster = 1:5
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
for i_cluster = 1:5
    cluster_neurons = sig_cells.GMM_cluster == i_cluster;

    % pull out reward kernel coefficients
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize});
    max(kernel_coeffs,2);
    [~,ix] = max(kernel_coeffs,[],2);
    [~,peaksort_kernel] = sort(ix);
    
    % pull out peaks from transients sort
    rew0_peak = transients_table.Rew0_peak_pos;
    rew0_peak3 = rew0_peak(transients_table.gmm_cluster == i_cluster);
    [~,peaksort_rew0_driscoll] = sort(rew0_peak3);
    
    rew1plus_peak = transients_table.Rew1plus_peak_pos;
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    [~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    
    subplot(3,5,i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(peaksort_kernel,:),[],2)))
%     imagesc(flipud(kernel_coeffs(peaksort_kernel,:)))
    title(sprintf("Cluster %i",i_cluster)) 
    ylabel("Sort by Peak Kernel Coeff")
    subplot(3,5,5 + i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(peaksort_rew0_driscoll,:),[],2))) 
%     imagesc(flipud(kernel_coeffs(peaksort_rew0_driscoll,:))) 
    ylabel("Sort by Shuffle Test Using Rew0")
    subplot(3,5,10 + i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2))) 
%     imagesc(flipud((kernel_coeffs(peaksort_rew1plus_driscoll,:))))
    ylabel("Sort by Shuffle Test Using Rew 1plus") 
    xlabel("Reward Kernel")
end

%% 1ii.5) Tiling in dip timing?  

% first across clusters
i_rewsize = 3;
figure() 
for i_cluster = 1:5
    cluster_neurons = sig_cells.GMM_cluster == i_cluster;

    % pull out reward kernel coefficients
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize});
    max(kernel_coeffs,2);
    [~,ix] = min(kernel_coeffs,[],2);
    [~,troughsort_kernel] = sort(ix);
    
    % pull out peaks from transients sort
    rew0_peak = transients_table.Rew0_peak_neg;
    rew0_peak3 = rew0_peak(transients_table.gmm_cluster == i_cluster);
    [~,troughsort_rew0_driscoll] = sort(rew0_peak3);
    
    rew1plus_peak = transients_table.Rew1plus_peak_neg;
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    [~,troughsort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    
    subplot(3,5,i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(troughsort_kernel,:),[],2)))
%     imagesc(flipud(kernel_coeffs(troughsort_kernel,:)))
    title(sprintf("Cluster %i",i_cluster)) 
    ylabel("Sort by Min Kernel Coeff")
    subplot(3,5,5 + i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(troughsort_rew0_driscoll,:),[],2))) 
%     imagesc(flipud(kernel_coeffs(peaksort_rew0_driscoll,:))) 
    ylabel("Sort by Shuffle Test Using Negative Rew0")
    subplot(3,5,10 + i_cluster)
    imagesc(flipud(zscore(kernel_coeffs(troughsort_rew1plus_driscoll,:),[],2))) 
%     imagesc(flipud((kernel_coeffs(peaksort_rew1plus_driscoll,:))))
    ylabel("Sort by Shuffle Test Using Negative 1plus") 
    xlabel("Reward Kernel")
end

%% 1ii.75 Visualize trough and peak tiling together, both with 1plus reward transient discovery 
i_rewsize = 3;
rewsizes = [1 2 4];
peak_binary = [0 1 0 1]; % look at peak or trough
figure()
for i_cluster = 1:4
    cluster_neurons = sig_cells.GMM_cluster == i_cluster;
    % pull out reward kernel coefficients
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize});
    
    rew1plus_peak = transients_table.Rew1plus_peak_pos;
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    [~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    
    rew1plus_trough = transients_table.Rew1plus_peak_neg;
    rew1plus_trough_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
    [~,troughsort_rew1plus_driscoll] = sort(rew1plus_trough_cluster);
    
    subplot(1,4,i_cluster)  
    if peak_binary(i_cluster) == true
        imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))  
        title(sprintf("Cluster %i \n Peak Sorted",i_cluster))
    else 
        imagesc(flipud(zscore(kernel_coeffs(troughsort_rew1plus_driscoll,:),[],2)))  
        title(sprintf("Cluster %i \n Trough Sorted",i_cluster))
    end
    if i_cluster == 1
        ylabel(sprintf("%i uL \n Transient Response-Sorted",rewsizes(i_rewsize)))
    end
    
    set(gca,'fontsize',13)
    xticks(2:2:11) 
    xticklabels(.2 * (2:2:11)) 
    xlabel("Reward Kernel Center (sec)")
    
%     
%     subplot(2,4,4 + i_cluster)
%     imagesc(flipud(zscore(kernel_coeffs(troughsort_rew1plus_driscoll,:),[],2)))
%     if i_cluster == 1
%         ylabel(sprintf("%i uL \n Trough Sorted",rewsizes(i_rewsize)))
%     end
%     set(gca,'fontsize',13)
%     xticks(2:2:11) 
%     xticklabels(.2 * (2:2:11)) 
%     xlabel("Reward Kernel Center (sec)")
end


%% 1iii) Now visualize kernels, across reward sizes
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rew1plus_trough = transients_table.Rew1plus_peak_neg;
rewsizes = [1 2 4]; 
peak_binary = [0 1 0 1]; % look at peak or trough
figure()
for i_cluster = 1:4
    cluster_neurons = cluster == i_cluster; 
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    [~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);
    rew1plus_trough_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
    [~,troughsort_rew1plus_driscoll] = sort(rew1plus_trough_cluster);
    
    for i_rewsize = 1:numel(rewsizes)
        subplot(3,4,4 * (i_rewsize - 1) + i_cluster)
        kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize});  
        if peak_binary(i_cluster) == true
%             imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))
            imagesc(flipud(kernel_coeffs(peaksort_rew1plus_driscoll,:)))
            if i_rewsize == 1
                title(sprintf("Cluster %i \n Peak Sorted",i_cluster))
            end
        else
%             imagesc(flipud(zscore(kernel_coeffs(troughsort_rew1plus_driscoll,:),[],2))) 
            imagesc(flipud(kernel_coeffs(troughsort_rew1plus_driscoll,:))) 
            if i_rewsize == 1
                title(sprintf("Cluster %i \n Trough Sorted",i_cluster))
            end
        end
%         imagesc(flipud(kernel_coeffs(peaksort_rew1plus_driscoll,:)))
        if i_cluster == 1
            ylabel(sprintf("%i uL \n Transient Sorted",rewsizes(i_rewsize)))
        end 
        set(gca,'fontsize',13)
        xticks(2:2:11)
        xticklabels(.2 * (2:2:11)) 
        if i_rewsize == numel(rewsizes)
            xlabel("Reward Kernel Center (sec)")
        end
    end
end

%% 1iv) Cross reward size differences
rew1plus_peak = transients_table.Rew1plus_peak_neg;
i_cluster = 3; 
cluster_neurons = cluster == i_cluster;  
nNeurons = length(find(cluster_neurons)); 
rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
[~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);

peak_ix = nan(3,nNeurons); 
mdls = cell(3,1); 
figure()
for i_rewsize = 1:3
    kernel_coeffs = beta_sig_cells_cat(cluster_neurons,rew_kernel_bool{i_rewsize}); 
    [~,peak_ix(i_rewsize,:)] =  min(kernel_coeffs,[],2);  
    peak_ix(i_rewsize,:) = peak_ix(i_rewsize,peaksort_rew1plus_driscoll);
    peak_ix(i_rewsize,peak_ix(i_rewsize,:) == 1) = NaN;
    
    subplot(3,2,1 + 2 * (i_rewsize-1))
    imagesc(flipud(zscore(kernel_coeffs(peaksort_rew1plus_driscoll,:),[],2)))  
%     imagesc(flipud(kernel_coeffs(peaksort_rew1plus_driscoll,:)))  
    if i_rewsize == 1
        title(sprintf("Cluster %i Reward Kernel Coefficients",i_cluster))
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
        title(sprintf("Cluster %i Peak Rew Kernel",i_cluster))
    end 
end

% figure();hold on
% for i_rewsize = 1:3
%     bar(i_rewsize,mdls{i_rewsize}.Coefficients.Estimate(2),'FaceColor',cool3(i_rewsize,:));
%     errorbar(i_rewsize,mdls{i_rewsize}.Coefficients.Estimate(2),mdls{i_rewsize}.Coefficients.SE(2))
% end

%% 1v) Distribution of peaks and troughs between clusters 
% 1) Differences in dimensionality of reward response? 
% 2) Differences in timing of reward response?
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rew1plus_trough = transients_table.Rew1plus_peak_neg;
figure()
for i_cluster = 1:5
    nNeurons_cluster = length(find(transients_table.gmm_cluster == i_cluster)); 
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    nNeurons_cluster_sigPeak = length(find(~isnan(rew1plus_peak_cluster))); 
    rew1plus_trough_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
    nNeurons_cluster_sigTrough = length(find(~isnan(rew1plus_trough_cluster))); 
    subplot(2,5,i_cluster)  
    histogram(rew1plus_peak_cluster,0:.2:2) 
    title(sprintf("Cluster %i \n %i%% Sig Peak Cells (%i/%i) \n H = %.3f",i_cluster,round(100*nNeurons_cluster_sigPeak/nNeurons_cluster),nNeurons_cluster_sigPeak,nNeurons_cluster,calc_shannonH(rew1plus_peak_cluster,0:.2:2)))
    xlabel("Peak Time (sec)")
    if i_cluster == 1
        ylabel("Distribution of Peak Time")
    end
    set(gca,'fontsize',12)
    subplot(2,5,5 + i_cluster) 
    histogram(rew1plus_trough_cluster,0:.2:2)
    title(sprintf("%i%% Sig Trough Cells (%i/%i) \n H = %.3f",round(100*nNeurons_cluster_sigTrough/nNeurons_cluster),nNeurons_cluster_sigTrough,nNeurons_cluster,calc_shannonH(rew1plus_trough_cluster,0:.2:2)))
    xlabel("Trough Time (sec)")
    if i_cluster == 1
        ylabel("Distribution of Trough Time")
    end 
    set(gca,'fontsize',12)
end

%% 1v.1) Distribution of peaks and troughs between clusters, only show peak or trough per cluster
% 1) Differences in dimensionality of reward response? 
% 2) Differences in timing of reward response?
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rew1plus_trough = transients_table.Rew1plus_peak_neg;
peak_binary = [0 1 0 1]; % look at peak or trough
figure()
for i_cluster = 1:4
    nNeurons_cluster = length(find(transients_table.gmm_cluster == i_cluster)); 
    rew1plus_peak_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    nNeurons_cluster_sigPeak = length(find(~isnan(rew1plus_peak_cluster))); 
    rew1plus_trough_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
    nNeurons_cluster_sigTrough = length(find(~isnan(rew1plus_trough_cluster))); 
    subplot(1,4,i_cluster)  
    if peak_binary(i_cluster) == true
        histogram(rew1plus_peak_cluster,0:.2:2) 
        title(sprintf("Cluster %i \n %i%% Sig Peak Cells (%i/%i) \n H = %.3f",i_cluster,round(100*nNeurons_cluster_sigPeak/nNeurons_cluster),nNeurons_cluster_sigPeak,nNeurons_cluster,calc_shannonH(rew1plus_peak_cluster,0:.2:2)))
        xlabel("Peak Time (sec)")
        ylabel("Distribution of Peak Time")
    else 
        histogram(rew1plus_trough_cluster,0:.2:2)
        title(sprintf("Cluster %i \n %i%% Sig Trough Cells (%i/%i) \n H = %.3f",i_cluster,round(100*nNeurons_cluster_sigTrough/nNeurons_cluster),nNeurons_cluster_sigTrough,nNeurons_cluster,calc_shannonH(rew1plus_trough_cluster,0:.2:2)))
        xlabel("Trough Time (sec)")
        ylabel("Distribution of Trough Time")
    end
    set(gca,'fontsize',12)

end

%% 1v.25) Distribution of peaks and troughs between clusters 
% 1) Differences in dimensionality of reward response? 
% 2) Differences in timing of reward response?
[~,rew1plus_peak] = max(beta_sig_cells_cat(:,rew_kernel_bool{3}),[],2); 
[~,rew1plus_trough] = min(beta_sig_cells_cat(:,rew_kernel_bool{3}),[],2); 
figure()
for i_cluster = 1:5
    rew1plus_peak_cluster = rew1plus_peak(sig_cells.GMM_cluster == i_cluster);
    rew1plus_trough_cluster = rew1plus_trough(sig_cells.GMM_cluster == i_cluster);
    subplot(2,5,i_cluster)  
    histogram(.2*rew1plus_peak_cluster,.2*(1:11)) 
    title(sprintf("Cluster %i \n H = %.3f",i_cluster,calc_shannonH(.2*rew1plus_peak_cluster,.2*(1:11))))
    xlabel("Peak Kernel (sec)")
    if i_cluster == 1
        ylabel("Distribution of Peak Kernel")
    end
    set(gca,'fontsize',12)
    subplot(2,5,5 + i_cluster) 
    histogram(.2*rew1plus_trough_cluster,.2*(1:11))
    title(sprintf("H = %.3f",calc_shannonH(.2*rew1plus_trough_cluster,.2*(1:11))))
    xlabel("Trough Kernel (sec)")
    if i_cluster == 1
        ylabel("Distribution of Trough Kernel")
    end 
    set(gca,'fontsize',12)
end

%% 1v.5) Distribution of peaks and troughs between clusters overlaid, using KDE
% 1) Differences in dimensionality of reward response? 
% 2) Differences in timing of reward response?
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rew1plus_trough = transients_table.Rew1plus_peak_neg;
gmm_colors = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255;
peak_binary = [0 1 0 1]; % look at peak or trough

if ~ismember('numeric_waveform_type',transients_table.Properties.VariableNames)
    % load waveform stuff if we don't have it 
    paths.waveform_clusters = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/waveform_cluster.mat';
    paths.waveforms = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/waveforms';
    load(paths.waveform_clusters) 
    waveforms = []; 
    session_all = unique(sig_cells.Session);
    for sIdx = 1:numel(session_all) 
        dat = load(fullfile(paths.waveforms,session_all{sIdx}));   
        waveforms = [waveforms ; dat.mean_waveform]; 
    end 
    %  Across all cells, not just sig cells
    waveform_types = ["Narrow","Regular","TriPhasic"]; 
    numeric_waveform_types = nan(size(waveform_cluster.WaveformType,1),1);
    for i_waveform_type = 1:numel(waveform_types)
        this_waveform_type = waveform_types(i_waveform_type);  
        this_waveform_type = this_waveform_type{:}; 
        these_cells = cellfun(@(x) strcmp(this_waveform_type,x),waveform_cluster.WaveformType);  
        numeric_waveform_types(these_cells) = i_waveform_type;
    end
    transients_table.numeric_waveform_types = numeric_waveform_types;
end

figure();hold on 
% lines4 = lines(4); 
p = []; 
H = [];
peak_distns = cell(5,1); 
counter = 1;
for i_cluster = 1:4
%     kernel_coeffs = beta_sig_cells_cat(sig_cells.GMM_cluster == i_cluster,rew_kernel_bool{3}); 
    if i_cluster ~= 2
        if peak_binary(i_cluster) == true
            rew_resp_ix_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
    %         [~,rew_resp_ix_cluster] =  max(kernel_coeffs,[],2); 
        else
            rew_resp_ix_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
    %         [~,rew_resp_ix_cluster] =  min(kernel_coeffs,[],2); 
        end
    %     histogram(rew_resp_ix_cluster) % ,0:.1:2)
        %     disp(nanmean(rew_resp_ix_cluster)) (rew_resp_ix_cluster >= .2)
    %     if i_cluster == 2
    %         i_pdf = pdf(fitdist(rew_resp_ix_cluster(rew_resp_ix_cluster >= .2),'kernel','Kernel','Normal'), (0:.01:2));
    %         plot((0:.01:2),i_pdf,'color',lines4(i_cluster,:),'linewidth',2,'linestyle',':');
    %         xline(mean(fitdist(rew_resp_ix_cluster(rew_resp_ix_cluster >= .2),'kernel','Kernel','Normal')),'linewidth',2,'color',lines4(i_cluster,:),'linestyle',':');
    %     end

        i_pdf = pdf(fitdist(rew_resp_ix_cluster,'kernel','Kernel','Normal'), (0:.01:2)); 
        p(i_cluster) = plot((0:.01:2),i_pdf,'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle','-');
%         h = histogram(rew_resp_ix_cluster,(0:.05:2),'normalization','pdf'); 
%         h.FaceColor = gmm_colors(i_cluster,:); 
%         h.FaceAlpha = .2;
       
%         xline(mean(fitdist(rew_resp_ix_cluster,'kernel','Kernel','Normal')),'linewidth',2,'color',gmm_colors(i_cluster,:),'linestyle','-');
        H = [H calc_shannonH(rew_resp_ix_cluster,0:.2:2)];
        peak_distns{counter} = rew_resp_ix_cluster; 
        counter = counter + 1; 
    elseif i_cluster == 2
        rew_resp_ix_cluster_ns = rew1plus_peak(transients_table.gmm_cluster == i_cluster & transients_table.numeric_waveform_types == 1);
        rew_resp_ix_cluster_rs = rew1plus_peak(transients_table.gmm_cluster == i_cluster & transients_table.numeric_waveform_types == 2);
        
        i_pdf = pdf(fitdist(rew_resp_ix_cluster_ns,'kernel','Kernel','Normal'), (0:.01:2)); 
        p(2) = plot((0:.01:2),i_pdf,'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle',':');
%         h = histogram(rew_resp_ix_cluster_ns,(0:.05:2),'normalization','pdf'); 
%         h.FaceColor = gmm_colors(i_cluster,:); 
%         h.LineStyle = ':'; 
        i_pdf = pdf(fitdist(rew_resp_ix_cluster_rs,'kernel','Kernel','Normal'), (0:.01:2)); 
        p(4) = plot((0:.01:2),i_pdf,'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle','--');
%         h = histogram(rew_resp_ix_cluster_rs,(0:.05:2),'normalization','pdf'); 
%         h.FaceColor = gmm_colors(i_cluster,:); 
%         h.LineStyle = '--'; 
        H = [H calc_shannonH(rew_resp_ix_cluster_ns,0:.2:2)];
        H = [H calc_shannonH(rew_resp_ix_cluster_rs,0:.2:2)];
        peak_distns{counter} = rew_resp_ix_cluster_ns; 
        peak_distns{counter + 1} = rew_resp_ix_cluster_rs; 
        counter = counter + 2;
    end
    ylabel("Distribution of Peak/Trough Time") 
    xlabel("Peak/Trough Time")
    
    
end
legend(p,sprintf("Cluster 1 Trough; H = %.3f",H(1)),sprintf("Cluster 2 NS Peak; H = %.3f",H(2)),sprintf("Cluster 3 Trough; H = %.3f",H(4)),...
sprintf("Cluster 2 WS Peak; H = %.3f",H(3)),'fontsize',16) 
set(gca,'FontSize',14)

%% 1v.5.1) Bootstrap entropy estimate to get p-value 
n_bootstrap_resamples = 1000; 
bootstrap_resample_size = round(min(cellfun(@length,peak_distns))); 

H = nan(n_bootstrap_resamples,5); 
for i_distn = 1:numel(peak_distns)
    this_distn = peak_distns{i_distn}; 
    for k_resample = 1:n_bootstrap_resamples 
        this_resample = datasample(this_distn,bootstrap_resample_size,'Replace',false); 
        H(k_resample,i_distn) = calc_shannonH(this_resample,0:.2:2); 
    end
end

p = nan(numel(peak_distns),numel(peak_distns)); 
for i_distn1 = 1:numel(peak_distns)
    for i_distn2 = 1:numel(peak_distns) 
        p(i_distn1,i_distn2) = length(find(H(:,i_distn1) < H(:,i_distn2))) / n_bootstrap_resamples; 
    end
end

% now visualize bootstrap results
b = bar(nanmean(H,1)','LineWidth',1.5);hold on
b.FaceColor = 'Flat'; 
b.CData(1,:) = gmm_colors(1,:); 
b.CData(2,:) = gmm_colors(2,:); 
b.CData(3,:) = gmm_colors(2,:); 
b.CData(4,:) = gmm_colors(3,:); 
b.CData(5,:) = gmm_colors(4,:); 
errorbar(1:5,nanmean(H,1),nanstd(H,1),'.k','linewidth',1.5)
fig = gca; 
xticklabels({"Cluster 1","Cluster 2 NS","Cluster 2 WS","Cluster 3","Cluster 4"})
ylabel(sprintf("Shannon Entropy of\n Peak/Trough Time"))
ylim([0 3])
set(fig,'fontsize',14)
p

%% 1v.ecdfs; test for significance
rew1plus_peak = transients_table.Rew1plus_peak_pos;
rew1plus_trough = transients_table.Rew1plus_peak_neg;
peak_binary = [0 1 0 1]; % look at peak or trough
figure();hold on 
lines4 = lines(4); 
rew_resp_ix = cell(4,1); 
counter = 1; 
for i_cluster = 1:3
    if i_cluster ~= 2
        if peak_binary(i_cluster) == true
            rew_resp_ix_cluster = rew1plus_peak(transients_table.gmm_cluster == i_cluster);
        else
            rew_resp_ix_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
        end
        [f,x,~,fup] = ecdf(rew_resp_ix_cluster);  
        shadedErrorBar(x,f,fup - f,'lineProps',{'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle','-'}) 
        rew_resp_ix{counter} = rew_resp_ix_cluster; 
        counter = counter + 1; 
    else 
        rew_resp_ix_cluster_ns = rew1plus_peak(transients_table.gmm_cluster == i_cluster & transients_table.numeric_waveform_types == 1);
        rew_resp_ix_cluster_rs = rew1plus_peak(transients_table.gmm_cluster == i_cluster & transients_table.numeric_waveform_types == 2);
        
        [f,x,~,fup] = ecdf(rew_resp_ix_cluster_ns);
        shadedErrorBar(x,f,fup - f,'lineProps',{'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle',':'})
        [f,x,~,fup] = ecdf(rew_resp_ix_cluster_rs);
        shadedErrorBar(x,f,fup - f,'lineProps',{'color',gmm_colors(i_cluster,:),'linewidth',2,'linestyle','--'})
        rew_resp_ix{counter} = rew_resp_ix_cluster_ns; 
        rew_resp_ix{counter + 1} = rew_resp_ix_cluster_rs; 
        counter = counter + 2; 
    end
end

legend("Cluster 1 Trough","Cluster 2 NS Peak","Cluster 2 WS Peak","Cluster 3 Trough",'fontsize',16)
ylabel("ECDF of Peak/Trough Time")
xlabel("Peak/Trough Time")
ylim([0 1])
set(gca,'FontSize',13)


[h,p] = kstest2(rew_resp_ix_cluster_ns,rew_resp_ix_cluster_rs); % use this to get significance!

%% Bootstrap to test for significance of entropy differences 




%% 1nah) Relationship between peak sort and time on patch encoding?   

i_cluster = 3; 
cluster_neurons = cluster == i_cluster;  
nNeurons = length(find(cluster_neurons)); 
rew1plus_peak_cluster = rew1plus_trough(transients_table.gmm_cluster == i_cluster);
[~,peaksort_rew1plus_driscoll] = sort(rew1plus_peak_cluster);

time_on_patch_ix = find(time_on_patch_bool);
% time_on_patch_ix = find(time_since_rew_bool);
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
    
%     if i_cluster == 3
%         xlim([-.5,.1])  
%     elseif i_cluster == 2 
%         xlim([-.2,.3])
%     end
    
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

% %% Change GLM Clustering to 5 cluster GMM
% sigCellCluster = cell(numel(mouse_grps),1); 
% sig_cell_sessions = sig_cells.Session;  
% for mIdx = 1:numel(mouse_grps) 
%     sigCellCluster{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
%     for i = 1:numel(mouse_grps{mIdx})  
%         sIdx = mouse_grps{mIdx}(i);
%         data = load(fullfile(paths.data,sessions{sIdx}));    
%         good_cells = data.sp.cids(data.sp.cgs==2);  
%         nNeurons = length(good_cells);  
%         % get GLM cluster vector per session
%         sig_cellIDs_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).CellID;   
%         sig_clusters_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).GMM_cluster;
%         
%         sigCellCluster{mIdx}{i} = nan(nNeurons,1); 
%         sigCellCluster{mIdx}{i}(ismember(good_cells,sig_cellIDs_session)) = sig_clusters_session; 
%     end
% end 
% 
% cat_sigCellCluster = cat(1,sigCellCluster{:}); 
% cat_sigCellCluster = cat(1,cat_sigCellCluster{:}); 
% transients_table.gmm_cluster = cat_sigCellCluster; 