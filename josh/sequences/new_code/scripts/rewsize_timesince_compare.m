%% Compare Peaksorted peth across reward sizes as an explanation of what's going on in the Naive Bayes decoding

% insert driscoll_transient_discovery2 loading here, copy from
% glm_peth_and_dimensionality

%% calculate peths for 1 uL, 2 uL, and 4 uL trials, sorts from 2
var_bins{3} = 0:.050:2;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

timesince_peth_1uL = nan(sum(s_nNeurons),length(var_bins{3})-1);  
pos_peak_ix_2uL = nan(sum(s_nNeurons),1); % [train test] 
timesince_peth_2uL = nan(sum(s_nNeurons),length(var_bins{3})-1);  
timesince_peth_4uL = nan(sum(s_nNeurons),length(var_bins{3})-1);  

iVar = 3; 

% this is only 1 because we don't care about transient significance.
transient_opt.nShuffles = 1; 
disp("Only one shuffle; dont care about transient significance here")

counter = 0;  
bar = waitbar(0,"PETHs for 3 reward sizes"); % progress tracking 
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})   
        % Load session information
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
        
        i_start = session_neuron_ranges{mIdx}{i}(1); 
        i_end = session_neuron_ranges{mIdx}{i}(2); 

        % Use rewsize 2, 4 uL trials for time since reward 
        rewsize1_trials = rewsize == 1; 
        rewsize2_trials = rewsize == 2; 
        rewsize4_trials = rewsize == 4; 
        
        for iVar = 3
            transient_opt.vars = iVar; 
            % First 1 uL peth
            [~,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},rewsize1_trials,tbin_ms,var_bins,transient_opt);
            timesince_peth_1uL(i_start:i_end,:) = taskvar_peth_cell{iVar};  
            % Then 2 uL peth and sort
            [transient_struct_tmp,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},rewsize2_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix_2uL(i_start:i_end,1) = transient_struct_tmp.peak_ix_pos_sigOrNot;
            timesince_peth_2uL(i_start:i_end,:) = taskvar_peth_cell{iVar};  
            % Last, 4 uL peth
            [~,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},rewsize4_trials,tbin_ms,var_bins,transient_opt);
            timesince_peth_4uL(i_start:i_end,:) = taskvar_peth_cell{iVar};  
        end  
        
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),bar) 
    end
end 
close(bar); 

%% Visualize PETHs for 3 reward sizes

% make some stuff for just GLM neurons
glm_peth_1uL = timesince_peth_1uL(~isnan(glm_cluster),:); 
glm_peth_2uL = timesince_peth_2uL(~isnan(glm_cluster),:); 
glm_peth_4uL = timesince_peth_4uL(~isnan(glm_cluster),:); 
glm_pos_peak_sec_2uL = pos_peak_ix_2uL(~isnan(glm_cluster)) / tbin_ms; 
n_glm_neurons = length(find(~isnan(glm_cluster)));

glm_cluster_id = glm_cluster(~isnan(glm_cluster));
[~,trials2uL_sort] = sort(glm_pos_peak_sec_2uL);

[glm_peth_2uL_sorted_normalized,mu,sigma] = zscore(glm_peth_2uL(trials2uL_sort,:),[],2);
glm_peth_1uL_sorted_normalized = zscore(glm_peth_1uL(trials2uL_sort,:),[],2); % (glm_peth_1uL(trials2uL_sort,:) - mu) ./ sigma;
glm_peth_4uL_sorted_normalized = zscore(glm_peth_4uL(trials2uL_sort,:),[],2); % (glm_peth_4uL(trials2uL_sort,:) - mu) ./ sigma;

all_nan = all(isnan(glm_peth_1uL_sorted_normalized),2) | all(isnan(glm_peth_2uL_sorted_normalized),2) | all(isnan(glm_peth_4uL_sorted_normalized),2);
glm_peth_1uL_sorted_normalized(all_nan,:) = [];
glm_peth_2uL_sorted_normalized(all_nan,:) = [];
glm_peth_4uL_sorted_normalized(all_nan,:) = [];

% for xval peak visualization
jitter = .025;

figure()
subplot(1,3,1)
imagesc(flipud(glm_peth_1uL_sorted_normalized))
caxis([-3 3])
xticks(1:10:numel(var_bins{3}))
xticklabels(var_bins{3}(1:10:end))
xlabel("Time Since Reward")
title("1 uL")
ylabel("2uL Sort")
set(gca,'fontsize',16)
subplot(1,3,2)
imagesc(flipud(glm_peth_2uL_sorted_normalized));
caxis([-3 3])
xticks(1:10:numel(var_bins{3}))
xticklabels(var_bins{3}(1:10:end))
xlabel("Time Since Reward")
ylabel("2uL Sort")
title("2 uL")
set(gca,'fontsize',16)
subplot(1,3,3)
imagesc(flipud(glm_peth_4uL_sorted_normalized));
caxis([-3 3])
xticks(1:10:numel(var_bins{3}))
xticklabels(var_bins{3}(1:10:end))
xlabel("Time Since Reward")
ylabel("2uL Sort")
title("4 uL")
set(gca,'fontsize',16)


