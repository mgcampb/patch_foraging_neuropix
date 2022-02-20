%% Figure 13
%  1) Xval peaksort all GLM neurons and visualize peth
%  2) Scatter train vs test peaks and report correlation
%  3) eigenspectrum analysis of true data vs 
%     i)   synthetic 1D ramp w/ Poisson noise
%     ii)  5-cluster mean PETH w/ Poisson noise 
%     iii) pure sequential encoding 
%     iv)  shuffled data?

%% set path
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
% paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 
sig_cell_sessions = sig_cells.Session;  

load('/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat')

% analysis options
calcFR_opt = struct;
calcFR_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calcFR_opt.smoothSigma_time = .05; % 50 msec smoothing
calcFR_opt.patch_leave_buffer = 0.50; % in seconds; only takes within patch times up to this amount before patch leave
calcFR_opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
calcFR_opt.cortex_only = true; 
calcFR_opt.tstart = 0;
tbin_ms = calcFR_opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 14:18 23 25];   
mouse_grps = {1:2,3:8,10:13,14:18,[23 25]};  
mouse_names = ["m75","m76","m78","m79","m80"]; 
session_titles = cell(numel(mPFC_sessions),1); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end 

%% Load neural data from all good cells into cell array
fr_mat_trials = cell(numel(mouse_grps),1); % cell array of nMice   
spikecounts_full = cell(numel(mouse_grps),1); % for discovery of higher persistent activites  .. no smoothing
vel_full = cell(numel(mouse_grps),1);  
accel_full = cell(numel(mouse_grps),1); 
cue_label = cell(numel(mouse_grps),1); 
cellIDs = cell(numel(mouse_grps),1);  
sigCellCluster = cell(numel(mouse_grps),1); 
brain_region = cell(numel(mouse_grps),1); 

for mIdx = 1:numel(mouse_grps)
    fr_mat_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);     
    spikecounts_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    vel_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    accel_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    cue_label{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    sigCellCluster{mIdx} = cell(numel(mouse_grps{mIdx}),1); % cluster or NaN
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));    
        good_cells = data.sp.cids(data.sp.cgs==2);  
        nNeurons = length(good_cells);  
        
        % get GLM cluster vector per session
        sig_cellIDs_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).CellID;   
        sig_clusters_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).GMM_cluster;
        
        calcFR_opt.tend = max(data.sp.st);
        fr_mat = calcFRVsTime(good_cells,data,calcFR_opt);  
         
        t = data.velt;
        spikecounts_full{mIdx}{i} = nan(numel(nNeurons),numel(t)-1);
        for cIdx = 1:nNeurons
            spike_t = data.sp.st(data.sp.clu==good_cells(cIdx));
            spikecounts_full{mIdx}{i}(cIdx,:) = histcounts(spike_t,t);
        end
        
        % Get trial event timings
        patchcue_ms = data.patchCSL(:,1)*1000;
        patchstop_ms = data.patchCSL(:,2)*1000;
        patchleave_ms = data.patchCSL(:,3)*1000; 
        patchcue_ix = round(patchcue_ms / tbin_ms) + 1; 
        patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
        patchleave_ix = round((patchleave_ms - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1;
        
        % Gather firing rate matrices in trial form
        fr_mat_trials{mIdx}{i} = cell(length(data.patchCSL),1);  
        cue_label{mIdx}{i} = zeros(size(fr_mat,2),1); 
        for iTrial = 1:length(data.patchCSL)
            fr_mat_trials{mIdx}{i}{iTrial} = fr_mat(:,patchcue_ix(iTrial):patchleave_ix(iTrial));   
            cue_label{mIdx}{i}(patchcue_ix(iTrial):patchstop_ix(iTrial)) = 1; % prox cue on
            cue_label{mIdx}{i}(patchstop_ix(iTrial):patchleave_ix(iTrial)) = 2; % patch cue on
        end

        % log information about cells 
        cellIDs{mIdx}{i} = good_cells;    
        brain_region{mIdx}{i} = data.brain_region_rough;    
        sigCellCluster{mIdx}{i} = nan(nNeurons,1); 
        sigCellCluster{mIdx}{i}(ismember(good_cells,sig_cellIDs_session)) = sig_clusters_session; 
        
        % Add speed information for motor confounds check later  
        vel_full{mIdx}{i} = data.vel;
        accel_full{mIdx}{i} = gradient(data.vel);
    end 
    fprintf("%s loading complete \n",mouse_names(mIdx))
end 

% Add task events: trialed 3 x trial length cell array 

task_vars_trialed = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    task_vars_trialed{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        
        % Load session information
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
        patchcue_sec = data.patchCSL(:,1);
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);  
        rew_sec = data.rew_ts;   
        cue_lens_ix = round((patchstop_sec - patchcue_sec) * 1000 / tbin_ms); % important for labeling task events  
        trial_lens_ix = cellfun(@(x) size(x,2),fr_mat_trials{mIdx}{i}); % note that this includes cue and preleave cutoff

        % Collect trial reward timings
        rew_ix_cell = cell(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < (patchleave_sec(iTrial) - calcFR_opt.patch_leave_buffer)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;  
            rew_ix_cell{iTrial}(rew_ix_cell{iTrial} == 0) = 1; 
            rew_ix_cell{iTrial} = rew_ix_cell{iTrial} + cue_lens_ix(iTrial); % add cue length
        end
        
        % Collect task events in nTrials x 1 cell array
        task_vars_trialed{mIdx}{i} = cell(nTrials,1); 
        for iTrial = 1:nTrials 
            iTrial_trial_len = trial_lens_ix(iTrial); 
            iTrial_cue_len = cue_lens_ix(iTrial);  
            task_vars_trialed{mIdx}{i}{iTrial} = nan(3,iTrial_trial_len); 
            
            % first add time since cue 
            task_vars_trialed{mIdx}{i}{iTrial}(1,1:iTrial_cue_len) = (1:iTrial_cue_len) * tbin_ms / 1000; 
            
            % Add time since reward information
            if length(rew_ix_cell{iTrial}) > 1 % more than one reward 
                task_vars_trialed{mIdx}{i}{iTrial}(2,iTrial_cue_len:rew_ix_cell{iTrial}(2)) = (0:(rew_ix_cell{iTrial}(2)-iTrial_cue_len)) * tbin_ms / 1000; 
                for r = 2:numel(rew_ix_cell{iTrial})
                    rew_ix = rew_ix_cell{iTrial}(r); 
                    task_vars_trialed{mIdx}{i}{iTrial}(3,rew_ix:end) = (0:(iTrial_trial_len-rew_ix)) * tbin_ms / 1000; 
                end 
            else % only one reward
                % only log time since first reward
                task_vars_trialed{mIdx}{i}{iTrial}(2,iTrial_cue_len:end) = (0:(iTrial_trial_len-iTrial_cue_len)) * tbin_ms / 1000;
            end
        end
    end
end


%% Collect number of neurons per session to allow for table pre-allocation  
s_nNeurons = [];  
counter = 1; 
for mIdx = 1:5  
    for i = 1:numel(mouse_grps{mIdx})
        s_nNeurons = [s_nNeurons; length(brain_region{mIdx}{i})];
        counter = counter + 1; 
    end
end 

session_neuron_starts = [0 ; cumsum(s_nNeurons)];  
session_neuron_ranges = cell(numel(mouse_grps),1);
counter = 1; 
for mIdx = 1:5  
    session_neuron_ranges{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        session_neuron_ranges{mIdx}{i} = [session_neuron_starts(counter)+1 session_neuron_starts(counter+1)];  
        counter = counter + 1; 
    end 
end

%% Calculate PETH
% separate training and visualization data 

var_bins{3} = 0:.050:2;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

timesince_peth = nan(sum(s_nNeurons),length(var_bins{3})-1);  
timesince_peth_test = nan(sum(s_nNeurons),length(var_bins{3})-1);  
pos_peak_ix = nan(sum(s_nNeurons),2); % [train test] 
neg_peak_ix = nan(sum(s_nNeurons),2); % [train test] 

% this is only 1 because we don't care about transient significance.
transient_opt.nShuffles = 1; 
disp("Only one shuffle; dont care about transient significance here")

counter = 0;  
bar = waitbar(0,"Making PETHs"); % progress tracking 
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
        rewsize24_trials = find(rewsize > 1); 
        sort_trials = rewsize24_trials(2:2:length(rewsize24_trials)); 
        vis_trials = rewsize24_trials(1:2:length(rewsize24_trials)); 
        
        for iVar = 3
            transient_opt.vars = iVar; 
            [transient_struct_tmp,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},vis_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix(i_start:i_end,1) = transient_struct_tmp.peak_ix_pos_sigOrNot;
            neg_peak_ix(i_start:i_end,1) = transient_struct_tmp.peak_ix_neg_sigOrNot;
            % add to taskvar peth
            timesince_peth(i_start:i_end,:) = taskvar_peth_cell{iVar};  

            [transient_struct_tmp,taskvar_peth_cell_test] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},sort_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix(i_start:i_end,2) = transient_struct_tmp.peak_ix_pos_sigOrNot;
            neg_peak_ix(i_start:i_end,2) = transient_struct_tmp.peak_ix_neg_sigOrNot;
            timesince_peth_test(i_start:i_end,:) = taskvar_peth_cell_test{iVar};
        end  
        
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),bar) 
    end
end 
close(bar); 

%% 1) and 2) Visualize cross-validated PETH and peaksort; don't care about significance to shuffle

gmm_colors = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255;

glm_cluster = transients_table.gmm_cluster; 
close all
% make some stuff for just GLM neurons
glm_peth = timesince_peth(~isnan(glm_cluster),:); 
glm_pos_peak_sec = pos_peak_ix(~isnan(glm_cluster),:) / tbin_ms; 
glm_neg_peak_sec = neg_peak_ix(~isnan(glm_cluster),:) / tbin_ms; 
n_glm_neurons = length(find(~isnan(glm_cluster)));

glm_cluster_id = glm_cluster(~isnan(glm_cluster));
[~,test_sort] = sort(glm_pos_peak_sec(:,2));
% to check out mid-responsive neurons
% [~,test_sort] = sort(glm_pos_peak_sec((glm_pos_peak_sec(:,2) > .5) & (glm_pos_peak_sec(:,2) < 1.5),2));
% glm_peth = glm_peth((glm_pos_peak_sec(:,2) > .5) & (glm_pos_peak_sec(:,2) < 1.5),:);

% for xval peak visualization
jitter = .025;

figure()
subplot(1,2,1)
imagesc(flipud(zscore(glm_peth(test_sort,:),[],2)))
caxis([-3 3])
xticks(1:10:numel(var_bins{3}))
xticklabels(var_bins{3}(1:10:end))
xlabel("Time Since Reward")
ylabel("Test Sort")
set(gca,'fontsize',16)

train_peaks_plot = jitter * randn(n_glm_neurons,1) + glm_pos_peak_sec(:,1);
test_peaks_plot = jitter * randn(n_glm_neurons,1) + glm_pos_peak_sec(:,2);

subplot(1,2,2)
gscatter(train_peaks_plot,test_peaks_plot,glm_cluster_id,gmm_colors,'o',5,0);hold on

% scatter(train_peaks_plot,test_peaks_plot,'ko','linewidth',.25) 
% set(gca,'colorOrder',gmm_colors);
[r,p] = corrcoef(glm_pos_peak_sec(:,1),glm_pos_peak_sec(:,2));

title(sprintf("Time Since Reward Train vs Test Peaks (n = %i) \n (r = %.3f, p = %.3f)",n_glm_neurons,r(2),p(2)))
xlabel("Time Since Reward Train Peak")
ylabel("Time Since Reward Test Peak")
ylim([0,2])
xlim([0,2])
xticks((1:10:numel(var_bins{3})) / tbin_ms)
xticklabels(var_bins{3}(1:10:end))
yticks((1:10:numel(var_bins{3})) / tbin_ms)
yticklabels(var_bins{3}(1:10:end))
set(gca,'fontsize',18)

%% visualize not cv for the figure 


%% Recalculate PETH not holding out trials
var_bins{3} = 0:.050:2;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

timesince_peth_allTrials = nan(sum(s_nNeurons),length(var_bins{3})-1);  
pos_peak_ix_allTrials = nan(sum(s_nNeurons),2); % [train test] 

% this is only 1 because we don't care about transient significance.
transient_opt.nShuffles = 1; 
disp("Only one shuffle; dont care about transient significance here")

counter = 0;  
bar = waitbar(0,"Making All trials PETH"); % progress tracking 
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
        rewsize24_trials = rewsize > 1; 
        
        for iVar = 3
            transient_opt.vars = iVar; 
            [transient_struct_tmp,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},rewsize24_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix_allTrials(i_start:i_end,1) = transient_struct_tmp.peak_ix_pos_sigOrNot;
            % add to taskvar peth
            timesince_peth_allTrials(i_start:i_end,:) = taskvar_peth_cell{iVar};  
        end  
        
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),bar) 
    end
end 
close(bar); 

%% Quickly visualize PETH from all trials
glm_cluster = transients_table.gmm_cluster; 
% make some stuff for just GLM neurons
glm_peth_allTrials = timesince_peth_allTrials(~isnan(glm_cluster),:); 
glm_pos_peak_sec_allTrials = pos_peak_ix_allTrials(~isnan(glm_cluster)) / tbin_ms; 
n_glm_neurons = length(find(~isnan(glm_cluster)));

glm_cluster_id = glm_cluster(~isnan(glm_cluster));
[~,all_sort] = sort(glm_pos_peak_sec_allTrials);
% to check out mid-responsive neurons
% [~,test_sort] = sort(glm_pos_peak_sec((glm_pos_peak_sec(:,2) > .5) & (glm_pos_peak_sec(:,2) < 1.5),2));
% glm_peth = glm_peth((glm_pos_peak_sec(:,2) > .5) & (glm_pos_peak_sec(:,2) < 1.5),:);

% for xval peak visualization
jitter = .025;

figure()
imagesc(flipud(zscore(glm_peth_allTrials(all_sort,:),[],2)))
caxis([-3 3])
xticks(1:10:numel(var_bins{3}))
xticklabels(var_bins{3}(1:10:end))
xlabel("Time Since Reward")
ylabel("Test Sort")
set(gca,'fontsize',16)

%% test for difference in peak locations separated by cluster 
p = anova1(glm_pos_peak_sec_allTrials,glm_cluster(~isnan(glm_cluster)))
pk = kruskalwallis(glm_pos_peak_sec_allTrials,glm_cluster(~isnan(glm_cluster)))

%% Entropy of peaks analysis
cluster_id = glm_cluster(~isnan(glm_cluster));
cluster1_peaks = glm_pos_peak_sec_allTrials(cluster_id == 1);
cluster2_peaks = glm_pos_peak_sec_allTrials(cluster_id == 2);

n_bootstrap_repeats = 1000; 

% Calculate shannon entropy of peaks between Cluster 1 and Cluster 2, bootstrap CI
cluster1_shannonH_distn = bootstrp(n_bootstrap_repeats,@(x)calc_shannonH(x,0:.2:2),cluster1_peaks);
cluster2_shannonH_distn = bootstrp(n_bootstrap_repeats,@(x)calc_shannonH(x,0:.2:2),cluster2_peaks);

cluster1_shannonH_CI = quantile(cluster1_shannonH_distn,[.025,.975])
cluster2_shannonH_CI = quantile(cluster2_shannonH_distn,[.025,.975])

%% 3) Embedding dimensionality of PETH
close all

glm_peth_allTrials = timesince_peth_allTrials(~isnan(glm_cluster),:); 
glm_peth_evenTrials = timesince_peth(~isnan(glm_cluster),:); 
glm_peth_oddTrials = timesince_peth_test(~isnan(glm_cluster),:); 

norm_glm_peth_allTrials = zscore(glm_peth_allTrials');
norm_glm_peth_evenTrials = zscore(glm_peth_evenTrials');
norm_glm_peth_oddTrials = zscore(glm_peth_oddTrials');

[coeff_all,score_all,latent,tsquared,expl_all] = pca(norm_glm_peth_allTrials);
[coeff_even,score_even,~,~,expl_even,mu] = pca(norm_glm_peth_evenTrials);
[coeff_odd,score_odd,~,~,expl_odd] = pca(norm_glm_peth_oddTrials);

%% construct mean per cluster peth

gmm_means = nan(5,size(glm_peth_allTrials,2));
gmm_cluster_id = glm_cluster(~isnan(glm_cluster));
n_glm_neurons = length(find(~isnan(glm_cluster)));

cluster_n_neurons = arrayfun(@(x) length(find(gmm_cluster_id == x)),(1:5));
cluster_neuron_starts = [0 cumsum(cluster_n_neurons)];  
cluster_neuron_ranges = cell(5,1);

for i_cluster = 1:5
    cluster_neuron_ranges{i_cluster} = [cluster_neuron_starts(i_cluster)+1 cluster_neuron_starts(i_cluster+1)];
end

gmm_peth = nan(n_glm_neurons,size(gmm_means,2));
for i_cluster = 1:5
    gmm_means(i_cluster,:) = mean(glm_peth_allTrials(gmm_cluster_id == i_cluster,:));
    this_start = cluster_neuron_ranges{i_cluster}(1);
    this_end = cluster_neuron_ranges{i_cluster}(2);
    gmm_peth(this_start:this_end,:) = repmat(gmm_means(i_cluster,:),[cluster_n_neurons(i_cluster) 1]);
end
[~,~,~,~,expl_gmm_means] = pca(zscore(gmm_peth'));

%% High and low embedding data to compare to
n_glm_neurons = length(find(~isnan(glm_cluster)));

phi_highD = linspace(0,10,n_glm_neurons); 
phi_lowD = 10 + zeros(n_glm_neurons,1); 
G = 5;
sigma_low = 20;
sigma_high = 1;
n_trials = 1;
S = repmat((1:.01:10)',[1 n_trials])';
X_lowEmbedding = gen_manifold_embedding_dataset(n_glm_neurons,phi_lowD,G,sigma_low,S); 
X_highEmbedding = gen_manifold_embedding_dataset(n_glm_neurons,phi_highD,G,sigma_high,S); 

figure()
subplot(1,2,1)
imagesc(flipud(X_lowEmbedding))
subplot(1,2,2)
imagesc(flipud(X_highEmbedding))

[~,~,~,~,expl_ramp] = pca(zscore(X_lowEmbedding'));
[~,~,~,~,expl_seq] = pca(zscore(X_highEmbedding'));

%% perform cvPCA on shuffled data

var_bins{3} = 0:.050:2;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

timesince_peth_shuffle_even = nan(sum(s_nNeurons),length(var_bins{3})-1);  
timesince_peth_shuffle_odd = nan(sum(s_nNeurons),length(var_bins{3})-1);  

% this is only 1 because we don't care about transient significance.
transient_opt.nShuffles = 1; 
disp("Only one shuffle; dont care about transient significance here")

counter = 0;  
bar = waitbar(0,"Making Shuffled PETHs"); % progress tracking 
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
        rewsize24_trials = find(rewsize == 4); 
        sort_trials = rewsize24_trials(2:2:length(rewsize24_trials)); 
        vis_trials = rewsize24_trials(1:2:length(rewsize24_trials)); 
        
        for iVar = 3
            transient_opt.vars = iVar; 
            [~,~,~,shuffle_peth_even] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},vis_trials,tbin_ms,var_bins,transient_opt);
            % add to taskvar peth
            timesince_peth_shuffle_even(i_start:i_end,:) = shuffle_peth_even;  

            [~,~,~,shuffle_peth_odd] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},sort_trials,tbin_ms,var_bins,transient_opt);
            timesince_peth_shuffle_odd(i_start:i_end,:) = shuffle_peth_odd;
        end  
        
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),bar) 
    end
end 
close(bar)
%% Perform cvPCA on shuffled data
glm_peth_evenTrials_shuffle = timesince_peth_shuffle_even(~isnan(glm_cluster),:); 
glm_peth_oddTrials_shuffle = timesince_peth_shuffle_odd(~isnan(glm_cluster),:); 
norm_glm_peth_evenTrials_shuffle = zscore(glm_peth_evenTrials_shuffle');
norm_glm_peth_oddTrials_shuffle = zscore(glm_peth_oddTrials_shuffle');

[coeff_even_shuffle,score_even_shuffle] = pca(norm_glm_peth_evenTrials_shuffle);
[coeff_odd_shuffle,score_odd_shuffle] = pca(norm_glm_peth_oddTrials_shuffle);

n_pcs_calc = 10;
var_expl_cv_heldout_shuffle = nan(n_pcs_calc,2);
for i_n_pcs = 1:n_pcs_calc
    this_pc_reconstr_even = score_even_shuffle(:,1:i_n_pcs) *  coeff_even_shuffle(:,1:i_n_pcs)';
    var_expl_cv_heldout_shuffle(i_n_pcs,1) = var_expl(norm_glm_peth_oddTrials,this_pc_reconstr_even);
    
    this_pc_reconstr_odd = score_odd_shuffle(:,1:i_n_pcs) *  coeff_odd_shuffle(:,1:i_n_pcs)';
    var_expl_cv_heldout_shuffle(i_n_pcs,2) = var_expl(norm_glm_peth_evenTrials,this_pc_reconstr_odd);
end
var_expl_cv_heldout_shuffle_mean = mean(var_expl_cv_heldout_shuffle,2);

%% Finally, cvPCA on actual data

glm_peth_evenTrials = timesince_peth(~isnan(glm_cluster),:); 
glm_peth_oddTrials = timesince_peth_test(~isnan(glm_cluster),:); 

norm_glm_peth_evenTrials = zscore(glm_peth_evenTrials');
norm_glm_peth_oddTrials = zscore(glm_peth_oddTrials');

[coeff_even,score_even,~,~,expl_even,mu] = pca(norm_glm_peth_evenTrials);
[coeff_odd,score_odd,~,~,expl_odd] = pca(norm_glm_peth_oddTrials);

n_pcs_vis = 5;
n_pcs_calc = 10;
var_expl_cv_heldout = nan(n_pcs_calc,2);
for i_n_pcs = 1:n_pcs_calc
    this_pc_reconstr_even = score_even(:,1:i_n_pcs) *  coeff_even(:,1:i_n_pcs)';
    var_expl_cv_heldout(i_n_pcs,1) = var_expl(norm_glm_peth_oddTrials,this_pc_reconstr_even);
    
    this_pc_reconstr_odd = score_odd(:,1:i_n_pcs) *  coeff_odd(:,1:i_n_pcs)';
    var_expl_cv_heldout(i_n_pcs,2) = var_expl(norm_glm_peth_evenTrials,this_pc_reconstr_odd);
end
var_expl_cv_heldout_mean = mean(var_expl_cv_heldout,2);

spectral9 = cbrewer('div','Spectral',9);

% visualize results
figure();hold on
plot(var_expl_cv_heldout_mean(1:n_pcs_vis) / max(var_expl_cv_heldout_mean(:)),'k','linewidth',3);
plot(cumsum(expl_all(1:n_pcs_vis)/100),'k--','linewidth',3,'color',[.4 .4 .4])
plot(cumsum(expl_gmm_means(1:n_pcs_vis)) / 100,'linewidth',3,'color',spectral9(4,:))
plot(cumsum(expl_ramp(1:n_pcs_vis)/100),'linewidth',3,'color',spectral9(9,:))
plot(cumsum(expl_seq(1:n_pcs_vis)/100),'linewidth',3,'color',spectral9(2,:))
legend(["cvPCA on Data","Non-CV PCA on Data","GMM Means","1D Ramping Activity","Sequential Activity"])
ylabel("Cumulative Variance Explained")
xlabel("PCs")
xlim([1 n_pcs_vis])
ylim([0 1])
set(gca,'fontsize',20)

%% Do cvPCA with k-folds to get errorbars 

% Calculate PETH
% separate training and visualization data 
iVar = 3; 
var_bins{3} = 0:.050:2;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

% this is only 1 because we don't care about transient significance.
transient_opt.nShuffles = 1; 
n_folds = 5; 
foldid = cell(numel(mouse_grps),1);

% first collect folds
for mIdx = 1:numel(mouse_grps)
    foldid{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    for i = 1:numel(mouse_grps{mIdx})   
        % Load session information
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        this_rewsize = mod(data.patches(:,2),10);   
        
        % folderinos
        rewsize24_trials = this_rewsize > 1;
        n_trials = length(rewsize24_trials);
%         this_foldid = nan(n_trials,1);
        % define folds
        too_many_folds = repmat((1:n_folds)',[ceil(n_trials / n_folds),1]);
        this_foldid = too_many_folds(1:length(find(rewsize24_trials)));
        foldid{mIdx}{i} = this_foldid;
    end
end

% preallocate train and test peths
timesince_train_peths = cell(n_folds,1);
timesince_test_peths = cell(n_folds,1);
for i_fold = 1:n_folds
    timesince_train_peths{i_fold} = nan(sum(s_nNeurons),length(var_bins{3})-1);  
    timesince_test_peths{i_fold} = nan(sum(s_nNeurons),length(var_bins{3})-1);  
end
% disp("Only one shuffle; dont care about transient significance here")
transient_opt.vars = iVar; 
%% Now get peths across folds
for i_fold = 1:n_folds
    counter = 0;  
    bar = waitbar(0,sprintf("Making PETHs fold %i",i_fold)); % progress tracking 
    for mIdx = 1:numel(mouse_grps)
        for i = 1:numel(mouse_grps{mIdx})   
            % Load session information
            sIdx = mouse_grps{mIdx}(i);   
            data = load(fullfile(paths.data,sessions{sIdx}));
            nTrials = length(data.patchCSL); 
            this_rewsize = mod(data.patches(:,2),10);   

            i_start = session_neuron_ranges{mIdx}{i}(1); 
            i_end = session_neuron_ranges{mIdx}{i}(2); 
            
            % get fold trials
            rewsize24_trials = find(this_rewsize > 1);
            train_trials = rewsize24_trials(foldid{mIdx}{i} ~= i_fold & ~isnan(foldid{mIdx}{i})); 
            test_trials = rewsize24_trials(foldid{mIdx}{i} == i_fold); 

            [~,taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},train_trials,tbin_ms,var_bins,transient_opt);
            % add to taskvar peth
            timesince_train_peths{i_fold}(i_start:i_end,:) = taskvar_peth_cell{iVar};  

            [~,taskvar_peth_cell_test] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},test_trials,tbin_ms,var_bins,transient_opt);
            timesince_test_peths{i_fold}(i_start:i_end,:) = taskvar_peth_cell_test{iVar};
            if ~isempty(find(isnan(timesince_test_peths{i_fold}(i_start:i_end,:)),1))
                fprintf("mIdx: %i i: %i nan! \n",mIdx,i)
            end

            counter = counter + 1; 
            waitbar(counter/numel(mPFC_sessions),bar) 
        end
    end 
    close(bar); 
end

%% Now do cvPCA on the k-folds
n_pcs_calc = 10;
var_expl_kfold = nan(n_pcs_calc,n_folds);
for i_fold = 1:n_folds
    glm_peth_train = timesince_train_peths{i_fold}(~isnan(glm_cluster),:);
    glm_peth_test = timesince_test_peths{i_fold}(~isnan(glm_cluster),:);
    norm_glm_peth_train = zscore(glm_peth_train');
    norm_glm_peth_test = zscore(glm_peth_test');
    
    disp(size(find(isnan(norm_glm_peth_train))))
    disp(size(find(isnan(norm_glm_peth_test))))
%     
    [coeff_train,score_train] = pca(norm_glm_peth_train);
    
    for i_n_pcs = 1:n_pcs_calc
        this_pc_reconstr_train = score_train(:,1:i_n_pcs) *  coeff_train(:,1:i_n_pcs)';
        non_nan = ~isnan(norm_glm_peth_test);
        var_expl_kfold(i_n_pcs,i_fold) = var_expl(norm_glm_peth_test(non_nan),this_pc_reconstr_train(non_nan));
    end
end
%% Visualize kfold cvPCA

var_expl_kfold_norm = var_expl_kfold ./ max(var_expl_kfold);
var_expl_kfold_norm_mean = mean(var_expl_kfold_norm,2);
var_expl_kfold_norm_sem = 1.96 * std(var_expl_kfold_norm,[],2) / sqrt(n_folds);

expl_all_frac = cumsum(expl_all/100);
expl_gmm_means_frac = cumsum(expl_gmm_means) / 100;
expl_ramp_frac = cumsum(expl_ramp) / 100; 
expl_seq_frac = cumsum(expl_seq) / 100; 

spectral9 = cbrewer('div','Spectral',9);
n_pcs_vis = 8;

p_vals = nan(n_pcs_vis,1);
for i = 1:n_pcs_vis
    [h,p] = ttest(var_expl_kfold_norm(i,:) - expl_gmm_means_frac(i));
    p_vals(i) = p;
end

% visualize results
figure();hold on
% plot(var_expl_cv_heldout_mean(1:n_pcs_vis) / max(var_expl_cv_heldout_mean(:)),'k','linewidth',3);

% shadedErrorBar(1:n_pcs_vis,var_expl_kfold_norm_mean(1:n_pcs_vis),var_expl_kfold_norm_sem(1:n_pcs_vis),'lineProps',{'linewidth',2,'color','k'})
% scatter(1:n_pcs_vis,var_expl_kfold_norm_mean(1:n_pcs_vis),'.')
errorbar(1:n_pcs_vis,var_expl_kfold_norm_mean(1:n_pcs_vis),var_expl_kfold_norm_sem(1:n_pcs_vis),'marker','.','markerSize',20,'linewidth',3,'color','k')
plot(expl_all_frac(1:n_pcs_vis),'k--','linewidth',3,'color',[.4 .4 .4])
plot(expl_gmm_means_frac(1:n_pcs_vis),'linewidth',3,'color',spectral9(4,:))
plot(expl_ramp_frac(1:n_pcs_vis),'linewidth',3,'color',spectral9(9,:))
plot(expl_seq_frac,'linewidth',3,'color',spectral9(2,:))
legend(["cvPCA on Data","Non-CV PCA on Data","GMM Means","1D Ramping Activity","Sequential Activity"])
ylabel("Cumulative Variance Explained")
xlabel("PCs")
xlim([1 n_pcs_vis])
ylim([.3 1])
set(gca,'fontsize',20)

%% cvPCA testing
% check what the pcs look like 
% figure()
% plot(score_all(:,1:5),'linewidth',2)

% check that variance explained calculation is working
n_pcs_vis = 10;
var_expl_calc_manual = nan(n_pcs_vis,1);
for i_n_pcs = 1:n_pcs_vis
    this_pc_reconstr = score_all(:,1:i_n_pcs) *  coeff_all(:,1:i_n_pcs)';
    var_expl_calc_manual(i_n_pcs) = var_expl(norm_glm_peth_allTrials,this_pc_reconstr);
end
figure()
hold on; 
plot(cumsum(expl_all(1:n_pcs_vis)) / 100)
plot(var_expl_calc_manual)



%% Functions
function this_var_expl = var_expl(X,Y)
    % Calculate proportion variance in X explained by variance in Y
    X_centered = X - nanmean(X,1);
    Y_centered = Y - nanmean(Y,1);
    X_2 = nanmean(vecnorm(X_centered,2,2))^2 ;
    X_minus_Y_2 =  nanmean(vecnorm(X_centered - Y_centered,2,2))^2;
    this_var_expl = (X_2 - X_minus_Y_2) / X_2;
end

function R = gen_manifold_embedding_dataset(n_neurons,phi,G,sigma,S) 
    % Generate neural population response from neurons w/ gaussian
    % responsivity
    % ----
    % Arguments
    % phi: stimulus preference per neuron [n_neurons x 1]
    % G: gain
    % S: 1D stimulus [n_trials x t_len]
    % ----
    % Returns
    % R: [n_neurons x t_len x n_trials]
    
    [n_trials,t_len] = size(S); 
    
    R = nan(n_neurons,t_len,n_trials); 
    
    % iterate over trials
    for i_trial = 1:n_trials
        for i_timestep = 1:t_len
            R(:,i_timestep,i_trial) = G * exp(-((S(i_trial,i_timestep) - phi) / (sqrt(2.0 * sigma))).^2);
%             R(:,i_timestep,i_trial) = poissrnd(R(:,i_timestep,i_trial)); % add poisson noise
        end
    end
end
