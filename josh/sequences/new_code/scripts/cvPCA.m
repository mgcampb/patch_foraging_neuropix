%% cvPCA to formalize dimensionality argument 

% use call to driscoll_transient_discovery2 with diff trial selections to
% get avg peths using many folds to get errorbars

%% set path
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
% paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
% paths.transients_table = 
load(paths.sig_cells);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 
sig_cell_sessions = sig_cells.Session;  

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
rewsize = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    task_vars_trialed{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        
        % Load session information
        nTrials = length(data.patchCSL); 
        i_rewsize = mod(data.patches(:,2),10);   
        rewsize{mIdx}{i} = i_rewsize; 
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

%% Now iterate over sessions and get unshuffled peths 
%  Comparing 2 and 4 uL trials, select 
n_folds = 1; 
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);
train_peths = cell(n_folds,1); 
test_peths = cell(n_folds,1); 
pos_peak_train_ix = cell(n_folds,1); 
transient_opt.nShuffles = 0; 
rew1plus_ix = 3; 
transient_opt.vars = rew1plus_ix; % t = 1+ reward

var_bins = {};
var_bins{rew1plus_ix} = 0:.050:2;  

for i_fold = 1:n_folds
    train_peths{i_fold} = nan(sum(s_nNeurons),length(var_bins{rew1plus_ix})-1);  
    test_peths{i_fold} = nan(sum(s_nNeurons),length(var_bins{rew1plus_ix})-1);
end

for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})
        this_rewsize = rewsize{mIdx}{i};
        
        i_start = session_neuron_ranges{mIdx}{i}(1);
        i_end = session_neuron_ranges{mIdx}{i}(2);
        
        rewsize24_trials = this_rewsize > 1;
        n_trials = length(rewsize24_trials); 
        foldid = nan(n_trials,1); 
        % define folds
        too_many_folds = repmat((1:n_folds)',[ceil(n_trials / n_folds),1]); 
        foldid(rewsize24_trials) = too_many_folds(1:length(find(rewsize24_trials))); 
        
        if n_folds > 1
            for i_fold = 1:n_folds
                % get train trials
                train_trials = foldid ~= i_fold;
                test_trials = foldid == i_fold; % only need this when we go back 

                % get peth on train trials
                [train_transient_struct,train_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},train_trials,tbin_ms,var_bins,transient_opt);
                train_peths{i_fold}(i_start:i_end,:) = train_peth_cell{rew1plus_ix};

                % get peth on test trials
                [~,test_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},test_trials,tbin_ms,var_bins,transient_opt);
                test_peths{i_fold}(i_start:i_end,:) = test_peth_cell{rew1plus_ix};
            end
        else 
            train_trials = 1:n_trials; 
            test_trials = 1:n_trials; 
            
            % get peth on train trials
            [train_transient_struct,train_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},train_trials,tbin_ms,var_bins,transient_opt);
            train_peths{i_fold}(i_start:i_end,:) = train_peth_cell{rew1plus_ix};

            % get peth on test trials
            [~,test_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},test_trials,tbin_ms,var_bins,transient_opt);
            test_peths{i_fold}(i_start:i_end,:) = test_peth_cell{rew1plus_ix};
        end
    end
    fprintf("%s peth calculation complete \n",mouse_names(mIdx))
end

%% Quickly check that the data looks right by visualizing peaksort peths

% first all data
% figure()
% [~,peaksort] = sort(transients_table.Rew1plus_peak_pos); 
% for i_fold = 1:n_folds
%     subplot(1,n_folds,i_fold)
%     imagesc(flipud(zscore(train_peths{i_fold}(peaksort,:),[],2)))
% end

gmm_cluster = transients_table.gmm_cluster;
figure() 
for i_gmm_cluster = 1:4 
    cluster_peth = test_peths{i_fold}(gmm_cluster == i_gmm_cluster,:); 
    [~,peaksort] = sort(transients_table.Rew1plus_peak_pos(gmm_cluster == i_gmm_cluster)); 
    for i_fold = 1:n_folds 
        subplot(4,n_folds,i_fold + (i_gmm_cluster - 1) * n_folds)
        imagesc(flipud(zscore(cluster_peth(peaksort,:),[],2)))
    end
end

%% load numic waveform types
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

%% Now perform cvPCA to get variance explained per cvPC
gmm_cluster = transients_table.gmm_cluster;
n_subsample_repeats = 10000; % repeat because we subsample some populations

% subsample_size = min(arrayfun(@(i_gmm_cluster) length(find(gmm_cluster == i_gmm_cluster)),1:4)); 
subsample_size = length(find(gmm_cluster == 2 & transients_table.numeric_waveform_types == 1)); % tough hardcoding here
n_bins = length(var_bins{rew1plus_ix})-1; % this is n_stimuli

test_var_expl = cell(5,1); 
train_var_expl = cell(5,1); 
train_var_expl_b1 = nan(5,n_folds,n_subsample_repeats); 
train_var_expl_b0 = nan(5,n_folds,n_subsample_repeats); 
n_pcs_var_expl = 20; 
counter = 1; 
for i_gmm_cluster = [1 2 2 3 4]
    cluster_n_neurons = length(find(gmm_cluster == i_gmm_cluster)); 
    test_var_expl{counter} = nan(n_bins,n_folds,n_subsample_repeats);
    train_var_expl{counter} = nan(n_bins-1,n_folds,n_subsample_repeats);
    for i_fold = 1:n_folds
        if i_gmm_cluster ~= 2
            this_train_peth = zscore(train_peths{i_fold}(gmm_cluster == i_gmm_cluster,:),[],2);
            this_train_peth(isnan(this_train_peth)) = 0;
            this_test_peth = zscore(test_peths{i_fold}(gmm_cluster == i_gmm_cluster,:),[],2);
            this_test_peth(isnan(this_test_peth)) = 0;
        elseif i_gmm_cluster == 2 && counter == 2 % subselect WS
            this_train_peth = zscore(train_peths{i_fold}(gmm_cluster == i_gmm_cluster & transients_table.numeric_waveform_types == 1,:),[],2);
            this_train_peth(isnan(this_train_peth)) = 0;
            this_test_peth = zscore(test_peths{i_fold}(gmm_cluster == i_gmm_cluster & transients_table.numeric_waveform_types == 1,:),[],2);
            this_test_peth(isnan(this_test_peth)) = 0;
        elseif i_gmm_cluster == 2 && counter == 3 % subselect NS
            this_train_peth = zscore(train_peths{i_fold}(gmm_cluster == i_gmm_cluster & transients_table.numeric_waveform_types == 2,:),[],2);
            this_train_peth(isnan(this_train_peth)) = 0;
            this_test_peth = zscore(test_peths{i_fold}(gmm_cluster == i_gmm_cluster & transients_table.numeric_waveform_types == 2,:),[],2);
            this_test_peth(isnan(this_test_peth)) = 0;
        end
        
        for i_subsample = 1:n_subsample_repeats
            % downsample to get the same number of neurons for each population
            this_resample = datasample((1:size(this_train_peth,1)),subsample_size,'Replace',false);
            subsampled_train_peth = this_train_peth(this_resample,:);
            subsampled_test_peth = this_test_peth(this_resample,:);
            
            [u,sv] = svd(subsampled_train_peth);
            sv_inv = 1 ./ sv; sv_inv(isinf(sv_inv)) = 0;
            projection = subsampled_train_peth' * (u * sv_inv);
            
            % projection onto train and test sets
            train_proj = subsampled_train_peth * projection;
            test_proj = subsampled_test_peth * projection;
            % sum of squares for test
            test_ss = sum(train_proj' * test_proj,1);
            this_test_var_expl = cumsum(test_ss); %  ./ sum(test_ss);
            test_var_expl{counter}(:,i_fold,i_subsample) = this_test_var_expl ./ sum(test_ss);
            % sum of squares for train
            train_ss = sum(train_proj' * train_proj,1);
            [~,~,~,~,this_train_var_expl] = pca(subsampled_train_peth');
            train_var_expl{counter}(:,i_fold,i_subsample) = cumsum(this_train_var_expl * .01);
            b0_b1_var_expl = [ones(1,n_pcs_var_expl) ; 1:(n_pcs_var_expl)]' \ log(.01 * this_train_var_expl(1:n_pcs_var_expl)); % 
            train_var_expl_b0(counter,i_fold,i_subsample) = b0_b1_var_expl(1); % (2); 
            train_var_expl_b1(counter,i_fold,i_subsample) = b0_b1_var_expl(2); 
        end
    end
    test_var_expl{counter} = reshape(test_var_expl{counter},[n_bins,n_folds*n_subsample_repeats]);
    train_var_expl{counter} = reshape(train_var_expl{counter},[n_bins-1,n_folds*n_subsample_repeats]);
    counter = counter + 1; 
end
train_var_expl_b0 = reshape(train_var_expl_b0,[5,n_folds * n_subsample_repeats]); 
train_var_expl_b1 = reshape(train_var_expl_b1,[5,n_folds * n_subsample_repeats]);

%% Now visualize
figure()
subplot(1,2,1)
for i_gmm_cluster = 1:4
    shadedErrorBar(1:(n_bins-1),mean(train_var_expl{i_gmm_cluster},2),std(train_var_expl{i_gmm_cluster},[],2),'lineprops',{'linewidth',1.5,'color',gmm_colors(i_gmm_cluster,:)})
end
yl = ylim();
xlim([0,20])
subplot(1,2,2)
for i_gmm_cluster = 1:4
    shadedErrorBar(1:n_bins,mean(test_var_expl{i_gmm_cluster},2),std(test_var_expl{i_gmm_cluster},[],2),'lineprops',{'linewidth',1.5,'color',gmm_colors(i_gmm_cluster,:)})
end
ylim(yl)
xlim([0,20])
legend("Cluster 1","cluster 2","cluster 3","cluster 4")

%% Just visualize un-CV var_expl 
close all
% two quantifications: 1) var expl by 1st component, 2) beta1 exp fit
% reorg train_var_expl to just look at first component 
expl1 = nan(5,n_folds * n_subsample_repeats);
for i_gmm_cluster = 1:5 
    expl1(i_gmm_cluster,:) = train_var_expl{i_gmm_cluster}(1,:); 
end

% calc p-values for expl1
p_expl1 = nan(5,5); 
for i_pop1 = 1:5
    for i_pop2 = 1:5
        p_expl1(i_pop1,i_pop2) = length(find(expl1(i_pop1,:) < expl1(i_pop2,:))) / (n_folds * n_subsample_repeats); 
    end
end
% calc p-values for beta1 exp fit
p_beta1 = nan(5,5); 
for i_pop1 = 1:5
    for i_pop2 = 1:5
        p_beta1(i_pop1,i_pop2) = length(find(train_var_expl_b1(i_pop1,:) < train_var_expl_b1(i_pop2,:))) / (n_folds * n_subsample_repeats); 
    end
end

figure(); 
gmm_colors = [68 119 170; 238 102 119; 238 102 119 ; 34 136 51; 204 187 68; 102 204 238]/255;
gmm_linestyles = ["-",":","--","-","-"];
xlim([0,20])
for i_gmm_cluster = 1:5
    shadedErrorBar(1:(n_bins-1),mean(train_var_expl{i_gmm_cluster},2),std(train_var_expl{i_gmm_cluster},[],2),'lineprops',{'linewidth',1.5,'color',gmm_colors(i_gmm_cluster,:),'linestyle',gmm_linestyles(i_gmm_cluster)})
end
ylim([0.3,1.05])
xlim([1,10])
legend(["Cluster 1","Cluster 2 NS","Cluster 2 WS","Cluster 3","Cluster 4"])
fig = gca;
xlabel("Principal Component")
ylabel("Cumulative Proportion of Variance Explained")
set(fig,'fontsize',14)
% Quantify using linear fit to log var_expl

figure()
% now visualize bootstrap results
b = bar(mean(train_var_expl_b1,2)','LineWidth',1.5);hold on
b.FaceColor = 'Flat'; 
b.CData(1,:) = gmm_colors(1,:); 
b.CData(2,:) = gmm_colors(2,:); 
b.CData(3,:) = gmm_colors(3,:); 
b.CData(4,:) = gmm_colors(4,:); 
b.CData(5,:) = gmm_colors(5,:); 
errorbar(1:5,mean(train_var_expl_b1,2),std(train_var_expl_b1,[],2),'.k','linewidth',1.5)
fig = gca; 
xticklabels(["Cluster 1","Cluster 2 NS","Cluster 2 WS","Cluster 3","Cluster 4"])
ylabel(sprintf("Eigenspectrum exponential fit coefficient"))
ylim([-.5,0])
set(fig,'fontsize',14)

