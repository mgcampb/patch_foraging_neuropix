%% Now decoding with single cell selection 

%% Set paths
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 
sig_cell_sessions = sig_cells.Session;  

% analysis options
calcFR_opt = struct;
calcFR_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calcFR_opt.smoothSigma_time = .05; % 50 msec smoothing
calcFR_opt.patch_leave_buffer = 0; % in seconds; only takes within patch times up to this amount before patch leave
calcFR_opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
calcFR_opt.cortex_only = true; 
calcFR_opt.tstart = 0;
tbin_ms = calcFR_opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 15:18 23 25];   
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]};  
mouse_names = ["m75","m76","m78","m79","m80"];
session_titles = cell(numel(mouse_grps),1);
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];
        session_titles{mIdx}{i} = session_title;
    end
end  

%% Load neural data from sigCells, concatenate within mice
X = cell(numel(mouse_grps),1); % one per task variable
X_vel = cell(numel(mouse_grps),1); % one per task variable
X_accel = cell(numel(mouse_grps),1); % one per task variable
X_pos = cell(numel(mouse_grps),1); % one per task variable
X_clusters = cell(numel(mouse_grps),1); % one vector of cluster identities per session 
X_cellIDs = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_accel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_clusters{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    X_cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));  
        acceleration = gradient(data.vel);
        sig_cellIDs_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).CellID;   
        sig_clusters_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).KMeansCluster;
        calcFR_opt.tend = max(data.sp.st);
        fr_mat = calcFRVsTime(sig_cellIDs_session,data,calcFR_opt); 
        
        patchstop_ms = data.patchCSL(:,2)*1000;
        patchleave_ms = data.patchCSL(:,3)*1000;
        patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
        patchleave_ix = round((patchleave_ms - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1;
        
        % Gather firing rate matrices in trial form
        fr_mat_trials = cell(length(data.patchCSL),1); 
        vel_trials = cell(length(data.patchCSL),1);
        accel_trials = cell(length(data.patchCSL),1);
        pos_trials = cell(length(data.patchCSL),1);
        for iTrial = 1:length(data.patchCSL)
            fr_mat_trials{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));  
            vel_trials{iTrial} = data.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));   
            accel_trials{iTrial} = acceleration(patchstop_ix(iTrial):patchleave_ix(iTrial));
            pos_trials{iTrial} = data.patch_pos(patchstop_ix(iTrial):patchleave_ix(iTrial));  
        end  
        
        % we could think about just doing a pooling step here
        X{mIdx}{i,1} = fr_mat_trials; 
        X{mIdx}{i,2} = fr_mat_trials;  
        X_vel{mIdx}{i,1} = vel_trials; 
        X_vel{mIdx}{i,2} = vel_trials;  
        X_pos{mIdx}{i,1} = pos_trials; 
        X_pos{mIdx}{i,2} = pos_trials;  
        X_accel{mIdx}{i,1} = accel_trials; 
        X_accel{mIdx}{i,2} = accel_trials; 
        X_clusters{mIdx}{i} = sig_clusters_session;  
        X_cellIDs{mIdx}{i} = sig_cellIDs_session; 
    end
end 

%% Load task variable data and bin according to some discretization 
% maybe throw out labels that don't get used a certain number of times
y = cell(numel(mouse_grps),1); % one cell per reward size  
y_rewsize = cell(numel(mouse_grps),1);
xval_table = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    y{mIdx} = cell(numel(mouse_grps{mIdx}),3); % 3 variables to decode 
    y_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per sesion  
    xval_table{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx}(1:end-4); 
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        % Load session information
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);  
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);  
        prts = patchleave_sec - patchstop_sec; 
        rew_sec = data.rew_ts;  
        % index vectors
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round((data.patchCSL(:,3)*1000 - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1; 
        prts_ix = patchleave_ix - patchstop_ix + 1;

        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            last_rew_ix = max(rew_indices);
            rew_sec_cell{iTrial} = rew_indices(rew_indices >= 1);
        end
        
        % Create task variables and bin according to some discretization  
        var_bins{1} = 0:.05:2; % bins to classify timeSinceReward (sec) 
        var_bins{2} = 0:.05:2; % bins to classify timeOnPatch (sec) 
        var_bins{3} = 0:.05:2; % bins to classify time2Leave (sec) 
        
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);  
        fr_mat_postRew = X{mIdx}{i,1}; 
        vel_postRew = X_vel{mIdx}{i,1}; 
        accel_postRew = X_accel{mIdx}{i,1}; 
        pos_postRew = X_pos{mIdx}{i,1}; 
        for iTrial = 1:nTrials
            trial_len_ix = prts_ix(iTrial);
            timeSinceReward_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
            timeOnPatch_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
            for r = 1:numel(rew_sec_cell{iTrial})
                rew_ix = (rew_sec_cell{iTrial}(r)) * 1000 / tbin_ms; 
                timeSinceReward_binned{iTrial}(rew_ix:end) =  (1:length(timeSinceReward_binned{iTrial}(rew_ix:end))) * tbin_ms / 1000;
                if r == numel(rew_sec_cell{iTrial}) % if it is our last reward
                    time2Leave_binned{iTrial} = fliplr(timeSinceReward_binned{iTrial}(rew_ix:end)); % add to time2leave 
                    fr_mat_postRew{iTrial} = fr_mat_postRew{iTrial}(:,rew_ix:end); % cut down to only after final reward 
                    vel_postRew{iTrial} = vel_postRew{iTrial}(rew_ix:end); 
                    accel_postRew{iTrial} = accel_postRew{iTrial}(rew_ix:end); 
                    pos_postRew{iTrial} = pos_postRew{iTrial}(rew_ix:end); 
                end
            end   
            % handle no extra rewards case
            if numel(rew_sec_cell{iTrial}) == 0 
                time2Leave_binned{iTrial} = fliplr(timeSinceReward_binned{iTrial}); 
            end
            
            % Now use histcounts to actually bin task variables  
            % Set out of bounds (0 label) as NaN to denote missing data
            [~,~,timeSinceReward_binned{iTrial}] = histcounts(timeSinceReward_binned{iTrial},var_bins{1});  
            timeSinceReward_binned{iTrial}(timeSinceReward_binned{iTrial} == 0) = NaN; 
            [~,~,timeOnPatch_binned{iTrial}] = histcounts(timeOnPatch_binned{iTrial},var_bins{2});  
            timeOnPatch_binned{iTrial}(timeOnPatch_binned{iTrial} == 0) = NaN; 
            [~,~,time2Leave_binned{iTrial}] = histcounts(time2Leave_binned{iTrial},var_bins{3});  
            time2Leave_binned{iTrial}(time2Leave_binned{iTrial} == 0) = NaN; 
        end

        % add post reward firing rate data
        X{mIdx}{i,3} = fr_mat_postRew;  
        X_vel{mIdx}{i,3} = vel_postRew;   
        X_accel{mIdx}{i,3} = accel_postRew;  
        X_pos{mIdx}{i,3} = pos_postRew;   
        % task variables
        y{mIdx}{i,1} = timeSinceReward_binned; 
        y{mIdx}{i,2} = timeOnPatch_binned; 
        y{mIdx}{i,3} = time2Leave_binned;   
        y_rewsize{mIdx}{i} = rewsize; 
        
        % xval table variables
        SessionName = repmat(session,[nTrials,1]);
        SessionIx = repmat(i,[nTrials,1]);
        Rewsize = rewsize;
        TrialNum = (1:nTrials)'; 
        FoldID = nan(nTrials,1); 
        % make xval table for this session
        xval_table{mIdx}{i} = table(SessionName,SessionIx,TrialNum,Rewsize,FoldID); 
    end 
end 

%% Create datasets for training classifiers
% This may seem a bit repetitive, but workflow allows us to get everything
% together, then specify different options for classification training 
% Also, this cell runs really quickly
dataset_opt = struct;
% select the features to add in different
dataset_opt.features = cell(numel(mouse_grps),1); 
% iterate within sessions, add cellIDs
for mIdx = 1:numel(mouse_grps) 
    dataset_opt.features{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i_session = 1:numel(mouse_grps{mIdx})  
        dataset_opt.features{mIdx}{i_session} = cell(numel(X_cellIDs{mIdx}{i_session}),1); 
        for i_cellID = 1:numel(X_cellIDs{mIdx}{i_session}) 
            cellID = X_cellIDs{mIdx}{i_session}(i_cellID);
            dataset_opt.features{mIdx}{i_session}{i_cellID} = struct;
            dataset_opt.features{mIdx}{i_session}{i_cellID}.type = "CellID";
            dataset_opt.features{mIdx}{i_session}{i_cellID}.ix = cellID; % cellID
            dataset_opt.features{mIdx}{i_session}{i_cellID}.shuffle = false; % shuffle?
            dataset_opt.features{mIdx}{i_session}{i_cellID}.name = sprintf("%s Cell %i",session_titles{mIdx}{i_session},cellID); % name for visualizations
        end 
        % add all clusters to end of the features
        dataset_opt.features{mIdx}{i_session}{numel(dataset_opt.features{mIdx}{i_session})+1} = struct;
        dataset_opt.features{mIdx}{i_session}{numel(dataset_opt.features{mIdx}{i_session})}.type = "KMeans Clusters";
        dataset_opt.features{mIdx}{i_session}{numel(dataset_opt.features{mIdx}{i_session})}.ix = [1 2 3]; % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{numel(dataset_opt.features{mIdx}{i_session})}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{numel(dataset_opt.features{mIdx}{i_session})}.name = "All Clusters"; % name for visualizations
    end 
end

% other options
dataset_opt.rewsizes = [1 2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 5; % xval folds  
dataset_opt.vars = [1 2 3];    

[X_dataset,y_dataset,xval_table] = gen_multiclassDataset_singleSession(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,y_rewsize,xval_table,dataset_opt,mouse_grps);

%% Now use classification datasets to train classifiers, holding out test folds
dataset_opt.distribution = 'normal';
models = fit_dataset_singleSession(X_dataset,y_dataset,xval_table,dataset_opt,mouse_names);

% Now evaluate classifier performance on test folds 

[y_true_full,y_hat_full] = predict_dataset_singleSession(X_dataset,y_dataset,models,xval_table,dataset_opt,mouse_names);

%% Analyze model performance   
nMice = numel(X_dataset); 
abs_error_mean_givenTrue = cell(nMice,1); 
abs_error_mean_givenTrueAll = cell(nMice,1); 
abs_error_mean_givenHat = cell(nMice,1); 
abs_error_mean_givenHatAll = cell(nMice,1); 
cond_means = cell(nMice,1);  
confusion_mats = cell(nMice,1); 
MI_cells = cell(nMice,1);   
MI_sessions = cell(nMice,1);

for mIdx = 1:nMice
    abs_error_mean_givenTrue{mIdx} = cell(numel(X_dataset{mIdx}),1); 
    abs_error_mean_givenTrueAll{mIdx} = cell(numel(X_dataset{mIdx}),1);  
    abs_error_mean_givenHat{mIdx} = cell(numel(X_dataset{mIdx}),1); 
    abs_error_mean_givenHatAll{mIdx} = cell(numel(X_dataset{mIdx}),1);  
    cond_means{mIdx} = cell(numel(X_dataset{mIdx}),1); 
    confusion_mats{mIdx} = cell(numel(X_dataset{mIdx}),1);
    MI_cells{mIdx} = cell(numel(X_dataset{mIdx}),1); 
    MI_sessions{mIdx} = cell(numel(X_dataset{mIdx}),1); 
    for i = 1:numel(X_dataset{mIdx}) 
        abs_error_mean_givenTrue{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        abs_error_mean_givenTrueAll{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        abs_error_mean_givenHat{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        abs_error_mean_givenHatAll{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        cond_means{mIdx}{i} = cell(numel(dataset_opt.vars),1);  
        confusion_mats{mIdx}{i} = cell(numel(dataset_opt.vars),1);
        MI_cells{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        MI_sessions{mIdx}{i} = cell(numel(dataset_opt.vars),1); 
        for iVar = 1:numel(dataset_opt.vars)  
            abs_error_mean_givenTrue{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
            abs_error_mean_givenTrueAll{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            abs_error_mean_givenHat{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
            abs_error_mean_givenHatAll{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            cond_means{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
            confusion_mats{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            MI_cells{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            MI_sessions{mIdx}{i}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            for iRewsize = 1:numel(dataset_opt.rewsizes)    
                % get y_true and y_hat to eval errors
                y_true = y_true_full{mIdx}{i}{iVar}{iRewsize};
                % preallocate metrics 
                abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i})-1,length(var_bins{iVar})-1);  
                abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i})-1,length(var_bins{iVar})-1);  
                cond_means{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i})-1,length(var_bins{iVar})-1);   
                confusion_mats{mIdx}{i}{iVar}{iRewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
                MI_cells{mIdx}{i}{iVar}{iRewsize} = nan(numel(dataset_opt.features{mIdx}{i}),1); 
                for iFeature = 1:(numel(dataset_opt.features{mIdx}{i})-1)   
                    confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature} = confusionmat(y_true_full{mIdx}{i}{iVar}{iRewsize},... 
                                                                                     y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature});   
                    MI_cells{mIdx}{i}{iVar}{iRewsize}(iFeature) = MI_confusionmat(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature});                                      
                                                                                 
                    y_hat = y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature};
                    errors = y_true - y_hat;
                    for true_label = 1:max(y_true)
                        abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize}(iFeature,true_label) = nanmean(abs(errors(y_true == true_label)));
                        abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize}(iFeature,true_label) = nanmean(abs(errors(y_hat == true_label)));
                    end  
                    
                    % Now get conditional means per neuron 
                    cond_means_tmp = cell(dataset_opt.numFolds,1);
                    for kFold = 1:dataset_opt.numFolds 
                        if dataset_opt.distribution == "normal"
                            cond_means_tmp{kFold} = cellfun(@(x) x(1),models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold}.DistributionParameters); 
                        else 
                            % get mean of kernel distribution 
                            cond_means_tmp{kFold} = cellfun(@(x) mean(x),models{mIdx}{i}{iVar}{iRewsize}{iFeature}{kFold}.DistributionParameters); 
                        end
                    end
                    
                    cond_means{mIdx}{i}{iVar}{iRewsize}(iFeature,:) = mean(cat(3,cond_means_tmp{:}),3); 
                end
                
                % Now do same for feature that has all neurons in it  
                last_feature = numel(dataset_opt.features{mIdx}{i});
                y_hat = y_hat_full{mIdx}{i}{iVar}{iRewsize}{last_feature};
                errors = y_true - y_hat;
                abs_error_mean_givenTrueAll{mIdx}{i}{iVar}{iRewsize} = nan(length(var_bins{iVar})-1,1); 
                abs_error_mean_givenHatAll{mIdx}{i}{iVar}{iRewsize} = nan(length(var_bins{iVar})-1,1); 
                for true_label = 1:max(y_true) 
                    abs_error_mean_givenTrueAll{mIdx}{i}{iVar}{iRewsize}(true_label) = nanmean(abs(errors(y_true == true_label)));
                    abs_error_mean_givenHatAll{mIdx}{i}{iVar}{iRewsize}(true_label) = nanmean(abs(errors(y_hat == true_label)));
                end 
                confusion_mats{mIdx}{i}{iVar}{iRewsize}{last_feature} = confusionmat(y_true_full{mIdx}{i}{iVar}{iRewsize},... 
                                                                                     y_hat_full{mIdx}{i}{iVar}{iRewsize}{last_feature});  
                                                                                 
                MI_sessions{mIdx}{i}{iVar}{iRewsize} = MI_confusionmat(confusion_mats{mIdx}{i}{iVar}{iRewsize}{last_feature});   
            end
        end 
    end 
end 

% 
% % Pool within mice
% abs_error_pooled = cell(numel(mouse_grps),1); 
% cond_means_pooled = cell(numel(mouse_grps),1); 
% clusters_pooled = cell(numel(mouse_grps),1); 
% for mIdx = 1:numel(mouse_grps) 
%     abs_error_pooled{mIdx} = cell(numel(dataset_opt.vars),1); 
%     cond_means_pooled{mIdx} = cell(numel(dataset_opt.vars),1); 
%     clusters_pooled{mIdx} = []; 
%     for iVar = 1:numel(dataset_opt.vars) 
%         abs_error_pooled{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
%         cond_means_pooled{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
%         for iRewsize = 1:numel(dataset_opt.rewsizes) 
%             abs_error_tmp = cell(numel(mouse_grps{mIdx},1)); 
%             cond_means_tmp = cell(numel(mouse_grps{mIdx},1)); 
%             for i = 1:numel(mouse_grps{mIdx})
%                 abs_error_tmp{i} = abs_error_mean{mIdx}{i}{iVar}{iRewsize};
%                 cond_means_tmp{i} = cond_means{mIdx}{i}{iVar}{iRewsize};
%                 if iVar == numel(dataset_opt.vars) && iRewsize == numel(dataset_opt.rewsizes) 
%                     clusters_pooled{mIdx} = [clusters_pooled{mIdx} ; X_clusters{mIdx}{i}]; % log cluster IDs
%                 end
%             end 
%             abs_error_pooled{mIdx}{iVar}{iRewsize} = cat(1,abs_error_tmp{:});
%             cond_means_pooled{mIdx}{iVar}{iRewsize} = cat(1,cond_means_tmp{:});
%         end
%     end
% end 
% 
% Pool across all mice 
abs_error_givenTrue_pooled = cell(numel(dataset_opt.vars),1); 
abs_error_givenHat_pooled = cell(numel(dataset_opt.vars),1); 
MI_cells_pooled = cell(numel(dataset_opt.vars),1);  
MI_sessions_pooled = cell(numel(dataset_opt.vars),1);  
cond_means_pooled = cell(numel(dataset_opt.vars),1); 
clusters_allPooled = []; 
s_nNeurons = []; 
for iVar = 1:numel(dataset_opt.vars) 
    abs_error_givenTrue_pooled{iVar} = cell(numel(dataset_opt.rewsizes),1); 
    abs_error_givenHat_pooled{iVar} = cell(numel(dataset_opt.rewsizes),1);  
    MI_cells_pooled{iVar} = cell(numel(dataset_opt.rewsizes),1); 
    MI_sessions_pooled{iVar} = cell(numel(dataset_opt.rewsizes),1); 
    cond_means_pooled{iVar} = cell(numel(dataset_opt.rewsizes),1); 
    for iRewsize = 1:numel(dataset_opt.rewsizes)
        abs_error_givenTrue_pooled{iVar}{iRewsize} = []; 
        abs_error_givenHat_pooled{iVar}{iRewsize} = [];  
        MI_cells_pooled{iVar}{iRewsize} = []; 
        MI_sessions_pooled{iVar}{iRewsize} = []; 
        cond_means_pooled{iVar}{iRewsize} = []; 
        for mIdx = 1:numel(mouse_grps) 
            for i = 1:numel(mouse_grps{mIdx})
                abs_error_givenTrue_pooled{iVar}{iRewsize} = [abs_error_givenTrue_pooled{iVar}{iRewsize} ; abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize}]; 
                abs_error_givenHat_pooled{iVar}{iRewsize} = [abs_error_givenHat_pooled{iVar}{iRewsize} ; abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize}]; 
                MI_cells_pooled{iVar}{iRewsize} = [MI_cells_pooled{iVar}{iRewsize} ; MI_cells{mIdx}{i}{iVar}{iRewsize}];
                MI_sessions_pooled{iVar}{iRewsize} = [MI_sessions_pooled{iVar}{iRewsize} ; MI_sessions{mIdx}{i}{iVar}{iRewsize}];
                
                cond_means_pooled{iVar}{iRewsize} = [cond_means_pooled{iVar}{iRewsize} ; cond_means{mIdx}{i}{iVar}{iRewsize}]; 
                if iVar == numel(dataset_opt.vars) && iRewsize == numel(dataset_opt.rewsizes) 
                    clusters_allPooled = [clusters_allPooled ; X_clusters{mIdx}{i}]; % log cluster IDs 
                    s_nNeurons = [s_nNeurons ; length(X_clusters{mIdx}{i})]; % log session number of neurons
                end
            end
        end
    end
end

%% Visualize single cell decoding performance  
var_names = ["Time Since Reward","Time on Patch","Time Until Leave"];
figure()
for mIdx = 5
    for i = 1
        for iVar = 3
            for iRewsize = 3 
                for iCell = 1:25
                    subplot(5,5,iCell)
                    imagesc(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iCell+0}) %./sum(confusion_mats{mIdx}{i}{iVar}{iRewsize}{iCell+0},2)) 
                    xticks([])
                    yticks([]) 
                    if iCell == 23
                        xlabel(sprintf("Predicted %s",var_names(iVar)))
                    end 
                    if iCell == 11
                        ylabel(sprintf("True %s",var_names(iVar)))
                    end
                end 
            end
        end 
        suptitle(sprintf("%s Single Cell Confusion Matrices (Un-Normalized)",session_titles{mIdx}{i}))
    end 
end 

%% Vis single cells vs population decoding accuracy  
colors = lines(3);
for mIdx = 5
    for i = 1
        for iVar = 1
            for iRewsize = 3
                figure() ;hold on
                for iCell = 1:(numel(dataset_opt.features{mIdx}{i})-1)   
                    plot(abs_error_mean_givenHat{mIdx}{i}{iVar}{iRewsize}(iCell,:),'color',colors(X_clusters{mIdx}{i}(iCell),:))
                end
                plot(abs_error_mean_givenHatAll{mIdx}{i}{iVar}{iRewsize},'linewidth',2)
            end
        end
    end
end 

%% Visualize decoding vs encoding peak relationships
close all
for iVar = 1
    for iRewsize = 3
        [~,min_error_ix] = min(abs_error_givenHat_pooled{iVar}{iRewsize},[],2);
        [~,peak_fr_ix] = min(cond_means_pooled{iVar}{iRewsize},[],2);
        [~,decoding_sort] = sort(min_error_ix);
        [~,encoding_sort] = sort(peak_fr_ix);
        
        figure()
        s(1) = subplot(2,2,1);
        imagesc(flipud(zscore(cond_means_pooled{iVar}{iRewsize}(encoding_sort,:),[],2))) 
        caxis([-3,3])  
        ylabel("Sort by min FR bin") 
        title("Z-scored Time Since Reward Encoding")
        s(2) = subplot(2,2,2);
        imagesc(flipud(abs_error_givenHat_pooled{iVar}{iRewsize}(encoding_sort,:))) 
        colorbar() 
        caxis([0 10]) 
        title("Mean Abs Decoding Error")
        s(3) = subplot(2,2,3);
        imagesc(flipud(zscore(cond_means_pooled{iVar}{iRewsize}(decoding_sort,:),[],2))) 
        caxis([-3,3]) 
        ylabel("Sort by min Decoding Error") 
        xlabel("True Time Since Reward")
        s(4) = subplot(2,2,4);
        imagesc(flipud(abs_error_givenHat_pooled{iVar}{iRewsize}(decoding_sort,:))) 
        colorbar() 
        caxis([0 10])  
        xlabel("Predicted Time Since Reward")

        colormap(s(1),parula)
        colormap(s(2),copper)
        colormap(s(3),parula)
        colormap(s(4),copper)
    end 
    suptitle("Time Since Reward Encoding vs Decoding")
end

%% Relationship between dip in FR and incr in accuracy acr rewsizes
for iVar = 1
    figure()
    for iRewsize_encoding = 1:3 
        [~,min_fr_ix] = min(cond_means_pooled{iVar}{iRewsize_encoding},[],2); 
        [~,encoding_sort] = sort(min_fr_ix);
        for iRewsize_decoding = 1:3  
            [~,min_error_ix] = min(abs_error_givenTrue_pooled{iVar}{iRewsize_decoding},[],2); 
            subplot(3,3,iRewsize_encoding + (iRewsize_decoding-1)*3)
            imagesc(flipud(abs_error_givenTrue_pooled{iVar}{iRewsize_decoding}(encoding_sort,:)))  
%             imagesc(flipud(zscore(cond_means_allPooled{iVar}{iRewsize_decoding}(decoding_sort,:),[],2))) 
            colorbar() 
            colormap(copper)
            caxis([0 10])   
%             caxis([-3,3])
            lm = fitlm(min_fr_ix,min_error_ix);
            beta = lm.Coefficients.Estimate(2);
            [r,p] = corrcoef(min_fr_ix,min_error_ix); 
            title(sprintf("r = %.3f, p = %.3f, β = %.3f",r(2),p(2),beta))
        end
    end
end 

%% Visualize single cell decoding performance in individual sessions 
colors = lines(3); 
for mIdx = 1:numel(mouse_grps)
    for iVar = 3
        for iRewsize = 3
            figure()
            for i = 1:numel(mouse_grps{mIdx})
                for iClust = 1:3 
                    subplot(3,numel(mouse_grps{mIdx}),i + (iClust-1)*numel(mouse_grps{mIdx}))
                    plot(abs_error_mean_givenTrueAll{mIdx}{i}{iVar}{iRewsize},'k','linewidth',2) ;hold on 
                end  
                for iNeuron = 1:size(abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize},1) 
                    subplot(3,numel(mouse_grps{mIdx}),i + (X_clusters{mIdx}{i}(iNeuron)-1) * numel(mouse_grps{mIdx}))
                    plot(abs_error_mean_givenTrue{mIdx}{i}{iVar}{iRewsize}(iNeuron,:),'color',colors(X_clusters{mIdx}{i}(iNeuron),:),'linewidth',.5)
                end
            end
        end
    end
end

%% Visualize relationship between population confusionmats and mutual information  
close all
n = length(var_bins{1})-1;
for iVar = 1
    for mIdx = 1:5
        figure()
        for i = 1:numel(mouse_grps{mIdx})
            for iRewsize = 1:3
                subplot(3,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}) * (iRewsize-1))
                imagesc(flipud(confusion_mats{mIdx}{i}{iVar}{iRewsize}{end} ./ sum(confusion_mats{mIdx}{i}{iVar}{iRewsize}{end},1)))  
                if iRewsize == 1
                    title(sprintf("%s N = %i \n MI = %.3f",session_titles{mIdx}{i},length(X_clusters{mIdx}{i}),MI_sessions{mIdx}{i}{iVar}{iRewsize}/log(n)))
                else 
                    title(sprintf("MI = %.3f",MI_sessions{mIdx}{i}{iVar}{iRewsize}/log(n)))
                end  
                yticks([])  
                xticks([])
                if i == 1
                    ylabel(sprintf("%i uL",dataset_opt.rewsizes(iRewsize)))
                end
            end 
        end 
        suptitle(sprintf("Decoding %s",var_names(iVar)))
    end
end

%% Visualize information content across sessions
% for both single cells and full sessions 
n = length(var_bins{1})-1;
figure()
for iVar = 1:3 
    for iRewsize = 1:3  
        [r,p] = corrcoef(s_nNeurons,MI_sessions_pooled{iVar}{iRewsize}/log(n));
        subplot(3,3,iRewsize + 3 * (iVar-1)) 
        scatter(s_nNeurons,MI_sessions_pooled{iVar}{iRewsize}/log(n))  
        ylim([.05,.3]) 
        title(sprintf("%i uL %s \n (r = %.3f p = %.3f)",dataset_opt.rewsizes(iRewsize),var_names{iVar},r(2),p(2)))  
        
        if iVar == 3
            xlabel("Session # Sig Cells")
        end
        if iRewsize == 1
            ylabel("Mutual Information")
        end
    end 
end 

%% Visualize single cell information vs population information 
close all
n = length(var_bins{1})-1;
for iVar = 1
    for mIdx = 1:5
        f = figure();
        for i = 1:numel(mouse_grps{mIdx})
            for iRewsize = 1:3
                subplot(3,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}) * (iRewsize-1)) 
                clust1 = [MI_cells{mIdx}{i}{iVar}{iRewsize}(X_clusters{mIdx}{i} == 1) 1+ zeros(length(find((X_clusters{mIdx}{i} == 1))),1)];
                clust2 = [MI_cells{mIdx}{i}{iVar}{iRewsize}(X_clusters{mIdx}{i} == 2) 2+ zeros(length(find((X_clusters{mIdx}{i} == 2))),1)];
                clust3 = [MI_cells{mIdx}{i}{iVar}{iRewsize}(X_clusters{mIdx}{i} == 3) 3+ zeros(length(find((X_clusters{mIdx}{i} == 3))),1)];
                MI_data = [clust1 ; clust2 ; clust3];
%                 violinplot(MI_data(:,1),MI_data(:,2),'ViolinColor',lines(3));hold on 
                gscatter(.2 * rand(size(MI_data,1),1) + MI_data(:,2),.01 * randn(size(MI_data,1),1) + MI_data(:,1)/log(n),MI_data(:,2),lines(3),'o',5) 
                xticks([1 2 3]) 
                xlim([0 4]) 
                ylim([0 .3]) 
                hLeg=findobj(f(1,1),'type','legend'); set(hLeg,'visible','off')
%                 histogram(MI_cells{mIdx}{i}{iVar}{iRewsize}) ;hold on
                yline(MI_sessions{mIdx}{i}{iRewsize}{iVar}/log(n),'linewidth',2) 
%                 ylim([0 MI_sessions{mIdx}{i}{iRewsize}{iVar}/log(n)+.1])
                if iRewsize == 1
                    title(sprintf("%s N = %i \n Session MI = %.3f",session_titles{mIdx}{i},length(X_clusters{mIdx}{i}),MI_sessions{mIdx}{i}{iVar}{iRewsize}/log(n)))
                else 
                    title(sprintf("Session MI = %.3f",MI_sessions{mIdx}{i}{iVar}{iRewsize}/log(n)))
                end  

                if i == 1
                    ylabel(sprintf("%i uL \n MI",dataset_opt.rewsizes(iRewsize)))
                end
            end 
        end 
        suptitle(sprintf("Decoding %s",var_names(iVar)))
    end
end

