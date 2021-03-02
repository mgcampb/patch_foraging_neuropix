%% Cross reward time decoding 
%  Question: does neural state evolve differentially at later rewards on
%  patch? 
%  - Divide 1, 2, 4 uL reward events at t = 0,t = 1,t = 2
%  - Only decode time since reward  
%  - This could theoretically rule out model 4, which has to time on patch
%    information 

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
session_titles = cell(numel(mPFC_sessions),1); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end  

%% Load neural data from sigCells, concatenate within mice
X = cell(numel(mouse_grps),1); % trials, reward events
X_vel = cell(numel(mouse_grps),1); % trials, reward events
X_pos = cell(numel(mouse_grps),1); % trials, reward events
X_clusters = cell(numel(mouse_grps),1); % one vector of cluster identities per session
for mIdx = 1:5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_clusters{mIdx} = cell(numel(mouse_grps{mIdx}),2); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx})); 
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
        pos_trials = cell(length(data.patchCSL),1);
        for iTrial = 1:length(data.patchCSL)
            fr_mat_trials{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));  
            vel_trials{iTrial} = data.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));  
            pos_trials{iTrial} = data.patch_pos(patchstop_ix(iTrial):patchleave_ix(iTrial));  
        end  
        
        % we could think about just doing a pooling step here
        X{mIdx}{i,1} = fr_mat_trials; 
        X_vel{mIdx}{i,1} = vel_trials; 
        X_pos{mIdx}{i,1} = pos_trials; 
        X_clusters{mIdx}{i} = sig_clusters_session; 
    end
end 

%% Load task variable data and bin according to some discretization 
% maybe throw out labels that don't get used a certain number of times
y = cell(numel(mouse_grps),1); % one cell per reward size  
y_rewsize = cell(numel(mouse_grps),1); 
y_rewTime = cell(numel(mouse_grps),1); 
xval_table = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    y{mIdx} = cell(numel(mouse_grps{mIdx}),1); % just decoding time since rew
    y_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per sesion 
    y_rewTime{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewtimes per sesion 
    SessionName = []; 
    SessionIx = []; 
    Rewsize = [];  
    RewTime = [];
    TrialNum = [];
    FoldID = []; 
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
        rew_sec = data.rew_ts(2:end);    
        
        % restrict analyzed rewards to those delivered leave_buffer_sec before patch leave
        leave_buffer_sec = 1;  
        keep = nan(length(rew_sec),1);
        for iRew = 1:length(rew_sec)  
            keep(iRew) = ~(min(abs(rew_sec(iRew)-patchleave_sec)) < leave_buffer_sec);
        end  
        rew_sec = rew_sec(logical(keep));
        nRews = length(rew_sec);  

        % index vectors
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round((data.patchCSL(:,3)*1000 - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1; 
        prts_ix = patchleave_ix - patchstop_ix + 1;

        % Collect trial reward timings 
        rew_sec_cell = cell(nTrials,1);  
        rew_ix_cell = cell(nTrials,1);  
        rewsize_perRew = cell(nTrials,1); 
        trial_perRew = nan(nRews,1);   
        for iTrial = 1:nTrials
            rew_sec_cell{iTrial} = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_sec_cell{iTrial} * 1000 / tbin_ms;  
            rew_ix_cell{iTrial}(rew_ix_cell{iTrial} == 0) = 1;
            rewsize_perRew{iTrial} = repmat(rewsize(iTrial),[length(rew_sec_cell{iTrial}) 1]); 
            trial_perRew(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) = iTrial; 
        end   
        rewTimes = cat(1,rew_sec_cell{:}); 
        rewsize_perRew = cat(1,rewsize_perRew{:}); 
        
        % Create task variables and bin according to some discretization  
        var_bins = 0:.05:1; % bins to classify timeSinceReward (sec) 
        
        timeSinceReward_binned = cell(nRews,1); 
        fr_mat_rewEvents = cell(nRews,1); 
        vel_rewEvents = cell(nRews,1); 
        pos_rewEvents = cell(nRews,1);  
        iRew = 1;
        for iTrial = 1:nTrials
            rew_event_edges = [rew_ix_cell{iTrial} ; prts_ix(iTrial)];
            for r = 1:(numel(rew_event_edges)-1)
                rew_event_start = rew_event_edges(r) + (r > 1); % add 1 if we are past the t0 reward
                rew_event_end = rew_event_edges(r + 1);
                
                % log binned time since reward
                timeSinceReward_binned{iRew} = (0:(rew_event_end - rew_event_start)) * tbin_ms / 1000;
                [~,~,timeSinceReward_binned{iRew}] = histcounts(timeSinceReward_binned{iRew},var_bins);  
                timeSinceReward_binned{iRew}(timeSinceReward_binned{iRew} == 0) = NaN; % set out of bounds time to NaN
                % log neural activity and behavior 
                fr_mat_rewEvents{iRew} = X{mIdx}{i,1}{iTrial}(:,rew_event_start:rew_event_end);
                vel_rewEvents{iRew} = X_vel{mIdx}{i,1}{iTrial}(rew_event_start:rew_event_end);
                pos_rewEvents{iRew} = X_pos{mIdx}{i,1}{iTrial}(rew_event_start:rew_event_end); 
                iRew = iRew + 1;
            end
        end
        
        % add to xval table data
        SessionName = [SessionName ; repmat(session,[nRews,1])];
        SessionIx = [SessionIx ; repmat(i,[nRews,1])]; 
        Rewsize = [Rewsize ; rewsize_perRew]; 
        RewTime = [RewTime ; rewTimes];  
        TrialNum = [TrialNum ; trial_perRew];
        FoldID = [FoldID ; nan(nRews,1)]; 
        
        % add post reward firing rate data 
        % just overwrite trialed data... we don't care about that anymore
        X{mIdx}{i,2} = fr_mat_rewEvents;  
        X_vel{mIdx}{i,2} = vel_rewEvents;  
        X_pos{mIdx}{i,2} = pos_rewEvents;  
        % task variables
        y{mIdx}{i} = timeSinceReward_binned; 
        y_rewsize{mIdx}{i} = rewsize_perRew;  
        y_rewTime{mIdx}{i} = rewTimes;
    end 
    xval_table{mIdx} = table(SessionName,SessionIx,Rewsize,RewTime,TrialNum,FoldID); 
end 

%% Create datasets for training classifiers
% This may seem a bit repetitive, but workflow allows us to get everything
% together, then specify different options for classification training 
% Also, this cell runs really quickly
dataset_opt = struct;
% select the features to add in different 
dataset_opt.features = {}; 
for clust = 1:3
    dataset_opt.features{clust} = struct;
    dataset_opt.features{clust}.type = "KMeans Clusters"; 
    dataset_opt.features{clust}.ix = (clust); % indices within the feature type we selected
    dataset_opt.features{clust}.shuffle = false; % shuffle? 
    dataset_opt.features{clust}.name = sprintf("Cluster %i",clust); % name for visualizations
end 
dataset_opt.features{4} = struct;
dataset_opt.features{4}.type = "KMeans Clusters";
dataset_opt.features{4}.ix = [1 2 3]; % indices within the feature type we selected
dataset_opt.features{4}.shuffle = false; % shuffle?
dataset_opt.features{4}.name = "All Clusters"; % name for visualizations

dataset_opt.features{5} = struct;
dataset_opt.features{5}.type = "KMeans Clusters";
dataset_opt.features{5}.ix = [1 2 3]; % indices within the feature type we selected
dataset_opt.features{5}.shuffle = true; % shuffle?
dataset_opt.features{5}.name = "Shuffled Neural Data"; % name for visualizations

% other options
dataset_opt.rewsizes = [4]; % which reward size trials will we train to
dataset_opt.numFolds = 2; % xval folds  
dataset_opt.rewTimes = [0 1 2];     
X_dataset = cell(numel(mouse_grps),1);  
y_dataset = cell(numel(mouse_grps),1); 

for mIdx = 1:5 % iterate over mice
    X_dataset{mIdx} = cell(numel(dataset_opt.features),1);
    y_dataset{mIdx} = cell(numel(dataset_opt.features),1);
    for iFeature = 1:numel(dataset_opt.features)
        X_dataset{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        y_dataset{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions, collect data
            sIdx = mouse_grps{mIdx}(i);
            rewsize_perRew = y_rewsize{mIdx}{i};
            rewTimes = y_rewTime{mIdx}{i};
            nRews = length(rewTimes);
            
            % Feature handling
            if strcmp(dataset_opt.features{iFeature}.type,"KMeans Clusters")
                neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.features{iFeature}.ix); % neuron cluster mask
                X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,2},'UniformOutput',false); % X w/ neurons of interest
            elseif strcmp(dataset_opt.features{iFeature}.type,"Velocity")
                X_session_feature = X_vel{mIdx}{i,2};
            elseif strcmp(dataset_opt.features{iFeature}.type,"Position")
                X_session_feature = X_pos{mIdx}{i,2};
            end
            % Shuffle data?
            if dataset_opt.features{iFeature}.shuffle == true
                % save indexing so that we can throw this back in trialed form
                event_lens = cellfun(@(x) size(x,2),X_session_feature);
                end_ix = cumsum(event_lens);
                start_ix = end_ix - event_lens + 1;
                % concatenate, then circshift neurons independently
                X_session_feature_cat = cat(2,X_session_feature{:});
                shifts = randi(size(X_session_feature_cat,2),1,size(X_session_feature_cat,1));
                X_session_feature_cat_shuffle = cell2mat(arrayfun(@(x) circshift(X_session_feature_cat(x,:),[1 shifts(x)]),(1:numel(shifts))','un',0));
                X_session_feature = arrayfun(@(x) X_session_feature_cat_shuffle(:,start_ix(x):end_ix(x)),(1:nRews)','un',0);
            end
            
            % reward size loop
            for iRewsize = 1:numel(dataset_opt.rewsizes)  
                for iRewtime = 1:numel(dataset_opt.rewTimes)
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    this_rewtime = dataset_opt.rewTimes(iRewtime);
                    trials_keep = (rewsize_perRew == this_rewsize & rewTimes == this_rewtime); % rewsize and time mask
                    X_dataset{mIdx}{iFeature}{iRewsize,iRewtime} = [X_dataset{mIdx}{iFeature}{iRewsize,iRewtime};X_session_feature(trials_keep)];
                    y_dataset{mIdx}{iFeature}{iRewsize,iRewtime} = [y_dataset{mIdx}{iFeature}{iRewsize,iRewtime};y{mIdx}{i}(trials_keep)];
                end
            end
        end
    end
    
    % Make xval folds, evenly distributing sessions between folds
    for iRewsize = 1:numel(dataset_opt.rewsizes)
        for iRewtime = 1:numel(dataset_opt.rewTimes)
            this_rewsize = dataset_opt.rewsizes(iRewsize);
            this_rewtime = dataset_opt.rewTimes(iRewtime); 
            these_trials = xval_table{mIdx}.Rewsize == this_rewsize & xval_table{mIdx}.RewTime == this_rewtime;
            xval_table_thisRewsizeRewTime = xval_table{mIdx}(these_trials,:);
            iRewsize_foldid = nan(size(xval_table_thisRewsizeRewTime,1),1);
            shift_by = 0; % change which fold is the "last fold" to make sure one fold is not way smaller than the rest
            for i = 1:numel(mouse_grps{mIdx}) % evenly distribute trials from this session between folds
                keep_this = xval_table_thisRewsizeRewTime.SessionIx == i; % keep trials from this session
                i_nTrials = sum(keep_this); % to ensure proper assignment indexing
                iRewsize_foldid_this = repmat(circshift(1:dataset_opt.numFolds,shift_by),1,ceil(i_nTrials/dataset_opt.numFolds)*dataset_opt.numFolds);
                iRewsize_foldid(keep_this) = iRewsize_foldid_this(1:i_nTrials); % assign folds 1:k
                shift_by = shift_by - mod(i_nTrials,dataset_opt.numFolds); % shift which fold is getting fewer trials
            end
            % assign folds among trials of this reward size
            xval_table{mIdx}(these_trials,:).FoldID = iRewsize_foldid;
        end
    end
end

%% Now use classification datasets to train classifiers within reward size and time, holding out test folds

models = cell(numel(mouse_grps),1);
zero_sigma = 0.5; 
for mIdx = 1:5
    models{mIdx} = cell(numel(dataset_opt.features),1);
    % iterate over the variables we are decoding
    for iFeature = 1:numel(dataset_opt.features)
        models{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        % iterate over reward sizes of interest
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            for iRewtime = 1:numel(dataset_opt.rewTimes)
                models{mIdx}{iFeature}{iRewsize,iRewtime} = cell(dataset_opt.numFolds,1);
                this_rewsize = dataset_opt.rewsizes(iRewsize);  
                this_rewtime = dataset_opt.rewTimes(iRewtime);
                these_trials = xval_table{mIdx}.Rewsize == this_rewsize & xval_table{mIdx}.RewTime == this_rewtime;
               
                % xval folds for this mouse and reward size and time
                foldid = xval_table{mIdx}(these_trials,:).FoldID;
                sessionIx = xval_table{mIdx}(these_trials,:).SessionIx;
                
                % iterate over xval folds and train models
                for kFold = 1:dataset_opt.numFolds
                    [X_train,~,y_train,~] = kfold_split(X_dataset{mIdx}{iFeature}{iRewsize,iRewtime}, ...
                                                        y_dataset{mIdx}{iFeature}{iRewsize,iRewtime}, ...
                                                        foldid,kFold,sessionIx);
                    % Add some noise s.t. we can avoid zero variance gaussians
                    X_train(X_train == 0) = normrnd(0,zero_sigma,[length(find(X_train == 0)),1]); 
                    X_train = X_train + .001 * rand(size(X_train)); % same issue for nonzeros... 
                    models{mIdx}{iFeature}{iRewsize,iRewtime}{kFold} = fitcnb(X_train',y_train,'Prior','uniform');
                end
            end
        end
    end
    fprintf("%s Model Fitting Complete \n",mouse_names(mIdx))
end 

%% Now evaluate on test folds within rewsize/time, just to see how well we're doing baseline 

confusion_mats = cell(numel(mouse_grps),1);
generative_models = cell(numel(mouse_grps),1);
y_true_full = cell(numel(mouse_grps),1);
y_hat_full = cell(numel(mouse_grps),1);
for mIdx = 1:5
    confusion_mats{mIdx} = cell(numel(dataset_opt.features),1);
    generative_models{mIdx} = cell(numel(dataset_opt.features),1);
    y_true_full{mIdx} = cell(numel(dataset_opt.features),1);
    y_hat_full{mIdx} = cell(numel(dataset_opt.features),1);
    for iFeature = 1:numel(dataset_opt.features)
        confusion_mats{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        generative_models{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        y_true_full{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        y_hat_full{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            for iRewtime = 1:numel(dataset_opt.rewTimes)
                confusion_mats_tmp = cell(numel(dataset_opt.numFolds),1);
                generative_models_tmp = cell(numel(dataset_opt.numFolds),1);
                y_true_full_tmp = cell(numel(dataset_opt.numFolds),1);
                y_hat_full_tmp = cell(numel(dataset_opt.numFolds),1);
                this_rewsize = dataset_opt.rewsizes(iRewsize);  
                this_rewtime = dataset_opt.rewTimes(iRewtime);
                these_trials = xval_table{mIdx}.Rewsize == this_rewsize & xval_table{mIdx}.RewTime == this_rewtime;
               
                % xval folds for this mouse and reward size and time
                foldid = xval_table{mIdx}(these_trials,:).FoldID;
                sessionIx = xval_table{mIdx}(these_trials,:).SessionIx;
                for kFold = 1:dataset_opt.numFolds
                    [~,X_test,~,y_test] = kfold_split(X_dataset{mIdx}{iFeature}{iRewsize,iRewtime}, ...
                                                      y_dataset{mIdx}{iFeature}{iRewsize,iRewtime}, ...
                                                      foldid,kFold,sessionIx);
                    [y_hat,Posterior] = predict(models{mIdx}{iFeature}{iRewsize}{kFold},X_test');
                    confusion_mats_tmp{kFold} = confusionmat(y_test,y_hat);
                    generative_models_tmp{kFold} = rot90(cellfun(@(x) x(1),models{mIdx}{iFeature}{iRewsize,iRewtime}{kFold}.DistributionParameters));
                    y_true_full_tmp{kFold} = y_test;
                    y_hat_full_tmp{kFold} = y_hat;
                end
                confusion_mats{mIdx}{iFeature}{iRewsize,iRewtime} = sum(cat(3,confusion_mats_tmp{:}),3);
                generative_models{mIdx}{iFeature}{iRewsize,iRewtime} = mean(cat(3,generative_models_tmp{:}),3);
                y_true_full{mIdx}{iFeature}{iRewsize,iRewtime} = cat(1,y_true_full_tmp{:});
                y_hat_full{mIdx}{iFeature}{iRewsize,iRewtime} = cat(1,y_hat_full_tmp{:});
            end
        end
    end
    fprintf("%s Model Evaluation Complete \n",mouse_names(mIdx))
end 

%% Analyze decoding performance in terms of time prediction 

error_bin_width = .05;
error_hmap = cell(numel(mouse_grps),1);
yhat_mean_withinRewsize = cell(numel(mouse_grps),1);
yhat_sem_withinRewsize = cell(numel(mouse_grps),1);
absError_mean = cell(numel(mouse_grps),1);
absError_sem = cell(numel(mouse_grps),1);
learned_representations = cell(numel(mouse_grps),1);
for mIdx = 1:5
    error_hmap{mIdx} = cell(numel(dataset_opt.features),1);
    yhat_mean_withinRewsize{mIdx} = cell(numel(dataset_opt.features),1);
    yhat_sem_withinRewsize{mIdx} = cell(numel(dataset_opt.features),1);
    absError_mean{mIdx} = cell(numel(dataset_opt.features),1);
    absError_sem{mIdx} = cell(numel(dataset_opt.features),1);
    learned_representations{mIdx} = cell(numel(dataset_opt.features),1);
    for iFeature = 1:numel(dataset_opt.features)
        error_hmap{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        yhat_mean_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        yhat_sem_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        absError_mean{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        absError_sem{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        error_bins = -max(var_bins):error_bin_width:max(var_bins);
        msec_conversion = diff(var_bins(1:2));
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            for iRewtime = 1:numel(dataset_opt.rewTimes)
                i_y_true = y_true_full{mIdx}{iFeature}{iRewsize,iRewtime};
                i_y_true_msec = msec_conversion * i_y_true;
                i_y_hat_msec = msec_conversion * y_hat_full{mIdx}{iFeature}{iRewsize,iRewtime};
                errors_msec = i_y_hat_msec - i_y_true_msec; % so that positive errors are late predictions
                error_hmap{mIdx}{iFeature}{iRewsize,iRewtime} = nan(length(error_bins)-1,max(i_y_true));
                yhat_mean_withinRewsize{mIdx}{iFeature}{iRewsize,iRewtime} = nan(max(i_y_true),1);
                yhat_sem_withinRewsize{mIdx}{iFeature}{iRewsize,iRewtime} = nan(max(i_y_true),1);
                absError_mean{mIdx}{iFeature}{iRewsize,iRewtime} = nan(max(i_y_true),1);
                absError_sem{mIdx}{iFeature}{iRewsize,iRewtime} = nan(max(i_y_true),1);
                for true_label = 1:max(i_y_true)
                    error_hmap{mIdx}{iFeature}{iRewsize,iRewtime}(:,true_label) = histcounts(errors_msec(i_y_true == true_label),error_bins,'Normalization', 'probability');
                    yhat_mean_withinRewsize{mIdx}{iFeature}{iRewsize,iRewtime}(true_label) = mean(i_y_hat_msec(i_y_true == true_label));
                    yhat_sem_withinRewsize{mIdx}{iFeature}{iRewsize,iRewtime}(true_label) = std(i_y_hat_msec(i_y_true == true_label)) / sqrt(length(find(i_y_true == true_label)));
                    absError_mean{mIdx}{iFeature}{iRewsize,iRewtime}(true_label) = mean(abs(errors_msec(i_y_true == true_label)));
                    absError_sem{mIdx}{iFeature}{iRewsize,iRewtime}(true_label) = std(abs(errors_msec(i_y_true == true_label))) / sqrt(length(find(i_y_true == true_label)));
                end
            end
        end
    end
end

%% Visualize encoding representations
close all
for iRewsize = 1
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 3
        figure()
        for mIdx = 1:numel(mouse_grps)
            for iRewtime = [1 2 3] % [2 1 3] %  1:numel(dataset_opt.rewTimes)
                subplot(numel(dataset_opt.rewTimes),5,mIdx + 5 * (iRewtime - 1))  
                if iRewtime == 1
                    [~,max_ix] = max(generative_models{mIdx}{iFeature}{iRewsize,iRewtime},[],2);    
                    [~,peaksort] = sort(max_ix);
                end
                
                imagesc(flipud(zscore(generative_models{mIdx}{iFeature}{iRewsize,iRewtime}(peaksort,:),[],2)))
                xticks(1:5:(numel(var_bins)))
                xticklabels(var_bins(1:5:end))
                
                if mIdx == 1
                    ylabel("Neurons")
                end
                if iRewtime == 1
                    title(mouse_names(mIdx))
                end
                xlabel(sprintf("Time since t=%i reward (sec)",dataset_opt.rewTimes(iRewtime)))
            end
        end
        suptitle(sprintf("%s, %i uL Trials",dataset_opt.features{iFeature}.name,this_rewsize))
    end
end

%% Visualize decoding performance with heatmap
close all
vis_mice = 1:5;
for iRewsize = 1
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 2
        figure()
        for m = 1:numel(vis_mice)
            mIdx = vis_mice(m); 
            for iRewtime = 1:numel(dataset_opt.rewTimes) 
                this_rewtime = dataset_opt.rewTimes(iRewtime);
                subplot(numel(dataset_opt.rewTimes),numel(vis_mice),m + numel(vis_mice) * (iRewtime - 1))
                
                imagesc(flipud(confusion_mats{mIdx}{iFeature}{iRewsize,iRewtime} ./ sum(confusion_mats{mIdx}{iFeature}{iRewsize,iRewtime},2)))
                xticks(1:5:(numel(var_bins)))
                xticklabels(var_bins(1:5:end))
                yticks(1:5:(numel(var_bins)))
                yticklabels(fliplr(var_bins(1:5:end)))
                
                if m == 1
                    ylabel(sprintf("True t=%i Time since reward",this_rewtime))
                end 
                
                if iRewtime == 1
                    title(mouse_names(mIdx))
                end
                
                xlabel(sprintf("Decoded t=%i Time since reward",this_rewtime))
            end
        end
        suptitle(sprintf("%s, %i uL Trials",dataset_opt.features{iFeature}.name,this_rewsize))
    end
end

%% Visualize decoding performance betw reward sizes with line plot of abs error
colors = winter(3);
close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
vis_mice = 1:5;
shuffle_ix = 5;
for iFeature = 2
    figure()
    for m = 1:numel(vis_mice)
        mIdx = vis_mice(m);
        for iRewsize = 1:numel(dataset_opt.rewsizes) 
            this_rewsize = dataset_opt.rewsizes(iRewsize);
            subplot(numel(dataset_opt.rewsizes),numel(vis_mice),m + numel(vis_mice) * (iRewsize - 1))
            for iRewtime = 1:numel(dataset_opt.rewTimes)
                shadedErrorBar(var_bins(1:end-1),absError_mean{mIdx}{iFeature}{iRewsize,iRewtime},absError_sem{mIdx}{iFeature}{iRewsize,iRewtime},'lineprops',{'Color',colors(iRewtime,:)});
            end
            
            shadedErrorBar(var_bins(1:end-1),absError_mean{mIdx}{shuffle_ix}{iRewsize,iRewtime},absError_sem{mIdx}{shuffle_ix}{iRewsize,iRewtime},'lineprops',{'Color',[0 0 0]});
            
            ylim([0,1])
            
            if mIdx == 1
                ylabel("Time since reward |Decoding Error|") 
                legend(["t = 0 Reward","t = 1 Reward","t = 2 Reward","Shuffled Neural Data"])
            end
            if iRewsize == 1
                title(mouse_names(mIdx))
            end
            xlabel(sprintf("True time since %i uL reward",this_rewsize))
        end
    end
    suptitle(sprintf("%s Decoding time-course",dataset_opt.features{iFeature}.name))
end 

%% Cross reward time decoding 
vis_rewTimes = 2:3; 
confusion_mats_xRewtime = cell(numel(mouse_grps),1); 
yhat_mean = cell(numel(mouse_grps),1); 
yhat_sem = cell(numel(mouse_grps),1); 
for mIdx = 5
    confusion_mats_xRewtime{mIdx} = cell(numel(dataset_opt.features),1);  
    yhat_mean{mIdx} = cell(numel(dataset_opt.features),1);  
    yhat_sem{mIdx} = cell(numel(dataset_opt.features),1);  
    for iFeature = 1:numel(dataset_opt.features)
        confusion_mats_xRewtime{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),1);
        yhat_mean{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        yhat_sem{mIdx}{iFeature} = cell(numel(dataset_opt.rewsizes),numel(dataset_opt.rewTimes));
        for iRewsize = 1:numel(dataset_opt.rewsizes) 
            this_rewsize = dataset_opt.rewsizes(iRewsize); 
            msec_conversion = diff(var_bins(1:2));
            for i_heldout_rewTime = 1:numel(vis_rewTimes) % numel(dataset_opt.rewTimes)     
                heldout_rewTime = vis_rewTimes(i_heldout_rewTime);
                % initialize cells for predictions from different models 
                confusion_mats_xRewtime{mIdx}{iFeature}{iRewsize,heldout_rewTime} = cell(numel(dataset_opt.rewTimes),1);
                yhat_mean{mIdx}{iFeature}{iRewsize,heldout_rewTime} = cell(numel(dataset_opt.rewTimes),1);
                yhat_sem{mIdx}{iFeature}{iRewsize,heldout_rewTime} = cell(numel(dataset_opt.rewTimes),1); 
                
                % get the data from this reward time
                this_rewTime = dataset_opt.rewTimes(heldout_rewTime); % dataset_opt.rewTimes(i_heldout_rewTime);
                heldout_rewTime_trials = (xval_table{mIdx}.Rewsize == this_rewsize & xval_table{mIdx}.RewTime == this_rewTime);
                sessionIx = xval_table{mIdx}(heldout_rewTime_trials,:).SessionIx; 
                X_heldout_full = padCat(X_dataset{mIdx}{iFeature}{iRewsize,heldout_rewTime},sessionIx);
                y_heldout_full = cat(2,y_dataset{mIdx}{iFeature}{iRewsize,heldout_rewTime}{:});  
                
                % now iterate over the other reward times
                trained_rewTimes = setdiff(1:numel(vis_rewTimes),i_heldout_rewTime);  
                for i_trained_rewTime = 1:numel(trained_rewTimes)
                    trained_rewTime = trained_rewTimes(i_trained_rewTime); 
%                     disp(trained_rewTime)
                    confusion_mats_xRewtime_tmp = cell(numel(dataset_opt.numFolds),1); 
                    y_hat_tmp = cell(numel(dataset_opt.numFolds),1); 
                    for kFold = 1:dataset_opt.numFolds
                        this_model = models{mIdx}{iFeature}{iRewsize,trained_rewTime}{kFold};
                        y_hat_tmp{kFold} = predict(this_model,X_heldout_full');
                        confusion_mats_xRewtime_tmp{kFold} = confusionmat(y_heldout_full,y_hat_tmp{kFold});
                    end
%                     disp([i_heldout_rewTime trained_rewTime])
                    confusion_mats_xRewtime{mIdx}{iFeature}{iRewsize,heldout_rewTime}{trained_rewTime} = sum(cat(3,confusion_mats_xRewtime_tmp{:}),3); 
                    y_hat = msec_conversion * mean(cat(2,y_hat_tmp{:}),2);
                    
                    for true_label = 1:max(y_heldout_full)
                        yhat_mean{mIdx}{iFeature}{iRewsize,heldout_rewTime}{trained_rewTime}(true_label) = mean(y_hat(y_heldout_full == true_label));
                        yhat_sem{mIdx}{iFeature}{iRewsize,heldout_rewTime}{trained_rewTime}(true_label) = std(y_hat(y_heldout_full == true_label)) / sqrt(length(find(y_heldout_full == true_label)));
                    end
                end
                
                yhat_mean{mIdx}{iFeature}{iRewsize,heldout_rewTime}{i_heldout_rewTime} = yhat_mean_withinRewsize{mIdx}{iFeature}{iRewsize,heldout_rewTime}';
                yhat_sem{mIdx}{iFeature}{iRewsize,heldout_rewTime}{i_heldout_rewTime} = yhat_sem_withinRewsize{mIdx}{iFeature}{iRewsize,heldout_rewTime}';
            end
        end
    end
    fprintf("%s Cross-Rewsize Analysis Complete \n",mouse_names(mIdx))
end 

%% First visualize cross reward size using confusionmat heatmap  
% vis_rewTimes = 1:3; 
close all
for mIdx = 5
    for iFeature = 4
        for iRewsize = 1
            figure()
            for i_heldout_rewTime = 1:numel(vis_rewTimes) 
                heldout_rewTime = vis_rewTimes(i_heldout_rewTime);
                for i_trained_rewTime = 1:numel(vis_rewTimes) 
                    trained_rewTime = vis_rewTimes(i_trained_rewTime);
                    subplot(numel(vis_rewTimes),numel(vis_rewTimes),i_trained_rewTime + numel(vis_rewTimes) * (i_heldout_rewTime - 1))  
                    if heldout_rewTime ~= trained_rewTime  
                        imagesc(flipud(confusion_mats_xRewtime{mIdx}{iFeature}{iRewsize,heldout_rewTime}{trained_rewTime})) 
                    else 
                        % trained on this rewsize
                        imagesc(flipud(confusion_mats{mIdx}{iFeature}{iRewsize,heldout_rewTime}))
                    end
                    
                    xticks(1:5:(numel(var_bins)))
                    xticklabels(var_bins(1:5:end))
                    yticks(1:5:(numel(var_bins)))
                    yticklabels(fliplr(var_bins(1:5:end)))

                    if i_trained_rewTime == 1 
                        ylabel(sprintf("True Time Since t = %i Reward",dataset_opt.rewTimes(heldout_rewTime)))
                    end 
                    if i_heldout_rewTime == numel(vis_rewTimes)
                        xlabel(sprintf("t = %i Model Decoded Time Since Reward",dataset_opt.rewTimes(trained_rewTime)))
                    end
                end
            end
        end 
        suptitle(sprintf("%s %s Cross-Rewtime Time Since Reward Decoding",dataset_opt.features{iFeature}.name,mouse_names(mIdx)))
    end
end

%% Now visualize cross rewtime decoding as line plot 
colors = cbrewer('qual','Set1',3); 
% close all
vis_mice = 1:5;
vis_rewTimes = 1:3;
for iFeature = 4
    figure()
    for iRewsize = 1
        for m = 1:numel(vis_mice)
            mIdx = vis_mice(m);
            for i_heldout_rewTime = vis_rewTimes
                subplot(numel(vis_rewTimes),numel(vis_mice),m + numel(vis_mice) * (i_heldout_rewTime - 1));hold on
                for i_trained_rewTime = vis_rewTimes
                    
                    shadedErrorBar(var_bins(1:end-1),yhat_mean{mIdx}{iFeature}{iRewsize,i_heldout_rewTime}{i_trained_rewTime},yhat_sem{mIdx}{iFeature}{iRewsize,i_heldout_rewTime}{i_trained_rewTime},'lineprops',{'Color',colors(i_trained_rewTime,:)});

                    % axis labels
                    if mIdx == 1 && i_trained_rewTime == 1
                        ylabel(sprintf("Decoded t = %i Time Since Reward",dataset_opt.rewTimes(i_heldout_rewTime)))
                    end
                end
                
                plot([0 max(var_bins(1:end-1))],[0 max(var_bins(1:end-1))],'k--','linewidth',1.5)
                
                if i_heldout_rewTime == 1
                    title(mouse_names(mIdx))
                end 
                
                if i_heldout_rewTime == numel(vis_rewTimes)
                    xlabel("True time since reward")
                end
            end
            if m == 1
                legend(["t=0 Trained Model" ...
                        "t=1 Trained Model", ... 
                        "t=2 Trained Model", ...
                        "Perfect Prediction"])
            end
        end
    end
    suptitle(sprintf("%s Cross-Rewtime Time Since %i uL  Reward Decoding",dataset_opt.features{iFeature}.name,dataset_opt.rewsizes(iRewsize)))
end

