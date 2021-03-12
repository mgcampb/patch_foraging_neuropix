%% Re-do population decoding framework so we look at one session at a time 
%  - Motivated by fact that population decoding fidelity looks very
%    different in different sessions, esp by diff # sig cells 
%  - Also use this script to introduce new evaluation procedures for time
%    since reward and time on patch decoding: 
%       i) Separate time since reward evaluation by reward time 
%       ii) Separate time on patch evaluation by reward delivery schedule 
%           - is there time on patch information that persists past reward delivery? esp. in Clu2/3
%       iii) 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table.mat';
load(paths.transients_table);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 
% sig_cell_sessions = sig_cells.Session;  

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

%% Load neural data from sigCells, concatenate within mice
X = cell(numel(mouse_grps),1); % one per task variable
X_vel = cell(numel(mouse_grps),1); % one per task variable
X_accel = cell(numel(mouse_grps),1); % one per task variable
X_pos = cell(numel(mouse_grps),1); % one per task variable
X_clusters = cell(numel(mouse_grps),1); % one vector of GLM cluster identities per session 
X_peak_pos = cell(numel(mouse_grps),1); % one vector of positive peak indices per session
X_cellIDs = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_accel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_clusters{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    X_peak_pos{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    X_cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        data = load(fullfile(paths.data,sessions{sIdx}));  
        acceleration = gradient(data.vel); 
        good_cells = data.sp.cids(data.sp.cgs==2);   
        
        calcFR_opt.tend = max(data.sp.st); 
        fr_mat = calcFRVsTime(good_cells,data,calcFR_opt);  
        
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
        
        session_table = transients_table(strcmp(transients_table.Session,session_title),:); 
        glm_clusters_session = session_table.GLM_Cluster; 
        
        % we could think about just doing a pooling step here
        X{mIdx}{i,1} = fr_mat_trials; 
        X{mIdx}{i,2} = fr_mat_trials;  
        X_vel{mIdx}{i,1} = vel_trials; 
        X_vel{mIdx}{i,2} = vel_trials;  
        X_pos{mIdx}{i,1} = pos_trials; 
        X_pos{mIdx}{i,2} = pos_trials;  
        X_accel{mIdx}{i,1} = accel_trials; 
        X_accel{mIdx}{i,2} = accel_trials; 
        X_clusters{mIdx}{i} = glm_clusters_session;   
        X_peak_pos{mIdx}{i} = [session_table.Rew0_peak_pos session_table.Rew1plus_peak_pos];
        X_cellIDs{mIdx}{i} = good_cells; 
    end
end  

%% Load task variable data and bin according to some discretization 
% maybe throw out labels that don't get used a certain number of times
y = cell(numel(mouse_grps),1); % one cell per reward size   
y_rewsize = cell(numel(mouse_grps),1);
RX = cell(numel(mouse_grps),1); 
RXX = cell(numel(mouse_grps),1);  
rew_time = cell(numel(mouse_grps),1);  
rew_num = cell(numel(mouse_grps),1);  
xval_table = cell(numel(mouse_grps),1);  
for mIdx = 1:5
    y{mIdx} = cell(numel(mouse_grps{mIdx}),3); % 3 variables to decode 
    y_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per sesion   
    RX{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    RXX{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_time{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_num{mIdx} = cell(numel(mouse_grps{mIdx}),1);
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
        floor_prts = floor(prts); % for rew barcode
        rew_sec = data.rew_ts;  
        % index vectors
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round((data.patchCSL(:,3)*1000 - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1; 
        prts_ix = patchleave_ix - patchstop_ix + 1;

        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        nTimesteps = 10; 
        nTrials = length(rewsize); 
        rew_barcode = zeros(length(data.patchCSL) , nTimesteps);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_sec_cell{iTrial} = rew_indices(rew_indices >= 1); 
            % make rew_barcode for time on patch evaluation separation
            % Note we add 1 to rew_indices here because we are now 1 indexing
            rew_barcode(iTrial , (max(rew_indices+1)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices+1) = rewsize(iTrial);
        end 
        
        % collect RX and RXX reward schedule labels
        session_RXX = nan(nTrials,1);
        for iRewsize = [1 2 4] 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0) = double(sprintf("%i00",iRewsize)); 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize) = double(sprintf("%i0%i",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize));
        end
        RX{mIdx}{i} = floor(session_RXX/10); 
        RXX{mIdx}{i} = session_RXX; 
        
        % Create task variables and bin according to some discretization  
        var_bins{1} = 0:.05:2; % bins to classify timeSinceReward (sec) 
        var_bins{2} = 0:.05:2; % bins to classify timeOnPatch (sec) 
        var_bins{3} = 0:.05:2; % bins to classify time2Leave (sec) 
        
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);   
        session_rewTime = cell(nTrials,1); 
        session_rewNum = cell(nTrials,1); 
        fr_mat_postRew = X{mIdx}{i,1}; 
        vel_postRew = X_vel{mIdx}{i,1}; 
        accel_postRew = X_accel{mIdx}{i,1}; 
        pos_postRew = X_pos{mIdx}{i,1}; 
        for iTrial = 1:nTrials
            trial_len_ix = prts_ix(iTrial);
            timeSinceReward_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
            timeOnPatch_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000; 
            session_rewTime{iTrial} = zeros(trial_len_ix,1); 
            session_rewNum{iTrial} = zeros(trial_len_ix,1); 
            for r = 1:numel(rew_sec_cell{iTrial})
                rew_ix = (rew_sec_cell{iTrial}(r)) * 1000 / tbin_ms; 
                timeSinceReward_binned{iTrial}(rew_ix:end) =  (1:length(timeSinceReward_binned{iTrial}(rew_ix:end))) * tbin_ms / 1000;
                if r == numel(rew_sec_cell{iTrial}) % if it is our last reward
                    time2Leave_binned{iTrial} = fliplr(timeSinceReward_binned{iTrial}(rew_ix:end)); % add to time2leave  
                    session_rewTime{iTrial}(rew_ix:end) = rew_sec_cell{iTrial}(r); 
                    session_rewNum{iTrial}(rew_ix:end) = r;
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
        rew_time{mIdx}{i} = session_rewTime;
        rew_num{mIdx}{i} = session_rewNum;

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

%% Create population datasets 
dataset_opt = struct;
% select the features to add in different
dataset_opt.features = cell(numel(mouse_grps),1); 
% iterate within sessions, add cellIDs
for mIdx = 1:numel(mouse_grps) 
    dataset_opt.features{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i_session = 1:numel(mouse_grps{mIdx})  
        sIdx = mouse_grps{mIdx}(i_session);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]); 
        dataset_opt.features{mIdx}{i_session} = cell(7,1); 
        for iCluster = 1:3 
            if ~isempty(find(X_clusters{mIdx}{i_session} == iCluster,1))
                dataset_opt.features{mIdx}{i_session}{iCluster} = struct;
                dataset_opt.features{mIdx}{i_session}{iCluster}.type = "KMeans Clusters";
                dataset_opt.features{mIdx}{i_session}{iCluster}.ix = iCluster; % indices within the feature type we selected
                dataset_opt.features{mIdx}{i_session}{iCluster}.shuffle = false; % shuffle?
                dataset_opt.features{mIdx}{i_session}{iCluster}.name = sprintf("Cluster %i",iCluster); % name for visualizations 
            end
        end 
        % add all clusters to end of the features
        dataset_opt.features{mIdx}{i_session}{4} = struct;
        dataset_opt.features{mIdx}{i_session}{4}.type = "KMeans Clusters";
        dataset_opt.features{mIdx}{i_session}{4}.ix = [1 2 3]; % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{4}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{4}.name = "All Clusters"; % name for visualizations 
        % add cells w/ sig transient to rew0 and rew1plus
        dataset_opt.features{mIdx}{i_session}{5} = struct;
        dataset_opt.features{mIdx}{i_session}{5}.type = "CellID";
        dataset_opt.features{mIdx}{i_session}{5}.ix = X_cellIDs{mIdx}{i}(all(~isnan(X_peak_pos{mIdx}{i}),2))';
        dataset_opt.features{mIdx}{i_session}{5}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{5}.name = "Transient Selection Cells"; % name for visualizations 
        % Velocity
        dataset_opt.features{mIdx}{i_session}{6} = struct;
        dataset_opt.features{mIdx}{i_session}{6}.type = "Velocity";
        dataset_opt.features{mIdx}{i_session}{6}.ix = []; % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{6}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{6}.name = "Velocity"; % name for visualizations
        % Velocity
        dataset_opt.features{mIdx}{i_session}{7} = struct;
        dataset_opt.features{mIdx}{i_session}{7}.type = "Position";
        dataset_opt.features{mIdx}{i_session}{7}.ix = []; % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{7}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{7}.name = "Position"; % name for visualizations
    end 
end

% other options
dataset_opt.rewsizes = [1 2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 5; % xval folds  
dataset_opt.vars = [1 2 3];    

[X_dataset,y_dataset,xval_table] = gen_multiclassDataset_singleSession(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,y_rewsize,xval_table,dataset_opt,mouse_grps);

%% Train and evaluate population classifiers 
% Now use classification datasets to train classifiers, holding out test folds
dataset_opt.distribution = 'normal';
models = fit_dataset_singleSession(X_dataset,y_dataset,xval_table,dataset_opt,mouse_names);

% Now evaluate classifier performance on test folds 
[y_true_full,y_hat_full] = predict_dataset_singleSession(X_dataset,y_dataset,models,xval_table,dataset_opt,mouse_names);

%% Need new temporally precise error metric. MSE | true time and predicted time are in some interval? 


%% Take error and information metrics from performance on heldout data
%  Functionalize this better!

[cond_means,confusion_mats,mutual_information,MSE, ...
 abs_error_mean_givenTrue,abs_error_mean_givenHat,... 
 abs_error_sem_givenTrue,abs_error_sem_givenHat,... 
 accuracy_mean_givenTrue,accuracy_mean_givenHat,...
 accuracy_sem_givenTrue,accuracy_sem_givenHat] = eval_dataset_singleSession(models,y_hat_full,y_true_full,dataset_opt,var_bins);


%% Visualize error over time metric  
% 1 variable, 1 mouse, session x panels, rewsize y panels, features plotted together

var_names = ["Time Since Reward","Time on Patch","Time to Leave"]; 
metric_name = "Accuracy given true";
vis_features = 1:4;  
feature_names = ["Cluster 1", "Cluster 2","Cluster 3","All GLM Neurons","Transient Selection Cells","Velocity","Position"];

colors = lines(10);  
vis_mice = 5; 
vis_var = 2; 
vis_errorOverTime(vis_mice,vis_var,vis_features,accuracy_mean_givenHat,accuracy_sem_givenHat,colors,dataset_opt,session_titles,feature_names,var_bins,var_names,metric_name) 

%% Visualize confusion matrices 
var_names = ["Time Since Reward","Time on Patch","Time to Leave"]; 
close all
vis_features = 1:2;  % 1:5;
iRewsize = 2; 
iVar = 1;  
vis_mice = [1 5];
vis_confusionMat(confusion_mats,X_dataset,session_titles,mutual_information,var_bins,var_names,iVar,iRewsize,vis_features,vis_mice,dataset_opt)


%% Evaluate time since reward decoding separated by reward number 
% the endgame here: get confusionmat(y_true(these_ix),y_pred(these_ix))
rewTime_confusion_mats = cell(nMice,1); 
rewNum_confusion_mats = cell(nMice,1); 
timesince_ix = 1; 
analyze_rewTimes = {0 1 2:10};  
rewTime_names = ["t = 0","t = 1","t = 2+"];
analyze_rewNums = {0 1 2:10};  
rewNum_names = ["rew 1","rew 2","rew 2+"];
for mIdx = 1:numel(mouse_grps)  
    rewTime_confusion_mats{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    rewNum_confusion_mats{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        rewTime_confusion_mats{mIdx}{i} = cell(numel(dataset_opt.rewsizes),1); 
        rewNum_confusion_mats{mIdx}{i} = cell(numel(dataset_opt.rewsizes),1);  
        foldID = xval_table{mIdx}{i}.FoldID;   
        session_rewsize = xval_table{mIdx}{i}.Rewsize;
        for i_rewsize = 1:numel(dataset_opt.rewsizes) 
            iRewsize = dataset_opt.rewsizes(i_rewsize); 
            rew_time_tmp = cell(dataset_opt.numFolds,1); 
            rew_num_tmp = cell(dataset_opt.numFolds,1); 
            for kFold =  1:dataset_opt.numFolds % this is how the test folds are added 
                rew_time_fold = rew_time{mIdx}{i}(foldID == kFold & session_rewsize == iRewsize);
                rew_time_tmp{kFold} = cat(1,rew_time_fold{:});  
                rew_num_fold = rew_num{mIdx}{i}(foldID == kFold & session_rewsize == iRewsize); 
                rew_num_tmp{kFold} = cat(1,rew_num_fold{:}); 
            end 
            rew_time_full = cat(1,rew_time_tmp{:}); 
            rew_num_full = cat(1,rew_num_tmp{:}); 
            timesince_true = y_true_full{mIdx}{i}{timesince_ix}{i_rewsize};      
              
            rewTime_confusion_mats{mIdx}{i}{i_rewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
            rewNum_confusion_mats{mIdx}{i}{i_rewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
            % iterate over features and get confusion matrices for diff rew times / rew numbers.. maybe pool rews 1+
            for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) 
                if ~isempty (dataset_opt.features{mIdx}{i}{iFeature})
                    rewTime_confusion_mats{mIdx}{i}{i_rewsize}{iFeature} = cell(numel(analyze_rewTimes),1); 
                    rewNum_confusion_mats{mIdx}{i}{i_rewsize}{iFeature} = cell(numel(analyze_rewNums),1); 
                    time_on_patch_hat = y_hat_full{mIdx}{i}{timesince_ix}{i_rewsize}{iFeature}; 

                    for i_rew_time = 1:numel(analyze_rewTimes)
                        iRewtime = analyze_rewTimes{i_rew_time}; 
                        rewtime_ix = ismember(rew_time_full,iRewtime); % so we can use ranges 
                        rewTime_confusion_mats{mIdx}{i}{i_rewsize}{iFeature}{i_rew_time} = confusionmat(timesince_true(rewtime_ix),time_on_patch_hat(rewtime_ix));  
                    end 

                    for i_rew_num = 1:numel(analyze_rewTimes)
                        iRewnum = analyze_rewTimes{i_rew_num}; 
                        rewnum_ix = ismember(rew_num_full,iRewnum); % so we can use ranges 
                        rewNum_confusion_mats{mIdx}{i}{i_rewsize}{iFeature}{i_rew_num} = confusionmat(timesince_true(rewnum_ix),time_on_patch_hat(rewnum_ix));  
                    end
                end
            end
        end
    end
end

%% Evaluate time on patch decoding separated by RXX
time_on_patch_ix = 2;  
analyze_tts = {[100 101 200 202 400 404] [110 111 220 222 440 444]};  
RX_confusion_mats = cell(nMice,1); 
for mIdx = 1:numel(mouse_grps)  
    RX_confusion_mats{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})  
        RX_confusion_mats{mIdx}{i} = cell(numel(dataset_opt.rewsizes),1); 
        session_RXX = RXX{mIdx}{i};   
        session_rewsize = xval_table{mIdx}{i}.Rewsize; 
        t_lens = cellfun(@length,rew_time{mIdx}{i}); 
        foldID = xval_table{mIdx}{i}.FoldID; 
        for i_rewsize = 1:numel(dataset_opt.rewsizes) 
            iRewsize = dataset_opt.rewsizes(i_rewsize);          
            RXX_ordered_tmp = cell(dataset_opt.numFolds,1);  
            t_lens_ordered_tmp = cell(dataset_opt.numFolds,1);  
            for kFold =  1:dataset_opt.numFolds % this is how the test folds are added 
                RXX_ordered_tmp{kFold} = session_RXX(foldID == kFold & session_rewsize == iRewsize); 
                t_lens_ordered_tmp{kFold} = t_lens(foldID == kFold & session_rewsize == iRewsize); 
            end 
            RXX_ordered = cat(1,RXX_ordered_tmp{:}); 
            t_lens_ordered = cat(1,t_lens_ordered_tmp{:});
            
            % Now do an arrayfun to get RX in trialed form 
            RXX_full = arrayfun(@(t) RXX_ordered(t) + zeros(t_lens_ordered(t),1),(1:length(find(session_rewsize == iRewsize)))','un',0);
            RXX_full = cat(1,RXX_full{:}); % now use this bitch to index

            time_on_patch_true = y_true_full{mIdx}{i}{time_on_patch_ix}{i_rewsize};  
              
%             disp(length(RX_full) == length(time_on_patch_true))
%             % iterate over features and get confusion matrices for diff
            
            RX_confusion_mats{mIdx}{i}{i_rewsize} = cell(numel(dataset_opt.features{mIdx}{i}),1); 
            % iterate over features and get confusion matrices for diff trial types
            for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) 
                if ~isempty (dataset_opt.features{mIdx}{i}{iFeature})
                    RX_confusion_mats{mIdx}{i}{i_rewsize}{iFeature} = cell(numel(analyze_tts),1); 
                    time_on_patch_hat = y_hat_full{mIdx}{i}{time_on_patch_ix}{i_rewsize}{iFeature}; 

                    for i_trialtype = 1:numel(analyze_tts)
                        this_trialtype = analyze_tts{i_trialtype}; 
                        trialtype_ix = ismember(RXX_full,this_trialtype); % so we can use ranges 
                        RX_confusion_mats{mIdx}{i}{i_rewsize}{iFeature}{i_trialtype} = confusionmat(time_on_patch_true(trialtype_ix),time_on_patch_hat(trialtype_ix));  
                    end
                end
            end
        end
    end
end

%% Now visualize time since reward separated by reward time / num 
rew_time_bool = false; % separate by reward time ? (else use number)
if rew_time_bool == true
    analyze_rew_ix = analyze_rewTimes; 
else 
    analyze_rew_ix = analyze_rewNums; 
end

close all
for iRewsize = 3
    for iFeature = 4
        for mIdx = 1:numel(mouse_grps)   
            figure()
            for i = 1:numel(mouse_grps{mIdx}) 
                for i_rew_ix = 1:numel(analyze_rew_ix)
                    subplot(numel(analyze_rew_ix),numel(mouse_grps{mIdx}),i + (i_rew_ix - 1) * numel(mouse_grps{mIdx}))  
                    if ~isempty(rewTime_confusion_mats{mIdx}{i}{iRewsize}{iFeature})
                        imagesc(flipud(rewTime_confusion_mats{mIdx}{i}{iRewsize}{iFeature}{i_rew_ix})) 
                        if i == 1 
                            if rew_time_bool == true
                                ylabel(sprintf("%s \n True Time",rewTime_names(i_rew_ix))) 
                            else 
                                ylabel(sprintf("%s \n True Time",rewNum_names(i_rew_ix)))
                            end 
                            yticks(1:10:(length(var_bins{timesince_ix})-1));
                            yticklabels(fliplr(var_bins{1}(1:10:end)))
                        else 
                            xticks([])
                            yticks([])
                        end
                    end  
                    if i_rew_ix == 1 
                        title(session_titles{mIdx}(i))
                    end
                    
                    if i_rew_ix == numel(analyze_rew_ix) 
                        xlabel("Predicted Time")
                        xticks(1:10:(length(var_bins{timesince_ix})-1));
                        xticklabels(var_bins{1}(1:10:end))
                    end
                end
            end 
            suptitle(sprintf("%s Time Since %i uL Rew \n Decoded by %s",mouse_names(mIdx),dataset_opt.rewsizes(iRewsize),dataset_opt.features{mIdx}{i}{iFeature}.name))
        end
    end 
end

%% And visualize time on patch separated by RXX 
close all
for iRewsize = 3
    tt_names = [sprintf("%i0",dataset_opt.rewsizes(iRewsize)) sprintf("%i%i",dataset_opt.rewsizes(iRewsize),dataset_opt.rewsizes(iRewsize))];
    for iFeature = 2:3
        for mIdx = 3 % 1:numel(mouse_grps)
            figure()
            for i = 1:numel(mouse_grps{mIdx}) 
                for i_trialtype = 1:numel(analyze_tts) 
                    subplot(numel(analyze_tts),numel(mouse_grps{mIdx}),i + (i_trialtype - 1) * numel(mouse_grps{mIdx}))  
%                     subplot(numel(analyze_tts),2,min(2,i) + (i_trialtype - 1) * 2)  
                    if ~isempty(rewNum_confusion_mats{mIdx}{i}{iRewsize}{iFeature})
                        imagesc(flipud(RX_confusion_mats{mIdx}{i}{iRewsize}{iFeature}{i_trialtype})) 
                        if i == 1 
                            ylabel(sprintf("%s \n True Time (sec)",tt_names(i_trialtype)))
                            yticks(1:10:(length(var_bins{timesince_ix})-1));
                            yticklabels(fliplr(var_bins{1}(1:10:end)))
                        else 
                            xticks([])
                            yticks([])
                        end
                    end
                    if i_trialtype == numel(analyze_tts) 
                        xlabel("Predicted Time (sec)")
                        xticks(1:10:(length(var_bins{timesince_ix})-1));
                        xticklabels(var_bins{1}(1:10:end))
                    end 
                    if i_trialtype == 1 
                        title(session_titles{mIdx}(i))
                    end
                end
            end 
            suptitle(sprintf("%s Time On %i uL Patch \n Decoded by %s",mouse_names(mIdx),dataset_opt.rewsizes(iRewsize),dataset_opt.features{mIdx}{i}{iFeature}.name))
        end
    end
end

%% First reformat datastructures to make x-session pooling easier
mutual_information_xSessions = cell(numel(mouse_grps),1); 
confusion_mats_xSessions = cell(numel(mouse_grps),1);  
rewTime_confusion_mats_xSessions = cell(numel(mouse_grps),1); 
rewNum_confusion_mats_xSessions = cell(numel(mouse_grps),1); 
RX_confusion_mats_xSessions = cell(numel(mouse_grps),1); 
y_hat_full_xSessions = cell(numel(mouse_grps),1); 
y_true_full_xSessions = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)  
    for i = 1:numel(mouse_grps{mIdx})
        for iVar = 1:numel(dataset_opt.vars)  
            for iRewsize = 1:numel(dataset_opt.rewsizes)  
                y_true_full_xSessions{mIdx}{iVar}{iRewsize}{i} = y_true_full{mIdx}{i}{iRewsize}{iVar};
                for iFeature = 1:numel(dataset_opt.features{mIdx}{i}) 
                    mutual_information_xSessions{mIdx}{iVar}{iRewsize}(iFeature,i) = mutual_information{mIdx}{i}{iVar}{iRewsize}(iFeature); 
                    confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}{i} = confusion_mats{mIdx}{i}{iVar}{iRewsize}{iFeature}; 
                    rewTime_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}{:,i} = rewNum_confusion_mats{mIdx}{i}{iRewsize}{iFeature};
                    rewNum_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}{:,i} = rewTime_confusion_mats{mIdx}{i}{iRewsize}{iFeature};
                    RX_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}{:,i} = RX_confusion_mats{mIdx}{i}{iRewsize}{iFeature};  
                    y_hat_full_xSessions{mIdx}{iVar}{iRewsize}{iFeature}{i} = y_hat_full{mIdx}{i}{iVar}{iRewsize}{iFeature};
                end
            end
        end
    end
end 

%% what is distn of information acr everything?  this doesnt mean much currently lol 
all_mi = []; 
for mIdx = 1:5 
    for iVar = 1:3 
        for iRewsize = 1:3
            all_mi = [all_mi ; mutual_information_xSessions{mIdx}{iVar}{iRewsize}(:)]; 
        end
    end
end

%% Perform some pooling across sessions within mice before performing more detailed analysis  

% session inclusion criterion
MI_threshold = .3; % 0.1; % this is arbitrary; figure something out real later. maybe based on shuffle control 
% prep data structures
pooled_confusion_mats = cell(numel(mouse_grps),1); 
pooled_RX_confusion_mats = cell(numel(mouse_grps),1); 
pooled_rewTime_confusion_mats = cell(numel(mouse_grps),1);  
pooled_rewNum_confusion_mats = cell(numel(mouse_grps),1);  
pooled_y_hat = cell(numel(mouse_grps),1);
pooled_y_true = cell(numel(mouse_grps),1);

for mIdx = 1:numel(mouse_grps)
    for iVar = 1:numel(dataset_opt.vars) 
        for iRewsize = 1:numel(dataset_opt.rewsizes)  
            for iFeature = 1:numel(dataset_opt.features{mIdx}{1}) % this isn't super legit. works unless diff nums of features per mouse
                this_xSession_mi = mutual_information_xSessions{mIdx}{iVar}{iRewsize}(iFeature,:);  
                include_sessions = this_xSession_mi >= MI_threshold;   
%                 disp(length(find(include_sessions)))
                pooled_confusion_mats_tmp = confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}(include_sessions); 
                pooled_confusion_mats{mIdx}{iVar}{iRewsize}{iFeature} = sum(cat(3,pooled_confusion_mats_tmp{:}),3);  
                
                % pool y hat and y true... note y_true has features now b/c
                % diff features are included per session 
                if ~isempty(find(include_sessions,1))
                    pooled_y_true_tmp = y_true_full_xSessions{mIdx}{iVar}{iRewsize}(include_sessions);
                    pooled_y_true{mIdx}{iVar}{iRewsize}{iFeature} = cat(1,pooled_y_true_tmp{:}); 
                    pooled_y_hat_tmp = y_hat_full_xSessions{mIdx}{iVar}{iRewsize}{iFeature}(include_sessions);
                    pooled_y_hat{mIdx}{iVar}{iRewsize}{iFeature} = cat(1,pooled_y_hat_tmp{:}); 
                end
                
                % handle rewNum and rewTime
                if iVar == timesince_ix
                    for iRewtime = 1:numel(analyze_rewTimes)   
                        pooled_confusion_mats_tmp1 = rewTime_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}(include_sessions); 
                        pooled_confusion_mats_tmp2 = cellfun(@(x) x{iRewtime},pooled_confusion_mats_tmp1,'un',0);  
                        size_inclusion = cellfun(@length,pooled_confusion_mats_tmp2) == length(var_bins{iVar})-1; 
                        pooled_confusion_mats_tmp3 = pooled_confusion_mats_tmp2(size_inclusion); 
                        pooled_rewTime_confusion_mats{mIdx}{iRewsize}{iFeature}{iRewtime} = sum(cat(3,pooled_confusion_mats_tmp3{:}),3); 
                    end
                    
                    for iRewnum = 1:numel(analyze_rewNums) 
                        pooled_confusion_mats_tmp = rewNum_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}(include_sessions);  
                        pooled_confusion_mats_tmp = cellfun(@(x) x{iRewnum},pooled_confusion_mats_tmp,'un',0);
                        size_inclusion = cellfun(@length,pooled_confusion_mats_tmp) == length(var_bins{iVar})-1; 
                        pooled_confusion_mats_tmp = pooled_confusion_mats_tmp(size_inclusion); 
                        pooled_rewNum_confusion_mats{mIdx}{iRewsize}{iFeature}{iRewnum} = sum(cat(3,pooled_confusion_mats_tmp{:}),3); 
                    end
                end 
                
                % handle RXX separated time on patch 
                if iVar == time_on_patch_ix 
                    for iTrialtype = 1:numel(analyze_tts)
                        pooled_confusion_mats_tmp = RX_confusion_mats_xSessions{mIdx}{iVar}{iRewsize}{iFeature}(include_sessions); 
                        pooled_confusion_mats_tmp = cellfun(@(x) x{iTrialtype},pooled_confusion_mats_tmp,'un',0);  
                        size_inclusion = cellfun(@length,pooled_confusion_mats_tmp) == length(var_bins{iVar})-1; 
                        pooled_confusion_mats_tmp = pooled_confusion_mats_tmp(size_inclusion); 
                        pooled_RX_confusion_mats{mIdx}{iRewsize}{iFeature}{iTrialtype} = sum(cat(3,pooled_confusion_mats_tmp{:}),3); 
                    end
                end
            end     
        end
    end
end

%% Visualize good session mouse-pooled confusionmats

vis_features = 1:5;
for iRewsize = 3
    for iVar = 1
        figure()
        for mIdx = 1:numel(mouse_grps)
            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature); 
                subplot(numel(vis_features),numel(mouse_grps),mIdx + (i_feature-1) * numel(mouse_grps))
                imagesc(flipud(pooled_confusion_mats{mIdx}{iVar}{iRewsize}{iFeature} ./ sum(pooled_confusion_mats{mIdx}{iVar}{iRewsize}{iFeature},1)))
                if i_feature == 1 
                    title(mouse_names(mIdx))
                end 
                if i_feature == numel(vis_features) 
                    xlabel("Predicted Time (sec)")
                    xticks(1:10:(length(var_bins{iVar})-1));
                    xticklabels(var_bins{1}(1:10:end)) 
                else 
                    xticks([])
                end  
                if mIdx == 1
                    ylabel(sprintf("%s \n Predicted Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                    yticks(1:10:(length(var_bins{iVar})-1));
                    yticklabels(var_bins{1}(1:10:end))
                else 
                    yticks([])
                end 
            end
        end 
        suptitle(sprintf("Decoding %s \n (MI Threshold: %.2f nats)",var_names(iVar),MI_threshold))
    end
end 

%% Pooled time since reward separated by reward time / num 
rew_time_bool = true; % separate by reward time ? (else use number)
if rew_time_bool == true
    analyze_rew_ix = analyze_rewTimes; 
else 
    analyze_rew_ix = analyze_rewNums; 
end

close all 
vis_mice = 2:5;
for iRewsize = 2:3
    for iFeature = 4
        figure()
        for m_ix = 1:numel(vis_mice)
            mIdx = vis_mice(m_ix); 
            for i_rew_ix = 1:numel(analyze_rew_ix)
                subplot(numel(analyze_rew_ix),numel(vis_mice),m_ix + (i_rew_ix - 1) * numel(vis_mice))
                if rew_time_bool == true
                    if ~isempty(pooled_rewTime_confusion_mats{mIdx}{iRewsize}{iFeature})
                        imagesc(flipud(pooled_rewTime_confusion_mats{mIdx}{iRewsize}{iFeature}{i_rew_ix} ./ sum(pooled_rewTime_confusion_mats{mIdx}{iRewsize}{iFeature}{i_rew_ix},1)))
                        if m_ix == 1
                            ylabel(sprintf("%s \n True Time",rewTime_names(i_rew_ix)))
                            yticks(1:10:(length(var_bins{timesince_ix})-1));
                            yticklabels(fliplr(var_bins{1}(1:10:end)))
                        else
                            xticks([])
                            yticks([])
                        end
                    end 
                else 
                    if ~isempty(pooled_rewNum_confusion_mats{mIdx}{iRewsize}{iFeature})
                        imagesc(flipud(pooled_rewNum_confusion_mats{mIdx}{iRewsize}{iFeature}{i_rew_ix}))
                        if m_ix == 1
                            ylabel(sprintf("%s \n True Time",rewNum_names(i_rew_ix)))
                            yticks(1:10:(length(var_bins{timesince_ix})-1));
                            yticklabels(fliplr(var_bins{1}(1:10:end)))
                        else
                            xticks([])
                            yticks([])
                        end
                    end 
                end
                if i_rew_ix == 1
                    title(session_titles{mIdx}(i))
                end
                
                if i_rew_ix == numel(analyze_rew_ix)
                    xlabel("Predicted Time")
                    xticks(1:10:(length(var_bins{timesince_ix})-1));
                    xticklabels(var_bins{1}(1:10:end))
                end
            end
        end
        suptitle(sprintf("Time Since %i uL Rew \n Decoded by %s",dataset_opt.rewsizes(iRewsize),dataset_opt.features{mIdx}{i}{iFeature}.name))
    end
end 

%% Pooled time on patch separated by RX 

% vis_mice = 1:5;
for iRewsize = 3
    tt_names = [sprintf("%i0",dataset_opt.rewsizes(iRewsize)) sprintf("%i%i",dataset_opt.rewsizes(iRewsize),dataset_opt.rewsizes(iRewsize))];
    for iFeature = 3
        figure()
        for mIdx = 1:numel(mouse_grps)
            for i_trialtype = 1:numel(analyze_tts)
                subplot(numel(analyze_tts),numel(mouse_grps),mIdx + (i_trialtype - 1) * numel(mouse_grps))
                if ~isempty(pooled_RX_confusion_mats{mIdx}{iRewsize}{iFeature})
                    imagesc(flipud(pooled_RX_confusion_mats{mIdx}{iRewsize}{iFeature}{i_trialtype}./sum(pooled_RX_confusion_mats{mIdx}{iRewsize}{iFeature}{i_trialtype},1)))
                    if mIdx == 1
                        ylabel(sprintf("%s \n True Time (sec)",tt_names(i_trialtype)))
                        yticks(1:10:(length(var_bins{time_on_patch_ix})-1));
                        yticklabels(fliplr(var_bins{1}(1:10:end)))
                    else
                        xticks([])
                        yticks([])
                    end
                end
                if i_rew_ix == numel(analyze_rew_ix)
                    xlabel("Predicted Time (sec)")
                    xticks(1:10:(length(var_bins{time_on_patch_ix})-1));
                    xticklabels(var_bins{1}(1:10:end))
                end
                if i_trialtype == 1
                    title(mouse_names(mIdx))
                end
            end
        end
        suptitle(sprintf("%s Time On %i uL Patch \n Decoded by %s",mouse_names(mIdx),dataset_opt.rewsizes(iRewsize),dataset_opt.features{mIdx}{i}{iFeature}.name))
    end
end

%% Switch to trialed format to make things easier

% X input struct to keep things a bit tidier
X_struct = struct; 
X_struct.X = X; 
X_struct.X_clusters = X_clusters; 
X_struct.X_vel = X_vel; 
X_struct.X_pos = X_pos; 

trial_decoding_features = 1:4; % for now just use the GLM clusters 

y_hat_trials = predict_dataset_trialed(trial_decoding_features,X_struct,models,xval_table,dataset_opt,mouse_names,session_titles);

%% We now have all the data we need to do some nice evaluation! 


