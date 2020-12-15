%% Decode events and choice from glm sig cells and potentially driscoll selection neurons
%  What does timecourse of reward delivery decoding look like? 
%  Do axes of reward decoding change over timecourse decoded?  
%  What cell groups can we decode reward events from?   

% Ok folks, this is our last idea: Naive bayes classification 
% It's a classification problem, but it's no longer imbalanced.
% ** Trained independently between reward sizes (we can check cross performance later) ** 
% The categories are binned (100 msec bins?): 
%   1. Time since reward 
%   2. Time since patch stop (Kinda tricky)
%   3. Time until patch leave (Only use after last reward) (Cluster 3?)
% X: FR of glm-significant cells (try driscoll selection later) 
%    Pool X across days for mice- insert blocks of NaN values to fill where other days
%    add their data 
% y: timebin of variable of interest (separated between rewards)  
%    Pool y across days for mice- just a matter of concatenating 

% Measure performance in terms of error in timebin decoding (xval)
% See what information and where we lose by dropping different glm cluster neurons

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
X = cell(numel(mouse_grps),1); % one per task variable
X_vel = cell(numel(mouse_grps),1); % one per task variable
X_pos = cell(numel(mouse_grps),1); % one per task variable
X_clusters = cell(numel(mouse_grps),1); % one vector of cluster identities per session
for mIdx = 1:5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_clusters{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
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
        X{mIdx}{i,2} = fr_mat_trials;  
        X_vel{mIdx}{i,1} = vel_trials; 
        X_vel{mIdx}{i,2} = vel_trials;  
        X_pos{mIdx}{i,1} = pos_trials; 
        X_pos{mIdx}{i,2} = pos_trials; 
        X_clusters{mIdx}{i} = sig_clusters_session; 
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
    SessionName = []; 
    SessionIx = []; 
    Rewsize = []; 
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
            rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        end 
        
        % Create task variables and bin according to some discretization  
        var_bins{1} = 0:.05:3; % bins to classify timeSinceReward (sec) 
        var_bins{2} = 0:.05:3; % bins to classify timeOnPatch (sec) 
        var_bins{3} = 0:.05:3; % bins to classify time2Leave (sec) 
        
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);  
        fr_mat_postRew = X{mIdx}{i,1}; 
        vel_postRew = X_vel{mIdx}{i,1}; 
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
        
        % add to xval table data
        SessionName = [SessionName ; repmat(session,[nTrials,1])];
        SessionIx = [SessionIx ; repmat(i,[nTrials,1])]; 
        Rewsize = [Rewsize ; rewsize]; 
        FoldID = [FoldID ; nan(nTrials,1)]; 
        
        % add post reward firing rate data
        X{mIdx}{i,3} = fr_mat_postRew;  
        X_vel{mIdx}{i,3} = vel_postRew;  
        X_pos{mIdx}{i,3} = pos_postRew;  
        % task variables
        y{mIdx}{i,1} = timeSinceReward_binned; 
        y{mIdx}{i,2} = timeOnPatch_binned; 
        y{mIdx}{i,3} = time2Leave_binned;   
        y_rewsize{mIdx}{i} = rewsize; 
    end 
    xval_table{mIdx} = table(SessionName,SessionIx,Rewsize,FoldID); 
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

% add shuffle control (could loop to do more than 1 shuffle) 
dataset_opt.features{5} = struct;
dataset_opt.features{5}.type = "KMeans Clusters";
dataset_opt.features{5}.ix = [1 2 3]; % indices within the feature type we selected
dataset_opt.features{5}.shuffle = true; % shuffle?
dataset_opt.features{5}.name = "Shuffled Neural Data"; % name for visualizations
% add velocity control
dataset_opt.features{6} = struct;
dataset_opt.features{6}.type = "Velocity";
dataset_opt.features{6}.shuffle = false; % shuffle?
dataset_opt.features{6}.name = "Velocity"; % name for visualizations
% add position control
dataset_opt.features{7} = struct;
dataset_opt.features{7}.type = "Position";
dataset_opt.features{7}.shuffle = false; % shuffle?
dataset_opt.features{7}.name = "Position"; % name for visualizations

% other options
dataset_opt.rewsizes = [2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 5; % xval folds  
dataset_opt.vars = [1 2 3];     
X_dataset = cell(numel(mouse_grps),1);  
y_dataset = cell(numel(mouse_grps),1); 

for mIdx = 1:5 % iterate over mice
    X_dataset{mIdx} = cell(numel(dataset_opt.features),1);
    y_dataset{mIdx} = cell(numel(dataset_opt.features),1); 
    for iFeature = 1:numel(dataset_opt.features)  
        X_dataset{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        y_dataset{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        for iVar = 1:numel(dataset_opt.vars) % iterate over variables
            X_dataset{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            y_dataset{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions, collect data
                sIdx = mouse_grps{mIdx}(i); 
                rewsize = y_rewsize{mIdx}{i};   
                nTrials = length(rewsize); 
                
                % Pull out feature
                if strcmp(dataset_opt.features{iFeature}.type,"KMeans Clusters")
                    neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.features{iFeature}.ix); % neuron cluster mask
                    X_session_feature = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest 
                elseif strcmp(dataset_opt.features{iFeature}.type,"Velocity") 
                    X_session_feature = X_vel{mIdx}{i,iVar};
                elseif strcmp(dataset_opt.features{iFeature}.type,"Position") 
                    X_session_feature = X_pos{mIdx}{i,iVar};
                end 
                
                % Shuffle data?
                if dataset_opt.features{iFeature}.shuffle == true   
                    % save indexing so that we can throw this back in trialed form
                    t_lens = cellfun(@(x) size(x,2),X_session_feature); 
                    leave_ix = cumsum(t_lens);
                    stop_ix = leave_ix - t_lens + 1;   
                    % concatenate, then circshift neurons independently
                    X_session_feature_cat = cat(2,X_session_feature{:});  
                    shifts = randi(size(X_session_feature_cat,2),1,size(X_session_feature_cat,1));
                    X_session_feature_cat_shuffle = cell2mat(arrayfun(@(x) circshift(X_session_feature_cat(x,:),[1 shifts(x)]),(1:numel(shifts))','un',0));
                    X_session_feature = arrayfun(@(x) X_session_feature_cat_shuffle(:,stop_ix(x):leave_ix(x)),(1:nTrials)','un',0);
                end
                
                for iRewsize = 1:numel(dataset_opt.rewsizes)
                    this_rewsize = dataset_opt.rewsizes(iRewsize);
                    trials_keep = rewsize == this_rewsize; % rewsize mask
                    X_dataset{mIdx}{iFeature}{iVar}{iRewsize} = [X_dataset{mIdx}{iFeature}{iVar}{iRewsize};X_session_feature(trials_keep)];
                    y_dataset{mIdx}{iFeature}{iVar}{iRewsize} = [y_dataset{mIdx}{iFeature}{iVar}{iRewsize};y{mIdx}{i,iVar}(trials_keep)];
                end
            end
        end
    end

    % Make xval folds, evenly distributing sessions between folds
    for iRewsize = 1:numel(dataset_opt.rewsizes)
        this_rewsize = dataset_opt.rewsizes(iRewsize);
        xval_table_thisRewsize = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:);
        iRewsize_foldid = nan(size(xval_table_thisRewsize,1),1);
        shift_by = 0; % change which fold is the "last fold" to make sure one fold is not way smaller than the rest
        for i = 1:numel(mouse_grps{mIdx}) % evenly distribute trials from this session between folds
            keep_this = xval_table_thisRewsize.SessionIx == i; % keep trials from this session
            i_nTrials = sum(keep_this); % to ensure proper assignment indexing
            iRewsize_foldid_this = repmat(circshift(1:dataset_opt.numFolds,shift_by),1,ceil(i_nTrials/dataset_opt.numFolds)*dataset_opt.numFolds);
            iRewsize_foldid(keep_this) = iRewsize_foldid_this(1:i_nTrials); % assign folds 1:k
            shift_by = shift_by - mod(i_nTrials,dataset_opt.numFolds); % shift which fold is getting fewer trials
        end
        % assign folds among trials of this reward size
        xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,4).FoldID = iRewsize_foldid;
    end
end

%% Now use classification datasets to train classifiers, holding out test folds

models = cell(numel(mouse_grps),1);
zero_sigma = 0.5; 
for mIdx = 1:5
    models{mIdx} = cell(numel(dataset_opt.features),1);
    % iterate over the variables we are decoding
    for iFeature = 1:numel(dataset_opt.features)
        models{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        for iVar = 1:3
            models{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            % iterate over reward sizes of interest
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                models{mIdx}{iFeature}{iVar}{iRewsize} = cell(dataset_opt.numFolds,1);
                this_rewsize = dataset_opt.rewsizes(iRewsize);
                % xval folds for this mouse and reward size
                foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).FoldID;
                sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx; 
                % iterate over xval folds and train models
                for kFold = 1:dataset_opt.numFolds 
                    [X_train,~,y_train,~] = kfold_split(X_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                        y_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                        foldid,kFold,sessionIx);  
                    % Add some noise s.t. we can avoid zero variance gaussians
                    X_train(X_train == 0) = normrnd(0,zero_sigma,[length(find(X_train == 0)),1]);
                    models{mIdx}{iFeature}{iVar}{iRewsize}{kFold} = fitcnb(X_train',y_train,'Prior','uniform');
                end
            end
        end
    end
    fprintf("%s Model Fitting Complete \n",mouse_names(mIdx))
end

%% Now evaluate classifier performance on test folds 
confusion_mats = cell(numel(mouse_grps),1);
generative_models = cell(numel(mouse_grps),1);
y_true_full = cell(numel(mouse_grps),1);
y_hat_full = cell(numel(mouse_grps),1);
for mIdx = 1:5
    confusion_mats{mIdx}{iFeature} = cell(numel(dataset_opt.features),1);
    generative_models{mIdx}{iFeature} = cell(numel(dataset_opt.features),1);
    y_true_full{mIdx}{iFeature} = cell(numel(dataset_opt.features),1);
    y_hat_full{mIdx}{iFeature} = cell(numel(dataset_opt.features),1);
    for iFeature = 1:numel(dataset_opt.features)
        confusion_mats{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        generative_models{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        y_true_full{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        y_hat_full{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        for iVar = 1:3
            confusion_mats{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            generative_models{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            y_true_full{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            y_hat_full{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                confusion_mats_tmp = cell(numel(dataset_opt.numFolds),1);
                generative_models_tmp = cell(numel(dataset_opt.numFolds),1);
                y_true_full_tmp = cell(numel(dataset_opt.numFolds),1);
                y_hat_full_tmp = cell(numel(dataset_opt.numFolds),1);
                this_rewsize = dataset_opt.rewsizes(iRewsize);
                % xval folds for this mouse and reward size
                foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,4).FoldID;
                sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx;
                for kFold = 1:dataset_opt.numFolds
                    [~,X_test,~,y_test] = kfold_split(X_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                      y_dataset{mIdx}{iFeature}{iVar}{iRewsize}, ...
                                                      foldid,kFold,sessionIx);
                    [y_hat,Posterior] = predict(models{mIdx}{iFeature}{iVar}{iRewsize}{kFold},X_test');
                    confusion_mats_tmp{kFold} = confusionmat(y_test,y_hat);
                    generative_models_tmp{kFold} = rot90(cellfun(@(x) x(1),models{mIdx}{iFeature}{iVar}{iRewsize}{kFold}.DistributionParameters));
                    y_true_full_tmp{kFold} = y_test;
                    y_hat_full_tmp{kFold} = y_hat;
                end
                confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} = sum(cat(3,confusion_mats_tmp{:}),3);
                generative_models{mIdx}{iFeature}{iVar}{iRewsize} = mean(cat(3,generative_models_tmp{:}),3);
                y_true_full{mIdx}{iFeature}{iVar}{iRewsize} = cat(1,y_true_full_tmp{:});
                y_hat_full{mIdx}{iFeature}{iVar}{iRewsize} = cat(1,y_hat_full_tmp{:});
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
        error_hmap{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        yhat_mean_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        yhat_sem_withinRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        absError_mean{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        absError_sem{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        learned_representations{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        for iVar = 1:3
            error_hmap{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            yhat_mean_withinRewsize{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            yhat_sem_withinRewsize{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            absError_mean{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            absError_sem{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            error_bins = -max(var_bins{iVar}):error_bin_width:max(var_bins{iVar});
            msec_conversion = diff(var_bins{iVar}(1:2));
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                i_y_true = y_true_full{mIdx}{iFeature}{iVar}{iRewsize};
                i_y_true_msec = msec_conversion * i_y_true;
                i_y_hat_msec = msec_conversion * y_hat_full{mIdx}{iFeature}{iVar}{iRewsize};
                errors_msec = i_y_hat_msec - i_y_true_msec; % so that positive errors are late predictions
                error_hmap{mIdx}{iFeature}{iVar}{iRewsize} = nan(length(error_bins)-1,max(i_y_true)); 
                yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                yhat_sem_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                absError_mean{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                absError_sem{mIdx}{iFeature}{iVar}{iRewsize} = nan(max(i_y_true),1);
                for true_label = 1:max(i_y_true) 
                    error_hmap{mIdx}{iFeature}{iVar}{iRewsize}(:,true_label) = histcounts(errors_msec(i_y_true == true_label),error_bins,'Normalization', 'probability');
                    yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = mean(i_y_hat_msec(i_y_true == true_label));
                    yhat_sem_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = std(i_y_hat_msec(i_y_true == true_label)) / sqrt(length(find(i_y_true == true_label)));
                    absError_mean{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = mean(abs(errors_msec(i_y_true == true_label)));
                    absError_sem{mIdx}{iFeature}{iVar}{iRewsize}(true_label) = std(abs(errors_msec(i_y_true == true_label))) / sqrt(length(find(i_y_true == true_label)));
                end
            end
        end
    end
end

%% Visualize encoding representation 
close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
for iRewsize = 2
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 1:3
        figure()
        for mIdx = 1:numel(mouse_grps)
            for iVar = 1:numel(dataset_opt.vars)
                subplot(3,5,mIdx + 5 * (iVar - 1))  
                if iVar == 1
                    [~,max_ix] = max(generative_models{mIdx}{iFeature}{iVar}{iRewsize},[],2);
                    [~,peaksort] = sort(max_ix);
                end
                if iVar ~= 3
                    imagesc(flipud(zscore(generative_models{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:),[],2)))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(var_bins{iVar}(1:10:end))
                else
                    imagesc(flipud(zscore(fliplr(generative_models{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:)),[],2)))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(-fliplr(var_bins{iVar}(1:10:end)) )
                end
                
                if mIdx == 1
                    ylabel("Neurons")
                end
                if iVar == 1
                    title(mouse_names(mIdx))
                end
                xlabel(sprintf("%s (sec)",var_names(iVar)))
            end
        end
        suptitle(sprintf("%s, %i uL Trials",dataset_opt.features{iFeature}.name,this_rewsize))
    end
end

%% Visualize decoding performance with heatmap
close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
vis_mice = 1:5;
for iRewsize = 1
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 4
        figure()
        for m = 1:numel(vis_mice)
            mIdx = vis_mice(m);
            for iVar = 1:numel(dataset_opt.vars)
                subplot(3,numel(vis_mice),m + numel(vis_mice) * (iVar - 1))
                
                if iVar ~= 3
                    imagesc(flipud(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} ./ sum(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize},2)))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(var_bins{iVar}(1:10:end))
                    yticks(1:10:(numel(var_bins{iVar})))
                    yticklabels(fliplr(var_bins{iVar}(1:10:end)))
                else
                    imagesc(flipud(rot90(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} ./ sum(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize},2),2)))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(-fliplr(var_bins{iVar}(1:10:end)) )
                    yticks(1:10:(numel(var_bins{iVar})))
                    yticklabels(-var_bins{iVar}(1:10:end))
                end
                
                if m == 1
                    ylabel(sprintf("True %s",var_names(iVar)))
                end
                if iVar == 1
                    title(mouse_names(mIdx))
                end

                xlabel(sprintf("Decoded %s",var_names(iVar)))
            end
        end
        suptitle(sprintf("%s, %i uL Trials",dataset_opt.features{iFeature}.name,this_rewsize))
    end
end

%% Visualize decoding performance betw reward sizes with line plot of abs error
colors = cool(3);
close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"];
this_rewsize = dataset_opt.rewsizes(iRewsize);
for iFeature = 4
    figure()
    for mIdx = 1:numel(mouse_grps)
        for iVar = 1:numel(dataset_opt.vars)
            subplot(3,5,mIdx + 5 * (iVar - 1))
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                if iVar ~= 3
                    shadedErrorBar(var_bins{iVar}(1:end-1),absError_mean{mIdx}{iFeature}{iVar}{iRewsize},absError_sem{mIdx}{iFeature}{iVar}{iRewsize},'lineprops',{'Color',colors(iRewsize,:)});
                else
                    shadedErrorBar(-fliplr(var_bins{iVar}(1:end-1)),flipud(absError_mean{mIdx}{iFeature}{iVar}{iRewsize}),flipud(absError_sem{mIdx}{iFeature}{iVar}{iRewsize}),'lineprops',{'Color',colors(iRewsize,:)});
                end
            end
            ylim([0,1])
            
            if mIdx == 1
                ylabel(sprintf("%s |Decoding Error|",var_names(iVar)))
            end
            if iVar == 1
                title(mouse_names(mIdx))
            end
            xlabel(sprintf("%s",var_names(iVar)))
        end
    end
    suptitle(sprintf("%s Decoding Timecourse",dataset_opt.features{iFeature}.name))
end 

%% Visualize decoding performance betw features in dataset 
colors = [lines(3) ; 0 .6 .2; 0 0 0; .4 .4 .4 ; .2 .2 .8];
% close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
vis_features = [4 6 7]; 
vis_mice = 1:5;

for iRewsize = 2
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    figure()
    for m = 1:numel(vis_mice) % numel(mouse_grps) 
        mIdx = vis_mice(m);
        for iVar = 1:numel(dataset_opt.vars)
            subplot(3,numel(vis_mice),m + numel(vis_mice) * (iVar - 1))
            for ix_Feature = 1:numel(vis_features) % numel(dataset_opt.features) 
                iFeature = vis_features(ix_Feature);
                if iVar ~= 3
                    shadedErrorBar(var_bins{iVar}(1:end-1),absError_mean{mIdx}{iFeature}{iVar}{iRewsize},absError_sem{mIdx}{iFeature}{iVar}{iRewsize},'lineprops',{'Color',colors(iFeature,:)});
                else
                    shadedErrorBar(-fliplr(var_bins{iVar}(1:end-1)),flipud(absError_mean{mIdx}{iFeature}{iVar}{iRewsize}),flipud(absError_sem{mIdx}{iFeature}{iVar}{iRewsize}),'lineprops',{'Color',colors(iFeature,:)});
                end
            end
            ylim([0,1.5])
            
            if m == 1
                ylabel(sprintf("%s |Decoding Error|",var_names(iVar)))
            end
            if iVar == 1
                title(mouse_names(mIdx))
            end
            xlabel(sprintf("%s",var_names(iVar)))
            if m == 1 && iVar == 1
                legend(cellfun(@(x) x.name,dataset_opt.features(vis_features)))
            end
        end
        
    end
    suptitle(sprintf("%i uL Trial Decoding Timecourse",this_rewsize))
end 

%% Cross reward size decoding 
%  What do decoders trained on diff rewsizes have to say about other
%  rewsize data 
    
confusion_mats_xRewsize = cell(numel(mouse_grps),1); 
yhat_mean = cell(numel(mouse_grps),1); 
yhat_sem = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    confusion_mats_xRewsize{mIdx} = cell(numel(dataset_opt.features),1);  
    yhat_mean{mIdx} = cell(numel(dataset_opt.features),1);  
    yhat_sem{mIdx} = cell(numel(dataset_opt.features),1);  
    for iFeature = 1:numel(dataset_opt.features)
        confusion_mats_xRewsize{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        yhat_mean{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        yhat_sem{mIdx}{iFeature} = cell(numel(dataset_opt.vars),1);
        for iVar = 1:3 
            msec_conversion = diff(var_bins{iVar}(1:2));
            confusion_mats_xRewsize{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            yhat_mean{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            yhat_sem{mIdx}{iFeature}{iVar} = cell(numel(dataset_opt.rewsizes),1);
            for i_heldout_rewsize = 1:numel(dataset_opt.rewsizes)     
                this_rewsize = dataset_opt.rewsizes(i_heldout_rewsize); 
                sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx; 
                X_heldout_full = padCat(X_dataset{mIdx}{iFeature}{iVar}{i_heldout_rewsize},sessionIx);
                y_heldout_full = cat(2,y_dataset{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{:});  
                
                % now iterate over the other reward sizes
                trained_rewsizes = setdiff(1:numel(dataset_opt.rewsizes),i_heldout_rewsize); 
                for i_trained_rewsize = 1:numel(trained_rewsizes) 
                    trained_rewsize = trained_rewsizes(i_trained_rewsize);  
                    confusion_mats_xRewsize_tmp = cell(numel(dataset_opt.numFolds),1); 
                    y_hat_tmp = cell(numel(dataset_opt.numFolds),1); 
                    for kFold = 1:dataset_opt.numFolds
                        this_model = models{mIdx}{iFeature}{iVar}{trained_rewsize}{kFold};
                        y_hat_tmp{kFold} = predict(this_model,X_heldout_full');
                        confusion_mats_xRewsize_tmp{kFold} = confusionmat(y_heldout_full,y_hat_tmp{kFold});
                    end 
                    confusion_mats_xRewsize{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{trained_rewsize} = sum(cat(3,confusion_mats_xRewsize_tmp{:}),3); 
                    y_hat = msec_conversion * mean(cat(2,y_hat_tmp{:}),2);
                    
                    for true_label = 1:max(y_heldout_full)
                        yhat_mean{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{trained_rewsize}(true_label) = mean(y_hat(y_heldout_full == true_label));
                        yhat_sem{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{trained_rewsize}(true_label) = std(y_hat(y_heldout_full == true_label)) / sqrt(length(find(y_heldout_full == true_label)));
                    end
                end
                
                yhat_mean{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{i_heldout_rewsize} = yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{i_heldout_rewsize};
                yhat_sem{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{i_heldout_rewsize} = yhat_sem_withinRewsize{mIdx}{iFeature}{iVar}{i_heldout_rewsize};
            end
        end
    end
    fprintf("%s Cross-Rewsize Analysis Complete \n",mouse_names(mIdx))
end 

%% First visualize cross reward size using confusionmat heatmap  

for mIdx = 1:5
    for iFeature = 4
        for iVar = 2
            figure() 
            for i_heldout_rewsize = 1:3   
                for i_trained_rewsize = 1:numel(dataset_opt.rewsizes)
                    subplot(3,3,i_trained_rewsize + 3 * (i_heldout_rewsize - 1))  
                    if i_heldout_rewsize ~= i_trained_rewsize  
                        imagesc(flipud(confusion_mats_xRewsize{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{i_trained_rewsize})) 
                    else 
                        % trained on this rewsize
                        imagesc(flipud(confusion_mats{mIdx}{iFeature}{iVar}{i_heldout_rewsize})) 
                    end 
                    if iVar ~= 3 
                        xticks(1:10:(numel(var_bins{iVar})))
                        xticklabels(var_bins{iVar}(1:10:end))
                        yticks(1:10:(numel(var_bins{iVar})))
                        yticklabels(fliplr(var_bins{iVar}(1:10:end)))
                    else
                        xticks(1:10:(numel(var_bins{iVar})))
                        xticklabels(-fliplr(var_bins{iVar}(1:10:end)) )
                        yticks(1:10:(numel(var_bins{iVar})))
                        yticklabels(-var_bins{iVar}(1:10:end))
                    end 
                    if i_trained_rewsize == 1 
                        ylabel(sprintf("True %i uL %s",dataset_opt.rewsizes(i_heldout_rewsize),var_names(iVar)))
                    end 
                    if i_heldout_rewsize == 3
                        xlabel(sprintf("%i uL Model Decoded %s",dataset_opt.rewsizes(i_trained_rewsize),var_names(iVar)))
                    end
                end
            end
        end 
        suptitle(sprintf("%s %s Cross-Rewsize Decoding of %s",dataset_opt.features{iFeature}.name,mouse_names(mIdx),var_names(iVar)))
    end 
end  

%% Now visualize decoding as line plot 
colors = cool(3); 
close all
for iFeature = 1:4
    figure()
    for iVar = 1
        for mIdx = 1:5
            for heldout_rewsize = 1:numel(dataset_opt.rewsizes) 
                subplot(3,5,mIdx + 5 * (heldout_rewsize - 1));hold on
                for trained_rewsize = 1:numel(dataset_opt.rewsizes)
                    if iVar ~= 3
                        shadedErrorBar(var_bins{iVar}(1:end-1),yhat_mean{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize},flipud(yhat_sem{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize}),'lineprops',{'Color',colors(trained_rewsize,:)});
                    else 
                        if heldout_rewsize == trained_rewsize
                            shadedErrorBar(-fliplr(var_bins{iVar}(1:end-1)),flipud(yhat_mean{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize}),flipud(yhat_sem{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize}),'lineprops',{'Color',colors(trained_rewsize,:)});
                        else 
                            shadedErrorBar(-fliplr(var_bins{iVar}(1:end-1)),fliplr(yhat_mean{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize}),fliplr(yhat_sem{mIdx}{iFeature}{iVar}{heldout_rewsize}{trained_rewsize}),'lineprops',{'Color',colors(trained_rewsize,:)});
                        end
                    end 
                    % axis labels
                    if mIdx == 1 && trained_rewsize == 1
                        ylabel(sprintf("Decoded %i uL %s",dataset_opt.rewsizes(heldout_rewsize),var_names(iVar)))
                    end
                end
                if iVar ~= 3
                    plot([0 max(var_bins{iVar}(1:end-1))],[0 max(var_bins{iVar}(1:end-1))],'k--','linewidth',1.5)  
                else 
                    plot([0 -max(var_bins{iVar}(1:end-1))],[0 max(var_bins{iVar}(1:end-1))],'k--','linewidth',1.5)  
                end   
                
                if heldout_rewsize == 1
                    title(mouse_names(mIdx))
                end 
                
                if heldout_rewsize == 3 
                    xlabel(sprintf("True %s",var_names(iVar)))
                end
            end
            if mIdx == 1
                legend(["1 uL Trained Model" ...
                        "2 uL Trained Model", ...
                        "4 uL Trained Model", ...
                        "Perfect Prediction"])
            end
        end
    end
    suptitle(sprintf("%s Cross-Rewsize Decoding of %s",dataset_opt.features{iFeature}.name,var_names(iVar)))
end

%% Testing  
mIdx = 5; 
iVar = 2;
iRewsize = 1;
this_rewsize = 1;
X_practice = X_dataset{mIdx}{iVar}{iRewsize};  
sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx;
X_padcat = padCat(X_practice,sessionIx); 

% get the generative model tracking learned representation across time 
kFold = 1;
rep = rot90(cellfun(@(x) x(1),models{mIdx}{iVar}{iRewsize}{1}.DistributionParameters));


%% Helper functions 

function [X_train,X_test,y_train,y_test] = kfold_split(X_dataset,y_dataset,foldid,kFold,sessionIx) 
% kfold_split splits a dataset (cell array) into train and test folds 
%   X (features array) is concatenated as blocks by session and padded to allow for missing values
%   y (labels array) is concatenated and returned as a column vector 

    % Concatenate and pad features  
    X_train = padCat(X_dataset(foldid ~= kFold),sessionIx(foldid ~= kFold)); 
    X_test = padCat(X_dataset(foldid == kFold),sessionIx(foldid == kFold));    
    
    % Concatenate labels
    y_train_cell = y_dataset(foldid ~= kFold);
    y_train = cat(2,y_train_cell{:})';
    y_test_cell = y_dataset(foldid == kFold);
    y_test = cat(2,y_test_cell{:})';
    
end

function X_full = padCat(X_cell,sessionIx)
% Concatenate neural data (from some folds) across sessions with NaN padding 
% Requires trials to be sorted by session
    t_lens = cellfun(@(x) size(x,2),X_cell); % sum trial lengths across sessions
%     [s_nNeurons,~,session_labels] = unique(cellfun(@(x) size(x,1),X_cell)); 
    % get the session lengths
    session_lengths = arrayfun(@(x) sum(cellfun(@(y) size(y,2),X_cell(sessionIx == x))),unique(sessionIx)); 
    session_ix_starts = [0 cumsum(session_lengths)']; % w/ the lengths, get starting index so we can drop in data at proper rows
    session_trial_starts = [cell2mat(arrayfun(@(x) find(sessionIx==x,1),unique(sessionIx),'un',false))',length(sessionIx)+1];   
    s_nNeurons = arrayfun(@(x) size(X_cell{x},1),session_trial_starts(1:end-1)); 
    session_neuron_starts = [0 cumsum(s_nNeurons)]; 
    X_full = nan(sum(s_nNeurons),sum(t_lens)); 
    % iterate over sessions, and fill in values where needed
    for s_ix = 1:(numel(session_ix_starts)-1) 
        X_session_cell = X_cell(session_trial_starts(s_ix):session_trial_starts(s_ix+1)-1); 
        X_session = cat(2,X_session_cell{:});
        X_full(session_neuron_starts(s_ix)+1:session_neuron_starts(s_ix+1),session_ix_starts(s_ix)+1:session_ix_starts(s_ix+1)) = X_session;
    end
end
    