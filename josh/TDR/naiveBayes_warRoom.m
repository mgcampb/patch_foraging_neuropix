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
X_clusters = cell(numel(mouse_grps),1); % one vector of cluster identities per session
for mIdx = 1:5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
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
        for iTrial = 1:length(data.patchCSL)
            fr_mat_trials{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial)); 
        end  
        
        % we could think about just doing a pooling step here
        X{mIdx}{i,1} = fr_mat_trials; 
        X{mIdx}{i,2} = fr_mat_trials; 
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
        timeSinceReward_bins = 0:.05:2; % bins to classify (sec)
        timeOnPatch_bins = 0:.05:2; % bins to classify (sec)
        time2Leave_bins = 0:.05:2; % bins to classify (sec)
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);  
        fr_mat_postRew = X{mIdx}{i,1};
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
                end
            end   
            % handle no extra rewards case
            if numel(rew_sec_cell{iTrial}) == 0 
                time2Leave_binned{iTrial} = fliplr(timeSinceReward_binned{iTrial}); 
            end
            
            % Now use histcounts to actually bin task variables  
            % Set out of bounds (0 label) as NaN to denote missing data
            % rather than having a huge poorly defined class
            [~,~,timeSinceReward_binned{iTrial}] = histcounts(timeSinceReward_binned{iTrial},timeSinceReward_bins);  
            timeSinceReward_binned{iTrial}(timeSinceReward_binned{iTrial} == 0) = NaN; 
            [~,~,timeOnPatch_binned{iTrial}] = histcounts(timeOnPatch_binned{iTrial},timeOnPatch_bins);  
            timeOnPatch_binned{iTrial}(timeOnPatch_binned{iTrial} == 0) = NaN; 
            [~,~,time2Leave_binned{iTrial}] = histcounts(time2Leave_binned{iTrial},time2Leave_bins);  
            time2Leave_binned{iTrial}(time2Leave_binned{iTrial} == 0) = NaN; 
        end   
        
        % add to xval table data
        SessionName = [SessionName ; repmat(session,[nTrials,1])];
        SessionIx = [SessionIx ; repmat(i,[nTrials,1])]; 
        Rewsize = [Rewsize ; rewsize]; 
        FoldID = [FoldID ; nan(nTrials,1)]; 
        
        % add post reward firing rate data
        X{mIdx}{i,3} = fr_mat_postRew;  
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
dataset_opt.clusters = [1 2 3]; % which neuron clusters to include in classification
dataset_opt.rewsizes = [1 2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 5; % xval folds  
dataset_opt.vars = [1 2 3];   
X_dataset = cell(numel(mouse_grps),1);  
y_dataset = cell(numel(mouse_grps),1); 

for mIdx = 1:5 % iterate over mice 
    X_dataset{mIdx} = cell(numel(dataset_opt.vars),1); 
    y_dataset{mIdx} = cell(numel(dataset_opt.vars),1); 
    for iVar = 1:numel(dataset_opt.vars) % iterate over variables
        X_dataset{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1);
        y_dataset{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
        for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions, collect data
            sIdx = mouse_grps{mIdx}(i);
            rewsize = y_rewsize{mIdx}{i};
            neurons_keep = ismember(X_clusters{mIdx}{i},dataset_opt.clusters); % neuron cluster mask
            X_session_cluster = cellfun(@(x) x(neurons_keep,:),X{mIdx}{i,iVar},'UniformOutput',false); % X w/ neurons of interest
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                this_rewsize = dataset_opt.rewsizes(iRewsize);
                trials_keep = rewsize == this_rewsize; % rewsize mask 
                X_dataset{mIdx}{iVar}{iRewsize} = [X_dataset{mIdx}{iVar}{iRewsize};X_session_cluster(trials_keep)]; 
                y_dataset{mIdx}{iVar}{iRewsize} = [y_dataset{mIdx}{iVar}{iRewsize};y{mIdx}{i,iVar}(trials_keep)];
            end
        end
    end
    
    % Now initialize folds in xval table 
    % Evenly distribute sessions between folds as we have previously done
    % for reward size 
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
    models{mIdx} = cell(numel(dataset_opt.vars),1);
    % iterate over the variables we are decoding 
    for iVar = 1:3
        models{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
        % iterate over reward sizes of interest
        for iRewsize = 1:numel(dataset_opt.rewsizes)   
            models{mIdx}{iVar}{iRewsize} = cell(dataset_opt.numFolds,1);  
            this_rewsize = dataset_opt.rewsizes(iRewsize);  
            % xval folds for this mouse and reward size
            foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,4).FoldID; 
            sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx; 
            
            % iterate over xval folds and train models
            for kFold = 1:dataset_opt.numFolds  
                [X_train,X_test,y_train,y_test] = kfold_split(X_dataset{mIdx}{iVar}{iRewsize}, ... 
                                                              y_dataset{mIdx}{iVar}{iRewsize}, ... 
                                                              foldid,kFold,sessionIx);
                % Add some noise s.t. we can avoid zero variance gaussians
                X_train(X_train == 0) = normrnd(0,zero_sigma,[length(find(X_train == 0)),1]);
                models{mIdx}{iVar}{iRewsize}{kFold} = fitcnb(X_train',y_train,'Prior','uniform');
                
                % fitcnb(X_train',y_train,'Prior','uniform');
            end
        end
    end
end


%% Now evaluate classifier performance in test folds 
confusion_mats = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    confusion_mats{mIdx} = cell(numel(dataset_opt.vars),1); 
    for iVar = 1:3 
        confusion_mats{mIdx}{iVar} = cell(numel(dataset_opt.rewsizes),1); 
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            confusion_mats{mIdx}{iVar}{iRewsize} = cell(numel(dataset_opt.numFolds),1);
            this_rewsize = dataset_opt.rewsizes(iRewsize);  
            % xval folds for this mouse and reward size
            foldid = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,4).FoldID; 
            sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx; 
            for kFold = 1:dataset_opt.numFolds
                [X_train,X_test,y_train,y_test] = kfold_split(X_dataset{mIdx}{iVar}{iRewsize}, ... 
                                                              y_dataset{mIdx}{iVar}{iRewsize}, ... 
                                                              foldid,kFold,sessionIx);
                [y_hat,Posterior] = predict(models{mIdx}{iVar}{iRewsize}{kFold},X_test'); 
                confusion_mats{mIdx}{iVar}{iRewsize}{kFold} = confusionmat(y_test,y_hat); 
            end
        end
    end
end
%% Testing  
mIdx = 5; 
iVar = 2;
iRewsize = 1;
this_rewsize = 1;
X_practice = X_dataset{mIdx}{iVar}{iRewsize};  
sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx;
X_padcat = padCat(X_practice,sessionIx);


%% Helper functions 

function [X_train,X_test,y_train,y_test] = kfold_split(X_dataset,y_dataset,foldid,kFold,sessionIx) 
% kfold_split splits a dataset (cell array) into train and test folds 
%   X (features array) is concatenated as blocks by session and padded to allow for missing values
%   y (labels array) is concatenated and returned as a column vector 

    % Concatenate and pad features  
    X_train_cell = X_dataset(foldid ~= kFold);
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
    