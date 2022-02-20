%% A third naive bayes script, cleaned up and adding fourth variable to decode- total reward

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
% paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  
% paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table.mat';
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat';
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
    X{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_accel{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),2);  
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
        glm_clusters_session = session_table.gmm_cluster; 
        
        % we could think about just doing a pooling step here
        X{mIdx}{i,1} = fr_mat_trials; 
        X_vel{mIdx}{i,1} = vel_trials; 
        X_pos{mIdx}{i,1} = pos_trials; 
        X_accel{mIdx}{i,1} = accel_trials; 
        X_clusters{mIdx}{i} = glm_clusters_session;   
        X_peak_pos{mIdx}{i} = [session_table.Rew0_peak_pos session_table.Rew1plus_peak_pos];
        X_cellIDs{mIdx}{i} = good_cells; 
    end
end  

%% Make bins to discretize task variables
var_bins = cell(numel(mouse_grps),1); % allow for different discretizations acr mice
for mIdx = 1:numel(mouse_grps)
    for i_rewsize = [1 2 4] % allow for different discretizations acr reward sizes
        for iVar = 1:3
            % Create task variables and bin according to some discretization
            if mIdx == 3 % longer bins for mouse 78
                if i_rewsize == 1
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:5; % bins to classify timeSinceReward (sec)
                else
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:5; % bins to classify timeOnPatch (sec)
                end 
            elseif mIdx == 4 % shorter bins for mouse 79 
                if i_rewsize == 1 || i_rewsize == 2
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:2; % bins to classify timeSinceReward (sec)
                else
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:3; % bins to classify time2Leave (sec)
                end 
            else 
                if i_rewsize == 1
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:2; % bins to classify timeSinceReward (sec)
                else
                    var_bins{mIdx}{i_rewsize}{iVar} = 0:.05:3; % bins to classify timeOnPatch (sec)
                end
            end
        end 
    end
end

%% Load task variable data and bin according to some discretizations defined above

y = cell(numel(mouse_grps),1); % one cell per reward size   
rewsize = cell(numel(mouse_grps),1);
xval_table = cell(numel(mouse_grps),1);  
for mIdx = 1:5
    y{mIdx} = cell(numel(mouse_grps{mIdx}),4); % 4 variables to decode
    rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per sesion   
    xval_table{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx}(1:end-4); 
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        % Load session information
        nTrials = length(data.patchCSL); 
        i_rewsize = mod(data.patches(:,2),10);  
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
        nTrials = length(i_rewsize); 
        rew_barcode = zeros(length(data.patchCSL) , nTimesteps);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_sec_cell{iTrial} = rew_indices(rew_indices >= 1); 
            % make rew_barcode for time on patch evaluation separation
            % Note we add 1 to rew_indices here because we are now 1 indexing
            rew_barcode(iTrial , (max(rew_indices+1)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices+1) = i_rewsize(iTrial);
        end 
        
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);   
        i_rewNum = cell(nTrials,1); 
        % initialize to the full trial, then pare down according to where last reward occurred
        fr_mat_postRew = X{mIdx}{i,1}; 
        vel_postRew = X_vel{mIdx}{i,1}; 
        accel_postRew = X_accel{mIdx}{i,1}; 
        pos_postRew = X_pos{mIdx}{i,1}; 
        for iTrial = 1:nTrials
            trial_len_ix = prts_ix(iTrial);
            timeSinceReward_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
            timeOnPatch_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000; 
            i_rewNum{iTrial} = zeros(1,trial_len_ix); 
            for r = 1:numel(rew_sec_cell{iTrial})
                rew_ix = (rew_sec_cell{iTrial}(r)) * 1000 / tbin_ms; 
                timeSinceReward_binned{iTrial}(rew_ix:end) =  (1:length(timeSinceReward_binned{iTrial}(rew_ix:end))) * tbin_ms / 1000;
                i_rewNum{iTrial}(rew_ix:end) = r;
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
            
            this_binning = var_bins{mIdx}{i_rewsize(iTrial)}{1};
            % Now use histcounts to actually bin task variables  
            % Set out of bounds (0 label) as NaN to denote missing data
            [~,~,timeSinceReward_binned{iTrial}] = histcounts(timeSinceReward_binned{iTrial},var_bins{mIdx}{i_rewsize(iTrial)}{1});  
            timeSinceReward_binned{iTrial}(timeSinceReward_binned{iTrial} == 0) = NaN; 
            [~,~,timeOnPatch_binned{iTrial}] = histcounts(timeOnPatch_binned{iTrial},var_bins{mIdx}{i_rewsize(iTrial)}{2});  
            timeOnPatch_binned{iTrial}(timeOnPatch_binned{iTrial} == 0) = NaN; 
            [~,~,time2Leave_binned{iTrial}] = histcounts(time2Leave_binned{iTrial},var_bins{mIdx}{i_rewsize(iTrial)}{3});  
            time2Leave_binned{iTrial}(time2Leave_binned{iTrial} == 0) = NaN; 
        end 

        % add post reward firing rate data
        X{mIdx}{i,2} = fr_mat_postRew;  
        X_vel{mIdx}{i,2} = vel_postRew;   
        X_accel{mIdx}{i,2} = accel_postRew;  
        X_pos{mIdx}{i,2} = pos_postRew;   
        % task variables
        y{mIdx}{i,1} = timeSinceReward_binned; 
        y{mIdx}{i,2} = timeOnPatch_binned; 
        y{mIdx}{i,3} = time2Leave_binned;   
        y{mIdx}{i,4} = i_rewNum;
        rewsize{mIdx}{i} = i_rewsize; 
        
        % xval table variables
        SessionName = repmat(session,[nTrials,1]);
        SessionIx = repmat(i,[nTrials,1]);
        Rewsize = i_rewsize;
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
        dataset_opt.features{mIdx}{i_session} = gen_std_features(X_clusters{mIdx}{i_session});
    end 
end

% other options
dataset_opt.rewsizes = [1 2 4]; % which reward size trials will we train to

% from training
% dataset_opt.numFolds = 5; % number of xval folds  
% dataset_opt.vars = [1 2 3 4];    

% for making supp fig
dataset_opt.numFolds = 5; % number of xval folds  
dataset_opt.vars = 1;    

[X_dataset,y_dataset,xval_table] = gen_nb_dataset3(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,rewsize,xval_table,dataset_opt);

%% Train and evaluate population classifiers 
% Now use classification datasets to train classifiers, holding out test folds
dataset_opt.distribution = 'normal';
models = fit_nb_dataset3(X_dataset,y_dataset,xval_table,dataset_opt,mouse_names);

% Now evaluate classifier performance on test folds 
% [y_true_full,y_hat_full] = predict_dataset_singleSession(X_dataset,y_dataset,models,xval_table,dataset_opt,mouse_names);

%% Generate predictions at trialed level

% X input struct to keep things a bit tidier
X_struct = struct; 
X_struct.X = X; 
X_struct.X_clusters = X_clusters; 
X_struct.X_vel = X_vel; 
X_struct.X_pos = X_pos; 

trial_decoding_features = 1:7; % which features to generate trialed predictions for

y_hat_trials = predict_dataset_trialed(trial_decoding_features,X_struct,models,xval_table,dataset_opt,mouse_names,session_titles);

%% Now package results up so that we have everything we need to analyze
nb_results = struct;
nb_results.y_true = y; 
nb_results.y_hat = y_hat_trials; 
nb_results.var_bins = var_bins; 
nb_results.dataset_opt = dataset_opt; 
this_date = strrep(date,'-','_');
save_path = ['./structs/nb_results' this_date '.mat']; 
save(save_path,'nb_results');

