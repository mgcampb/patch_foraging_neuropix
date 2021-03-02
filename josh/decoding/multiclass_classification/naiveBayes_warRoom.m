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
X_accel = cell(numel(mouse_grps),1); % one per task variable
X_pos = cell(numel(mouse_grps),1); % one per task variable
X_clusters = cell(numel(mouse_grps),1); % one vector of cluster identities per session
X_cellIDs = cell(numel(mouse_grps),1); % one vector of cellIDs
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
    SessionName = []; 
    SessionIx = []; 
    Rewsize = []; 
    FoldID = [];  
    TrialNum = []; 
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
        
        % add to xval table data
        SessionName = [SessionName ; repmat(session,[nTrials,1])];
        SessionIx = [SessionIx ; repmat(i,[nTrials,1])]; 
        Rewsize = [Rewsize ; rewsize]; 
        FoldID = [FoldID ; nan(nTrials,1)];  
        TrialNum = [TrialNum ; (1:nTrials)'];
        
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
    end 
    xval_table{mIdx} = table(SessionName,SessionIx,TrialNum,Rewsize,FoldID); 
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

% % add velocity control
% dataset_opt.features{6} = struct;
% dataset_opt.features{6}.type = "Velocity";
% dataset_opt.features{6}.shuffle = false; % shuffle?
% dataset_opt.features{6}.name = "Velocity"; % name for visualizations
% % add position control
% dataset_opt.features{7} = struct;
% dataset_opt.features{7}.type = "Position";
% dataset_opt.features{7}.shuffle = false; % shuffle?
% dataset_opt.features{7}.name = "Position"; % name for visualizations
% % add acceleration control
% dataset_opt.features{8} = struct;
% dataset_opt.features{8}.type = "Acceleration";
% dataset_opt.features{8}.shuffle = false; % shuffle?
% dataset_opt.features{8}.name = "Acceleration"; % name for visualizations

% other options
dataset_opt.rewsizes = [1 2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 5; % xval folds  
dataset_opt.vars = [1 2 3];   

[X_dataset,y_dataset,xval_table] = gen_multiclassDataset(X,X_vel,X_pos,X_accel,X_clusters,X_cellIDs,... 
                                                         y,y_rewsize,... 
                                                         xval_table,mouse_grps,dataset_opt);

%% Now use classification datasets to train classifiers, holding out test folds

models = fit_dataset(X_dataset,y_dataset,xval_table,mouse_names,dataset_opt);

%% Now evaluate classifier performance on test folds 

[confusion_mats,cond_means,cond_sds,y_true_full,y_hat_full] = predict_dataset(X_dataset,y_dataset,models,xval_table,mouse_names,dataset_opt);

%% Analyze decoding performance in terms of time prediction 

msec_conversion = diff(var_bins{1}(1:2));
accuracy_tolerance = round(.25 / msec_conversion);

[yhat_mean_withinRewsize,yhat_sem_withinRewsize,absError_mean,absError_sem,accuracy] = eval_dataset_predictions(y_true_full,y_hat_full,accuracy_tolerance,dataset_opt);

%% Shuffle control

% add shuffle control (could loop to do more than 1 shuffle) 
shuffle_dataset_opt.features{1} = struct;
shuffle_dataset_opt.features{1}.type = "KMeans Clusters";
shuffle_dataset_opt.features{1}.ix = [1 2 3]; % indices within the feature type we selected
shuffle_dataset_opt.features{1}.shuffle = true; % shuffle?
shuffle_dataset_opt.features{1}.name = "Shuffled Neural Data"; % name for visualizations 
% other options
shuffle_dataset_opt.rewsizes = dataset_opt.rewsizes; % which reward size trials will we train to
shuffle_dataset_opt.numFolds = dataset_opt.numFolds; % xval folds  
shuffle_dataset_opt.vars = dataset_opt.vars;   
shuffle_dataset_opt.suppressOutput = true; 

msec_conversion = diff(var_bins{1}(1:2));
accuracy_tolerance = round(.25 / msec_conversion); 

nShuffles = 10;
accuracies_shuffle = cell(nShuffles,1);
absError_means_shuffle = cell(nShuffles,1);

f = waitbar(0,'Creating shuffled decoding performance distribution');

for iShuffle = 1:nShuffles 
    f2 = waitbar(0,'Creating shuffled dataset');
    [X_dataset_shuffle,y_dataset_shuffle,xval_table_shuffle] = gen_multiclassDataset(X,X_vel,X_pos,X_accel,X_clusters,X_cellIDs,... 
                                                             y,y_rewsize,... 
                                                             xval_table,mouse_grps,shuffle_dataset_opt);
    waitbar(.25,f2,'Fitting models');
    models_shuffle = fit_dataset(X_dataset_shuffle,y_dataset_shuffle,xval_table_shuffle,mouse_names,shuffle_dataset_opt);     
    waitbar(.50,f2,'Evaluating models on heldout data');
    [~,~,~,y_true_full,y_hat_full] = predict_dataset(X_dataset_shuffle,y_dataset_shuffle,models_shuffle,xval_table_shuffle,mouse_names,shuffle_dataset_opt);
    waitbar(.75,f2,'Calculating a few more metrics on heldout predictions');                                                                         
    [~,~,absError_mean_iShuffle,~,accuracy_iShuffle] = eval_dataset_predictions(y_true_full,y_hat_full,accuracy_tolerance,shuffle_dataset_opt); 
    accuracies_shuffle{iShuffle} = accuracy_iShuffle;
    absError_means_shuffle{iShuffle} = absError_mean_iShuffle;             
    waitbar(iShuffle / nShuffles,f,'Creating shuffled decoding performance distribution'); 
    close(f2);
end 
close(f);

%% avg over shuffles to get mean and sem for absError and accuracy 
accuracies_tmp = cell(numel(mouse_grps),1);
absError_means_tmp = cell(numel(mouse_grps),1);

% initialize structures to avg over rewards
for mIdx = 1:numel(mouse_grps) 
    for iVar = 1:3 
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            accuracies_tmp{mIdx}{iVar}{iRewsize} = nan(numel(var_bins{iVar})-1,nShuffles);
            absError_means_tmp{mIdx}{iVar}{iRewsize} = nan(numel(var_bins{iVar})-1,nShuffles);
        end 
    end
end

for iShuffle = 1:nShuffles
    for mIdx = 1:numel(mouse_grps)
        % iterate over the variables we are decoding
        for iVar = 1:3
            % iterate over reward sizes of interest
            for iRewsize = 1:numel(dataset_opt.rewsizes)
                accuracies_tmp{mIdx}{iVar}{iRewsize}(:,iShuffle) = accuracies_shuffle{iShuffle}{mIdx}{1}{iVar}{iRewsize};
                absError_means_tmp{mIdx}{iVar}{iRewsize}(:,iShuffle) = absError_means_shuffle{iShuffle}{mIdx}{1}{iVar}{iRewsize};
            end
        end
    end
end 

% now get means and sems to report
accuracies_mean_shuffle = cell(numel(mouse_grps),1); 
accuracies_sem_shuffle = cell(numel(mouse_grps),1); 
absError_mean_shuffle = cell(numel(mouse_grps),1); 
absError_sem_shuffle = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    for iVar = 1:3
        for iRewsize = 1:numel(dataset_opt.rewsizes)
            accuracies_mean_shuffle{mIdx}{iVar}{iRewsize} = mean(accuracies_tmp{mIdx}{iVar}{iRewsize},2);
            accuracies_sem_shuffle{mIdx}{iVar}{iRewsize} = std(accuracies_tmp{mIdx}{iVar}{iRewsize},[],2) / sqrt(nShuffles);
            absError_mean_shuffle{mIdx}{iVar}{iRewsize} = mean(absError_means_tmp{mIdx}{iVar}{iRewsize},2);
            absError_sem_shuffle{mIdx}{iVar}{iRewsize} = std(absError_means_tmp{mIdx}{iVar}{iRewsize},[],2) / sqrt(nShuffles);
        end
    end
end

%% Visualize encoding representation 
close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
for iRewsize = 3
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 1:4
        figure()
        for mIdx = 1:numel(mouse_grps)
            for iVar = 1:numel(dataset_opt.vars)
                subplot(3,5,mIdx + 5 * (iVar - 1))  
                if iVar == 1
                    [~,max_ix] = max(cond_means{mIdx}{iFeature}{iVar}{iRewsize},[],2);
                    [~,peaksort] = sort(max_ix);
                end
                if iVar ~= 3
                    imagesc(flipud(zscore(cond_means{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:),[],2)))
%                     imagesc(flipud(cond_means{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:)))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(var_bins{iVar}(1:10:end)) 
                    colorbar()
                else
                    imagesc(flipud(zscore(fliplr(cond_means{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:)),[],2)))
%                     imagesc(flipud(fliplr(cond_means{mIdx}{iFeature}{iVar}{iRewsize}(peaksort,:))))
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(-fliplr(var_bins{iVar}(1:10:end)) ) 
                    colorbar()
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
cmap = parula(50); % cbrewer('seq',"Blues",50); 
% cmap(1:10,:) = repmat([1 1 1],[10,1]);
vis_mice = 1:5;
for iRewsize = 3
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    for iFeature = 4
        figure()
        for m = 1:numel(vis_mice)
            mIdx = vis_mice(m);
            for iVar = 1:numel(dataset_opt.vars)
                subplot(3,numel(vis_mice),m + numel(vis_mice) * (iVar - 1))
                if iVar ~= 3 % note this is norm over cols
                    imagesc(flipud(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} ./ sum(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize},2)))
%                     hold on;plot(fliplr(1:length(yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize})),yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}/msec_conversion,'w','linewidth',1)
                    colormap(cmap)
                    xticks(1:10:(numel(var_bins{iVar})))
                    xticklabels(var_bins{iVar}(1:10:end))
                    yticks(1:10:(numel(var_bins{iVar})))
                    yticklabels(fliplr(var_bins{iVar}(1:10:end)))
                else
                    imagesc(flipud(rot90(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize} ./ sum(confusion_mats{mIdx}{iFeature}{iVar}{iRewsize},2),2)))
%                     hold on;plot(1:length(yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}),flipud(yhat_mean_withinRewsize{mIdx}{iFeature}{iVar}{iRewsize}/msec_conversion),'w','linewidth',1)
                    colormap(cmap)
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
colors = [lines(3) ; 0 .6 .2; 0 0 0; .2 .4 .2 ; .2 .2 .8; .3 .6 .3];
% close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
vis_features = [5 8]; 
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
            ylim([0,max(var_bins{iVar}) / 2])
            
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

%% Visualize performance in terms of accuracy w/ tolerance 
colors = [lines(3) ; 0 .6 .2; 0 0 0; .2 .4 .2 ; .2 .2 .8; .3 .6 .3];
% close all
var_names = ["Time Since Reward","Time On Patch","Time Until Leave"]; 
vis_features = 1:4; 
vis_mice = 1:5;

for iRewsize = 1:2
    this_rewsize = dataset_opt.rewsizes(iRewsize);
    figure()
    for m = 1:numel(vis_mice) % numel(mouse_grps)
        mIdx = vis_mice(m);
        for iVar = 1:numel(dataset_opt.vars)
            subplot(3,numel(vis_mice),m + numel(vis_mice) * (iVar - 1));hold on
            for ix_Feature = 1:numel(vis_features) % numel(dataset_opt.features)
                iFeature = vis_features(ix_Feature);
                if iVar ~= 3
                    plot(var_bins{iVar}(1:end-1),accuracy{mIdx}{iFeature}{iVar}{iRewsize},'linewidth',2,'color',colors(iFeature,:))
                    if ix_Feature == numel(vis_features)
                        shadedErrorBar(var_bins{iVar}(1:end-1),accuracies_mean_shuffle{mIdx}{iVar}{iRewsize},accuracies_sem_shuffle{mIdx}{iVar}{iRewsize},'lineprops',{'Color',[0 0 0]});
                    end
                else
                    plot(var_bins{iVar}(1:end-1),flipud(accuracy{mIdx}{iFeature}{iVar}{iRewsize}),'linewidth',2,'color',colors(iFeature,:));
                    if ix_Feature == numel(vis_features)
                        shadedErrorBar(var_bins{iVar}(1:end-1),flipud(accuracies_mean_shuffle{mIdx}{iVar}{iRewsize}),flipud(accuracies_sem_shuffle{mIdx}{iVar}{iRewsize}),'lineprops',{'Color',[0 0 0]});
                    end
                end
            end
            ylim([0,1])
            
            if m == 1
                ylabel(sprintf("%s Fraction within %.2f sec",var_names(iVar),accuracy_tolerance*msec_conversion))
            end
            if iVar == 1
                title(mouse_names(mIdx))
            end
            xlabel(sprintf("%s",var_names(iVar)))
            if m == 1 && iVar == 1
                legend([cellfun(@(x) x.name,dataset_opt.features(vis_features)) "Shuffled Neural Data"])
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
    for iFeature = 1:4
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
                y_heldout_full = cat(2,y_dataset{mIdx}{iVar}{i_heldout_rewsize}{:});  
                
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
%  make this visualization better 
colors = cool(3);
for mIdx = 5
    for iFeature = 2:3
        for iVar = 1
            figure() 
            for i_heldout_rewsize = 1:numel(dataset_opt.rewsizes)   
                for i_trained_rewsize = 1:numel(dataset_opt.rewsizes)
                    subplot(3,3,i_trained_rewsize + 3 * (i_heldout_rewsize - 1)) ;hold on
                    if i_heldout_rewsize ~= i_trained_rewsize  
                        imagesc(confusion_mats_xRewsize{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{i_trained_rewsize})
                    else 
                        % trained on this rewsize
                        imagesc(confusion_mats{mIdx}{iFeature}{iVar}{i_heldout_rewsize})
                    end 
                    
                    if iVar ~= 3 
                        hold on 
                        for plot_trained_rewsize = 1:numel(dataset_opt.rewsizes)  
                            plot(1:numel(var_bins{iVar}(1:end-1)),yhat_mean{mIdx}{iFeature}{iVar}{plot_trained_rewsize}{i_heldout_rewsize} / msec_conversion,'color',colors(plot_trained_rewsize,:),'linewidth',1.5)
                        end
                        xticks(1:10:(numel(var_bins{iVar})))
                        xticklabels(var_bins{iVar}(1:10:end))
                        yticks(1:10:(numel(var_bins{iVar})))
                        yticklabels(fliplr(var_bins{iVar}(1:10:end))) 
                    else 
%                         hold on
%                         plot(1:numel(var_bins{iVar}(1:end-1)),yhat_mean{mIdx}{iFeature}{iVar}{i_heldout_rewsize}{i_trained_rewsize},'color',colors(i_trained_rewsize,:),'linewidth',1.5)
                        xticks(1:10:(numel(var_bins{iVar})))
                        xticklabels(-fliplr(var_bins{iVar}(1:10:end)))
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
        legend("1 uL Model Prediction","2 uL Model Prediction","4 uL Model Prediction")
    end 
end  

%% Now visualize decoding as line plot 
colors = cool(3); 
close all
for iFeature = 3
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

%% Now load predictions into trialed cell array 
%  Do changes in time decoding correlate with differences in behavior? 
close all  
y_hat_trials = cell(numel(mouse_grps),1);
for mIdx = 1:numel(mouse_grps) 
    y_hat_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    for i = 1:numel(mouse_grps{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);
        
        data = load(fullfile(paths.data,sessions{sIdx}));  

        % Load session information
        nTrials = length(data.patchCSL);
        rewsize = mod(data.patches(:,2),10);
        y_hat_trials{mIdx}{i} = cell(nTrials,1); 
        
        s_nNeurons = cell(numel(dataset_opt.features),1);  
        session_neuron_starts = cell(numel(dataset_opt.features),1); 
        for iFeature = 1:numel(dataset_opt.features) 
            % get number of neurons of these clusters per session
            s_nNeurons{iFeature} = cell2mat(cellfun(@(x) length(find(ismember(x,dataset_opt.features{iFeature}.ix))),X_clusters{mIdx},'un',0));
            session_neuron_starts{iFeature} = [0 cumsum(s_nNeurons{iFeature})']; 
        end
        
        for iTrial = 1:nTrials 
            trial_fold = xval_table{mIdx}(xval_table{mIdx}.SessionIx == i & xval_table{mIdx}.TrialNum == iTrial,:).FoldID;
            fr_mat_iTrial = X{mIdx}{i,1}{iTrial};     
            iRewsize = find(dataset_opt.rewsizes == rewsize(iTrial),1); % reward size index
            if ~isnan(trial_fold)
                y_hat_trials{mIdx}{i}{iTrial} = cell(numel(dataset_opt.vars),1); 
                % Iterate over decoded variables
                for iVar = 1:2
                    y_hat_trials{mIdx}{i}{iTrial}{iVar} = cell(numel(dataset_opt.features),1);
                    for iFeature = 1:numel(dataset_opt.features)
                        % pad so that we can throw into prediction model
                        fr_mat_iTrialPadded = nan(sum(s_nNeurons{iFeature}),size(fr_mat_iTrial,2));
                        fr_mat_iTrialPadded(session_neuron_starts{iFeature}(i)+1:session_neuron_starts{iFeature}(i+1),:) = fr_mat_iTrial(ismember(X_clusters{mIdx}{i},dataset_opt.features{iFeature}.ix),:);
                        
                        % Predict task variables from decoder trained on other trials
                        this_model = models{mIdx}{iFeature}{iVar}{iRewsize}{trial_fold};
                        y_hat_trials{mIdx}{i}{iTrial}{iVar}{iFeature} = predict(this_model,fr_mat_iTrialPadded'); 
                    end
                end
            end
        end
    end 
    fprintf("%s Single Trial Predictions Complete \n",mouse_names(mIdx))
end
%% Now analyze / visualize single trials 
%  Peaksort w/ some peaksort 
%  Stimuli  
%  Predictions
%  Reward events
close all
feature_colors = [lines(3) ; 0 .6 .2; 0 0 0; .2 .4 .2 ; .2 .2 .8; .3 .6 .3];  
rew_colors = cool(3);
var_names = ["Time since reward","Time on Patch"];
for mIdx = 3
    for i = 2 % 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4); 
        session_date = sprintf("%s/%s",session(end-2),session(end-1:end));
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        clusters = X_clusters{mIdx}{i};  
%         clust12_bool = ismember(clusters,[1,2]); 
%         clusts12 = clusters(clust12_bool);
%         [~,clusts12_sort] = sort(clusts12); 
        [~,clusts_sort] = sort(clusters); 
        
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
        rew_ix_cell = cell(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;  
%             rew_ix_cell{iTrial}(rew_ix_cell{iTrial} == 0) = 1;
            rew_sec_cell{iTrial} = rew_indices; % THIS SHOULD BE GREATER OR EQUAL TO!!!!! mystery fucking solved
        end
        
        vis_features = 2:3;
        smoothing_sigma = 3;
        
        for iTrial = 21:30
            figure() 
            fr_mat_iTrial = X{mIdx}{i,1}{iTrial};  
            subplot(3,1,1);hold on
            imagesc(flipud(zscore(fr_mat_iTrial(clusts_sort,:),[],2)))
            gscatter(zeros(numel(clusts_sort),1)-.5,1:numel(clusts_sort),flipud(clusters(clusts_sort)),lines(3)) 
            legend(["Cluster 1","Cluster 2","Cluster 3"])
            xticks((0:25:size(fr_mat_iTrial,2)))
            xticklabels((0:25:size(fr_mat_iTrial,2)) * tbin_ms / 1000)
            xlim([-.5 size(X{mIdx}{i,1}{iTrial},2)]) 
            xlabel("Time on Patch (sec)")
            ylim([1 numel(clusts_sort)])   
            title(sprintf("%s %s %i uL Trial %i",mouse_names(mIdx),session_date,rewsize(iTrial),iTrial))
            for iVar = 1:2 
                subplot(3,1,iVar + 1);hold on  
                plot(y{mIdx}{i,iVar}{iTrial},'k','linewidth',2)  
                for iFeature = vis_features
                    plot(gauss_smoothing(y_hat_trials{mIdx}{i}{iTrial}{iVar}{iFeature},smoothing_sigma),'color',feature_colors(iFeature,:),'linewidth',2);
                end 
                xticks((0:25:size(fr_mat_iTrial,2)))
                xticklabels((0:25:size(fr_mat_iTrial,2)) * tbin_ms / 1000) 
%                 xlim([-.5 size(fr_mat_iTrial,2)])   
                xlabel("Time on Patch (sec)")  
                yticks((1:5:length(var_bins{iVar})))
                yticklabels(var_bins{iVar}(1:5:end)) 
                ylabel(sprintf("Decoded %s",var_names(iVar))) 
                yl = ylim(); 
                % add reward events 
                for r_ix = 1:numel(rew_ix_cell{iTrial}) 
                    r = rew_ix_cell{iTrial}(r_ix); 
                    v = [ r yl(1);r yl(2) ; r + 3 yl(2);r + 3 yl(1) ];
                    f = [1 2 3 4];
                    patch('Faces',f,'Vertices',v,'FaceColor',rew_colors(min(3,rewsize(iTrial)),:),'FaceAlpha',.5)
                end
            end
        end
    end
end 

%% Now collect traces of decoded time, color by PRT, plot 
R_prt = cell(4,1);
p_prt = cell(4,1); 
feature_colors = [lines(3) ; 0 .6 .2; 0 0 0; .2 .4 .2 ; .2 .2 .8; .3 .6 .3];  

for mIdx = 5
    R_prt = cell(4,1);
    p_prt = cell(4,1);
    for iFeature = 4
%         figure()
        % only include sessions w/ more than 10 neurons
        s_nNeurons = cellfun(@length,X_clusters{mIdx});
        include_sessions = find(s_nNeurons > 10);
        
        prts_pooled = cell(3); % one for each reward size
        decodedVars_pooled = cell(3);
        for iRewsize = 1:3
            prts_pooled{iRewsize} = [];
            decodedVars_pooled{iRewsize} = cell(2);
            for iVar = 1:2
                decodedVars_pooled{iRewsize}{iVar} = [];
            end
        end
        
        for i_session = 1:numel(include_sessions)
            figure()
            i = include_sessions(i_session);
            sIdx = mouse_grps{mIdx}(i);
            session = sessions{sIdx}(1:end-4);
            session_date = sprintf("%s/%s",session(end-2),session(end-1:end));
            data = load(fullfile(paths.data,sessions{sIdx}));
            
            % Load session information
            nTrials = length(data.patchCSL);
            rewsize = mod(data.patches(:,2),10);
            rew_sec = data.rew_ts;
            patchstop_sec = data.patchCSL(:,2);
            patchleave_sec = data.patchCSL(:,3);
            prts = patchleave_sec - patchstop_sec;
            patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
            patchleave_ix = round((data.patchCSL(:,3)*1000) / tbin_ms) + 1;
            prts_ix = patchleave_ix - patchstop_ix + 1;
            
            % Collect trial reward timings
            rew_sec_cell = cell(nTrials,1);
            rew_ix_cell = cell(nTrials,1);
            for iTrial = 1:nTrials
                rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) - patchstop_sec(iTrial));
                rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;
                rew_sec_cell{iTrial} = rew_indices;
            end
            
            for iRewsize = [1 2 4]
                iRewsize_trials = find(rewsize == iRewsize);
                RNil_trials = cell2mat(cellfun(@(x) length(x) == 1,rew_sec_cell(rewsize == iRewsize),'un',0));
                RNil_trials = iRewsize_trials(RNil_trials);
                cmap = cbrewer('div',"RdBu",numel(RNil_trials));
                [~,prt_sorted_RNil_trials] = sort(prts(RNil_trials));
                prt_sorted_RNil_trials = RNil_trials(prt_sorted_RNil_trials);
                
                prts_pooled{min(3,iRewsize)} = [prts_pooled{min(3,iRewsize)} ; prts(prt_sorted_RNil_trials)];
                
                for iTrial = 1:numel(RNil_trials)
                    trial = prt_sorted_RNil_trials(iTrial);
                    for iVar = 1:2
                        subplot(2,3,3 * (iVar - 1) + min(3,iRewsize));hold on
                        plot(gauss_smoothing(y_hat_trials{mIdx}{i}{trial}{iVar}{iFeature}(1:min(prts_ix(trial),100)),1),'color',cmap(iTrial,:),'linewidth',.5)
                        
                        padded_decoding = [y_hat_trials{mIdx}{i}{trial}{iVar}{iFeature}(1:min(prts_ix(trial),100)) ; nan(max(0,100 - prts_ix(trial)),1)];
                        decodedVars_pooled{min(3,iRewsize)}{iVar} = [decodedVars_pooled{min(3,iRewsize)}{iVar} padded_decoding];
                    end
                end
                for iVar = 1:2
                    subplot(2,3,3 * (iVar-1) + min(3,iRewsize))
                    title(sprintf("%iNil Decoded %s",iRewsize,var_names(iVar)))
                    xticks((0:25:100))
                    xticklabels((0:25:100) * tbin_ms / 1000)
                    yticks((1:10:length(var_bins{1})))
                    yticklabels(var_bins{1}(1:10:end))
                end
            end
            
            for iRewsize = [1 2 4]
                for iVar = 1:2
                    subplot(2,3,3 * (iVar-1) + min(3,iRewsize))
                    xlabel(sprintf("True %i uL %s ",iRewsize,var_names(iVar)))
                end
            end
        end
        
        yl = ylim();
        
        % Now calculate PRT corrcoef at each timepoint
        R_prt{iFeature} = cell(3);
        p_prt{iFeature} = cell(3);
        for iRewsize = 1:3
            R_prt{iFeature}{iRewsize} = cell(2);
            p_prt{iFeature}{iRewsize} = cell(2);
            for iVar = 1:2
                max_time = size(decodedVars_pooled{iRewsize}{iVar},1);
                R_prt{iFeature}{iRewsize}{iVar} = nan(max_time,1);
                p_prt{iFeature}{iRewsize}{iVar} = nan(max_time,1);
                for iTime = 1:max_time
                    these_decodes = decodedVars_pooled{iRewsize}{iVar}(iTime,:);
                    [r,p] = corrcoef(these_decodes(~isnan(these_decodes)),prts_pooled{iRewsize}(~isnan(these_decodes)));
                    R_prt{iFeature}{iRewsize}{iVar}(iTime) = r(2);
                    p_prt{iFeature}{iRewsize}{iVar}(iTime) = p(2);
                    if p(2) < .01
                        subplot(2,3,3 * (iVar-1) + min(3,iRewsize));hold on
                        scatter(iTime,yl(2) + 5,'k*')
                    end
                end
            end
        end
        subplot(2,3,1)
        ylabel("Decoded Time Since Reward")
        subplot(2,3,4)
        ylabel("Decoded Time On Patch")
        suptitle(sprintf("%s %s Decoding",mouse_names(mIdx),dataset_opt.features{iFeature}.name))
    end
end  

%% Visualize results between features  
sig_thresh = .01;
figure()
for iRewsize = 1:3
    for iVar = 1:2
        subplot(2,3,3 * (iVar-1) + min(3,iRewsize));hold on
        for iFeature = 1:4
            plot(R_prt{iFeature}{iRewsize}{iVar},'color',feature_colors(iFeature,:),'linewidth',1.5) 
            scatter(find(p_prt{iFeature}{iRewsize}{iVar}<sig_thresh),R_prt{iFeature}{iRewsize}{iVar}(p_prt{iFeature}{iRewsize}{iVar}<sig_thresh),[],feature_colors(iFeature,:),'marker','*') 
        end 
        ylim([-.75 .75]) 
        xticks((0:25:100))
        xticklabels((0:25:100) * tbin_ms / 1000) 
        if iRewsize == 3 
            disp_rewsize = 4;  
        else 
            disp_rewsize = iRewsize;
        end
        xlabel(sprintf("%i uL %s ",disp_rewsize,var_names(iVar)))
    end 
    if iRewsize == 1
        subplot(2,3,1) 
        ylabel("RNil Trial Pearson Correlation Coefficient")
        subplot(2,3,4)   
        ylabel("RNil Trial Pearson Correlation Coefficient")
    end
end

%% Single trial, inter-cluster correlations? 
prt_colors = flipud(cbrewer('div',"RdBu",100)); 
R_prt = cell(4,1);
p_prt = cell(4,1); 
feature_colors = [lines(3) ; 0 .6 .2; 0 0 0; .2 .4 .2 ; .2 .2 .8; .3 .6 .3];   
vis_features = 2:3; 
% close all
prt_colors2 = flipud(cbrewer('div',"RdBu",3));  
prt_colors2(2,:) = [];

for mIdx = 5
    figure() 
    decoding_errors = cell(2,3,2);
    for i_session = 1:numel(include_sessions)
        i = include_sessions(i_session);
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        session_date = sprintf("%s/%s",session(end-2),session(end-1:end));
        data = load(fullfile(paths.data,sessions{sIdx}));
        
        % Load session information
        nTrials = length(data.patchCSL);
        rewsize = mod(data.patches(:,2),10);
        rew_sec = data.rew_ts;
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);
        prts = patchleave_sec - patchstop_sec;
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round((data.patchCSL(:,3)*1000) / tbin_ms) + 1;
        prts_ix = patchleave_ix - patchstop_ix + 1;
        
        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        rew_ix_cell = cell(nTrials,1);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;
            rew_sec_cell{iTrial} = rew_indices;
        end 
        
        for iRewsize = [1 2 4]
            iRewsize_trials = find(rewsize == iRewsize);
            RNil_trials = cell2mat(cellfun(@(x) length(x) == 1,rew_sec_cell(rewsize == iRewsize),'un',0));
            RNil_trials = iRewsize_trials(RNil_trials);
            cmap = cbrewer('div',"RdBu",numel(RNil_trials));
            [~,prt_sorted_RNil_trials] = sort(prts(RNil_trials));
            prt_sorted_RNil_trials = RNil_trials(prt_sorted_RNil_trials);
            
            prts_pooled{min(3,iRewsize)} = [prts_pooled{min(3,iRewsize)} ; prts(prt_sorted_RNil_trials)];

            for iTrial = 1:numel(RNil_trials) 
                trial = prt_sorted_RNil_trials(iTrial);  
                error_sign = sign(numel(RNil_trials)/2 - iTrial);
                for iVar = 1:2 
                    subplot(2,3,3 * (iVar - 1) + min(3,iRewsize));hold on
                    decode3 = gauss_smoothing(y_hat_trials{mIdx}{i}{trial}{iVar}{3}(1:min(prts_ix(trial),100)),1);
                    decode2 = gauss_smoothing(y_hat_trials{mIdx}{i}{trial}{iVar}{2}(1:min(prts_ix(trial),100)),1);
                    plot(1 * msec_conversion * (decode3 - decode2),'color',cmap(iTrial,:),'linewidth',.5)
                    title(sprintf("%iNil Cluster 3-2 Decoded %s",iRewsize,var_names(iVar)))
                    xticks((0:25:100))
                    xticklabels((0:25:100) * tbin_ms / 1000)
                    ylim([-2 2])
                    if prts_ix(trial) >= 100
                        decoding_errors{iVar,min(3,iRewsize),min(2,2+error_sign)} = [decoding_errors{iVar,min(3,iRewsize),min(2,2+error_sign)} 1 * msec_conversion * (decode3 - decode2)];
                    end
                end
            end
        end
    end  
    
    figure()
    for iRewsize = 1:3 
        for iVar = 1:2  
            subplot(2,3,3 * (iVar - 1) + iRewsize);hold on
            for iSign = 1:2 
                shadedErrorBar(1:100,mean(decoding_errors{iVar,iRewsize,iSign},2),2*std(decoding_errors{iVar,iRewsize,iSign},[],2) / sqrt(size(decoding_errors{iVar,iRewsize,iSign},2)),'lineprops',{'Color',prt_colors2(iSign,:)});
            end 
            ylim([-1 1])
        end
    end
end 

%% Does magnitude of Cluster 1 Predict changes in Cluster 2/3 time decoding? 
% 

for mIdx = 5
    % only include sessions w/ more than 10 neurons
    s_nNeurons = cellfun(@length,X_clusters{mIdx});
    include_sessions = find(s_nNeurons > 10); 
    
    iVar = 2;
    figure()
    for i_session = 1:numel(include_sessions)
        i = include_sessions(i_session);
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        session_date = sprintf("%s/%s",session(end-2),session(end-1:end));
        data = load(fullfile(paths.data,sessions{sIdx}));
        
        clusters = X_clusters{mIdx}{i};  
        % can make this programatic ... loop over start/stop ix
        start_ix = 1; 
        stop_ix = 25; 
        avg_cluster1 = cell2mat(cellfun(@(x) mean(x(clusters == 1,start_ix:stop_ix),'all'),X{mIdx}{i,1},'un',0));
        
        % Load session information
        nTrials = length(data.patchCSL);
        rewsize = mod(data.patches(:,2),10);
        rew_sec = data.rew_ts;
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);
        prts = patchleave_sec - patchstop_sec;
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round((data.patchCSL(:,3)*1000) / tbin_ms) + 1;
        prts_ix = patchleave_ix - patchstop_ix + 1;
        
        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        rew_ix_cell = cell(nTrials,1);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec <= patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;
            rew_sec_cell{iTrial} = rew_indices;
        end
        
        for iRewsize = [1 2 4]
            iRewsize_trials = find(rewsize == iRewsize);
            RNil_trials = cell2mat(cellfun(@(x) length(x) == 1,rew_sec_cell(rewsize == iRewsize),'un',0));
            RNil_trials = iRewsize_trials(RNil_trials);
            cmap = cbrewer('div',"RdBu",numel(RNil_trials));
            [~,cluster1_sorted_RNil_trials] = sort(avg_cluster1(RNil_trials));
            cluster1_sorted_RNil_trials = RNil_trials(cluster1_sorted_RNil_trials);
            
            for iTrial = 1:numel(RNil_trials)
                trial = cluster1_sorted_RNil_trials(iTrial);
                for iFeature = 2:3
                    subplot(2,3,3 * (iFeature - 1 - 1) + min(3,iRewsize));hold on
                    plot(gauss_smoothing(y_hat_trials{mIdx}{i}{trial}{iVar}{iFeature}(1:min(prts_ix(trial),100)),1),'color',cmap(iTrial,:),'linewidth',.5)
                end
            end
            
            for iFeature = 2:3
                subplot(2,3,3 * (iFeature - 1 -1) + min(3,iRewsize))
                title(sprintf("%iNil %s Decoded %s",iRewsize,dataset_opt.features{iFeature}.name,var_names(iVar)))
                xticks((0:25:100))
                xticklabels((0:25:100) * tbin_ms / 1000)
                yticks((1:10:length(var_bins{1})))
                yticklabels(var_bins{1}(1:10:end)) 
                xlabel(sprintf("True %i uL %s ",iRewsize,var_names(iVar)))
            end
        end
    end
end


%% Testing  
mIdx = 3; 
iVar = 2;
iRewsize = 1;
this_rewsize = 1;
X_practice = X_dataset{mIdx}{4}{iVar}{iRewsize};  
sessionIx = xval_table{mIdx}(xval_table{mIdx}.Rewsize == this_rewsize,:).SessionIx;
X_padcat = padCat(X_practice,sessionIx); 

% get the generative model tracking learned representation across time 
% kFold = 1;
% rep = rot90(cellfun(@(x) x(1),models{mIdx}{iVar}{iRewsize}{1}.DistributionParameters));

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
    