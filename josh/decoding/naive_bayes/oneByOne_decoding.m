%% The final codedown: Decode time since reward, adding one cell at a time
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat';
load(paths.transients_table);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 

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
brain_region = cell(numel(mouse_grps),1); 
for mIdx = 5
    X{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_vel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_accel{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_pos{mIdx} = cell(numel(mouse_grps{mIdx}),3);  
    X_clusters{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    X_peak_pos{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    X_cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1);   
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 2
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
        brain_region{mIdx}{i} = session_table.Region;
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
for mIdx = 5
    y{mIdx} = cell(numel(mouse_grps{mIdx}),3); % 3 variables to decode 
    y_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per sesion   
    RX{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    RXX{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_time{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_num{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    xval_table{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 2 % 1:numel(mouse_grps{mIdx}) 
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
for mIdx = 5 % 1:numel(mouse_grps) 
    dataset_opt.features{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i_session = 2 % 1:numel(mouse_grps{mIdx})  
        sIdx = mouse_grps{mIdx}(i_session);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]); 
        % First feature: all GLM selected neurons
        dataset_opt.features{mIdx}{i_session}{1} = struct;
        dataset_opt.features{mIdx}{i_session}{1}.type = "KMeans Clusters";
        dataset_opt.features{mIdx}{i_session}{1}.ix = 1:5; % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{1}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{1}.name = "All Clusters"; % name for visualizations  
        % Second feature: all PFC cells
        dataset_opt.features{mIdx}{i_session}{2} = struct;
        dataset_opt.features{mIdx}{i_session}{2}.type = "CellID";
        dataset_opt.features{mIdx}{i_session}{2}.ix = X_cellIDs{mIdx}{i_session}(brain_region{mIdx}{i_session} == "PFC"); % indices within the feature type we selected
        dataset_opt.features{mIdx}{i_session}{2}.shuffle = false; % shuffle?
        dataset_opt.features{mIdx}{i_session}{2}.name = "All PFC Cells"; % name for visualizations  
    end 
end

% other options
dataset_opt.rewsizes = [2 4]; % which reward size trials will we train to
dataset_opt.numFolds = 2; % 2 folds  
dataset_opt.vars = [1 2 3];    

[X_dataset,y_dataset,xval_table] = gen_multiclassDataset_singleSession(X,X_clusters,X_cellIDs,X_vel,X_accel,X_pos,y,y_rewsize,xval_table,dataset_opt,mouse_grps);

%% Train and evaluate population classifiers 
% Now use classification datasets to train classifiers, holding out test folds
dataset_opt.distribution = 'normal';
models = fit_dataset_singleSession(X_dataset,y_dataset,xval_table,dataset_opt,mouse_names);

% Now evaluate classifier performance on test folds 
[y_true_full,y_hat_full] = predict_dataset_singleSession(X_dataset,y_dataset,models,xval_table,dataset_opt,mouse_names);

%%
[cond_means,confusion_mats,mutual_information,MAE,...
 abs_error_mean_givenTrue,abs_error_mean_givenHat,... 
 abs_error_sem_givenTrue,abs_error_sem_givenHat] = eval_dataset_singleSession(models,y_hat_full,y_true_full,dataset_opt,var_bins);

%% Visualize asymptotic(?) time since reward decoding performance from full population w/ heatmap and mutual information
%  Let's just care about time since reward for now
%  Look at one session at a time (this code exists already)

var_names = ["Time Since Reward","Time on Patch","Time to Leave"]; 
close all
vis_features = 1:2;
iRewsize = 2; 
iVar = 1;  
vis_mice = 5;
vis_confusionMat(confusion_mats,X_dataset,session_titles,mutual_information,MAE,var_bins,var_names,iVar,iRewsize,vis_features,vis_mice,dataset_opt)

%% Forward search over cells

iVar = 1; % time since reward
iRewsize = 2; % 4 uL 
mIdx = 5;  
i_session = 2; 
iFeature = 1; % All cells or GLM cells
% Choose to search over a subset of the population?
% population = 1:size(X_dataset{mIdx}{i_session}{iVar}{iRewsize}{iFeature}{1},1);
% population = find(X_clusters{mIdx}{i}(~isnan(X_clusters{mIdx}{i})) == 3); 
search_depth = 15;
timecourse_save_steps = [1 7 15]; % save timecourse information per 10 timesteps

fwd_mi_cumulative = cell(3,1); 
fwd_mae_cumulative = cell(3,1); 
fwd_timecourse_results = cell(3,1); 

for i_cluster = [1 2 4]
    population = find(X_clusters{mIdx}{i}(~isnan(X_clusters{mIdx}{i})) == i_cluster);
    i_search_depth = min(length(population),search_depth); 
    [fwd_mi_cumulative{i_cluster},fwd_mae_cumulative{i_cluster},fwd_timecourse_results{i_cluster}] = NB_fwd_search(population,i_search_depth,timecourse_save_steps,... 
                                                                                                       mIdx,i_session,iVar,iRewsize,iFeature,... 
                                                                                                       X_dataset,y_dataset,models,xval_table,dataset_opt); 
end

%% Randomly pick cells one-by-one, measure increase in decoder performance 
%  See how this differs in different chosen populations

iVar = 1; % time since reward
iRewsize = 2; % 4 uL 
mIdx = 5;  
i_session = 2; 
iFeature = 1; % All cells or GLM cells
% Choose to search over a subset of the population?
% population = 1:size(X_dataset{mIdx}{i_session}{iVar}{iRewsize}{iFeature}{1},1);
n_searches = 10; 
search_depth = 15;
rnd_timecourse_save_steps = [1 7 15]; % save timecourse information per 10 timesteps

rnd_mi_cumulative = cell(3,1); 
rnd_mae_cumulative = cell(3,1); 
rnd_timecourse_results = cell(3,1); 

for i_cluster = 1:3 
    population = find(X_clusters{mIdx}{i}(~isnan(X_clusters{mIdx}{i})) == i_cluster);  
    i_search_depth = min(length(population),search_depth); 
    [rnd_mi_cumulative{i_cluster},rnd_mae_cumulative{i_cluster},rnd_timecourse_results{i_cluster}] = NB_rnd_search(population,n_searches,i_search_depth,rnd_timecourse_save_steps,... 
                                                                   mIdx,i_session,iVar,iRewsize,iFeature,... 
                                                                   X_dataset,y_dataset,models,xval_table,dataset_opt);
end

%% Visualize changes in confusion matrix over course of fwd search 
bin_dt = diff(var_bins{1}(1:2));
figure()
for i_cluster = 1:3 
    for i_savept = 1:numel(timecourse_save_steps)  
        subplot(3,numel(timecourse_save_steps),3 * (i_cluster-1) + i_savept)
        if ~isempty(fwd_timecourse_results{i_cluster}{i_savept}) % if we didnt have enough cells
            imagesc(flipud(fwd_timecourse_results{i_cluster}{i_savept}.confusionmat))
        end 
        if i_cluster == 1 
            title(sprintf("%i Cells Added by Fwd Search",timecourse_save_steps(i_savept)),'FontSize',12)
        end 
        if i_savept == 1 
            ylabel(sprintf("Cluster %i \n True Time Since Rew",i_cluster),'FontSize',13) 
            yticks(0:10:40) 
            yticklabels(bin_dt * fliplr((0:10:40)))  
        end
        if i_cluster == 3
            xlabel("Predicted Time Since Rew",'FontSize',13) 
            xticks(0:10:40) 
            xticklabels(bin_dt * (0:10:40)) 
        end 
        if i_cluster ~= 3
            xticks([]) 
        end 
        if i_savept ~= 1 
            yticks([])
        end
    end
end

%% Visualize changes in timecourse of prediction over course of fwd search
bin_dt = diff(var_bins{1}(1:2)); 
x = bin_dt * (1:(length(var_bins{1})-1));
cmap = lines(3); 
shading = [1.3 1 .8];
figure()
for i_cluster = 1:3 
    subplot(1,numel(timecourse_save_steps),i_cluster)
    for i_savept = 1:numel(timecourse_save_steps)   
        if ~isempty(fwd_timecourse_results{i_cluster}{i_savept}) % if we didnt have enough cells
            shadedErrorBar(x,fwd_timecourse_results{i_cluster}{i_savept}.yhat_mean_timecourse,... 
                             fwd_timecourse_results{i_cluster}{i_savept}.yhat_sem_timecourse,...
                             'lineProps',{'color',min(1,shading(i_savept) * cmap(i_cluster,:)),... 
                                          'linewidth',2})
        end  
        title(sprintf("Cluster %i Fwd Search",i_cluster))
        if i_cluster == 1
            ylabel("Predicted Time Since Rew (sec)",'FontSize',13)
        end
        yticks(0:10:40)
        yticklabels(bin_dt * (0:10:40)) 
        ylim([0 40])
    end 
    xlabel("True Time Since Rew (sec)",'FontSize',13)
end

%% Visualize changes in timecourse of MAE over course of fwd search
bin_dt = diff(var_bins{1}(1:2)); 
x = bin_dt * (1:(length(var_bins{1})-1));
cmap = lines(3); 
shading = [1.3 1 .8];
figure()
for i_cluster = 1:3 
    subplot(1,numel(timecourse_save_steps),i_cluster)
    for i_savept = 1:numel(timecourse_save_steps)   
        if ~isempty(fwd_timecourse_results{i_cluster}{i_savept}) % if we didnt have enough cells
            shadedErrorBar(x,fwd_timecourse_results{i_cluster}{i_savept}.rmse_mean_timecourse,... 
                             fwd_timecourse_results{i_cluster}{i_savept}.rmse_sem_timecourse,...
                             'lineProps',{'color',min(1,shading(i_savept) * cmap(i_cluster,:)),... 
                                          'linewidth',2})
        end  
        title(sprintf("Cluster %i Fwd Search",i_cluster))
        if i_cluster == 1
            ylabel("Mean Absolute Error (sec)",'FontSize',13)
        end
        yticks(0:10:40)
        yticklabels(bin_dt * (0:10:40)) 
        ylim([-5 40])
    end 
    xlabel("True Time Since Rew (sec)",'FontSize',13)
end 

%% Visualize timecourse of predictions, avging over random searches
bin_dt = diff(var_bins{1}(1:2)); 
x = bin_dt * (1:(length(var_bins{1})-1));
cmap = lines(3); 
shading = [1.3 1 .8];
figure()
for i_cluster = 1:3  
    subplot(1,numel(timecourse_save_steps),i_cluster)
    for i_savept = 1:numel(rnd_timecourse_save_steps)
        if ~isempty(rnd_timecourse_results{i_cluster}{1}{i_savept})
            i_yhat_mean_timecourse = arrayfun(@(i_search) rnd_timecourse_results{i_cluster}{i_search}{i_savept}.yhat_mean_timecourse,(1:n_searches)','un',0);
            i_yhat_mean_timecourse = cat(2,i_yhat_mean_timecourse{:});
            
            shadedErrorBar(x,mean(bin_dt * i_yhat_mean_timecourse'),...
                std(bin_dt * i_yhat_mean_timecourse'),...
                'lineProps',{'color',min(1,shading(i_savept) * cmap(i_cluster,:)),...
                'linewidth',2}) 
            
            title(sprintf("Cluster %i Random Addition Search",i_cluster))
            if i_cluster == 1
                ylabel("Predicted Time Since Rew (sec)",'FontSize',13)
            end 
            ylim([0 2])
%             yticklabels(bin_dt * (0:10:40))
        end
    end 
    xlabel("True Time Since Rew (sec)",'FontSize',13)
end 

%% Visualize timecourse of MAE, avging over random searches
bin_dt = diff(var_bins{1}(1:2)); 
x = bin_dt * (1:(length(var_bins{1})-1));
cmap = lines(3); 
shading = [1.3 1 .8];
figure()
for i_cluster = 1:3  
    subplot(1,numel(timecourse_save_steps),i_cluster)
    for i_savept = 1:numel(rnd_timecourse_save_steps)
        if ~isempty(rnd_timecourse_results{i_cluster}{1}{i_savept})
            i_yhat_mean_timecourse = arrayfun(@(i_search) rnd_timecourse_results{i_cluster}{i_search}{i_savept}.mae_mean_timecourse,(1:n_searches)','un',0);
            i_yhat_mean_timecourse = cat(2,i_yhat_mean_timecourse{:});
            
            shadedErrorBar(x,mean(bin_dt * i_yhat_mean_timecourse'),...
                std(bin_dt * i_yhat_mean_timecourse'),...
                'lineProps',{'color',min(1,shading(i_savept) * cmap(i_cluster,:)),...
                'linewidth',2}) 
            
            title(sprintf("Cluster %i Random Addition Search",i_cluster))
            if i_cluster == 1
                ylabel("Mean Absolute Error (sec)",'FontSize',13)
            end 
            ylim([0 2])
%             yticklabels(bin_dt * (0:10:40))
        end
    end 
    xlabel("True Time Since Rew (sec)",'FontSize',13)
end

%% Visualize increase in information in random addition vs fwd search
cmap = lines(3); % Color map for different features
figure()  
for i_cluster = 1:3
    subplot(2,3,i_cluster) 
    % visualize results from stochastic search 
    plot(rnd_mi_cumulative{i_cluster}','linewidth',1,'color',min(1,1.3 * cmap(i_cluster,:)) );hold on
    shadedErrorBar(1:size(rnd_mi_cumulative{i_cluster},2),nanmean(rnd_mi_cumulative{i_cluster}),nanstd(rnd_mi_cumulative{i_cluster}),'lineProps',{'linewidth',1.5,'color',min(1,.8 * cmap(i_cluster,:))})
    plot(fwd_mi_cumulative{i_cluster},'linewidth',3,'color',cmap(i_cluster,:)); 
    ylim([0 1]) 
    if i_cluster == 1 
        ylabel("Mutual Information (nats)")
    end 
    title(sprintf("Cluster %i",i_cluster))
    % Visualize MAE results
    subplot(2,3,3 + i_cluster)
    plot(bin_dt * rnd_mae_cumulative{i_cluster}','linewidth',1,'color',min(1,1.3 * cmap(i_cluster,:)) );hold on
    shadedErrorBar(1:size(rnd_mi_cumulative{i_cluster},2),nanmean(bin_dt * rnd_mae_cumulative{i_cluster}),nanstd(bin_dt * rnd_mae_cumulative{i_cluster}),'lineProps',{'linewidth',1.5,'color',min(1,.8 * cmap(i_cluster,:))})
    plot(bin_dt *fwd_mae_cumulative{i_cluster},'linewidth',3,'color',cmap(i_cluster,:)); 
    ylim([0 1]) 
    if i_cluster == 1 
        ylabel("Mean Absolute Error (sec)")
    end 
    xlabel("Forward Search Depth") 
end 

%% Slightly different direction: Pick cells to maximize entropy of peak distribution, measure increase in decoder performance 
%  Is this better than random? 
%  Potentially rigorous way to assess encoding vs decoding efficiency
%  relationship

% First, do random search over transient-selected mPFC cells
iVar = 1; % time since reward
iRewsize = 2; % 4 uL 
mIdx = 5;  
i_session = 2; 
iFeature = 2; % All cells or GLM cells
% Use all transient-selected mPFC cells as population of interest 
rew1_sig_transient_bool = ~isnan(X_peak_pos{mIdx}{i_session}(:,2)); 
PFC_bool = strcmp(brain_region{mIdx}{i_session},"PFC"); 
midresp_bool = X_peak_pos{mIdx}{i_session}(:,2) > .2 & X_peak_pos{mIdx}{i_session}(:,2) < 1.8;
population = find(rew1_sig_transient_bool & PFC_bool & midresp_bool);
search_depth = 30; 
n_searches = 10;
timecourse_save_steps = [1 5 10 15 20 25 30]; % save timecourse information per 10 timesteps

% Perform random search over pfc sig transient cells
[rnd_pfc_mi_cumulative,rnd_pfc_mae_cumulative,rnd_pfc_timecourse_results,rnd_cells_chosen]... 
                                    = NB_rnd_search(population,n_searches,search_depth,timecourse_save_steps,... 
                                                    mIdx,i_session,iVar,iRewsize,iFeature,... 
                                                    X_dataset,y_dataset,models,xval_table,dataset_opt);

% Now perform peak time entropy maximization search
% discretize peak time so that entropy of peak time is maybe more meaningful 
rew1plus_peak_time = X_peak_pos{mIdx}{i_session}(:,2); 
[~,~,discr_rew1plus_peak] = histcounts(rew1plus_peak_time,0:.2:2);

[maxH_mi_cumulative,maxH_mae_cumulative,maxH_timecourse_results,maxH_peak_distns]... 
                                    = NB_maxH_search(population,n_searches,search_depth,timecourse_save_steps,... 
                                                    mIdx,i_session,iVar,iRewsize,iFeature,... 
                                                    X_dataset,y_dataset,models,xval_table,dataset_opt,discr_rew1plus_peak);

%% Visualize rnd vs maxH adding
bin_dt = diff(var_bins{1}(1:2));   
cmap = lines(2); 
x = 1:search_depth;
figure()
subplot(2,2,2);hold on
shadedErrorBar(x,mean(maxH_mi_cumulative),std(maxH_mi_cumulative),'lineprops',{'linewidth',1.5,'color',cmap(1,:)}) 
% plot(x,maxH_mi_cumulative','linewidth',.5,'color',cmap(1,:))
shadedErrorBar(x,mean(rnd_pfc_mi_cumulative),std(rnd_pfc_mi_cumulative),'lineprops',{'linewidth',1.5,'color',[.5 .5 .5]})
% plot(x,rnd_pfc_mi_cumulative','linewidth',.5,'color',[.5 .5 .5])
ylabel("MI (nats)") 
xlabel("Search depth")
legend(["Max Peak Distn H","Random Adding"])
subplot(2,2,4) ; hold on
shadedErrorBar(x,mean(bin_dt * maxH_mae_cumulative),std(bin_dt* maxH_mae_cumulative),'lineprops',{'linewidth',1.5,'color',cmap(1,:)})
% plot(x,maxH_mae_cumulative','linewidth',.5,'color',cmap(1,:))
shadedErrorBar(x,mean(bin_dt * rnd_pfc_mae_cumulative),std(bin_dt * rnd_pfc_mae_cumulative),'lineprops',{'linewidth',1.5,'color',[.5 .5 .5]})  
xlabel("Search depth")
% plot(x,rnd_pfc_mae_cumulative','linewidth',.5,'color',[.5 .5 .5]) 
ylabel("MAE (sec)")  

subplot(2,2,[1 3]) 
rand_peak_distns = arrayfun(@(x) cellfun(@(y) discr_rew1plus_peak(y),rnd_cells_chosen{x},'un',0),(1:n_searches)','un',0);
x = 1:search_depth;
rand_H = arrayfun(@(y) cellfun(@(x) calc_shannonH(x),rand_peak_distns{y}),(1:n_searches)','un',0);
rand_H = cat(2,rand_H{:});
maxH_H = arrayfun(@(y) cellfun(@(x) calc_shannonH(x),maxH_peak_distns{y}),(1:n_searches)','un',0); 
maxH_H = cat(2,maxH_H{:});

shadedErrorBar(x,mean(maxH_H'),std(maxH_H'),'lineprops',{'linewidth',1.5,'color',cmap(1,:)}) 
shadedErrorBar(x,mean(rand_H'),std(rand_H'),'lineprops',{'linewidth',1.5,'color',[.5 .5 .5]}) 
ylabel("Peak distribution entropy (nats)")
xlabel("Search depth") 

suptitle("Random vs Max Peak Distn H Forward Search")

%% Visualize confusion matrices across # neurons added

protocol_timecourse_results = {rnd_pfc_timecourse_results maxH_timecourse_results}; 
bin_dt = diff(var_bins{1}(1:2));
protocol_names = ["Mid-Responsive Random","Mid-Responsive max H"];
figure()
for i_protocol = 1:2 
    protocol_results = protocol_timecourse_results{i_protocol}; 
    for i_savept = 1:numel(timecourse_save_steps)  
        subplot(2,numel(timecourse_save_steps),length(timecourse_save_steps) * (i_protocol-1) + i_savept)
        if ~isempty(protocol_results{i_savept}) % if we didnt have enough cells
            pooled_confusionmat = arrayfun(@(i_search) protocol_results{i_search}{i_savept}.confusionmat,(1:n_searches)','un',0); 
            pooled_confusionmat = sum(cat(3,pooled_confusionmat{:}),3);
            imagesc(flipud(pooled_confusionmat))
        end 
        if i_protocol == 1 
            title(sprintf("%i Cells Added by Fwd Search",timecourse_save_steps(i_savept)),'FontSize',12)
        end 
        if i_savept == 1 
            ylabel(sprintf("%s Protocol \n True Time Since Rew",protocol_names(i_protocol)),'FontSize',13) 
            yticks(0:10:40) 
            yticklabels(bin_dt * fliplr((0:10:40)))  
        end
        if i_protocol == length(protocol_timecourse_results)
            xlabel("Predicted Time Since Rew",'FontSize',13) 
            xticks(0:10:40) 
            xticklabels(bin_dt * (0:10:40)) 
        end 
        if i_protocol ~= length(protocol_timecourse_results)
            xticks([]) 
        end 
        if i_savept ~= 1 
            yticks([])
        end
    end
end
