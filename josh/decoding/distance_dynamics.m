%% Exploration of direct comparison of FR vectors between condition groups to describe dynamics
%  Aim to shore up attractor story

ths = struct;
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

%% Now concatenate matrices into a form where we have flat session-wise fr mats 
%  labeled w/ reward size, time since reward, and time until leave (going to need 2 here?)
%  then throw this in pdist squareform
time_since_rew_ix = 1; 
time_on_patch_ix = 2; 
time_until_leave_ix = 3; 
patchOnset_distance_medians = cell(length(mouse_grps),1);
patchOffset_distance_medians = cell(length(mouse_grps),1);
patchOnset_distance_variances = cell(length(mouse_grps),1);
patchOffset_distance_variances = cell(length(mouse_grps),1);

for mIdx = 1:5
    patchOnset_distance_medians{mIdx} = cell(length(mouse_grps{mIdx}),1); 
    patchOffset_distance_medians{mIdx} = cell(length(mouse_grps{mIdx}),1); 
    patchOnset_distance_variances{mIdx} = cell(length(mouse_grps{mIdx}),1); 
    patchOffset_distance_variances{mIdx} = cell(length(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        rewsize = y_rewsize{mIdx}{i};
        nTrials = length(rewsize); 
        
        % load full data for first second distance comparison
        fr_mat_full = cat(2,X{mIdx}{i,time_on_patch_ix}{:});
        time_on_patch_full = cat(2,y{mIdx}{i,time_on_patch_ix}{:}); 
        rewsize_trialed = arrayfun(@(x) rewsize(x) + zeros(1,length(y{mIdx}{i,time_on_patch_ix}{x})), (1:nTrials)','un',0);
        rewsize_full = cat(2,rewsize_trialed{:}); 
        % restrict timepoints for analysis 
        time_selection = ~isnan(time_on_patch_full);
        fr_mat_full = fr_mat_full(:,time_selection); 
        rewsize_full = rewsize_full(time_selection); 
        rewsize_full(rewsize_full == 4) = 3; % just for convenience
        time_on_patch_full = time_on_patch_full(time_selection); 
        
        % load data for pre-leave distance comparison
        fr_mat_postRew_full = cat(2,X{mIdx}{i,time_until_leave_ix}{:});
        time_until_leave_full = cat(2,y{mIdx}{i,time_until_leave_ix}{:});
        rewsize_trialed = arrayfun(@(x) rewsize(x) + zeros(1,length(y{mIdx}{i,time_until_leave_ix}{x})), (1:nTrials)','un',0);
        rewsize_postRew_full = cat(2,rewsize_trialed{:});
        % restrict timepoints for analysis
        time_selection = ~isnan(time_until_leave_full);
        fr_mat_postRew_full = fr_mat_postRew_full(:,time_selection);
        rewsize_postRew_full = rewsize_postRew_full(:,time_selection);
        rewsize_postRew_full(rewsize_postRew_full == 4) = 3;
        time_until_leave_full = time_until_leave_full(time_selection);
        
        % prep for distance calculations
        analysis_timepoints = unique(time_on_patch_full);
        neuron_selection = (1:size(fr_mat_postRew_full,1))'; % all neurons
%         neuron_selection = X_clusters{mIdx}{i} == 2; % cluster selection
        
        distance_metric = 'cosine';
        if length(find(neuron_selection)) > 10
            % now iterate over conditions for distance comparison
            for iRewsize1 = 1:3
                for iRewsize2 = 1:3
                    if iRewsize2 <= iRewsize1 % so we don't redo work
                        for i_time = 1:numel(analysis_timepoints)
                            iTime = analysis_timepoints(i_time);
                            
                            % first do patch onset
                            fr_mat1 = fr_mat_full(neuron_selection,rewsize_full == iRewsize1 & time_on_patch_full == iTime);
                            fr_mat2 = fr_mat_full(neuron_selection,rewsize_full == iRewsize2 & time_on_patch_full == iTime);
                            
                            % now concatenate matrices
                            fr_mat_concat = [fr_mat1 fr_mat2];
                            class1_ix = 1:size(fr_mat1,2);
                            class2_ix = (size(fr_mat1,2)+1):size(fr_mat_concat,2);
                            
                            % calculate distances
                            D = squareform(pdist(fr_mat_concat',distance_metric));
                            
                            % log summary stats
                            if iRewsize1 ~= iRewsize2
                                patchOnset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = median(D(class1_ix,class2_ix),'all');
                                patchOnset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = var(D(class1_ix,class2_ix),[],'all');
                            else
                                D_selected = D(class1_ix,class2_ix);
                                tril_mask = logical(tril(D_selected,-1));
                                D_selected(tril_mask) = nan;
                                patchOnset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = nanmedian(D_selected,'all');
                                patchOnset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = nanvar(D_selected,[],'all');
                            end
                            
                            % next do patch offset
                            fr_mat1 = fr_mat_postRew_full(neuron_selection,rewsize_postRew_full == iRewsize1 & time_until_leave_full == iTime);
                            fr_mat2 = fr_mat_postRew_full(neuron_selection,rewsize_postRew_full == iRewsize2 & time_until_leave_full == iTime);
                            
                            % now concatenate matrices
                            fr_mat_concat = [fr_mat1 fr_mat2];
                            class1_ix = 1:size(fr_mat1,2);
                            class2_ix = (size(fr_mat1,2)+1):size(fr_mat_concat,2);
                            
                            % calculate distances
                            D = squareform(pdist(fr_mat_concat',distance_metric));
                            
                            % log summary stats
                            if iRewsize1 ~= iRewsize2
                                patchOffset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = median(D(class1_ix,class2_ix),'all');
                                patchOffset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = var(D(class1_ix,class2_ix),[],'all');
                            else
                                D_selected = D(class1_ix,class2_ix);
                                tril_mask = logical(tril(D_selected,-1));
                                D_selected(tril_mask) = nan;
                                patchOffset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = nanmedian(D_selected,'all');
                                patchOffset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2}(i_time) = nanvar(D_selected,[],'all');
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Now visualize, first session-wise
x = var_bins{1}(1:end-1); 
cool3 = cool(3); 
hsv3 = hsv(3); 
colors = cell(3,3);
colors{1,1} = cool3(1,:); 
colors{2,2} = cool3(2,:); 
colors{3,3} = cool3(3,:); 
colors{2,1} = hsv3(1,:); 
colors{3,1} = hsv3(2,:); 
colors{3,2} = hsv3(3,:); 

close all
for mIdx = 1:5
    n_sessions = numel(mouse_grps{mIdx});
    figure()
    for i = 1:n_sessions
        if ~isempty(patchOnset_distance_medians{mIdx}{i})
            for iRewsize1 = 1:3
                for iRewsize2 = 1:3
                    if iRewsize2 <= iRewsize1 % so we don't redo work
                        subplot(2,n_sessions,i) % patch onset
                        shadedErrorBar(x,patchOnset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2},patchOnset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2},'lineprops',{'color',colors{iRewsize1,iRewsize2}})
                        ylim([0.2,.5])
                        subplot(2,n_sessions,n_sessions + i) % patch offset
                        shadedErrorBar(x,fliplr(patchOffset_distance_medians{mIdx}{i}{iRewsize1,iRewsize2}),fliplr(patchOffset_distance_variances{mIdx}{i}{iRewsize1,iRewsize2}),'lineprops',{'color',colors{iRewsize1,iRewsize2}})
                        ylim([0.2,.5])
                    end
                end
            end
        end
    end
end