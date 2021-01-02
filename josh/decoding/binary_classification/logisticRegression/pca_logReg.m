%% Use a variety of methods to classify dimensionally reduced neural data into 2 classes 
% class 1) within .5 sec of leaving (500 ms before running cut off) 
% class 2) not within .5 sec of leaving 

% will unbalanced data be a problem? (more of class 2 than class 1) 

% The motivation for this analysis is to determine whether there exists a
% decision boundary in state space that is predictive of leaving

% Some potentially interesting questions: 
% 1. What combinations of neurons/PCs do or don't improve classification
%    accuracy? 
% 2. Does the boundary change across trial types? See if accuracy increases
%    by separating by reward size or time in session. 

%% Basics
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs 
paths.rampIDs = 'Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/ramping_neurons';

% addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% FR mat calculation settings
pcCalcOpt = struct;
pcCalcOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
pcCalcOpt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
pcCalcOpt.patch_leave_buffer = .5; % in seconds; only takes within patch times up to this amount before patch leave 
patch_leave_buffer_ms = pcCalcOpt.patch_leave_buffer * 1000;
pcCalcOpt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
pcCalcOpt.cortex_only = true;   
pcCalcOpt.onPatchOnly = true;
tbin_ms = pcCalcOpt.tbin*1000;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
% mgcPFC_sessions = [1:2 5 7]; 
% mgcSTR_sessions = [3:4 6 8:9];

%% Acquire PC reductions and a binary classification vector

% mPFC_sessions = [1:8 10:13 15:18 23 25];
classification_struct = struct; 
prop10 = nan(numel(sessions),1); 
standard_scores = cell(numel(sessions),1); 
meanUpAll_full = cell(numel(sessions),1);
meanUpCommon_full = cell(numel(sessions),1);  
expl_mat = nan(numel(sessions),40);

for sIdx = 8
    session = sessions{sIdx}; 
    pcCalcOpt.session = session;
    dat = load(fullfile(paths.data,session)); 
    ramp_fname = [paths.rampIDs '/m' sessions{sIdx}(1:end-4) '_rampIDs.mat']; 
    if isfield(dat,'anatomy')  % && exist(ramp_fname,'file')
        % initialize structs
        session = sessions{sIdx}(1:end-4);

        % load data
        fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
        good_cells = dat.sp.cids(dat.sp.cgs==2); 
        
        % subselect cortex 
        good_cells = good_cells(dat.anatomy.cell_labels.Cortex);    

        % get indices of ramping neurons 
        ramp_file = load(ramp_fname);
        ramps = ramp_file.ramps;
        ramp_up_all_ix = find(ismember(good_cells,ramps.up_all));
        ramp_up_common_ix = find(ismember(good_cells,ramps.up_common));

        % time bins
        frCalc_opt.tstart = 0;
        frCalc_opt.tend = max(dat.sp.st);

        % behavioral events to align to
        patchcue_ms = dat.patchCSL(:,1) * 1000;
        patchstop_ms = dat.patchCSL(:,2)*1000;
        patchleave_ms = dat.patchCSL(:,3)*1000;  
        prts = patchleave_ms - patchstop_ms; 
        patchType = dat.patches(:,2);
        rewsize = mod(patchType,10);  
        rew_ms = dat.rew_ts * 1000;
        patchCSL = dat.patchCSL;
        nTrials = length(patchCSL);
        
%         if ismember(sIdx,mgcPFC_sessions)
%             pcCalcOpt.region_selection = "PFC";
%             pcCalcOpt.cortex_only = false;
%         elseif ismember(sIdx,mgcSTR_sessions)
%             pcCalcOpt.region_selection = "STR";
%             pcCalcOpt.cortex_only = false;
%         else
%             disp("Warning: no region for this session")
%         end
        
        % calculate PCs, generate firing rate matrix
        [coeffs,fr_mat,good_cells,score,score_full,expl] = standard_pca_fn(paths,pcCalcOpt);  
        standard_scores{sIdx} = score_full;
        
        % create index vectors from our update timestamp vectors
        patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
        patchleave_ix = round((patchleave_ms - patch_leave_buffer_ms) / tbin_ms) + 1;

        % Put PCA results and firing rate matrix into classification struct 
        classification_struct(sIdx).fr_mat_raw = cell(nTrials,1); 
        classification_struct(sIdx).PCs = cell(nTrials,1); 
        classification_struct(sIdx).meanRamp_upAll = cell(nTrials,1);
        classification_struct(sIdx).meanRamp_upCommon = cell(nTrials,1);
        meanRamp_upAll = mean(fr_mat(ramp_up_all_ix,:),1);
        meanRamp_upCommon = mean(fr_mat(ramp_up_common_ix,:),1); 
        for iTrial = 1:nTrials
            classification_struct(sIdx).fr_mat_raw{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial)); 
            classification_struct(sIdx).PCs{iTrial} = score_full(1:10,patchstop_ix(iTrial):patchleave_ix(iTrial));
            classification_struct(sIdx).meanRamp_upAll{iTrial} = meanRamp_upAll(patchstop_ix(iTrial):patchleave_ix(iTrial));
            classification_struct(sIdx).meanRamp_upCommon{iTrial} = meanRamp_upCommon(patchstop_ix(iTrial):patchleave_ix(iTrial));
        end

        fr_mat_onPatch = horzcat(classification_struct(sIdx).fr_mat_raw{:}); 
        
        meanUpAll_full{sIdx} = meanRamp_upAll;
        meanUpCommon_full{sIdx} = meanRamp_upCommon;

        fprintf("Proportion Variance explained by first 10 PCs: %f \n",sum(expl(1:10)) / sum(expl))
        expl_mat(sIdx,:) = expl(1:40);
        prop10(sIdx) = sum(expl(1:10)) / sum(expl);

        % Get reward timings, prepare classification struct 
        t_lens = cellfun(@(x) size(x,2),classification_struct(sIdx).fr_mat_raw); 
        new_patchleave_ix = cumsum(t_lens);
        new_patchstop_ix = new_patchleave_ix - t_lens + 1; 
        classification_zone = 1000; % how much time before leave we're labeling in ms
        classification_struct(sIdx).rew_ix = cell(nTrials,1); 
        classification_struct(sIdx).labels = cell(nTrials,1); 
        classification_struct(sIdx).vel = cell(nTrials,1); 

        pre_rew_buffer = classification_zone + patch_leave_buffer_ms;

        for iTrial = 1:nTrials
            rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial));
            classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms); 
            classification_struct(sIdx).labels{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - classification_zone / tbin_ms); 
            classification_struct(sIdx).vel{iTrial} = dat.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));   
            classification_struct(sIdx).rewsize{iTrial} = zeros(t_lens(iTrial),1) + rewsize(iTrial);

            % now take out timesteps that came right before reward to better train regression
            pre_rew_label = zeros(t_lens(iTrial),1);  
            rew_ix = classification_struct(sIdx).rew_ix{iTrial}; 
            for iRew_ix = 1:numel(classification_struct(sIdx).rew_ix{iTrial}) 
                pre_rew_label(max(1,(rew_ix(iRew_ix) - pre_rew_buffer / tbin_ms)) : rew_ix(iRew_ix)) = 1; % take off the full second of activity
            end

            non_pre_rew = find(pre_rew_label == 0);
            classification_struct(sIdx).PCs_noPreRew{iTrial} = classification_struct(sIdx).PCs{iTrial}(:,non_pre_rew);
            classification_struct(sIdx).labels_noPreRew{iTrial} = classification_struct(sIdx).labels{iTrial}(non_pre_rew);
            classification_struct(sIdx).meanRamp_upAll_noPreRew{iTrial} =  classification_struct(sIdx).meanRamp_upAll{iTrial}(non_pre_rew);
            classification_struct(sIdx).meanRamp_upCommon_noPreRew{iTrial} = classification_struct(sIdx).meanRamp_upCommon{iTrial}(non_pre_rew);
            classification_struct(sIdx).vel_noPreRew{iTrial} = classification_struct(sIdx).vel{iTrial}(non_pre_rew); 
            classification_struct(sIdx).rewsize_noPreRew{iTrial} = classification_struct(sIdx).rewsize{iTrial}(non_pre_rew);
        end
    end
end

%% Variance explained visualization   
close all
figure() 
histogram(prop10,6)
xlim([0,1])
ylim([0,10])
xlabel("Variance Explained by Top 10 PCs")
ylabel("Session Density")
title("Distribution of Variance Explained by Top 10 PCs Across Sessions")

figure(); hold on
plot((expl_mat(:,1:20) ./ sum(expl_mat,2))','k--')
plot(nanmean((expl_mat(:,1:20) ./ sum(expl_mat,2))',2),'b','linewidth',2) 
xline(10,'k--','linewidth',2) 
ylabel("Proportion Variance Explained") 
xlabel("PC#") 
title("Cross-Session Variance Explained Decay")


%% Generate "reward barcodes" to average firing rates  
rew_barcodes = cell(numel(sessions),1);
for sIdx = 8
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % Trial data
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(patches(:,2),10);
    
    % make barcode matrices also want to know where we have no more rewards
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    rew_barcodes{sIdx} = rew_barcode;
end

%% Visualize the classification problem on a few single trials  
close all
for sIdx = 8
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    test_trials = 1:9;  
    sp_counter = 1;
%     figure()
%     for iTrial = test_trials 
%         subplot(3,3,sp_counter) 
%         gscatter(classification_struct(sIdx).PCs{iTrial}(1,:), ...
%                  classification_struct(sIdx).PCs{iTrial}(3,:), ...
%                  classification_struct(sIdx).labels{iTrial}, ... 
%                  [],[],5)  
% %         grid()
%         title(sprintf("Trial %i",iTrial)) 
%         xlabel("PC1"); ylabel("PC3")  
%         xlim([-10 10]); ylim([-10 10])
% %         legend("Stay","Leave in 500-1000 msec") 
%         sp_counter = sp_counter + 1; 
%         b = gca; legend(b,'off');
%     end 

    % concatenate to show cross trial data
%     concat_PCs = classification_struct(sIdx).PCs_noPreRew(test_trials);
%     concat_PCs = horzcat(concat_PCs{:}); 
%     concat_labels = classification_struct(sIdx).labels_noPreRew(test_trials); 
%     concat_labels = horzcat(concat_labels{:}); 
%     figure() 
%     gscatter(concat_PCs(1,:),concat_PCs(3,:),concat_labels,[0 0 0; 1 0 0]) 
%     xlabel("PC1"); ylabel("PC2") 
%     title("Labeled Points in PC Space") 
%     legend("Stay","Leave in 500-1500 msec") 
    
    % total concat pca 
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs_noPreRew{:}); 
    all_concat_labels = horzcat(classification_struct(sIdx).labels_noPreRew{:});  
    figure() 
    gscatter(all_concat_PCs(1,:),all_concat_PCs(2,:),all_concat_labels,[0 0 0; 1 0 0],[],2)  
    xlabel("PC1"); ylabel("PC2") 
    title(sprintf("%s Labeled Points in PC Space",session_title)) 
    legend("Stay","Leave in 500-1500 msec")  
    
%     figure() 
%     % now look at velocity  
%     sp_counter = 1;
%     for iTrial = test_trials 
%         subplot(3,3,sp_counter) 
%         t_len = numel(classification_struct(sIdx).vel{iTrial});
%         gscatter(1:t_len,classification_struct(sIdx).vel{iTrial},classification_struct(sIdx).labels{iTrial},[],[],2)   
%         title(sprintf("Trial %i",iTrial)) 
%         xlabel("Time"); 
%         ylabel("Velocity")
%         sp_counter = sp_counter + 1; 
%         b = gca; legend(b,'off');
%     end 
%     
%     figure() 
%     % now look at mean ramp
%     sp_counter = 1;
%     for iTrial = test_trials 
%         subplot(3,3,sp_counter) 
%         t_len = numel(classification_struct(sIdx).meanRamp_upAll{iTrial});
%         gscatter(1:t_len,classification_struct(sIdx).meanRamp_upAll{iTrial},classification_struct(sIdx).labels{iTrial},[],[],2)   
%         title(sprintf("Trial %i",iTrial)) 
%         xlabel("Time"); 
%         ylabel("Mean Ramping Activity")
%         sp_counter = sp_counter + 1; 
%         b = gca; legend(b,'off');
%     end  
    
%     figure() 
%     % now look at mean ramp point cloud for all trials
%     all_concat_meanRamp_upAll_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upAll_noPreRew{:});  
%     all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1; 
%     session_len = numel(all_concat_meanRamp_upAll_noPreRew);
%     gscatter(1:session_len,all_concat_meanRamp_upAll_noPreRew,all_concat_labels_noPreRew,[],[],2)   
%     xlabel("Time"); 
%     ylabel("Mean Ramping Activity")
%     b = gca; legend(b,'off'); 
%     
%     figure() 
%     % now look at PC1 point cloud for all trials
%     gscatter(1:session_len,all_concat_PCs(1,:),all_concat_labels_noPreRew,[],[],2)   
%     xlabel("Time"); 
%     ylabel("PC1 Activity Activity")
%     b = gca; legend(b,'off'); 
    
end  

%% Perform logistic regression on mean ramping activity
close all
for sIdx = 25 
    all_concat_meanRamp_upAll_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upAll_noPreRew{:}); 
    mu_noPreRew = mean(all_concat_meanRamp_upAll_noPreRew); 
    std_noPreRew = std(all_concat_meanRamp_upAll_noPreRew); 
    all_concat_meanRamp_upAll_noPreRew = (all_concat_meanRamp_upAll_noPreRew - mu_noPreRew) / std_noPreRew;
    all_concat_meanRamp_upAll= (horzcat(classification_struct(sIdx).meanRamp_upAll{:}) - mu_noPreRew) / std_noPreRew;
    all_concat_meanRamp_upCommon_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upCommon_noPreRew{:});
    session_len = size(all_concat_meanRamp_upAll,2);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1; 
    [B,dev,stats] = mnrfit(all_concat_meanRamp_upAll_noPreRew,all_concat_labels_noPreRew);   
    pi_hat = mnrval(B,all_concat_meanRamp_upAll'); % get probability output
    
    figure();colormap('hot')
    scatter(1:session_len,all_concat_meanRamp_upAll,3,pi_hat(:,2)','.');  
    colorbar()
    xlabel("Time in session"); 
    ylabel("Mean Ramp Value")
    title("P(leave in .5-1.5 sec | mean ramping activity (all))")  
    xl = xlim;
    yl = ylim;
end   

%% Now perform true forward search protocol 
%  question: even in seemingly messy/high-dimensional sessions, do we have
%  predictability in just a few dimensions if we look at the right ones? 
close all  
forward_search = struct;
for sIdx = 8
    if ~isempty(classification_struct(sIdx).fr_mat_raw)
        session = sessions{sIdx}(1:end-4); 
        session_title = sessions{sIdx}([1:2 end-6:end-4]);  
        fprintf("Starting %s \n",session_title)
        data = load(fullfile(paths.data,session)); 
        patches = data.patches;
        patchCSL = data.patchCSL; 
        prts = patchCSL(:,3) - patchCSL(:,2); 
        nTrials = length(prts);
        patchType = patches(:,2);
        rewsize = mod(patchType,10);  

%         all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:});   
%         all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;  
%         all_concat_rewsize_noPrewRew = vertcat(classification_struct(sIdx).rewsize_noPreRew{:})'; 
%         all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:}); 

        % folds are over trials
%         points = 1:numel(all_concat_rewsize_noPrewRew); 
        trials = 1:nTrials;

        % make folds
        xval_opt = struct;
        xval_opt.numFolds = 10;
        xval_opt.rew_size = [1,2,4];
        % split trials into groups (num groups = opt.numFolds)
        [trials,~,IC] = unique(trials);
        data_grp = nan(size(trials)); 
        
        data_grp = nan(nTrials,1); 
        shift_by = 0; % to make sure equal numbers of trials end up in each fold
        for i = 1:numel(xval_opt.rew_size)
            keep_this = rewsize == xval_opt.rew_size(i);
            data_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
            data_grp(keep_this) = data_grp_this(1:sum(keep_this)); % assign folds 1:10
            shift_by = shift_by - mod(sum(keep_this),xval_opt.numFolds); % shift which fold is getting fewer trials
        end
        foldid = data_grp(IC)';  
        
        threshold_step = .05;
        thresholds = 0:threshold_step:1; 

        new_xval = true;
        if new_xval == true  
            % First get a baseline for performance by taking logreg on velocity
            accuracies_vel = nan(xval_opt.numFolds,numel(thresholds));
            precisions_vel = nan(xval_opt.numFolds,numel(thresholds));
            TP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
            FP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
            ROC_AUC_vel = nan(xval_opt.numFolds,1); 
            PR_AUC_vel = nan(xval_opt.numFolds,1);
            % Velocity classification to have comparison
            for fIdx = 1:xval_opt.numFolds
                %%%% separate training and test data %%%
                X_train = classification_struct(sIdx).vel_noPreRew(foldid~=fIdx);  
                X_train = cat(2,X_train{:})';
                y_train = classification_struct(sIdx).labels_noPreRew(foldid~=fIdx); 
                y_train = cat(2,y_train{:})' + 1; 
                
                X_test = classification_struct(sIdx).vel_noPreRew(foldid==fIdx);  
                X_test = cat(2,X_test{:})';
                y_test = classification_struct(sIdx).labels_noPreRew(foldid==fIdx); 
                y_test = cat(2,y_test{:})' + 1;

                % now fit logistic regression to our training data
                [B,~,~] = mnrfit(X_train,y_train);
                pi_test = mnrval(B,X_test);

                for tIdx = 1:numel(thresholds)
                    threshold = thresholds(tIdx);
                    model_labels = double(pi_test(:,2) > threshold);
                    cm = confusionmat(y_test - 1,model_labels);
                    TN = cm(1,1);
                    FN = cm(2,1);
                    TP = cm(2,2);
                    FP = cm(1,2);

                    % classification performance metrics
                    accuracy = (TP + TN) / sum(cm(:));
                    precision = TP / (TP + FP); % precision: P(Yhat = 1 | Y = 1)
                    TP_rate = TP / (TP + FN); % sensitivity or recall:  P(Yhat = 1 | Y = 1)
                    FP_rate = FP / (TN + FP); % 1 - sensitivity: P(Yhat = 1 | Y = 0)

                    % log metrics
                    accuracies_vel(fIdx,tIdx) = accuracy;
                    precisions_vel(fIdx,tIdx) = precision;
                    TP_rates_vel(fIdx,tIdx) = TP_rate;
                    FP_rates_vel(fIdx,tIdx) = FP_rate;
                end 

                ROC_AUC_dx_vel = -squeeze(diff(FP_rates_vel(fIdx,:)));
                ROC_AUC_vel(fIdx) = sum(ROC_AUC_dx_vel .* TP_rates_vel(fIdx,1:end-1)); 
                PR_AUC_dx_vel = -squeeze(diff(TP_rates_vel(fIdx,:)));
                PR_AUC_vel(fIdx) = sum(PR_AUC_dx_vel(~isnan(precisions_vel(fIdx,1:end-1))) .* precisions_vel(fIdx,~isnan(precisions_vel(fIdx,1:end-1))));
            end 

            % Set the goal criteria
            ROC_AUC_vel_95CI = mean(ROC_AUC_vel) + 1.95 * std(ROC_AUC_vel);
            PR_AUC_vel_95CI = mean(PR_AUC_vel) + 1.95 * std(PR_AUC_vel); 

            % store velocity decoding results 
            forward_search(sIdx).precisions_vel = precisions_vel;
            forward_search(sIdx).TP_rates_vel = TP_rates_vel;
            forward_search(sIdx).FP_rates_vel = FP_rates_vel;
            forward_search(sIdx).ROC_AUC_vel = ROC_AUC_vel;
            forward_search(sIdx).PR_AUC_vel = PR_AUC_vel;

            % Variables to keep track of forward search progress
            best_ROC_AUC = 0;
            best_PR_AUC = 0;
            pcs_left = 1:size(all_concat_PCs_noPreRew,1); % the PCs that are left to pick
            pcs_picked = []; % store PCs that we are keeping in forward search 
            surpass_vel_nPCs = nan;  
            % initialize struct session struct
            forward_search(sIdx).precisions = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
            forward_search(sIdx).TP_rates = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
            forward_search(sIdx).FP_rates = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
            forward_search(sIdx).ROC_AUC = nan(numel(pcs_left),xval_opt.numFolds);
            forward_search(sIdx).PR_AUC = nan(numel(pcs_left),xval_opt.numFolds);

            % while (we haven't looked at 3 PCs) or (the best AUCs are less than vel and we still have PCs to look at) 
            % while (length(pcs_picked) < 3) || ((best_ROC_AUC < ROC_AUC_vel_95CI) || (best_PR_AUC < PR_AUC_vel_95CI) && ~isempty(pcs_left))
            while ~isempty(pcs_left) % just iterate over all 10 and collect data to make a proper forward search
                % set up datastructures to measure classification fidelity
                precisions = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
                TP_rates = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
                FP_rates = nan(numel(pcs_left),xval_opt.numFolds,numel(thresholds));
                ROC_AUC = nan(numel(pcs_left),xval_opt.numFolds);
                PR_AUC = nan(numel(pcs_left),xval_opt.numFolds);

                % iterate over pcs left and find the best one to add in terms
                % of kfold xval ROC and PR AUC
                for pcIdx = 1:numel(pcs_left) 
                    pcs_picked_tmp = [pcs_picked pcs_left(pcIdx)];

                    % Iterate over folds to use as test data
                    for fIdx = 1:xval_opt.numFolds
                        % separate training and test data
%                         data_train = all_concat_PCs_noPreRew(pcs_picked_tmp,foldid~=fIdx);
%                         labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
%                         data_test = all_concat_PCs_noPreRew(pcs_picked_tmp,foldid==fIdx);
%                         labels_test = all_concat_labels_noPreRew(foldid==fIdx); 
                        
                        X_train = classification_struct(sIdx).PCs_noPreRew(foldid~=fIdx);  
                        X_train = cat(2,X_train{:})'; 
                        X_train = X_train(:,pcs_picked_tmp);
                        y_train = classification_struct(sIdx).labels_noPreRew(foldid~=fIdx); 
                        y_train = cat(2,y_train{:})' + 1; 

                        X_test = classification_struct(sIdx).PCs_noPreRew(foldid==fIdx);  
                        X_test = cat(2,X_test{:})'; 
                        X_test = X_test(:,pcs_picked_tmp);
                        y_test = classification_struct(sIdx).labels_noPreRew(foldid==fIdx); 
                        y_test = cat(2,y_test{:})' + 1;

                        % now fit logistic regression to our training data
                        [B,~,~] = mnrfit(X_train,y_train);
                        pi_test = mnrval(B,X_test);

                        for tIdx = 1:numel(thresholds)
                            threshold = thresholds(tIdx);
                            model_labels = double(pi_test(:,2) > threshold);
                            cm = confusionmat(y_test' - 1,model_labels);
                            TN = cm(1,1);
                            FN = cm(2,1);
                            TP = cm(2,2);
                            FP = cm(1,2);

                            % classification performance metrics
                            accuracy = (TP + TN) / sum(cm(:));
                            precision = TP / (TP + FP); % precision: P(Yhat = 1 | Y = 1)
                            TP_rate = TP / (TP + FN); % sensitivity or recall:  P(Yhat = 1 | Y = 1)
                            FP_rate = FP / (TN + FP); % 1 - sensitivity: P(Yhat = 1 | Y = 0)

                            % log performance metrics
                            precisions(pcIdx,fIdx,tIdx) = precision;
                            TP_rates(pcIdx,fIdx,tIdx) = TP_rate;
                            FP_rates(pcIdx,fIdx,tIdx) = FP_rate;
                        end

                        ROC_AUC_dx = -squeeze(diff(FP_rates(pcIdx,fIdx,:)));
                        ROC_AUC(pcIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates(pcIdx,fIdx,1:end-1)));
                        PR_AUC_dx = -squeeze(diff(TP_rates(pcIdx,fIdx,:)));
                        PR_AUC(pcIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions(pcIdx,fIdx,1:end-1))) .* squeeze(precisions(pcIdx,fIdx,~isnan(precisions(pcIdx,fIdx,1:end-1)))));
                    end
                    if mod(pcIdx,2) == 0
                        fprintf("PC %i/%i Complete \n",pcIdx,numel(pcs_left))
                    end
                end

                % take mean across folds
                mean_ROC_AUC = mean(ROC_AUC,2); 
                mean_PR_AUC = mean(PR_AUC,2); 
                % select the best candidate and update search variables
                pc_new_ix = argmax(mean_PR_AUC); % choose best in PR because this is more competitive 
                pc_new = pcs_left(pc_new_ix);
                best_ROC_AUC = max(best_ROC_AUC,mean_ROC_AUC(pc_new_ix));
                best_PR_AUC = max(best_PR_AUC,mean_PR_AUC(pc_new_ix)); 
                pcs_left = setdiff(pcs_left,pc_new); 
                pcs_picked = [pcs_picked pc_new];   

                % log how many PCs we needed to do better than velocity
                if (best_ROC_AUC > ROC_AUC_vel_95CI) && (best_PR_AUC > PR_AUC_vel_95CI) && isnan(surpass_vel_nPCs)
                    surpass_vel_nPCs = numel(pcs_picked); 
                end 

                % log the ROC and PR information for the best choice 
                forward_search(sIdx).precisions(numel(pcs_picked),:,:) = squeeze(precisions(pc_new_ix,:,:));
                forward_search(sIdx).TP_rates(numel(pcs_picked),:,:) = squeeze(TP_rates(pc_new_ix,:,:)); 
                forward_search(sIdx).FP_rates(numel(pcs_picked),:,:) = squeeze(FP_rates(pc_new_ix,:,:)); 
                forward_search(sIdx).ROC_AUC(numel(pcs_picked),:) = squeeze(ROC_AUC(pc_new_ix,:)); 
                forward_search(sIdx).PR_AUC(numel(pcs_picked),:) = squeeze(PR_AUC(pc_new_ix,:));
            end
        end

        % log forward search data to struct
        forward_search(sIdx).pc_decodingOrder = pcs_picked;  
        forward_search(sIdx).surpass_vel_nPCs = surpass_vel_nPCs; 
        forward_search(sIdx).best_ROC_AUC = best_ROC_AUC; 
        forward_search(sIdx).best_PR_AUC = best_PR_AUC;

        fprintf("Session %s PC decoding order: \n",session_title) 
        disp(forward_search(sIdx).pc_decodingOrder) 

        fprintf("Session %s PCs to surpass velocity fidelity: \n",session_title) 
        disp(forward_search(sIdx).surpass_vel_nPCs)  
    end
end

%% Visualize results of forward search 
close all  
figcounter = 1;
pc_ranges = 1:10;
for sIdx = 8
    session = sessions{sIdx}(1:end-4); 
    session_title = sessions{sIdx}([1:2 end-6:end-4]); 
    
    forwardSearchVis(forward_search,classification_struct,sIdx,session_title,figcounter)
    
    figcounter = figcounter + 2; 
end   

%% Visualize forward search results across sessions 
close all
forward_search_cell = squeeze(struct2cell(forward_search))';
pcs_to_surpass = cat(1,forward_search_cell{:,12});   
best_auROC = cat(1,forward_search_cell{:,13});   
best_auPR = cat(1,forward_search_cell{:,14});   
mean_auROC_vel = mean(cat(2,forward_search_cell{:,4}));
mean_auPR_vel = mean(cat(2,forward_search_cell{:,5}));

figure() 
histogram(pcs_to_surpass) 
xlim([0 10]) 
ylim([0 10]) 
title("Distribution of PCs Needed to Surpass Velocity auROC and auPR") 
xlabel("#PCs Needed to Surpass Velocity auROC and auPR") 
ylabel("Session Density") 

% mouse_groups = {1:2,3:8,9:12,13:16,17:18}; % {1:2,3:8,10:13,15:18,[23 25]};   
mouse_groups = {1:2,3,[5 7],[4 6 8:9]}; % for malcolm
mean_mouse_auROC_vel = nan(numel(mouse_groups),1); 
mean_mouse_auROC_best = nan(numel(mouse_groups),1);
mean_mouse_auPR_vel = nan(numel(mouse_groups),1); 
mean_mouse_auPR_best = nan(numel(mouse_groups),1); 
for m = 1:numel(mouse_groups) 
    mean_mouse_auROC_vel(m) = mean(mean_auROC_vel(mouse_groups{m}));
    mean_mouse_auPR_vel(m) = mean(mean_auPR_vel(mouse_groups{m}));
    mean_mouse_auROC_best(m) = mean(best_auROC(mouse_groups{m}));
    mean_mouse_auPR_best(m) = mean(best_auPR(mouse_groups{m}));
end 

% mouse_names = ["m75","m76","m78","m79","m80"]; 
mouse_names = ["mc2 PFC","mc2 STR","mc4 PFC","mc4 STR"];  
figure()  
subplot(1,2,1);hold on
b = bar([mean_mouse_auROC_vel mean_mouse_auROC_best],'FaceColor','flat'); 
b(1).CData = [.5 .5 .5];  
b(2).CData = [0.8500    0.3250    0.2] + .1; 
ylim([.65,1]) 
x_vel = b(1).XEndPoints;
x_best = b(2).XEndPoints;  
for m = 1:numel(mouse_groups)  
    scatter(x_vel(m) + zeros(numel(mouse_groups{m}),1),mean_auROC_vel(mouse_groups{m}),20,'filled','d','MarkerFaceColor','k','MarkerEdgeColor','w')
    scatter(x_best(m) + zeros(numel(mouse_groups{m}),1),best_auROC(mouse_groups{m}),20,'filled','d','MarkerFaceColor','r','MarkerEdgeColor','k')
end 
xticklabels(mouse_names)  
ylabel("auROC")
legend(b(1:2),["Velocity","Best PC Classifier"]) 
title("Velocity vs Best PC Classifier auROC")

subplot(1,2,2);hold on
b = bar([mean_mouse_auPR_vel mean_mouse_auPR_best],'FaceColor','flat');
b(1).CData = [.5 .5 .5]; 
b(2).CData = [0.8500    0.3250    0.2] + .1; 
ylim([.45,1]) 
x_vel = b(1).XEndPoints;
x_best = b(2).XEndPoints;  
for m = 1:numel(mouse_groups)  
    scatter(x_vel(m) + zeros(numel(mouse_groups{m}),1),mean_auPR_vel(mouse_groups{m}),20,'filled','d','MarkerFaceColor','k','MarkerEdgeColor','w')
    scatter(x_best(m) + zeros(numel(mouse_groups{m}),1),best_auPR(mouse_groups{m}),20,'filled','d','MarkerFaceColor','r','MarkerEdgeColor','k')
end
xticklabels(mouse_names) 
ylabel("auPR")
legend(b(1:2),["Velocity","Best PC Classifier"]) 
title("Velocity vs Best PC Classifier auPR") 

%% Perform logistic regression on labelled PCs from forward search 
close all
for i = 1:8
    sIdx = mPFC_sessions(i);
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})';   
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';  
    session_len = size(all_concat_PCs,1);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1; 
    if exist('forward_search','var') 
        if ~isnan(forward_search(sIdx).surpass_vel_nPCs) 
            pcs_to_use = forward_search(sIdx).pc_decodingOrder(1:max(2,forward_search(sIdx).surpass_vel_nPCs));
        else 
            pcs_to_use = 1:10; 
            disp("We didn't surpass velocity decoding")
        end
    else
        pcs_to_use = 1:10;
    end
    
    [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew(:,pcs_to_use),all_concat_labels_noPreRew);   
    [~,pc_sort_by_logW] = sort(abs(B(2:end)),'descend');
    pi_hat = mnrval(B,all_concat_PCs(:,pcs_to_use));    
    
    if isequal(pcs_to_use,1:10)
        decode_pc1 = pc_sort_by_logW(1); 
        decode_pc2 = pc_sort_by_logW(2);  
        decode_pc3 = pc_sort_by_logW(3);  
    else
        decode_pc1 = pcs_to_use(1); 
        decode_pc2 = pcs_to_use(2); 
    end
    
    figure();colormap('hot')
    scatter(all_concat_PCs(:,decode_pc1),all_concat_PCs(:,decode_pc2),3,pi_hat(:,2)','.');  
    colorbar()
    xlabel(sprintf("PC%i",decode_pc1));
    ylabel(sprintf("PC%i",decode_pc2));  
    title("P(leave in .5-1.5 sec | PC1:10)")  
    xl = xlim;
    yl = ylim;  
    
    % now add p_leave to our classification struct 
    t_lens = cellfun(@(x) size(x,2),classification_struct(sIdx).fr_mat_raw); 
    patchleave_ix = cumsum(t_lens);
    patchstop_ix = patchleave_ix - t_lens + 1;  
    for iTrial = 1:numel(t_lens)
        classification_struct(sIdx).p_leave{iTrial} = pi_hat(patchstop_ix(iTrial):patchleave_ix(iTrial),2);
    end
end   

%% Visualize forward search PCs using HGK plot_timecourse 
close all
% mPFC_sessions = [1:8 10:13 15:18 23 25];
for sIdx = 1:numel(sessions)
%     sIdx = mPFC_sessions(i); 
    rew_barcode = rew_barcodes{sIdx};
    session = sessions{sIdx}(1:end-4); 
%     session_title = ['m' session(1:2) ' ' session(end-2) '/' session([end-1:end])];  
    session_title = [session(1:3) ' ' session(end-3:end-2) '/' session(end-1:end)]; % malcolm mice
    figpath = [paths.figs '/logRegPCs/m' session([1:2 end-2:end]) '_PCs.png']; 
    data = load(fullfile(paths.data,session)); 
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2) * 1000;
    patchleave_ms = data.patchCSL(:,3) * 1000;
    rew_size = mod(data.patches(:,2),10);
    
    decoding_order = forward_search(sIdx).pc_decodingOrder;
    score = standard_scores{sIdx}(forward_search(sIdx).pc_decodingOrder(1:6),:);
    
    % Make 4X group 
    RX_group = nan(length(patchstop_ms),1);
    trials40 = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 0);
    trials44 = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4); 
    RX_group(trials40) = 1; 
    RX_group(trials44) = 2; 
    
    % alignment values:
    % stop
    t_align{2} = patchstop_ms;
    t_start{2} = patchstop_ms;
    
    % leave
    t_align{3} = patchleave_ms;
    t_end{3} = patchleave_ms+1000;
    
    % for plotting up to X # of seconds max, w attrition for trials w lower PRTs
    for i = 1:20
        t_endmax = patchleave_ms - 500;
        t_endmax(patchleave_ms > i*1000 + patchstop_ms) = patchstop_ms(patchleave_ms > i*1000 + patchstop_ms) + i*1000;
        t_end{2}{i} = t_endmax;
        
        t_startmax = patchstop_ms;
        t_startmax(patchstop_ms < patchleave_ms - i*1000) = patchleave_ms(patchstop_ms < patchleave_ms - i*1000) - i*1000;
        t_start{3}{i} = t_startmax;
    end
    
    % grouping variables
    gr.uL = rew_size; 
    gr.RX_group = RX_group;
    
    % global variable for use w plot_timecourse
    global gP
    gP.cmap{3} = cool(3);
    
    % plot 6 PCs
    maxTime = 3; %
    
    hfig = figure('Position',[100 100 2300 700]);
    hfig.Name = 'test_plot';
    
    aIdx = 2;
    for pIdx = 1:6
        subplot(3,6,pIdx);
        PC6 = plot_timecourse('stream',score(pIdx,:),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}{maxTime}/tbin_ms,gr.uL,'resample_bin',1);
        
        PC6(2).XTick = [0 .05 .1];
        PC6(2).XTickLabel = {[0 1 2]};
        PC6(2).XLabel.String = 'time since patch stop (s)';
        
        PC6(2).Legend.String = {['1uL'] ['2uL'] ['4uL']};  
        if pIdx >= forward_search(sIdx).surpass_vel_nPCs
            title(['**PC' num2str(decoding_order(pIdx)) '**'],'FontSize',18);  
        else 
            title(['PC' num2str(decoding_order(pIdx))],'FontSize',18);
        end
    end
    
    aIdx = 3; maxTime = 3;
    for pIdx = 1:6
        subplot(3,6,pIdx+6);
        PC6 = plot_timecourse('stream',score(pIdx,:),t_align{aIdx}/tbin_ms,t_start{aIdx}{maxTime}/tbin_ms,t_end{aIdx}/tbin_ms,gr.uL,'resample_bin',1);
        
        PC6(2).XTick = [-.1 -.05 0];
        PC6(2).XTickLabel = {[-2 -1 0]};
        PC6(2).XLabel.String = 'time before leave (s)';
        
        PC6(2).Legend.String = {['1uL'] ['2uL'] ['4uL']};
        if pIdx >= forward_search(sIdx).surpass_vel_nPCs
            title(['**PC' num2str(decoding_order(pIdx)) '**'],'FontSize',18);  
        else 
            title(['PC' num2str(decoding_order(pIdx))],'FontSize',18);
        end
    end

    gP.cmap{2} = [0 0 0 ; 1 0 1];
    
    aIdx = 2; maxTime = 3;
    for pIdx = 1:6
        subplot(3,6,pIdx+12);
        PC6 = plot_timecourse('stream',score(pIdx,:),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}{maxTime}/tbin_ms,gr.RX_group,'resample_bin',1);
        
        PC6(2).XTick = [0 .05 .1];
        PC6(2).XTickLabel = {[0 1 2]};
        PC6(2).XLabel.String = 'time since patch stop (s)';
        
        PC6(2).Legend.String = {['40'] ['44']};
        if pIdx >= forward_search(sIdx).surpass_vel_nPCs
            title(['**PC' num2str(decoding_order(pIdx)) '**'],'FontSize',18);  
        else 
            title(['PC' num2str(decoding_order(pIdx))],'FontSize',18);
        end
    end
    
    suptitle(session_title)  
    saveas(gcf,figpath)
    close(gcf)
end

%% Look at P(leave) on single trials (decision variable structure?) 
close all
for sIdx = 24:24
    n_trials = length(classification_struct(sIdx).fr_mat_raw);
    test_trials = randsample(n_trials,9);
    
    %%% add ticks to show reward deliver locations, fix x axis
    sp_counter = 1; 
    figure()
    for i = 1:numel(test_trials) 
        subplot(3,3,sp_counter)
        iTrial = test_trials(i);  
        plot(classification_struct(sIdx).p_leave{iTrial},'linewidth',2)   
    
        if ~isempty(classification_struct(sIdx).rew_ix{iTrial}) 
            for iRew = 1:numel(classification_struct(sIdx).rew_ix{iTrial})
                xline(classification_struct(sIdx).rew_ix{iTrial}(iRew),'k--','linewidth',1.5)  
            end
        end
        title(sprintf("Trial %i",iTrial))   
        ylabel("P(Leave)") 
        xlabel("Time (sec)")  
        t_len = length(classification_struct(sIdx).p_leave{iTrial});
        xticks((1:(1/tbin_ms * 1000):t_len) - 1) 
        xticklabels(((1:(1/tbin_ms):t_len) - 1) * tbin_ms)
        sp_counter = sp_counter + 1;
    end
end 

%% P(leave) on RXX trials 
close all
for sIdx = 17
    session = sessions{sIdx}(1:end-4); 
    session_title = ['m' sessions{sIdx}([1:2]) ' ' sessions{sIdx}(end-6:end-4)];
    data = load(fullfile(paths.data,session));
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3); 
    prts = patchleave_ms - patchstop_ms;
    rew_ms = data.rew_ts;

    sec3ix = 3000/tbin_ms;
    
    rew_barcode = rew_barcodes{sIdx};
  
    rew_counter = 1;

    figure() 
    for iRewsize = [2,4] 
        trialsr00x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.5);
        trialsrr0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.5);
        trialsr0rx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.5);
        trialsrrrx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.5); 
        
        temp_prob_mat = {length(trialsr00x)};
        for j = 1:numel(trialsr00x)
            iTrial = trialsr00x(j);
            temp_prob_mat{j} = classification_struct(sIdx).p_leave{iTrial}(1:sec3ix);
        end
        mean_pLeaveR00x = mean(cat(3,temp_prob_mat{:}),3); % concatenate in third dimension, average over it
        
        temp_prob_mat = {length(trialsrr0x)};
        for j = 1:numel(trialsrr0x)
            iTrial = trialsrr0x(j);
            temp_prob_mat{j} = classification_struct(sIdx).p_leave{iTrial}(1:sec3ix);
        end
        mean_pLeaveRR0x = mean(cat(3,temp_prob_mat{:}),3); % concatenate in third dimension, average over it
        
        temp_prob_mat = {length(trialsr0rx)};
        for j = 1:numel(trialsr0rx)
            iTrial = trialsr0rx(j);
            temp_prob_mat{j} = classification_struct(sIdx).p_leave{iTrial}(1:sec3ix);
        end
        mean_pLeaveR0Rx = mean(cat(3,temp_prob_mat{:}),3); % concatenate in third dimension, average over it
        
        temp_prob_mat = {length(trialsrrrx)};
        for j = 1:numel(trialsrrrx)
            iTrial = trialsrrrx(j);
            temp_prob_mat{j} = classification_struct(sIdx).p_leave{iTrial}(1:sec3ix);
        end
        mean_pLeaveRRRx = mean(cat(3,temp_prob_mat{:}),3); % concatenate in third dimension, average over it
        
        subplot(2,2,1) 
        plot(mean_pLeaveR00x,'linewidth',2)   
        hold on
%         title(sprintf("Mean %i00 P(Leave)",iRewsize))
        subplot(2,2,2) 
        plot(mean_pLeaveRR0x,'linewidth',2)   
        hold on
%         title(sprintf("Mean %i%i0 P(Leave)",iRewsize,iRewsize))
        subplot(2,2,3) 
        plot(mean_pLeaveR0Rx,'linewidth',2)   
        hold on
%         title(sprintf("Mean %i0%i P(Leave)",iRewsize,iRewsize))
        subplot(2,2,4) 
        plot(mean_pLeaveRRRx,'linewidth',2)  
        hold on
%         title(sprintf("Mean %i%i%i P(Leave)",iRewsize,iRewsize,iRewsize))
    end 
    subplot(2,2,1) 
    legend("200","400") 
    title("Mean R00 P(Leave)") 
    xticks((1:(1/tbin_ms * 1000):sec3ix) - 1) 
    xticklabels(((1:(1/tbin_ms * 1000):sec3ix) - 1) * tbin_ms)   
    ylim([0,1])
    ylabel("P(Leave)")
    xlabel("Time (msec)")
    subplot(2,2,2) 
    legend("220","440") 
    title("Mean RR0 P(Leave)")
    xticks((1:(1/tbin_ms * 1000):sec3ix) - 1) 
    xticklabels(((1:(1/tbin_ms * 1000):sec3ix) - 1) * tbin_ms) 
    ylim([0,1])
    xlabel("Time (msec)") 
    ylabel("P(Leave)")
    subplot(2,2,3) 
    legend("202","404") 
    title("Mean R0R P(Leave)")
    xticks((1:(1/tbin_ms * 1000):sec3ix) - 1) 
    xticklabels(((1:(1/tbin_ms * 1000):sec3ix) - 1) * tbin_ms)
    ylim([0,1])
    xlabel("Time (msec)") 
    ylabel("P(Leave)")
    subplot(2,2,4) 
    legend("222","444") 
    title("Mean RRR P(Leave)")
    xticks((1:(1/tbin_ms * 1000):sec3ix) - 1) 
    xticklabels(((1:(1/tbin_ms * 1000):sec3ix) - 1) * tbin_ms)
    ylim([0,1])
    xlabel("Time (msec)") 
    ylabel("P(Leave)") 
    suptitle(session_title)
    
end 

%% Visualize P(Leave) using plot_timecourse 
for sIdx = 24:24 
    global gP
    gP.cmap{3} = cool(3);
    gr = struct;
    gr.rewsize = rewsize;
    gr.rewsize(rewsize == 4) = 3;
    maxTime = 2; % 2 seconds 
    
    concat_pLeave = classification_struct(sIdx).p_leave(:);
    
    colormap(cool(3))
    for pIdx = 1:3
        figure() 
        subplot(1,2,1)
        plot_timecourse('stream', score_full(pIdx,:),t_align{2}/tbin_ms,t_start{2}/tbin_ms,t_end{2}{maxTime}/tbin_ms, gr.rewsize, 'resample_bin',1);
        subplot(1,2,2)
        plot_timecourse('stream', score_full(pIdx,:),t_align{3}/tbin_ms,t_start{3}{maxTime}/tbin_ms,t_end{3}/tbin_ms, gr.rewsize, 'resample_bin',1);
        suptitle(sprintf("PC %i",pIdx))
    end

end

%% Visualize forward search results 

mPFC_sessions = [1:8 10:13 15:18 23 25];
surpass_vel = nan(numel(mPFC_sessions),1);

for i = 1:numel(mPFC_sessions)  
    sIdx = mPFC_sessions(i);
    surpass_vel(i) = forward_search(sIdx).surpass_vel_nPCs;
end

% surpass_vel(isnan(surpass_vel)) = 10;

figure()
h = histogram(surpass_vel);
title("Number of Forward Search PCs to Surpass Velocity AUCPR Across Sessions") 
xlabel("Number of Forward Search PCs to Surpass Velocity AUCPR")

%% Ramping neuron regressions compared to velocity

close all 
figcounter = 1;
for sIdx = 1:numel(sessions)
    if ~isempty(classification_struct(sIdx).fr_mat_raw)
        session = sessions{sIdx}(1:end-4); 
        session_title = sessions{sIdx}([1:2 end-6:end-4]);
        data = load(fullfile(paths.data,session)); 
        patches = data.patches;
        patchCSL = data.patchCSL;
        prts = patchCSL(:,3) - patchCSL(:,2);
        patchType = patches(:,2);
        rewsize = mod(patchType,10);  

        all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;  
        all_concat_rewsize_noPrewRew = vertcat(classification_struct(sIdx).rewsize_noPreRew{:})'; 
        all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:});  
        all_concat_meanRamp_upAll_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upAll_noPreRew{:});
        all_concat_meanRamp_upCommon_noPreRew = horzcat(classification_struct(sIdx).meanRamp_upCommon_noPreRew{:});

        % folds are going to be over points that did not directly precede reward
        points = 1:numel(all_concat_rewsize_noPrewRew);

        % make folds
        xval_opt = struct;
        xval_opt.numFolds = 10;
        xval_opt.rew_size = [1,2,4];
        % split trials into groups (num groups = opt.numFolds)
        [points,~,IC] = unique(points); 
        data_grp = nan(size(points));
        shift_by = 0; % to make sure equal numbers of trials end up in each fold
        % make sure all folds have roughly equal numbers of points from every rewsize
        for i = 1:numel(xval_opt.rew_size)
            keep_this = all_concat_rewsize_noPrewRew == xval_opt.rew_size(i);
            data_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
            data_grp(keep_this) = data_grp_this(1:sum(keep_this)); % assign folds 1:10
            shift_by = shift_by - mod(sum(keep_this),xval_opt.numFolds); % shift which fold is getting fewer trials
        end
        foldid = data_grp(IC)';   
        xval_opt.foldid = foldid;
        threshold_step = .05;
        thresholds = 0:threshold_step:1; 

        new_xval = true;
        if new_xval == true 
            [accuracies_vel,precisions_vel, ...
             TP_rates_vel,FP_rates_vel, ... 
             ROC_AUC_vel,PR_AUC_vel] = logReg_eval(all_concat_vel_noPreRew,all_concat_labels_noPreRew,thresholds,xval_opt); 
         
            [accuracies_ramp_all,precisions_ramp_all, ...
             TP_rates_ramp_all,FP_rates_ramp_all, ... 
             ROC_AUC_ramp_all,PR_AUC_ramp_all] = logReg_eval(all_concat_meanRamp_upAll_noPreRew,all_concat_labels_noPreRew,thresholds,xval_opt);
         
            [accuracies_ramp_common,precisions_ramp_common, ...
             TP_rates_ramp_common,FP_rates_ramp_common, ... 
             ROC_AUC_ramp_common,PR_AUC_ramp_common] = logReg_eval(all_concat_meanRamp_upCommon_noPreRew,all_concat_labels_noPreRew,thresholds,xval_opt);
        end

        figure(figcounter)  
        subplot(1,2,1) 
        hold on
        errorbar(mean(FP_rates_vel),mean(TP_rates_vel),1.96 * std(TP_rates_vel),'k','linewidth',1.5) 
        errorbar(mean(FP_rates_ramp_all),mean(TP_rates_ramp_all),1.96 * std(TP_rates_ramp_all),'linewidth',1.5)
        errorbar(mean(FP_rates_ramp_common),mean(TP_rates_ramp_common),1.96 * std(TP_rates_ramp_common),'linewidth',1.5)
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s Receiver Operator Characteristic Curve",session_title))
        plot([0,1],[0,1],'k--','linewidth',1.5) 
        ylim([0,1])
        legend("Velocity","Mean Ramping (All)","Mean Ramping (Common Threshold)","Naive Performance")
        subplot(1,2,2) ;hold on
        errorbar(mean(TP_rates_vel),mean(precisions_vel),1.96 * std(precisions_vel),'k','linewidth',1.5) 
        errorbar(mean(TP_rates_ramp_all),mean(precisions_ramp_all),1.96 * std(precisions_ramp_all),'linewidth',1.5)
        errorbar(mean(TP_rates_ramp_common),mean(precisions_ramp_common),1.96 * std(precisions_ramp_common),'linewidth',1.5)
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s Precision Recall Curve",session_title))
        yline(.5,'k--','linewidth',1.5)
        legend("Velocity","Mean Ramping (All)","Mean Ramping (Common Threshold)","Naive Performance")
        ylim([0,1])

        fprintf("Session %s Complete \n",session_title)

        figcounter = figcounter + 2; 
    end
end 
