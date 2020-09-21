%% Some tests to take off a certain amount of time before reward reception
%  - measure this in terms of AUC in ROC and PR
%  - prior is that this is not going to be super interesting

%% Basics
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% FR mat calculation settings
frCalc_opt = struct;
frCalc_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = frCalc_opt.tbin * 1000;
frCalc_opt.smoothSigma_time = 0.050; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};  

%% Create struct 
classification_struct = struct; 

for sIdx = 22:24
    % initialize structs
    session = sessions{sIdx}(1:end-4);
    tbin_ms = frCalc_opt.tbin*1000;
    
    % load data
    dat = load(fullfile(paths.data,session));
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    good_cells = dat.sp.cids(dat.sp.cgs==2); 
    
    % time bins
    frCalc_opt.tstart = 0;
    frCalc_opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;  
    prts = patchleave_ms - patchstop_ms; 
    patchType = dat.patches(:,2);
    rewsize = mod(patchType,10);  
    rew_ms = dat.rew_ts * 1000;
    
    % Trial level features for decision variable creation
    patchCSL = dat.patchCSL; 
    nTrials = length(patchCSL);

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, ~] = calcFRVsTime(good_cells,dat,frCalc_opt); % calc from full matrix
        toc
    end 
    
    % no buffer; deal with this directly later
    buffer = 500; % fixed t1, msec before leave to exclude in analysis of neural data
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - buffer) / tbin_ms) + 1; 

    % Make on patch FR_mat, then perform PCA 
    classification_struct(sIdx).fr_mat_raw = {nTrials};
    for iTrial = 1:nTrials
        classification_struct(sIdx).fr_mat_raw{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
    end 
    
    fr_mat_onPatch = horzcat(classification_struct(sIdx).fr_mat_raw{:}); 
    fr_mat_onPatchZscore = zscore(fr_mat_onPatch,[],2); 
    tic
    [coeffs,score,~,~,expl] = pca(fr_mat_onPatchZscore');
    toc  
    score = score'; % reduced data
    
    fprintf("Proportion Variance explained by first 10 PCs: %f \n",sum(expl(1:10)) / sum(expl))

    % Get reward timings
    t_lens = cellfun(@(x) size(x,2),classification_struct(sIdx).fr_mat_raw); 
    new_patchleave_ix = cumsum(t_lens);
    new_patchstop_ix = new_patchleave_ix - t_lens + 1; 
    classification_struct(sIdx).rew_ix = {nTrials}; 
    classification_struct(sIdx).PCs = {nTrials};   
    
    % Fixed t2 window of 1000 msec
    t2 = 1000;  
    % a few preRew windows
    pre_rew_windows = [0,250,500,750,1000]; % keep this fixed for now
    
    classification_struct(sIdx).PCs_noPreRew = {length(pre_rew_windows)};
    classification_struct(sIdx).labels_noPreRew = {length(pre_rew_windows)};
    classification_struct(sIdx).vel_noPreRew = {length(pre_rew_windows)};
    classification_struct(sIdx).rewsize_noPreRew = {length(pre_rew_windows)};
    for wIdx = 1:length(pre_rew_windows)
        classification_struct(sIdx).PCs_noPreRew{wIdx} = {{nTrials}}; 
        classification_struct(sIdx).labels_noPreRew{wIdx} = {{nTrials}}; 
        classification_struct(sIdx).vel_noPreRew{wIdx} = {{nTrials}}; 
        classification_struct(sIdx).rewsize_noPreRew{wIdx} = {{nTrials}};
    end 
    
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial) - buffer) - patchstop_ms(iTrial));
        classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms);
        classification_struct(sIdx).PCs{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial));
        classification_struct(sIdx).labels{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - t2 / tbin_ms);
        classification_struct(sIdx).vel{iTrial} = dat.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));
        classification_struct(sIdx).rewsize{iTrial} = zeros(t_lens(iTrial),1) + rewsize(iTrial);
        
        % now do some fancy stuff to differentiate between preRew
        for wIdx = 1:length(pre_rew_windows)
            % now take out timesteps that came right before reward to better train regression
            pre_rew_label = zeros(t_lens(iTrial),1);
            rew_ix = classification_struct(sIdx).rew_ix{iTrial};
            for iRew_ix = 1:numel(classification_struct(sIdx).rew_ix{iTrial})
                pre_rew_label(max(1,(rew_ix(iRew_ix) - floor(pre_rew_windows(wIdx) / tbin_ms))) : rew_ix(iRew_ix)) = 1; % take off pre-rew activity
            end
            
            non_pre_rew = find(pre_rew_label == 0);
            
            classification_struct(sIdx).PCs_noPreRew{wIdx}{iTrial} = classification_struct(sIdx).PCs{iTrial}(:,non_pre_rew);
            classification_struct(sIdx).labels_noPreRew{wIdx}{iTrial} = classification_struct(sIdx).labels{iTrial}(non_pre_rew);
            classification_struct(sIdx).vel_noPreRew{wIdx}{iTrial} = classification_struct(sIdx).vel{iTrial}(non_pre_rew);
            classification_struct(sIdx).rewsize_noPreRew{wIdx}{iTrial} = classification_struct(sIdx).rewsize{iTrial}(non_pre_rew);
        end
    end
end  

%% Now display results again w/ ROC / PR 

close all 
figcounter = 1;
for sIdx = 22:22
    session = sessions{sIdx}(1:end-4);   
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    data = load(fullfile(paths.data,session)); 
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    
    all_concat_PCs_noPreRew = {length(pre_rew_windows)}; 
    all_concat_labels_noPreRew = {length(pre_rew_windows)};  
    all_concat_rewsize_noPrewRew = {length(pre_rew_windows)};   
    all_concat_vel_noPreRew = {length(pre_rew_windows)};   
    for iWindow = 1:length(pre_rew_windows) 
        all_concat_PCs_noPreRew{iWindow} = horzcat(classification_struct(sIdx).PCs_noPreRew{iWindow}{:});   
        all_concat_labels_noPreRew{iWindow} = horzcat(classification_struct(sIdx).labels_noPreRew{iWindow}{:}) + 1;   
        all_concat_rewsize_noPrewRew{iWindow} = vertcat(classification_struct(sIdx).rewsize_noPreRew{iWindow}{:})'; 
        all_concat_vel_noPreRew{iWindow} = horzcat(classification_struct(sIdx).vel_noPreRew{iWindow}{:}); 
    end
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    xval_opt.rew_size = [1,2,4];
    
    threshold_step = .05;
    thresholds = 0:threshold_step:1; 
    
    new_xval = true;
    if new_xval == true 
        % set up datastructures to measure classification fidelity
        accuracies = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        precisions = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        TP_rates = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        FP_rates = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds)); 
        ROC_AUC = nan(numel(pre_rew_windows),xval_opt.numFolds); 
        PR_AUC = nan(numel(pre_rew_windows),xval_opt.numFolds); 
        % same for velocity
        accuracies_vel = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        precisions_vel = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        TP_rates_vel = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        FP_rates_vel = nan(numel(pre_rew_windows),xval_opt.numFolds,numel(thresholds));
        ROC_AUC_vel = nan(numel(pre_rew_windows),xval_opt.numFolds);
        PR_AUC_vel = nan(numel(pre_rew_windows),xval_opt.numFolds);
        
        for wIdx = 1:numel(pre_rew_windows)
            % Need to re-do xval division every new window
            points = 1:numel(all_concat_rewsize_noPrewRew{wIdx});
            % split trials into groups (num groups = opt.numFolds)
            [points,~,IC] = unique(points); % don't reeeeally know what's going on here
            data_grp = nan(size(points));
            shift_by = 0; % to make sure equal numbers of trials end up in each fold
            % make sure all folds have roughly equal numbers of points from every rewsize
            for i = 1:numel(xval_opt.rew_size)
                keep_this = all_concat_rewsize_noPrewRew{wIdx} == xval_opt.rew_size(i);
                data_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
                data_grp(keep_this) = data_grp_this(1:sum(keep_this)); % assign folds 1:10
                shift_by = shift_by - mod(sum(keep_this),xval_opt.numFolds); % shift which fold is getting fewer trials
            end
            foldid = data_grp(IC)';
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data for both pcs and vel
                data_train = all_concat_PCs_noPreRew{wIdx}(1:10,foldid~=fIdx); 
                data_train_vel = all_concat_vel_noPreRew{wIdx}(foldid~=fIdx);
                labels_train = all_concat_labels_noPreRew{wIdx}(foldid~=fIdx); 
                
                data_test = all_concat_PCs_noPreRew{wIdx}(1:10,foldid==fIdx); 
                data_test_vel = all_concat_vel_noPreRew{wIdx}(foldid==fIdx);
                labels_test = all_concat_labels_noPreRew{wIdx}(foldid==fIdx);
                
                % now fit logistic regression to our training data
                [B,~,~] = mnrfit(data_train',labels_train);
                pi_test = mnrval(B,data_test');
                
                % now fit logistic regression to velocity training data
                [B_vel,~,~] = mnrfit(data_train_vel',labels_train);
                pi_test_vel = mnrval(B_vel,data_test_vel');
                
                for tIdx = 1:numel(thresholds)
                    threshold = thresholds(tIdx);
                    model_labels = double(pi_test(:,2) > threshold);
                    cm = confusionmat(labels_test' - 1,model_labels);
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
                    accuracies(wIdx,fIdx,tIdx) = accuracy;
                    precisions(wIdx,fIdx,tIdx) = precision;
                    TP_rates(wIdx,fIdx,tIdx) = TP_rate;
                    FP_rates(wIdx,fIdx,tIdx) = FP_rate; 
                    
                    %%%  now do the same for vel %%%
                    model_labels = double(pi_test_vel(:,2) > threshold);
                    cm = confusionmat(labels_test' - 1,model_labels);
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
                    accuracies_vel(wIdx,fIdx,tIdx) = accuracy;
                    precisions_vel(wIdx,fIdx,tIdx) = precision;
                    TP_rates_vel(wIdx,fIdx,tIdx) = TP_rate;
                    FP_rates_vel(wIdx,fIdx,tIdx) = FP_rate;
                    
                end 
                
                ROC_AUC(wIdx,fIdx) = threshold_step * sum(TP_rates(wIdx,fIdx,:));
                PR_AUC(wIdx,fIdx) = threshold_step * sum(precisions(wIdx,fIdx,~isnan(precisions(wIdx,fIdx,:)))); 
                
                ROC_AUC_vel(wIdx,fIdx) = threshold_step * sum(TP_rates_vel(wIdx,fIdx,:));
                PR_AUC_vel(wIdx,fIdx) = threshold_step * sum(precisions_vel(wIdx,fIdx,~isnan(precisions_vel(wIdx,fIdx,:))));
            end
            fprintf("Window %i / %i Complete \n",wIdx,numel(pre_rew_windows))
        end 
    end
    
    % visualize results with AUROC and Precision-Recall Curve
    for wIdx = 1:numel(pre_rew_windows)
        figure(figcounter)
        errorbar(thresholds,squeeze(mean(accuracies(wIdx,:,:))),1.96 * squeeze(std(accuracies(wIdx,:,:))),'linewidth',1.5) 
        hold on
        xlabel("Threshold")
        ylabel("Mean Test Set Accuracy")
        title("10-fold Test Accuracy Across Thresholds")
        
        figure(figcounter + 1)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates(wIdx,:,:))),squeeze(mean(TP_rates(wIdx,:,:))),1.96 * squeeze(std(TP_rates(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s PCA Receiver Operator Characteristic Curve",session_title))
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates(wIdx,:,:))),squeeze(mean(precisions(wIdx,:,:))),1.96 * squeeze(std(precisions(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s PCA Precision Recall Curve",session_title)) 
        
        % make same plot for velocity
        figure(figcounter + 2)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates_vel(wIdx,:,:))),squeeze(mean(TP_rates_vel(wIdx,:,:))),1.96 * squeeze(std(TP_rates_vel(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s Velocity Receiver Operator Characteristic Curve",session_title))
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates_vel(wIdx,:,:))),squeeze(mean(precisions_vel(wIdx,:,:))),1.96 * squeeze(std(precisions_vel(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s Velocity Precision Recall Curve",session_title))
    end
    
    figure(figcounter + 1)  
    subplot(1,2,1) 
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("Remove 0 ms pre-rew","Remove 250 ms pre-rew","Remove 500 ms pre-rew","Remove 750 ms pre-rew","Remove 1000 ms pre-rew","Naive Performance") 
    subplot(1,2,2) 
    yline(.5,'k--','linewidth',1.5)
    ylim([0,1])
    legend("Remove 0 ms pre-rew","Remove 250 ms pre-rew","Remove 500 ms pre-rew","Remove 750 ms pre-rew","Remove 1000 ms pre-rew","Naive Performance") 
    
    figure(figcounter + 2)  
    subplot(1,2,1)
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("Remove 0 ms pre-rew","Remove 250 ms pre-rew","Remove 500 ms pre-rew","Remove 750 ms pre-rew","Remove 1000 ms pre-rew","Naive Performance") 
    subplot(1,2,2) 
    ylim([0,1])
    yline(.5,'k--','linewidth',1.5)
    legend("Remove 0 ms pre-rew","Remove 250 ms pre-rew","Remove 500 ms pre-rew","Remove 750 ms pre-rew","Remove 1000 ms pre-rew","Naive Performance") 
    
    figcounter = figcounter + 3; 
end