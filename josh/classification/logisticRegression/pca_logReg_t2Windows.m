%% A script to do some analysis of questions regarding timewindow choices 

% 1. how do B coefficients across PCs change w.r.t. timewindow of
%    prediction?  
% 2. what is the appropriate window to remove pre-rew to maximize
%    clustering fidelity? measure w/ AUROC 

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

%% Acquire PC reductions and a binary classification vector

classification_struct = struct; 

for sIdx = 22:22
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
    
    buffer = 500; % ms before leave to exclude in analysis of neural data
    
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
    % make labels vectors with t2 windows
    test_windows = [250,500,750,1000]; 
    classification_struct(sIdx).labels = {length(test_windows)};
    for iWindow = 1:length(test_windows)
        classification_struct(sIdx).labels{iWindow} = {{nTrials}};  
    end
    classification_struct(sIdx).vel = {nTrials};  
    
    pre_rew_window = 1500; % keep this fixed for now
    
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial));
        classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms); 
        classification_struct(sIdx).PCs{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial));  
        for iWindow = 1:length(test_windows)
            classification_struct(sIdx).labels{iWindow}{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - test_windows(iWindow) / tbin_ms);  
        end
        classification_struct(sIdx).vel{iTrial} = dat.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));   
        classification_struct(sIdx).rewsize{iTrial} = zeros(t_lens(iTrial),1) + rewsize(iTrial);
        
        % now take out timesteps that came right before reward to better train regression
        pre_rew_label = zeros(t_lens(iTrial),1);  
        rew_ix = classification_struct(sIdx).rew_ix{iTrial}; 
        for iRew_ix = 1:numel(classification_struct(sIdx).rew_ix{iTrial}) 
            pre_rew_label(max(1,(rew_ix(iRew_ix) - pre_rew_window / tbin_ms)) : rew_ix(iRew_ix)) = 1; % take off the full second of activity
        end 
        
        non_pre_rew = find(pre_rew_label == 0);
        classification_struct(sIdx).PCs_noPreRew{iTrial} = classification_struct(sIdx).PCs{iTrial}(:,non_pre_rew); 
        for iWindow = 1:length(test_windows)
            classification_struct(sIdx).labels_noPreRew{iWindow}{iTrial} = classification_struct(sIdx).labels{iWindow}{iTrial}(non_pre_rew); 
        end
        classification_struct(sIdx).vel_noPreRew{iTrial} = classification_struct(sIdx).vel{iTrial}(non_pre_rew); 
        classification_struct(sIdx).rewsize_noPreRew{iTrial} = classification_struct(sIdx).rewsize{iTrial}(non_pre_rew);
    end
end 

%% Now perform classification with logistic regression, using k-fold x-val  
%  add velocity classification as a control 

close all 
figcounter = 1;
for sIdx = 22:22
    session = sessions{sIdx}(1:end-4); 
    data = load(fullfile(paths.data,session)); 
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:});   
    all_concat_labels_noPreRew = {length(test_windows)};
    for iWindow = 1:length(test_windows)
        all_concat_labels_noPreRew{iWindow} = horzcat(classification_struct(sIdx).labels_noPreRew{iWindow}{:}) + 1;   
    end
    all_concat_rewsize_noPrewRew = vertcat(classification_struct(sIdx).rewsize_noPreRew{:})'; 
    all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:}); 
    
    % folds are going to be over points that did not directly precede reward
    points = 1:numel(all_concat_rewsize_noPrewRew);
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    xval_opt.rew_size = [1,2,4];
    % split trials into groups (num groups = opt.numFolds)
    [points,~,IC] = unique(points); % don't reeeeally know what's going on here
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
    threshold_step = .05;
    thresholds = 0:threshold_step:1; 
    
    new_xval = true;
    if new_xval == true 
        % set up datastructures to measure classification fidelity
        accuracies = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        precisions = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        TP_rates = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        FP_rates = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds)); 
        ROC_AUC = nan(numel(test_windows),xval_opt.numFolds); 
        PR_AUC = nan(numel(test_windows),xval_opt.numFolds); 
        % same for velocity
        accuracies_vel = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        precisions_vel = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        TP_rates_vel = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        FP_rates_vel = nan(numel(test_windows),xval_opt.numFolds,numel(thresholds));
        ROC_AUC_vel = nan(numel(test_windows),xval_opt.numFolds); 
        PR_AUC_vel = nan(numel(test_windows),xval_opt.numFolds);
        
        for wIdx = 1:numel(test_windows) 
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data for both pcs and vel
                data_train = all_concat_PCs_noPreRew(:,foldid~=fIdx); 
                data_train_vel = all_concat_vel_noPreRew(foldid~=fIdx);
                labels_train = all_concat_labels_noPreRew{wIdx}(foldid~=fIdx); 
                labels_test = all_concat_labels_noPreRew{wIdx}(foldid==fIdx);
                data_test = all_concat_PCs_noPreRew(:,foldid==fIdx); 
                data_test_vel = all_concat_vel_noPreRew(foldid==fIdx);
                
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
            fprintf("Window %i / %i Complete \n",wIdx,numel(test_windows))
        end 
    end
    
    % visualize results with AUROC and Precision-Recall Curve
    for wIdx = 1:numel(test_windows)
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
        title("PCA Receiver Operator Characteristic Curve")
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates(wIdx,:,:))),squeeze(mean(precisions(wIdx,:,:))),1.96 * squeeze(std(precisions(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title("PCA Precision Recall Curve") 
        
        % make same plot for velocity
        figure(figcounter + 2)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates_vel(wIdx,:,:))),squeeze(mean(TP_rates_vel(wIdx,:,:))),1.96 * squeeze(std(TP_rates_vel(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title("Velocity Receiver Operator Characteristic Curve")
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates_vel(wIdx,:,:))),squeeze(mean(precisions_vel(wIdx,:,:))),1.96 * squeeze(std(precisions_vel(wIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title("Velocity Precision Recall Curve")
    end
    
    figure(figcounter + 1)  
    subplot(1,2,1) 
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("Classify 250 msec pre-leave","Classify 500 msec pre-leave","Classify 750 msec pre-leave","Classify 1000 msec pre-leave","Naive Performance") 
    subplot(1,2,2) 
    yline(.5,'k--','linewidth',1.5)
    legend("Classify 250 msec pre-leave","Classify 500 msec pre-leave","Classify 750 msec pre-leave","Classify 1000 msec pre-leave","Naive Performance") 
    
    figure(figcounter + 2)  
    subplot(1,2,1)
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("Classify 250 msec pre-leave","Classify 500 msec pre-leave","Classify 750 msec pre-leave","Classify 1000 msec pre-leave","Naive Performance") 
    subplot(1,2,2) 
    yline(.5,'k--','linewidth',1.5)
    legend("Classify 250 msec pre-leave","Classify 500 msec pre-leave","Classify 750 msec pre-leave","Classify 1000 msec pre-leave","Naive Performance") 
    
    figcounter = figcounter + 3;
end

%% Now just check how logreg weights on PCs change by prediction window

close all 
for sIdx = 22:22
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})';   
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';  
    session_len = size(all_concat_PCs,1);  
    
    B_coeffs = nan(size(all_concat_PCs_noPreRew,2),numel(test_windows));
    
    for iWindow = 1:numel(test_windows)
        all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{iWindow}{:}) + 1;  
        [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew,all_concat_labels_noPreRew);
        B_coeffs(:,iWindow) = B(2:end);
        [~,pc_sort_by_logW] = sort(abs(B(2:end)),'descend');
        pi_hat = mnrval(B,all_concat_PCs);    
        
        decode_pc1 = pc_sort_by_logW(1); 
        decode_pc2 = pc_sort_by_logW(2);  
        decode_pc3 = pc_sort_by_logW(3); 
% 
%         figure();colormap('hot')
%         scatter3(all_concat_PCs(:,decode_pc1),all_concat_PCs(:,decode_pc2),all_concat_PCs(:,decode_pc3),3,pi_hat(:,2)','.')
%         colorbar()
%         xlabel(sprintf("PC%i",decode_pc1))
%         ylabel(sprintf("PC%i",decode_pc2))
%         zlabel(sprintf("PC%i",decode_pc3))
    end

    figure()
    b = bar(B_coeffs);
    b(4).FaceColor = [.2 .6 .5];
    legend("Classify 250 msec pre-leave","Classify 500 msec pre-leave","Classify 750 msec pre-leave","Classify 1000 msec pre-leave")
    xlabel("PC")
    ylabel("Logistic Regression Weight")
    title("PC Logistic Regression Results")
    
end
