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

for sIdx = 1:numel(sessions)
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
    classification_zone = 1000; % how much time before leave we're labeling in ms
    classification_struct(sIdx).rew_ix = {nTrials}; 
    classification_struct(sIdx).PCs = {nTrials};  
    classification_struct(sIdx).labels = {nTrials}; 
    classification_struct(sIdx).vel = {nTrials}; 
    
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial));
        classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms); 
        classification_struct(sIdx).PCs{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial)); 
        classification_struct(sIdx).labels{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - classification_zone / tbin_ms); 
        classification_struct(sIdx).vel{iTrial} = dat.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));   
        classification_struct(sIdx).rewsize{iTrial} = zeros(t_lens(iTrial),1) + rewsize(iTrial);
        
        % now take out timesteps that came right before reward to better train regression
        pre_rew_label = zeros(t_lens(iTrial),1);  
        rew_ix = classification_struct(sIdx).rew_ix{iTrial}; 
        for iRew_ix = 1:numel(classification_struct(sIdx).rew_ix{iTrial}) 
            pre_rew_label(max(1,(rew_ix(iRew_ix) - 1000 / tbin_ms)) : rew_ix(iRew_ix)) = 1; % take off the full second of activity
        end 
        
        non_pre_rew = find(pre_rew_label == 0);
        classification_struct(sIdx).PCs_noPreRew{iTrial} = classification_struct(sIdx).PCs{iTrial}(:,non_pre_rew);
        classification_struct(sIdx).labels_noPreRew{iTrial} = classification_struct(sIdx).labels{iTrial}(non_pre_rew);
        classification_struct(sIdx).vel_noPreRew{iTrial} = classification_struct(sIdx).vel{iTrial}(non_pre_rew); 
        classification_struct(sIdx).rewsize_noPreRew{iTrial} = classification_struct(sIdx).rewsize{iTrial}(non_pre_rew);
    end
end 

%% Visualize the classification problem on a few single trials  
close all
for sIdx = 1:1
    test_trials = 11:19;  
    sp_counter = 1;
    figure()
    for iTrial = test_trials 
        subplot(3,3,sp_counter)
        gscatter(classification_struct(sIdx).PCs_noPreRew{iTrial}(1,:), ...
                 classification_struct(sIdx).PCs_noPreRew{iTrial}(3,:), ...
                 classification_struct(sIdx).labels_noPreRew{iTrial}, ... 
                 [],[],5) 
        title(sprintf("Trial %i",iTrial)) 
        xlabel("PC1"); ylabel("PC3") 
%         legend("Stay","Leave in 500-1000 msec") 
        sp_counter = sp_counter + 1; 
        b = gca; legend(b,'off');
    end 
    
    % concatenate to show cross trial data
    concat_PCs = classification_struct(sIdx).PCs_noPreRew(test_trials);
    concat_PCs = horzcat(concat_PCs{:}); 
    concat_labels = classification_struct(sIdx).labels_noPreRew(test_trials); 
    concat_labels = horzcat(concat_labels{:}); 
    figure() 
    gscatter(concat_PCs(1,:),concat_PCs(3,:),concat_labels) 
    xlabel("PC1"); ylabel("PC3") 
    title("Labeled Points in PC Space") 
    legend("Stay","Leave in 500-1000 msec") 
    
    % total concat pca 
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs_noPreRew{:}); 
    all_concat_labels = horzcat(classification_struct(sIdx).labels_noPreRew{:});  
    figure() 
    gscatter(all_concat_PCs(1,:),all_concat_PCs(3,:),all_concat_labels,[],[],2) 
    xlabel("PC1"); ylabel("PC3") 
    title("Labeled Points in PC Space") 
    legend("Stay","Leave in 500-1000 msec")  
    
    figure() 
    % now look at velocity  
    sp_counter = 1;
    for iTrial = test_trials 
        subplot(3,3,sp_counter) 
        t_len = numel(classification_struct(sIdx).vel{iTrial});
        gscatter(1:t_len,classification_struct(sIdx).vel{iTrial},classification_struct(sIdx).labels{iTrial},[],[],2) 
        title(sprintf("Trial %i",iTrial)) 
        xlabel("Time"); 
        ylabel("Velocity")
        sp_counter = sp_counter + 1; 
        b = gca; legend(b,'off');
    end 
    
end 

%% Perform logistic regression on labelled PCs
close all
for sIdx = 1:2 
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:})';   
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';  
    session_len = size(all_concat_PCs,1);
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;  
    [B,dev,stats] = mnrfit(all_concat_PCs_noPreRew,all_concat_labels_noPreRew);   
    [~,pc_sort_by_logW] = sort(abs(B(2:end)),'descend');
    pi_hat = mnrval(B,all_concat_PCs);    
    
    decode_pc1 = pc_sort_by_logW(1); 
    decode_pc2 = pc_sort_by_logW(2);  
    decode_pc3 = pc_sort_by_logW(3); 
    
    figure();colormap('hot')
    scatter(all_concat_PCs(:,decode_pc1),all_concat_PCs(:,decode_pc2),3,pi_hat(:,2)','.');  
    colorbar()
    xlabel(sprintf("PC%i",decode_pc1));
    ylabel(sprintf("PC%i",decode_pc2));  
    title("P(leave in .5-1 sec | PC1:10)")  
    xl = xlim;
    yl = ylim;
    
    % now just show where P(leave) > .1 
%     ix = find(pi_hat(:,2) > .1); 
%     figure();colormap('hot')
%     scatter(all_concat_PCs(ix,decode_pc1),all_concat_PCs(ix,decode_pc2),3,pi_hat(ix,2)','.');  
%     colorbar()
%     xlabel(sprintf("PC%i",decode_pc1));ylabel(sprintf("PC%i",decode_pc2))  
%     title("P(leave in .5-1 sec | PC1:10)")   
    
%     figure();colormap('hot')
%     scatter3(all_concat_PCs(ix,decode_pc1),all_concat_PCs(ix,decode_pc2),all_concat_PCs(ix,decode_pc3),3,pi_hat(ix,2)','.') 
%     colorbar() 
%     xlabel(sprintf("PC%i",decode_pc1)) 
%     ylabel(sprintf("PC%i",decode_pc2)) 
%     zlabel(sprintf("PC%i",decode_pc3)) 
    
    figure()
    bar(B(2:end)) 
    xlabel("PC") 
    ylabel("Logistic Regression Weight") 
    title("PC Logistic Regression Results") 
    
    % show results using meshgrid
    [x,y] = meshgrid(xl(1):.1:xl(2),yl(1):.1:yl(2)); 
    x = x(:);
    y = y(:); 
    pihat_mesh = mnrval(B,[x zeros(size(x,1),1) y zeros(size(x,1),7)]); 
    figure();colormap("hot")
    scatter(x,y,[],pihat_mesh(:,2),'.');colorbar() 
    xlabel(sprintf("PC%i",decode_pc1));ylabel(sprintf("PC%i",decode_pc2)) 
    title("Logistic Regression Results as Meshgrid") 
    
    % now add p_leave to our classification struct 
    t_lens = cellfun(@(x) size(x,2),classification_struct(sIdx).fr_mat_raw); 
    patchleave_ix = cumsum(t_lens);
    patchstop_ix = patchleave_ix - t_lens + 1;  
    for iTrial = 1:numel(t_lens)
        classification_struct(sIdx).p_leave{iTrial} = pi_hat(patchstop_ix(iTrial):patchleave_ix(iTrial),2);
    end
end   

%% Now perform classification with logistic regression, using k-fold x-val  
% add velocity classification as a control
close all 
figcounter = 1;
for sIdx = 11:24
    session = sessions{sIdx}(1:end-4); 
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    data = load(fullfile(paths.data,session)); 
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    
    all_concat_PCs_noPreRew = horzcat(classification_struct(sIdx).PCs_noPreRew{:});   
    all_concat_labels_noPreRew = horzcat(classification_struct(sIdx).labels_noPreRew{:}) + 1;  
    all_concat_rewsize_noPrewRew = vertcat(classification_struct(sIdx).rewsize_noPreRew{:})'; 
    all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:}); 
    
    % folds are going to be over points that did not directly precede
    % reward
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
    pc_ranges = 1:10;
    
    new_xval = true;
    if new_xval == true 
        % set up datastructures to measure classification fidelity
        accuracies = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        precisions = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        TP_rates = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds));
        FP_rates = nan(numel(pc_ranges),xval_opt.numFolds,numel(thresholds)); 
        ROC_AUC = nan(numel(pc_ranges),xval_opt.numFolds); 
        PR_AUC = nan(numel(pc_ranges),xval_opt.numFolds);
        
        for pcIdx = 1:numel(pc_ranges) 
            last_pc = pc_ranges(pcIdx);
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data
                data_train = all_concat_PCs_noPreRew(1:last_pc,foldid~=fIdx);
                labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
                labels_test = all_concat_labels_noPreRew(foldid==fIdx);
                data_test = all_concat_PCs_noPreRew(1:last_pc,foldid==fIdx);
                
                % now fit logistic regression to our training data
                [B,~,~] = mnrfit(data_train',labels_train);
                pi_test = mnrval(B,data_test');
                
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
                    accuracies(pcIdx,fIdx,tIdx) = accuracy;
                    precisions(pcIdx,fIdx,tIdx) = precision;
                    TP_rates(pcIdx,fIdx,tIdx) = TP_rate;
                    FP_rates(pcIdx,fIdx,tIdx) = FP_rate;
                end 
                
                ROC_AUC(pcIdx,fIdx) = threshold_step * sum(TP_rates(pcIdx,fIdx,:));
                PR_AUC(pcIdx,fIdx) = threshold_step * sum(precisions(pcIdx,fIdx,~isnan(precisions(pcIdx,fIdx,:))));
            end
            fprintf("PC 1:%i Complete \n",pcIdx)
        end 
        
        accuracies_vel = nan(xval_opt.numFolds,numel(thresholds));
        precisions_vel = nan(xval_opt.numFolds,numel(thresholds));
        TP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
        FP_rates_vel = nan(xval_opt.numFolds,numel(thresholds));
        ROC_AUC_vel = nan(xval_opt.numFolds,1); 
        PR_AUC_vel = nan(xval_opt.numFolds,1);
        % Now repeat quickly for velocity to have comparison
        for fIdx = 1:xval_opt.numFolds
            % separate training and test data
            data_train = all_concat_vel_noPreRew(foldid~=fIdx);
            labels_train = all_concat_labels_noPreRew(foldid~=fIdx);
            labels_test = all_concat_labels_noPreRew(foldid==fIdx);
            data_test = all_concat_vel_noPreRew(foldid==fIdx); 
            
            % now fit logistic regression to our training data
            [B,~,~] = mnrfit(data_train',labels_train);
            pi_test = mnrval(B,data_test');
            
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
                accuracies_vel(fIdx,tIdx) = accuracy;
                precisions_vel(fIdx,tIdx) = precision;
                TP_rates_vel(fIdx,tIdx) = TP_rate;
                FP_rates_vel(fIdx,tIdx) = FP_rate;
            end 
            
            ROC_AUC_vel(fIdx) = threshold_step * sum(TP_rates_vel(fIdx,:));
            PR_AUC_vel(fIdx) = threshold_step * sum(precisions_vel(fIdx,~isnan(precisions_vel(fIdx,:))));
        end 
        disp("Velocity Complete")
    end
    
    % visualize results with AUROC and Precision-Recall Curve
    for pcIdx = [1,5,10]
%         figure(figcounter)
%         last_pc = pc_ranges(pcIdx);
%         errorbar(thresholds,squeeze(mean(accuracies(pcIdx,:,:))),1.96 * squeeze(std(accuracies(pcIdx,:,:))),'linewidth',1.5) 
%         hold on
%         xlabel("Threshold")
%         ylabel("Mean Test Set Accuracy")
%         title("10-fold Test Accuracy Across Thresholds")
        
        figure(figcounter + 1)
        subplot(1,2,1)
        errorbar(squeeze(mean(FP_rates(pcIdx,:,:))),squeeze(mean(TP_rates(pcIdx,:,:))),1.96 * squeeze(std(TP_rates(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean False Positive Rate Across Folds")
        ylabel("Mean True Positive Rate Across Folds")
        title(sprintf("%s Receiver Operator Characteristic Curve",session_title))
        subplot(1,2,2)
        errorbar(squeeze(mean(TP_rates(pcIdx,:,:))),squeeze(mean(precisions(pcIdx,:,:))),1.96 * squeeze(std(precisions(pcIdx,:,:))),'linewidth',1.5)
        hold on
        xlabel("Mean True Positive Rate Across Folds")
        ylabel("Mean Precision Across Folds")
        title(sprintf("%s Precision Recall Curve",session_title))
    end 
    
%     % now add velocity-based classification
%     figure(figcounter) 
%     errorbar(thresholds,mean(accuracies_vel),1.96 * std(accuracies_vel),'linewidth',1.5)  
%     legend("PC 1:1","PC 1:5","PC 1:10","Velocity")
    
    figure(figcounter + 1)  
    subplot(1,2,1)
    errorbar(mean(FP_rates_vel),mean(TP_rates_vel),1.96 * std(TP_rates_vel),'linewidth',1.5)  
    plot([0,1],[0,1],'k--','linewidth',1.5) 
    ylim([0,1])
    legend("PC 1:1","PC 1:5","PC 1:10","Velocity","Naive Performance") 
    subplot(1,2,2) 
    errorbar(mean(TP_rates_vel),mean(precisions_vel),1.96 * std(precisions_vel),'linewidth',1.5)
    yline(.5,'k--','linewidth',1.5)
    legend("PC 1:1","PC 1:5","PC 1:10","Velocity","Naive Performance") 
    ylim([0,1])
    
    % Now plot AUC 
    figure(figcounter + 2) 
    subplot(1,2,1)
    errorbar(pc_ranges,mean(ROC_AUC,2),1.96 * std(ROC_AUC,[],2),'linewidth',1.5)
    hold on 
    yline(mean(ROC_AUC_vel),'k--','linewidth',1.5)  
    title(sprintf("%s AUROC Forward Search",session_title)) 
    legend("AUROC for PCs","AUROC for Velocity") 
    xlabel("PCs Used In Logistic Regression") 
    ylabel("AUROC") 
    ylim([0,1])
    subplot(1,2,2)
    errorbar(pc_ranges,mean(PR_AUC,2),1.96 * std(PR_AUC,[],2),'linewidth',1.5) 
    hold on 
    yline(mean(PR_AUC_vel),'k--','linewidth',1.5)  
    title(sprintf("%s AUPR Forward Search",session_title)) 
    legend("AUPR for PCs","AUPR for Velocity") 
    xlabel("PCs Used In Logistic Regression")
    ylabel("AUPR") 
    ylim([0,1])
    
    figcounter = figcounter + 3;
end

%% Look at P(leave) on single trials (decision variable structure?) 
close all
for sIdx = 3:3
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
        xlabel("Time")  
        t_len = length(classification_struct(sIdx).p_leave{iTrial});
        xticks((1:(1/tbin_ms * 1000):t_len) - 1) 
        xticklabels(((1:(1/tbin_ms * 1000):t_len) - 1) * tbin_ms)
        sp_counter = sp_counter + 1;
    end
end 

%% P(leave) on RXX trials 
close all
for sIdx = 1:3 
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;

    sec3ix = 3000/tbin_ms;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(prts);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
  
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
    
end
%% practicing xval 

xval_opt = struct; 
xval_opt.numFolds = 10;
patches = 1:160; 
xval_opt.rew_size = [1,2,4];
% split trials into groups (num groups = opt.numFolds)
[trial,~,IC] = unique(patches);
trial_grp = nan(size(trial));
shift_by = 0; % to make sure equal numbers of trials end up in each fold

for i = 1:numel(xval_opt.rew_size)
    keep_this = rewsize==xval_opt.rew_size(i);
    trial_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
    trial_grp(keep_this) = trial_grp_this(1:sum(keep_this));
    shift_by = shift_by-mod(sum(keep_this),xval_opt.numFolds);
end

foldid = trial_grp(IC); 

% then this is how we do the xval procedure in loop!
for fIdx = 1:xval_opt.numFolds
    y_train = y(foldid~=fIdx);
    y_test = y(foldid==fIdx); 
end
