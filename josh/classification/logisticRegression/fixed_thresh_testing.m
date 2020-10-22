%% Computational experiment to explore the possibility that there is a non-fixed threshold to leave 

% procedure: 1) take a n-sized sample of points w/ all the same rew size,
%               get k-fold xval AUC for ROC and PR 
%            2) take a n-sized sample of points w/ different rew sizes, get
%               k-fold xval AUC for ROC and PR
% if there is a non-constant threshold, AUC1 > AUC2 

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

for sIdx = 1:24
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
    % 2 new guys to add to 
    classification_struct(sIdx).nRews = {nTrials}; 
    classification_struct(sIdx).prt_quartile = {nTrials};  
    prt_quartiles = [0 quantile(prts,3) max(prts)];   
    pre_rew_time = 1500;
    
    prt_quartile_labels = nan(nTrials,1);
    for q = 1:(numel(prt_quartiles)-1)
        q_ix = prts <= prt_quartiles(q+1) & prts >= prt_quartiles(q); 
        prt_quartile_labels(q_ix) = q;
    end
    
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)); 
        classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms); 
        classification_struct(sIdx).PCs{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial)); 
        classification_struct(sIdx).labels{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - classification_zone / tbin_ms); 
        classification_struct(sIdx).vel{iTrial} = dat.vel(patchstop_ix(iTrial):patchleave_ix(iTrial));    
        
        % stuff to check out w/ classification fidelity tests
        classification_struct(sIdx).prt_quartile{iTrial} = zeros(t_lens(iTrial),1) + prt_quartile_labels(iTrial);
        classification_struct(sIdx).nRews{iTrial} = zeros(t_lens(iTrial),1) + length(classification_struct(sIdx).rew_ix{iTrial}); 
        classification_struct(sIdx).rewsize{iTrial} = zeros(t_lens(iTrial),1) + rewsize(iTrial);
        
        % now take out timesteps that came right before reward to better train regression
        pre_rew_label = zeros(t_lens(iTrial),1);  
        rew_ix = classification_struct(sIdx).rew_ix{iTrial}; 
        for iRew_ix = 1:numel(classification_struct(sIdx).rew_ix{iTrial}) 
            pre_rew_label(max(1,(rew_ix(iRew_ix) - pre_rew_time / tbin_ms)) : rew_ix(iRew_ix)) = 1; % take off the full second of activity
        end 
        
        non_pre_rew = find(pre_rew_label == 0);
        classification_struct(sIdx).PCs_noPreRew{iTrial} = classification_struct(sIdx).PCs{iTrial}(:,non_pre_rew);
        classification_struct(sIdx).labels_noPreRew{iTrial} = classification_struct(sIdx).labels{iTrial}(non_pre_rew);
        classification_struct(sIdx).vel_noPreRew{iTrial} = classification_struct(sIdx).vel{iTrial}(non_pre_rew); 
        classification_struct(sIdx).rewsize_noPreRew{iTrial} = classification_struct(sIdx).rewsize{iTrial}(non_pre_rew); 
        classification_struct(sIdx).nRews_noPreRew{iTrial} = classification_struct(sIdx).nRews{iTrial}(non_pre_rew); 
        classification_struct(sIdx).prt_quartile_noPreRew{iTrial} = classification_struct(sIdx).prt_quartile{iTrial}(non_pre_rew);
    end
end 

%% Now do reward size constant threshold testing 
close all
figcounter = 1;

for sIdx = 1
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
    
    % folds are going to be over points that did not directly precede reward
    points = 1:numel(all_concat_rewsize_noPrewRew);
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    
    threshold_step = .05;
    thresholds = 0:threshold_step:1;
    pc_ranges = 1:10;
    N_samples = 50;
    rewsizes = [1,2,4];
    pc_range = 1:10; 
    
    % get sample size as min(nPts) / 2
    rewsize_nPts = nan(3,1);
    for iRewsize = 1:3
        rewsize_nPts(iRewsize) = length(find(all_concat_rewsize_noPrewRew == rewsizes(iRewsize)));
    end 
    K = round(min(rewsize_nPts) / 2); 
    
    new_xval = true;
    if new_xval == true
        % set up datastructures to measure classification fidelity
        accuracies_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds);
        PR_AUC_1rewsize = nan(numel(rewsizes),N_samples,xval_opt.numFolds);
        
        for iRewsize = 1:3
            this_rewsize = rewsizes(iRewsize);
            rewsize_PCs = all_concat_PCs_noPreRew(:,all_concat_rewsize_noPrewRew == this_rewsize);
            rewsize_labels = all_concat_labels_noPreRew(:,all_concat_rewsize_noPrewRew == this_rewsize); 
            
            for sampleIdx = 1:N_samples 
                sample_ix = randsample(size(rewsize_PCs,2),K); 
                sample_PCs = rewsize_PCs(:,sample_ix); 
                sample_labels = rewsize_labels(sample_ix);
                
                foldid = repmat(1:xval_opt.numFolds,1,ceil(size(rewsize_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
                foldid = foldid(1:size(rewsize_PCs,2));
                
                % Iterate over folds to use as test data
                for fIdx = 1:xval_opt.numFolds                    
                    % separate training and test data (both from our sample)
                    data_train = rewsize_PCs(pc_range,foldid~=fIdx);
                    labels_train = rewsize_labels(foldid~=fIdx);
                    data_test = rewsize_PCs(pc_range,foldid==fIdx);
                    labels_test = rewsize_labels(foldid==fIdx);
                    
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
                        accuracies_1rewsize(iRewsize,sampleIdx,fIdx,tIdx) = accuracy;
                        precisions_1rewsize(iRewsize,sampleIdx,fIdx,tIdx) = precision;
                        TP_rates_1rewsize(iRewsize,sampleIdx,fIdx,tIdx) = TP_rate;
                        FP_rates_1rewsize(iRewsize,sampleIdx,fIdx,tIdx) = FP_rate;
                    end
                    
                    ROC_AUC_dx = -squeeze(diff(FP_rates_1rewsize(iRewsize,sampleIdx,fIdx,:)));
                    ROC_AUC_1rewsize(iRewsize,sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_1rewsize(iRewsize,sampleIdx,fIdx,1:end-1)));
                    PR_AUC_dx = -squeeze(diff(TP_rates_1rewsize(iRewsize,sampleIdx,fIdx,:)));
                    PR_AUC_1rewsize(iRewsize,sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_1rewsize(iRewsize,sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_1rewsize(iRewsize,sampleIdx,fIdx,~isnan(precisions_1rewsize(iRewsize,sampleIdx,fIdx,1:end-1)))));
                end
            end 
            fprintf("Rewsize %i Complete \n",this_rewsize)
        end
        
        % here, perform control sampling w/ points chosen from all reward sizes
        accuracies_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_allRewsize = nan(N_samples,xval_opt.numFolds);
        PR_AUC_allRewsize = nan(N_samples,xval_opt.numFolds);
        
        for sampleIdx = 1:N_samples
            sample_ix = randsample(size(all_concat_PCs_noPreRew,2),K);
            sample_PCs = all_concat_PCs_noPreRew(:,sample_ix);
            sample_labels = all_concat_labels_noPreRew(sample_ix);
            
            foldid = repmat(1:xval_opt.numFolds,1,ceil(size(sample_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
            foldid = foldid(1:size(sample_PCs,2));
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data (both from our sample)
                data_train = sample_PCs(pc_range,foldid~=fIdx);
                labels_train = sample_labels(foldid~=fIdx);
                data_test = sample_PCs(pc_range,foldid==fIdx);
                labels_test = sample_labels(foldid==fIdx);
                
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
                    accuracies_allRewsize(sampleIdx,fIdx,tIdx) = accuracy;
                    precisions_allRewsize(sampleIdx,fIdx,tIdx) = precision;
                    TP_rates_allRewsize(sampleIdx,fIdx,tIdx) = TP_rate;
                    FP_rates_allRewsize(sampleIdx,fIdx,tIdx) = FP_rate;
                end
                
                ROC_AUC_dx = -squeeze(diff(FP_rates_allRewsize(sampleIdx,fIdx,:)));
                ROC_AUC_allRewsize(sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_allRewsize(sampleIdx,fIdx,1:end-1)));
                PR_AUC_dx = -squeeze(diff(TP_rates_allRewsize(sampleIdx,fIdx,:)));
                PR_AUC_allRewsize(sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_allRewsize(sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_allRewsize(sampleIdx,fIdx,~isnan(precisions_allRewsize(sampleIdx,fIdx,1:end-1)))));
            end
        end 
        disp("Control Complete")
    end
    
    mean_ROC_AUC_1rewsize = mean(ROC_AUC_1rewsize,[2,3]); 
    sem_ROC_AUC_1rewsize = 1.96 * std(ROC_AUC_1rewsize,[],[2,3]);
    mean_PR_AUC_1rewsize = mean(PR_AUC_1rewsize,[2,3]); 
    sem_PR_AUC_1rewsize = 1.96 * std(PR_AUC_1rewsize,[],[2,3]);
    
    mean_data = [mean_ROC_AUC_1rewsize mean_PR_AUC_1rewsize]; 
    sem_data = [sem_ROC_AUC_1rewsize sem_PR_AUC_1rewsize]; 
    
    control_mean_ROC_AUC = mean(ROC_AUC_allRewsize(:)); 
    control_sem_ROC_AUC = 1.96 * std(ROC_AUC_allRewsize(:)); 
    control_mean_PR_AUC = mean(PR_AUC_allRewsize(:)); 
    control_sem_PR_AUC = 1.96 * std(PR_AUC_allRewsize(:)); 
    
    figure() 
    hb = bar([1,2,3],mean_data);  
    xticklabels(["1 uL","2 uL","4 uL"])
    title("Fixed decision threshold test acr rewsizes")
%     hold on
%     % For each set of bars, find the centers of the bars, and write error bars
%     pause(0.1); %pause allows the figure to be created
%     for ib = 1:numel(hb)
%         %XData property is the tick labels/group centers; XOffset is the offset
%         %of each distinct group
%         xData = hb(ib).XData+hb(ib).XOffset; 
%         disp(xData)
%         errorbar(xData,mean_data(:,ib),sem_data(:,ib),'k.')
%     end 
    
    hold on; 
    yline(control_mean_ROC_AUC,'b-','linewidth',1.5)
    yline(control_mean_ROC_AUC - control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_ROC_AUC + control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_PR_AUC,'r-','linewidth',1.5)
    yline(control_mean_PR_AUC - control_sem_PR_AUC,'r:','linewidth',1)
    yline(control_mean_PR_AUC + control_sem_PR_AUC,'r:','linewidth',1)
end

%% Now do nRews threshold testing
close all
figcounter = 1;

for sIdx = 23:23
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
    all_concat_nRews_noPrewRew = vertcat(classification_struct(sIdx).nRews_noPreRew{:})';
    all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:});
    
    % folds are going to be over points that did not directly precede reward
    points = 1:numel(all_concat_nRews_noPrewRew);
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    
    threshold_step = .05;
    thresholds = 0:threshold_step:1;
    pc_ranges = 1:10;
    N_samples = 50;
    nRew_range = [1,2,3];
    pc_range = 1:10; 
    
    % get sample size as min(nPts) / 2
    nRews_nPts = nan(3,1);
    for i_nRews = 1:3
        nRews_nPts(i_nRews) = length(find(all_concat_nRews_noPrewRew == nRew_range(i_nRews)));
    end 
    K = round(min(nRews_nPts) / 2); 
    
    new_xval = true;
    if new_xval == true
        % set up datastructures to measure classification fidelity
        accuracies_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds);
        PR_AUC_1rewsize = nan(numel(nRew_range),N_samples,xval_opt.numFolds);
        
        for i_nRews = 1:3
            this_rewsize = nRew_range(i_nRews);
            rewsize_PCs = all_concat_PCs_noPreRew(:,all_concat_nRews_noPrewRew == this_rewsize);
            rewsize_labels = all_concat_labels_noPreRew(:,all_concat_nRews_noPrewRew == this_rewsize); 
            
            for sampleIdx = 1:N_samples 
                sample_ix = randsample(size(rewsize_PCs,2),K); 
                sample_PCs = rewsize_PCs(:,sample_ix); 
                sample_labels = rewsize_labels(sample_ix);
                
                foldid = repmat(1:xval_opt.numFolds,1,ceil(size(rewsize_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
                foldid = foldid(1:size(rewsize_PCs,2));
                
                % Iterate over folds to use as test data
                for fIdx = 1:xval_opt.numFolds                    
                    % separate training and test data (both from our sample)
                    data_train = rewsize_PCs(pc_range,foldid~=fIdx);
                    labels_train = rewsize_labels(foldid~=fIdx);
                    data_test = rewsize_PCs(pc_range,foldid==fIdx);
                    labels_test = rewsize_labels(foldid==fIdx);
                    
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
                        accuracies_1rewsize(i_nRews,sampleIdx,fIdx,tIdx) = accuracy;
                        precisions_1rewsize(i_nRews,sampleIdx,fIdx,tIdx) = precision;
                        TP_rates_1rewsize(i_nRews,sampleIdx,fIdx,tIdx) = TP_rate;
                        FP_rates_1rewsize(i_nRews,sampleIdx,fIdx,tIdx) = FP_rate;
                    end
                    
                    ROC_AUC_dx = -squeeze(diff(FP_rates_1rewsize(i_nRews,sampleIdx,fIdx,:)));
                    ROC_AUC_1rewsize(i_nRews,sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_1rewsize(i_nRews,sampleIdx,fIdx,1:end-1)));
                    PR_AUC_dx = -squeeze(diff(TP_rates_1rewsize(i_nRews,sampleIdx,fIdx,:)));
                    PR_AUC_1rewsize(i_nRews,sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_1rewsize(i_nRews,sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_1rewsize(i_nRews,sampleIdx,fIdx,~isnan(precisions_1rewsize(i_nRews,sampleIdx,fIdx,1:end-1)))));
                end
            end 
            fprintf("nRews %i Complete \n",this_rewsize)
        end
        
        % here, perform control sampling w/ points chosen from all reward sizes
        accuracies_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_allRewsize = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_allRewsize = nan(N_samples,xval_opt.numFolds);
        PR_AUC_allRewsize = nan(N_samples,xval_opt.numFolds);
        
        for sampleIdx = 1:N_samples
            sample_ix = randsample(size(all_concat_PCs_noPreRew,2),K);
            sample_PCs = all_concat_PCs_noPreRew(:,sample_ix);
            sample_labels = all_concat_labels_noPreRew(sample_ix);
            
            foldid = repmat(1:xval_opt.numFolds,1,ceil(size(sample_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
            foldid = foldid(1:size(sample_PCs,2));
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data (both from our sample)
                data_train = sample_PCs(pc_range,foldid~=fIdx);
                labels_train = sample_labels(foldid~=fIdx);
                data_test = sample_PCs(pc_range,foldid==fIdx);
                labels_test = sample_labels(foldid==fIdx);
                
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
                    accuracies_allRewsize(sampleIdx,fIdx,tIdx) = accuracy;
                    precisions_allRewsize(sampleIdx,fIdx,tIdx) = precision;
                    TP_rates_allRewsize(sampleIdx,fIdx,tIdx) = TP_rate;
                    FP_rates_allRewsize(sampleIdx,fIdx,tIdx) = FP_rate;
                end
                
                ROC_AUC_dx = -squeeze(diff(FP_rates_allRewsize(sampleIdx,fIdx,:)));
                ROC_AUC_allRewsize(sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_allRewsize(sampleIdx,fIdx,1:end-1)));
                PR_AUC_dx = -squeeze(diff(TP_rates_allRewsize(sampleIdx,fIdx,:)));
                PR_AUC_allRewsize(sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_allRewsize(sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_allRewsize(sampleIdx,fIdx,~isnan(precisions_allRewsize(sampleIdx,fIdx,1:end-1)))));
            end
        end 
        disp("Control Complete")
    end
    
    mean_ROC_AUC_1rewsize = mean(ROC_AUC_1rewsize,[2,3]); 
    sem_ROC_AUC_1rewsize = 1.96 * std(ROC_AUC_1rewsize,[],[2,3]);
    mean_PR_AUC_1rewsize = mean(PR_AUC_1rewsize,[2,3]); 
    sem_PR_AUC_1rewsize = 1.96 * std(PR_AUC_1rewsize,[],[2,3]);
    
    mean_data = [mean_ROC_AUC_1rewsize mean_PR_AUC_1rewsize]; 
    sem_data = [sem_ROC_AUC_1rewsize sem_PR_AUC_1rewsize]; 
    
    control_mean_ROC_AUC = mean(ROC_AUC_allRewsize(:)); 
    control_sem_ROC_AUC = 1.96 * std(ROC_AUC_allRewsize(:)); 
    control_mean_PR_AUC = mean(PR_AUC_allRewsize(:)); 
    control_sem_PR_AUC = 1.96 * std(PR_AUC_allRewsize(:)); 
    
    figure() 
    hb = bar([1,2,3],mean_data);  
    xticklabels(["1 reward","2 rewards","3 rewards"])
    title("Fixed decision threshold test acr # rewards received")
%     hold on
%     % For each set of bars, find the centers of the bars, and write error bars
%     pause(0.1); %pause allows the figure to be created
%     for ib = 1:numel(hb)
%         %XData property is the tick labels/group centers; XOffset is the offset
%         %of each distinct group
%         xData = hb(ib).XData+hb(ib).XOffset; 
%         disp(xData)
%         errorbar(xData,mean_data(:,ib),sem_data(:,ib),'k.')
%     end 
    
    hold on; 
    yline(control_mean_ROC_AUC,'b-','linewidth',1.5)
    yline(control_mean_ROC_AUC - control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_ROC_AUC + control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_PR_AUC,'r-','linewidth',1.5)
    yline(control_mean_PR_AUC - control_sem_PR_AUC,'r:','linewidth',1)
    yline(control_mean_PR_AUC + control_sem_PR_AUC,'r:','linewidth',1)
end

%% Now do PRT quartile threshold testing
close all
figcounter = 1;

for sIdx = 24:24
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
    all_concat_prt_quartile_noPrewRew = vertcat(classification_struct(sIdx).prt_quartile_noPreRew{:})';
    all_concat_vel_noPreRew = horzcat(classification_struct(sIdx).vel_noPreRew{:});
    
    % folds are going to be over points that did not directly precede reward
    points = 1:numel(all_concat_prt_quartile_noPrewRew);
    
    % make folds
    xval_opt = struct;
    xval_opt.numFolds = 10;
    
    threshold_step = .05;
    thresholds = 0:threshold_step:1;
    pc_ranges = 1:10;
    N_samples = 50;
    prt_quartile_range = 1:4;
    pc_range = 1:10; 
    
    % get sample size as min(nPts) / 2
    prt_quartile_nPts = nan(3,1);
    for i_prt_quartile = 1:4
        prt_quartile_nPts(i_prt_quartile) = length(find(all_concat_prt_quartile_noPrewRew == prt_quartile_range(i_prt_quartile)));
    end 
    K = round(min(prt_quartile_nPts) / 2); 
    
    new_xval = false;
    if new_xval == true
        % set up datastructures to measure classification fidelity
        accuracies_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds);
        PR_AUC_1quartile = nan(numel(prt_quartile_range),N_samples,xval_opt.numFolds);
        
        for i_prt_quartile = 1:4
            this_prt_quartile = prt_quartile_range(i_prt_quartile);
            prt_quartile_PCs = all_concat_PCs_noPreRew(:,all_concat_prt_quartile_noPrewRew == this_prt_quartile);
            prt_quartile_labels = all_concat_labels_noPreRew(:,all_concat_prt_quartile_noPrewRew == this_prt_quartile); 
            
            for sampleIdx = 1:N_samples 
                sample_ix = randsample(size(prt_quartile_PCs,2),K); 
                sample_PCs = prt_quartile_PCs(:,sample_ix); 
                sample_labels = prt_quartile_labels(sample_ix);
                
                foldid = repmat(1:xval_opt.numFolds,1,ceil(size(prt_quartile_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
                foldid = foldid(1:size(prt_quartile_PCs,2));
                
                % Iterate over folds to use as test data
                for fIdx = 1:xval_opt.numFolds                    
                    % separate training and test data (both from our sample)
                    data_train = prt_quartile_PCs(pc_range,foldid~=fIdx);
                    labels_train = prt_quartile_labels(foldid~=fIdx);
                    data_test = prt_quartile_PCs(pc_range,foldid==fIdx);
                    labels_test = prt_quartile_labels(foldid==fIdx);
                    
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
                        accuracies_1quartile(i_prt_quartile,sampleIdx,fIdx,tIdx) = accuracy;
                        precisions_1quartile(i_prt_quartile,sampleIdx,fIdx,tIdx) = precision;
                        TP_rates_1quartile(i_prt_quartile,sampleIdx,fIdx,tIdx) = TP_rate;
                        FP_rates_1quartile(i_prt_quartile,sampleIdx,fIdx,tIdx) = FP_rate;
                    end
                    
                    ROC_AUC_dx = -squeeze(diff(FP_rates_1quartile(i_prt_quartile,sampleIdx,fIdx,:)));
                    ROC_AUC_1quartile(i_prt_quartile,sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_1quartile(i_prt_quartile,sampleIdx,fIdx,1:end-1)));
                    PR_AUC_dx = -squeeze(diff(TP_rates_1quartile(i_prt_quartile,sampleIdx,fIdx,:)));
                    PR_AUC_1quartile(i_prt_quartile,sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_1quartile(i_prt_quartile,sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_1quartile(i_prt_quartile,sampleIdx,fIdx,~isnan(precisions_1quartile(i_prt_quartile,sampleIdx,fIdx,1:end-1)))));
                end
            end 
            fprintf("PRT Quartile %i Complete \n",i_prt_quartile)
        end
        
        % here, perform control sampling w/ points chosen from all reward sizes
        accuracies_allPRTs = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        precisions_allPRTs = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        TP_rates_allPRTs = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        FP_rates_allPRTs = nan(N_samples,xval_opt.numFolds,numel(thresholds));
        ROC_AUC_allPRTs = nan(N_samples,xval_opt.numFolds);
        PR_AUC_allPRTs = nan(N_samples,xval_opt.numFolds);
        
        for sampleIdx = 1:N_samples
            sample_ix = randsample(size(all_concat_PCs_noPreRew,2),K);
            sample_PCs = all_concat_PCs_noPreRew(:,sample_ix);
            sample_labels = all_concat_labels_noPreRew(sample_ix);
            
            foldid = repmat(1:xval_opt.numFolds,1,ceil(size(sample_PCs,2) / xval_opt.numFolds)); % just repeat 1:nFolds
            foldid = foldid(1:size(sample_PCs,2));
            
            % Iterate over folds to use as test data
            for fIdx = 1:xval_opt.numFolds
                % separate training and test data (both from our sample)
                data_train = sample_PCs(pc_range,foldid~=fIdx);
                labels_train = sample_labels(foldid~=fIdx);
                data_test = sample_PCs(pc_range,foldid==fIdx);
                labels_test = sample_labels(foldid==fIdx);
                
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
                    accuracies_allPRTs(sampleIdx,fIdx,tIdx) = accuracy;
                    precisions_allPRTs(sampleIdx,fIdx,tIdx) = precision;
                    TP_rates_allPRTs(sampleIdx,fIdx,tIdx) = TP_rate;
                    FP_rates_allPRTs(sampleIdx,fIdx,tIdx) = FP_rate;
                end
                
                ROC_AUC_dx = -squeeze(diff(FP_rates_allPRTs(sampleIdx,fIdx,:)));
                ROC_AUC_allPRTs(sampleIdx,fIdx) = sum(ROC_AUC_dx .* squeeze(TP_rates_allPRTs(sampleIdx,fIdx,1:end-1)));
                PR_AUC_dx = -squeeze(diff(TP_rates_allPRTs(sampleIdx,fIdx,:)));
                PR_AUC_allPRTs(sampleIdx,fIdx) = sum(PR_AUC_dx(~isnan(precisions_allPRTs(sampleIdx,fIdx,1:end-1))) .* squeeze(precisions_allPRTs(sampleIdx,fIdx,~isnan(precisions_allPRTs(sampleIdx,fIdx,1:end-1)))));
            end
        end 
        disp("Control Complete")
    end
    
    mean_ROC_AUC_1quartile = mean(ROC_AUC_1quartile,[2,3]); 
    sem_ROC_AUC_1quartile = 1.96 * std(ROC_AUC_1quartile,[],[2,3]);
    mean_PR_AUC_1quartile = mean(PR_AUC_1quartile,[2,3]); 
    sem_PR_AUC_1quartile = 1.96 * std(PR_AUC_1quartile,[],[2,3]);
    
    mean_data = [mean_ROC_AUC_1quartile mean_PR_AUC_1quartile]; 
    sem_data = [sem_ROC_AUC_1quartile sem_PR_AUC_1quartile]; 
    
    control_mean_ROC_AUC = mean(ROC_AUC_allPRTs(:)); 
    control_sem_ROC_AUC = 1.96 * std(ROC_AUC_allPRTs(:)); 
    control_mean_PR_AUC = mean(PR_AUC_allPRTs(:)); 
    control_sem_PR_AUC = 1.96 * std(PR_AUC_allPRTs(:)); 
    
    figure() 
    hb = bar([1,2,3,4],mean_data);  
    xticklabels(["PRT Quartile 1","PRT Quartile 2","PRT Quartile 3","PRT Quartile 4"])
    title("Fixed decision threshold test acr PRT quartile")
%     hold on
%     % For each set of bars, find the centers of the bars, and write error bars
%     pause(0.1); %pause allows the figure to be created
%     for ib = 1:numel(hb)
%         %XData property is the tick labels/group centers; XOffset is the offset
%         %of each distinct group
%         xData = hb(ib).XData+hb(ib).XOffset; 
%         disp(xData)
%         errorbar(xData,mean_data(:,ib),sem_data(:,ib),'k.')
%     end 
    
    hold on; 
    yline(control_mean_ROC_AUC,'b-','linewidth',1.5)
    yline(control_mean_ROC_AUC - control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_ROC_AUC + control_sem_ROC_AUC,'b:','linewidth',1)
    yline(control_mean_PR_AUC,'r-','linewidth',1.5)
    yline(control_mean_PR_AUC - control_sem_PR_AUC,'r:','linewidth',1)
    yline(control_mean_PR_AUC + control_sem_PR_AUC,'r:','linewidth',1)
end
