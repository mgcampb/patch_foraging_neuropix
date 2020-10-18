%% Basics
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% FR mat calculation settings
frCalc_opt = struct;
frCalc_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = frCalc_opt.tbin * 1000;
frCalc_opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};


%% Acquire PC reductions and a binary classification vector

classification_struct = struct; 

for sIdx = 3:3
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
    rew_ms = dat.rew_ts * 1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
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
    t_lens = cellfun(@(x) size(x,2),classification_struct(3).fr_mat_raw); 
    new_patchleave_ix = cumsum(t_lens);
    new_patchstop_ix = new_patchleave_ix - t_lens + 1; 
    classification_zone = 500; % how much time before leave we're labeling in ms
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
    end
end 

%% Now try out LDA 
close all 
for sIdx = 3:3
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:})';   
    all_concat_vel = horzcat(classification_struct(sIdx).vel{:})'; 
    all_concat_labels = horzcat(classification_struct(sIdx).labels{:}) + 1;    
    
    % fit LDA to all data
    lda = fitcdiscr(all_concat_PCs,all_concat_labels); 
    classes_pca = resubPredict(lda);  
    classes_pca_confusion = classes_pca;
    
    % now do k-fold xval to determine predictiveness
    cp = cvpartition(all_concat_labels,'KFold',10);
    cvlda = crossval(lda,'CVPartition',cp);
    ldaCVErr = kfoldLoss(cvlda);  
    
    misclassified = classes_pca_confusion ~= all_concat_labels'; 
    classes_pca_confusion(misclassified & classes_pca_confusion == 1) = 3;
    classes_pca_confusion(misclassified & classes_pca_confusion == 2) = 4; 
    
    % velocity control
    lda_vel = fitcdiscr(all_concat_vel,all_concat_labels); 
    cp_vel = cvpartition(all_concat_labels,'KFold',10); 
    cvlda_vel = crossval(lda_vel,'CVPartition',cp_vel);
    ldaCVErr_vel = kfoldLoss(cvlda_vel);   
    classes_vel = resubPredict(lda_vel);

    figure()  
%     gscatter(all_concat_PCs(misclassified,1),all_concat_PCs(misclassified,3),ldaClass(misclassified),[],'x') 
    hold on
    gscatter(all_concat_PCs(:,1),all_concat_PCs(:,3),classes_pca_confusion,'brcy','.') 
    xlabel("PC 1"); ylabel("PC 3") 
    title(sprintf("PC 1:10 Full Session LDA Results (10-fold X-Val Accuracy: %f)",1-ldaCVErr))
    legend("Predict Stay","Predict Leave in 500-1000 msec","Misclassified as Stay","Misclassified as Leave")  
    
    % now add PCs 1 by 1 and see how much X-val accuracy increases  
    xPC_ldaCVAcc = nan(10,1);
    for iPC = 1:10 
        lda = fitcdiscr(all_concat_PCs(:,1:iPC),all_concat_labels); 
        cp = cvpartition(all_concat_labels,'KFold',10);
        cvlda = crossval(lda,'CVPartition',cp);
        xPC_ldaCVAcc(iPC) = 1 - kfoldLoss(cvlda);  
    end  
    
    figure() 
    plot(xPC_ldaCVAcc,'linewidth',2)  
    title("LDA 10-Fold X-Val Accuracy using different ranges of PCs")
    xlabel("PCs included in LDA") 
    ylabel("10-fold X-Val Accuracy")
    
    % here, use kfold x-val to see if we can make some claims about
    % predictiveness 
    session = sessions{sIdx}(1:end-4); 
    data = load(fullfile(paths.data,session));
    patches = data.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10); 
    
    % confusion matrices for PCA, velocity 
    C_vel = confusionmat(all_concat_labels,classes_vel); 
    C_pca = confusionmat(all_concat_labels,classes_pca); 
    figure() 
    subplot(1,2,1) 
    confusionchart(C_vel,'RowSummary','row-normalized')  
    title("Confusion Matrix for Velocity LDA")
    subplot(1,2,2) 
    confusionchart(C_pca,'RowSummary','row-normalized') 
    title("Confusion Matrix for PCA LDA")
    
    
end