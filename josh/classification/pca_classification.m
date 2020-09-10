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

    new_fr_mat = false;
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
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial));
        classification_struct(sIdx).rew_ix{iTrial} = round(rew_indices(rew_indices > 1) / tbin_ms); 
        classification_struct(sIdx).PCs{iTrial} = score(1:10,new_patchstop_ix(iTrial):new_patchleave_ix(iTrial)); 
        classification_struct(sIdx).labels{iTrial} = 1:t_lens(iTrial) > (t_lens(iTrial) - classification_zone / tbin_ms);
    end
end 

%% Visualize the classification problem on a few single trials  
close all
for sIdx = 3:3 
    test_trials = 11:19;  
    sp_counter = 1;
    figure()
    for iTrial = test_trials 
        subplot(3,3,sp_counter)
        gscatter(classification_struct(sIdx).PCs{iTrial}(1,:), ...
                 classification_struct(sIdx).PCs{iTrial}(2,:), ...
                 classification_struct(sIdx).labels{iTrial}, ... 
                 [],[],5) 
        title(sprintf("Trial %i",iTrial)) 
        xlabel("PC1"); ylabel("PC2")
        sp_counter = sp_counter + 1; 
        disp(sp_counter)
    end 
    
    % concatenate to show cross trial data
    concat_PCs = classification_struct(sIdx).PCs(test_trials);
    concat_PCs = horzcat(concat_PCs{:}); 
    concat_labels = classification_struct(sIdx).labels(test_trials); 
    concat_labels = horzcat(concat_labels{:}); 
    figure() 
    gscatter(concat_PCs(1,:),concat_PCs(2,:),concat_labels) 
    xlabel("PC1"); ylabel("PC2") 
    title("Labeled Points in PC Space") 
    
    % total concat pca 
    all_concat_PCs = horzcat(classification_struct(sIdx).PCs{:}); 
    all_concat_labels = horzcat(classification_struct(sIdx).labels{:});  
    figure() 
    gscatter(all_concat_PCs(1,:),all_concat_PCs(4,:),all_concat_labels) 
    xlabel("PC1"); ylabel("PC2") 
    title("Labeled Points in PC Space") 
end 

%% Perform logistic regression on labelled PCs

for sIdx = 3:3
    
    
end





