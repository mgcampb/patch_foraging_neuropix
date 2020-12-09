%% Decode reward events via logistic regression... potentially with polynomial features?  
%  what does timecourse of reward delivery decoding look like? 
%  do axes of reward decoding change over timecourse decoded? 

%% Generic setup
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = .5; % in seconds; only takes within patch times up to this amount before patch leave
opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
opt.cortex_only = true;
tbin_ms = opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 15:18 23 25]; 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end

%% Load firing rate matrices, perform PCA
pca_trialed = cell(numel(sessions),1); 
TDR_struct = struct; 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);
    % Get the session name
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session));
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    
    % Get standardized PC transformation and smoothed fr mat
    opt.session = session; % session to analyze   
    new_load = true; % just for development purposes
    if new_load == true 
        [coeffs,fr_mat,good_cells,score,~,expl] = standard_pca_fn(paths,opt); 
    end
    % Get times to index firing rate matrix
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - 1000 * opt.patch_leave_buffer) / tbin_ms) + 1;
    
    % Gather firing rate matrices in trial form
    fr_mat_trials = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        fr_mat_trials{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial)); 
    end  
    score_full = coeffs' * zscore(horzcat(fr_mat_trials{:}),[],2); % s.t. indexing will line up 
    
    % Get new indexing vectors for our just on patch matrix
    t_lens = cellfun(@(x) size(x,2),fr_mat_trials); 
    new_patchleave_ix = cumsum(t_lens);
    new_patchstop_ix = new_patchleave_ix - t_lens + 1;   
    
    % Similarly gather PCA projections to explain 75% variance in cell arra
    surpass_75 = find(cumsum(expl / sum(expl)) > .75,1);
    pca_trialed{sIdx} = cell(length(dat.patchCSL),1);
    for iTrial = 1:length(dat.patchCSL)
        pca_trialed{sIdx}{iTrial} = score_full(1:surpass_75,new_patchstop_ix(iTrial,1):new_patchleave_ix(iTrial,1)); 
    end 
    
    TDR_struct(sIdx).pca_trials = pca_trialed{sIdx}; 
end    

%% Generate "reward barcodes" to average firing rates later
rew_barcodes = cell(numel(sessions),1);
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);
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

%% Set up classification problem
% - time on patch
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session)); 
    nTrials = length(dat.patchCSL); 
    rewsize = mod(dat.patches(:,2),10); 
    patchstop_sec = dat.patchCSL(:,2);
    patchleave_sec = dat.patchCSL(:,3);  
    prts = patchleave_sec - patchstop_sec; 
    floor_prts = floor(prts); 
    rew_sec = dat.rew_ts;  
    % index vectors 
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - 1000 * opt.patch_leave_buffer) / tbin_ms) + 1; 
    prts_ix = patchleave_ix - patchstop_ix + 1;
    
    % make barcode matrices to make task variables
    nTimesteps = 15;
    rew_barcode = zeros(nTrials , nTimesteps); 
    rew_sec_cell = cell(nTrials,1);
    for iTrial = 1:nTrials
        rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_sec_cell{iTrial} = rew_indices; % use all rewards (rew_indices > 1); 
    end
    
    % Make decision variables
    time_on_patch = cell(nTrials,1);
    time_since_rew = cell(nTrials,1);   
    time2leave = cell(nTrials,1);  
    rew_num = cell(nTrials,1); 
    all_rewsizes_labels = cell(nTrials,1);  
    sep_rewsizes_labels = cell(nTrials,1); 
    for iTrial = 1:nTrials
        trial_len_ix = prts_ix(iTrial); 
        time_on_patch{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        time_since_rew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;   
        rew_num{iTrial} = zeros(trial_len_ix,1); 
        time2leave{iTrial} = fliplr(time_since_rew{iTrial});
        all_rewsizes_labels{iTrial} = zeros(trial_len_ix,1); 
        sep_rewsizes_labels{iTrial} = zeros(trial_len_ix,1); 
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = max(1,(rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms);
            time_since_rew{iTrial}(rew_ix:end) =  (1:length(time_since_rew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
            all_rewsizes_labels{iTrial}(rew_ix:min(trial_len_ix,rew_ix + round(1000/tbin_ms))) = 1;
            sep_rewsizes_labels{iTrial}(rew_ix:min(trial_len_ix,rew_ix + round(1000/tbin_ms))) = min(3,rewsize(iTrial)); % 4 -> 3 
            rew_num{iTrial}(rew_ix:end) = r;  
        end  
    end  
    TDR_struct(sIdx).time_on_patch = time_on_patch;
    TDR_struct(sIdx).time_since_rew = time_since_rew;  
    TDR_struct(sIdx).rew_num = rew_num;
    TDR_struct(sIdx).time2leave = time2leave; 
    TDR_struct(sIdx).all_rewsizes_labels = all_rewsizes_labels; 
    TDR_struct(sIdx).sep_rewsizes_labels = sep_rewsizes_labels; 
end 

%% Now fit logistic regression between PCA and reward events 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);    
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    nTrials = length(dat.patchCSL); 
    rewsize = mod(dat.patches(:,2),10);  
    
    % concatenate task variable and pca data
    all_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).all_rewsizes_labels{:}));
    sep_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).sep_rewsizes_labels{:}));
    pca_full = cat(2,TDR_struct(sIdx).pca_trials{:})'; 
    
    % Make folds for xval to evaluate reward event decoding fidelity
    xval_opt = struct;
    xval_opt.numFolds = 2;
    xval_opt.rew_size = [1,2,4];
    % split trials into groups (num groups = opt.numFolds) 
    trials = 1:nTrials; 
    [trials,~,IC] = unique(trials);
    data_grp = nan(size(trials));
    shift_by = 0; % to make sure equal numbers of trials end up in each fold
    for iRewsize = 1:numel(xval_opt.rew_size)
        keep_this = (rewsize == xval_opt.rew_size(iRewsize));
        data_grp_this = repmat(circshift(1:xval_opt.numFolds,shift_by),1,ceil(sum(keep_this)/xval_opt.numFolds)*xval_opt.numFolds);
        data_grp(keep_this) = data_grp_this(1:sum(keep_this)); % assign folds 1:10
        shift_by = shift_by - mod(sum(keep_this),xval_opt.numFolds); % shift which fold is getting fewer trials
    end
    foldid = data_grp(IC)';   
    
    % fit linear logistic regression model first   
    B_all = cell(xval_opt.numFolds,1); 
    B_sep = cell(xval_opt.numFolds,1); 
    for test_fold = 1:xval_opt.numFolds 
        train_pca = TDR_struct(sIdx).pca_trials(foldid ~= test_fold); 
        train_pca = cat(2,train_pca{:})'; 
        train_all_labels = TDR_struct(sIdx).all_rewsizes_labels(foldid ~= test_fold); 
        train_all_labels = categorical(1 + cat(1,train_all_labels{:})); 
        train_sep_labels = TDR_struct(sIdx).sep_rewsizes_labels(foldid ~= test_fold); 
        train_sep_labels = categorical(1 + cat(1,train_sep_labels{:}));  
        % fit models on training data
        [B_all_fold,~,~] = mnrfit(train_pca,train_all_labels);  
        [B_sep_fold,~,~] = mnrfit(train_pca,train_sep_labels);   
        B_all{test_fold} = B_all_fold;  
        B_sep{test_fold} = B_sep_fold;
    end
    
%     [B_all,~,~] = mnrfit(pca_full,all_labels_full);  
%         [B_sep,~,~] = mnrfit(pca_full,sep_labels_full);  
%     pihat_all = mnrval(B_all,pca_full);  
%     pihat_sep = mnrval(B_sep,pca_full);   
    
    TDR_struct(sIdx).foldid = foldid; % save fold identities
    TDR_struct(sIdx).B_all = B_all; % save fold decoders
    TDR_struct(sIdx).B_sep = B_sep; % save fold decoders
    
    fprintf("Session %s Complete \n",session_titles{i})
    
end   

%% Put predictions into trialed form
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i); 
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    prts = dat.patchCSL(:,3) - dat.patchCSL(:,2); 
    rewsize = mod(dat.patches(:,2),10);  
    t_lens = cellfun(@(x) size(x,2),TDR_struct(sIdx).pca_trials); 
    foldid = TDR_struct(sIdx).foldid;
    
    pihat_all = cell(length(t_lens),1); 
    pihat_sep = cell(length(t_lens),1);   
    predictions_all = cell(length(t_lens),1);  
    predictions_sep = cell(length(t_lens),1);  
    projections_all = cell(length(t_lens),1); 
    projections_sep = cell(length(t_lens),1); 
    rewsize_trialed = cell(length(t_lens),1); 
    prt_trialed = cell(length(t_lens),1); 
    
    for iTrial = 1:numel(t_lens)  
        % Create probability predictions from xval data
        pihat_all_trial = mnrval(TDR_struct(sIdx).B_all{foldid(iTrial)},TDR_struct(sIdx).pca_trials{iTrial}');
        pihat_sep_trial = mnrval(TDR_struct(sIdx).B_sep{foldid(iTrial)},TDR_struct(sIdx).pca_trials{iTrial}'); 
        [~,predictions_all_iTrial] = max(pihat_all_trial,[],2); 
        [~,predictions_sep_iTrial] = max(pihat_sep_trial,[],2);  
        
        ones_pca_trial = [ones(1,size(TDR_struct(sIdx).pca_trials{iTrial},2)) ; TDR_struct(sIdx).pca_trials{iTrial}];
        projections_all{iTrial} = TDR_struct(sIdx).B_all{foldid(iTrial)}' * ones_pca_trial; 
        projections_sep{iTrial} = TDR_struct(sIdx).B_sep{foldid(iTrial)}' * ones_pca_trial; 
        
        pihat_all{iTrial} = pihat_all_trial;
        pihat_sep{iTrial} = pihat_sep_trial; 
        predictions_all{iTrial} = predictions_all_iTrial; 
        predictions_sep{iTrial} = predictions_sep_iTrial;
        rewsize_trialed{iTrial} = rewsize(iTrial) + zeros(t_lens(iTrial),1); 
        prt_trialed{iTrial} = prts(iTrial) + zeros(t_lens(iTrial),1); 
    end 
    TDR_struct(sIdx).pihat_all = pihat_all; 
    TDR_struct(sIdx).pihat_sep = pihat_sep;   
    TDR_struct(sIdx).predictions_all = predictions_all;
    TDR_struct(sIdx).predictions_sep = predictions_sep; 
    TDR_struct(sIdx).projections_all = projections_all; 
    TDR_struct(sIdx).projections_sep = projections_sep; 
    TDR_struct(sIdx).rewsize_trialed = rewsize_trialed;
    TDR_struct(sIdx).prt_trialed = prt_trialed;
end 

%% Visualize confusion matrix for classification 
close all 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    % concatenate task variable and pca data
    all_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).all_rewsizes_labels{:}));
    sep_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).sep_rewsizes_labels{:}));
    all_predictions = cat(1,TDR_struct(sIdx).predictions_all{:});
    sep_predictions = cat(1,TDR_struct(sIdx).predictions_sep{:});
    
    figure() 
    subplot(1,2,2)
    cm1 = confusionchart(double(sep_labels_full),sep_predictions);
    cm1.RowSummary = 'row-normalized';
    cm1.ColumnSummary = 'column-normalized';
    subplot(1,2,1)
    cm2 = confusionchart(double(all_labels_full),all_predictions);
    cm2.RowSummary = 'row-normalized';
    cm2.ColumnSummary = 'column-normalized';  
    suptitle(session_titles(i)) 
end 

%% Average predictions on RX trials
RX_predictions_all = cell(numel(mPFC_sessions),6); 
RX_predictions_sep = cell(numel(mPFC_sessions),6);  
R0_predictions_sep_prt = cell(numel(mPFC_sessions),6);  
sec2ix = round(2000 / tbin_ms);  

for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    dat = load(fullfile(paths.data,sessions{sIdx}));  
    prts = dat.patchCSL(:,3) - dat.patchCSL(:,2); 
    rew_barcode = rew_barcodes{sIdx};  
    
    rew_counter = 1; 
    % separate by RX and PRT short vs long
    for iRewsize = [1,2,4]
        trialsr0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55); 
        r0x_quantiles = [0 quantile(prts(trialsr0x),2) max(prts(trialsr0x))];   
        if ~isempty(trialsr0x)
            [~,~,bin] = histcounts(prts(trialsr0x),r0x_quantiles);  
            trialsr0x_short = trialsr0x(bin == 1);
            trialsr0x_long = trialsr0x(bin == 3); 
        end
        trialsrrx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);  

        % first binary predictions
        tmp_r0_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_all(trialsr0x),'UniformOutput',false); 
        mean_r0 = mean(cat(3,tmp_r0_cell{:}),3);    
        tmp_rr_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_all(trialsrrx),'UniformOutput',false); 
        mean_rr = mean(cat(3,tmp_rr_cell{:}),3);    
        RX_predictions_all{sIdx,rew_counter} = mean_r0; 
        RX_predictions_all{sIdx,rew_counter+3} = mean_rr;  
        
        % next, reward size specific predictions 
        tmp_r0_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_sep(trialsr0x),'UniformOutput',false); 
        mean_r0 = mean(cat(3,tmp_r0_cell{:}),3);    
        tmp_rr_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_sep(trialsrrx),'UniformOutput',false); 
        mean_rr = mean(cat(3,tmp_rr_cell{:}),3);    
        RX_predictions_sep{sIdx,rew_counter} = mean_r0;
        RX_predictions_sep{sIdx,rew_counter+3} = mean_rr;
        
        if ~isempty(trialsr0x)
            % last, r0 but separate by prt short/long
            tmp_r0xshort_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_sep(trialsr0x_short),'UniformOutput',false);
            mean_r0short = mean(cat(3,tmp_r0xshort_cell{:}),3);
            tmp_r0xlong_cell = cellfun(@(x) x(1:sec2ix,:),TDR_struct(sIdx).pihat_sep(trialsr0x_long),'UniformOutput',false);
            mean_r0long = mean(cat(3,tmp_r0xlong_cell{:}),3);
            R0_predictions_sep_prt{sIdx,rew_counter} = mean_r0short;
            R0_predictions_sep_prt{sIdx,rew_counter+3} = mean_r0long;
        end
        
        rew_counter = rew_counter + 1;
    end 
end  

%% Visualize RX traces binary rewarded
colors = {[0 0 0],[1 0 0]};   
conditions = ["10","20","40","11","22","44"]; 
close all
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    figure();hold on
    for condIdx = 1:6 
        if ~isempty(RX_predictions_sep{sIdx,condIdx})
            subplot(2,3,condIdx);hold on
            for k = 1:2
                plot(RX_predictions_all{sIdx,condIdx}(:,k),'color',colors{k},'linewidth',2)
            end
            title(conditions(condIdx))
            ylim([0 1])
            xticks([0 25 50 75 100])
            xticklabels([0 .5 1 1.5 2])
        end
    end 
    legend(["P(Not Rewarded)","P(Rewarded)"])
    suptitle(session_titles{i}) 
end

%% Visualize RX traces rewsize separated
colors = {[0 0 0],[0 1 1],[.5 .5 1],[1 0 1]};   
conditions = ["10","20","40","11","22","44"];
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i); 
    figure();hold on
    for condIdx = 1:6 
        if ~isempty(RX_predictions_sep{sIdx,condIdx})
            subplot(2,3,condIdx);hold on
            for k = 1:4
                plot(RX_predictions_sep{sIdx,condIdx}(:,k),'color',colors{k},'linewidth',2)
            end
            title(conditions(condIdx))
            ylim([0 1])
            xticks([0 25 50 75 100])
            xticklabels([0 .5 1 1.5 2])
        end
    end  
    suptitle(session_titles{i}) 
end 

%% Visualize R0 traces rewsize separated, split by PRT

colors = {[0 0 0],[0 1 1],[.5 .5 1],[1 0 1]};   
conditions = ["10 Short","20 Short","40 Short","10 Long","20 Long","40 Long"];
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i); 
    figure();hold on
    for condIdx = 1:6 
        if ~isempty(R0_predictions_sep_prt{sIdx,condIdx})
            subplot(2,3,condIdx);hold on
            for k = [1 4]
                plot(R0_predictions_sep_prt{sIdx,condIdx}(:,k),'color',colors{k},'linewidth',2)
            end
            title(conditions(condIdx))
            ylim([0 1])
            xticks([0 25 50 75 100])
            xticklabels([0 .5 1 1.5 2])
        end
    end  
    suptitle(session_titles{i}) 
end

%% More direct PRT correlation; fit linear model between rewsize probabilities and time2leave 
% look within reward sizes and across reward sizes  
% PRT or time2leave? difference?
for i = 18 % 1:numel(mPFC_sessions)  
    sIdx = mPFC_sessions(i); 
    dat = load(fullfile(paths.data,sessions{sIdx}));  
    prts = dat.patchCSL(:,3) - dat.patchCSL(:,2); 
    rew_barcode = rew_barcodes{sIdx};
    
    time_bins = 0:.1:2; 
    regression_data = cell(3,numel(time_bins));
  
    trials1Nil = find(rew_barcode(:,1) == 1 & rew_barcode(:,2) < 0);    
    trials2Nil = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) < 0);   
    trials4Nil = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) < 0);    
    rNil_trials = {trials1Nil trials2Nil trials4Nil};  
    Rsquared = nan(3,numel(time_bins));  
    p = nan(3,numel(time_bins)); 
    coeffs = nan(3,numel(time_bins),4); 
    for iRewsize = 1:numel(rNil_trials)  
        pihat_trialGroup = TDR_struct(sIdx).projections_sep(rNil_trials{iRewsize}); 
        pihat_trialGroup = cat(2,pihat_trialGroup{:})'; 
        timeOnPatch_trialGroup = TDR_struct(sIdx).time_on_patch(rNil_trials{iRewsize});  
        timeOnPatch_trialGroup = cat(2,timeOnPatch_trialGroup{:});   
        time2leave_trialGroup = TDR_struct(sIdx).time2leave(rNil_trials{iRewsize});   
        time2leave_trialGroup = cat(2,time2leave_trialGroup{:});
        prt_trialGroup = TDR_struct(sIdx).prt_trialed(rNil_trials{iRewsize});  
        prt_trialGroup = cat(1,prt_trialGroup{:}); 
        % bin time since reward
        [~,~,time_on_patch_bins] = histcounts(timeOnPatch_trialGroup,time_bins); 
        for kBin = 1:max(time_on_patch_bins) 
            pihat_kBin = pihat_trialGroup(time_on_patch_bins == kBin,:); 
            time2leave_kBin = time2leave_trialGroup(time_on_patch_bins == kBin); 
            kBin_lm = fitlm(pihat_kBin,time2leave_kBin,'intercept',true); 
            p(iRewsize,kBin) = anova(kBin_lm,'summary').pValue(2); 
            Rsquared(iRewsize,kBin) = kBin_lm.Rsquared(1).Adjusted; 
            coeffs(iRewsize,kBin,:) = kBin_lm.Coefficients.Estimate(2:end);
        end
    end
end

%% How does accuracy change w.r.t. time since reward? 
close all
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    rewsize = mod(dat.patches(:,2),10);  
    time_on_patch_bins = linspace(0,2,21);  
    
    % get the data
    time_since_full = cat(2,TDR_struct(sIdx).time_since_rew{:}); 
    time_patch_full = cat(2,TDR_struct(sIdx).time_on_patch{:}); 
    labels_all_full = 1 + cat(1,TDR_struct(sIdx).all_rewsizes_labels{:}); 
    labels_sep_full = 1 + cat(1,TDR_struct(sIdx).sep_rewsizes_labels{:});   
    rewsize_full  =  cat(1,TDR_struct(sIdx).rewsize_trialed{:})';
    predictions_all_full = cat(1,TDR_struct(sIdx).predictions_all{:});
    predictions_sep_full = cat(1,TDR_struct(sIdx).predictions_sep{:});

    no_t0 = double(true);
    
    % this is where we do our little evaluation metric 
    accuracies_all_all = nan(length(time_on_patch_bins),1); % binary predictions on all trials
    accuracies_sep_all = nan(length(time_on_patch_bins),1); % rewsize specific predictions on all trials
    accuracies_all_sep = nan(length(time_on_patch_bins),3); % binary predictions separated by rewsize
    accuracies_sep_sep = nan(length(time_on_patch_bins),3); % rewsize specific predictions separated by rewsize
    [~,~,bin] = histcounts(time_since_full,time_on_patch_bins); 
    for iBin = 1:max(bin)  
        accuracies_all_all(iBin) = length(find(labels_all_full(bin == iBin) == predictions_all_full(bin == iBin))) / length(labels_all_full(bin == iBin)); 
        accuracies_sep_all(iBin) = length(find(labels_sep_full(bin == iBin) == predictions_sep_full(bin == iBin))) / length(labels_sep_full(bin == iBin)); 
        for iRewsize = [1,2,4] 
            accuracies_all_sep(iBin,min(3,iRewsize)) = length(find(labels_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin) ...
                                                                == predictions_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin))) ... 
                                                                      / length(labels_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin)); 
            accuracies_sep_sep(iBin,min(3,iRewsize)) = length(find(labels_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin) ... 
                                                                == predictions_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin))) ... 
                                                                      / length(labels_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & bin == iBin)); 
        end
    end
    colors = cool(3);
    
    figure()  
    subplot(2,2,1)
    plot(accuracies_all_all,'linewidth',2)  
    xticks(1:5:numel(time_on_patch_bins))
    xticklabels(time_on_patch_bins(1:5:end))  
    title("Binary Classification Accuracy") 
    xlabel("Time Since Reward") 
    ylim([0,1])
    subplot(2,2,3);hold on
    for k = 1:3
        plot(accuracies_all_sep(:,k),'linewidth',2,'color',colors(k,:))   
    end
    xticks(1:5:numel(time_on_patch_bins))
    xticklabels(time_on_patch_bins(1:5:end))  
    title("Binary Classification Accuracy") 
    legend(["1 uL","2 uL","4 uL"]) 
    xlabel("Time Since Reward") 
    ylim([0,1])
    subplot(2,2,2) 
    plot(accuracies_sep_all,'linewidth',2)   
    xticks(1:5:numel(time_on_patch_bins))
    xticklabels(time_on_patch_bins(1:5:end))   
    title("Reward size Classification Accuracy") 
    xlabel("Time Since Reward") 
    ylim([0,1])
    subplot(2,2,4);hold on
    for k = 1:3
        plot(accuracies_sep_sep(:,k),'linewidth',2,'color',colors(k,:))   
    end
    xticks(1:5:numel(time_on_patch_bins))
    xticklabels(time_on_patch_bins(1:5:end))   
    title("Reward size Classification Accuracy")
    legend(["1 uL","2 uL","4 uL"]) 
    xlabel("Time Since Reward") 
    ylim([0,1])
    
    suptitle(session_titles{i})
end

%% And reward number 
close all
for i = numel(mPFC_sessions)
    sIdx = mPFC_sessions(i); 
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    rewsize = mod(dat.patches(:,2),10);  
    time_on_patch_bins = linspace(0,1,21);  
    
    % get the data
    time_since_full = cat(2,TDR_struct(sIdx).time_since_rew{:}); 
    time_patch_full = cat(2,TDR_struct(sIdx).time_on_patch{:});  
    rew_num_full = cat(1,TDR_struct(sIdx).rew_num{:})';  
    labels_all_full = 1 + cat(1,TDR_struct(sIdx).all_rewsizes_labels{:}); 
    labels_sep_full = 1 + cat(1,TDR_struct(sIdx).sep_rewsizes_labels{:});   
    rewsize_full  =  cat(1,TDR_struct(sIdx).rewsize_trialed{:})';
    [~,predictions_all_full] = max(cat(1,TDR_struct(sIdx).predictions_all{:}),[],2); 
    [~,predictions_sep_full] = max(cat(1,TDR_struct(sIdx).predictions_sep{:}),[],2);  

    no_t0 = double(false);
    
    rew_num_range = 1:5; 
    % this is where we do our little evaluation metric 
    accuracies_all_all = zeros(length(rew_num_range),1); % binary predictions on all trials
    accuracies_sep_all = zeros(length(rew_num_range),1); % rewsize specific predictions on all trials
    accuracies_all_sep = zeros(length(rew_num_range),3); % binary predictions separated by rewsize
    accuracies_sep_sep = zeros(length(rew_num_range),3); % rewsize specific predictions separated by rewsize
    for iRew_num = 1:max(rew_num_range)  
        accuracies_all_all(iRew_num) = length(find(labels_all_full(rew_num_full == iRew_num) == predictions_all_full(rew_num_full == iRew_num))) / length(labels_all_full(rew_num_full == iRew_num)); 
        accuracies_sep_all(iRew_num) = length(find(labels_sep_full(rew_num_full == iRew_num) == predictions_sep_full(rew_num_full == iRew_num))) / length(labels_sep_full(rew_num_full == iRew_num)); 
        for iRewsize = [1,2,4] 
            accuracies_all_sep(iRew_num,min(3,iRewsize)) = length(find(labels_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num) ...
                                                                == predictions_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num))) ... 
                                                                      / length(labels_all_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num)); 
            accuracies_sep_sep(iRew_num,min(3,iRewsize)) = length(find(labels_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num) ... 
                                                                == predictions_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num))) ... 
                                                                      / length(labels_sep_full(time_patch_full > no_t0 & rewsize_full == iRewsize & rew_num_full == iRew_num)); 
        end
    end  
    
    colors = [.4 .4 .4 ; cool(3)]; 
    figure() 
    subplot(2,1,1)
    b1 = bar([accuracies_all_all accuracies_all_sep],'FaceColor',"Flat");  
    title("Binary classification accuracy across reward number")
    subplot(2,1,2) 
    b2 = bar([accuracies_sep_all accuracies_sep_sep],'FaceColor',"Flat");   
    title("Reward size classification accuracy across reward number")
    suptitle(session_titles{i}) 
    
    for k = 1:numel(b1) 
        b1(k).CData = colors(k,:); 
        b2(k).CData = colors(k,:); 
    end
    
end

%% Try with ECOC multiclass SVM model 
% takes a long time
for i = 18 % 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);    
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    nTrials = length(dat.patchCSL); 
    rewsize = mod(dat.patches(:,2),10);  
    
    % concatenate task variable and pca data
    all_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).all_rewsizes_labels{:}));
    sep_labels_full = categorical(1 + cat(1,TDR_struct(sIdx).sep_rewsizes_labels{:}));
    pca_full = cat(2,TDR_struct(sIdx).pca_trials{:})'; 
    
    t = templateSVM('Standardize',true,'KernelFunction','gaussian'); % kernel SVMs
    Mdl = fitcecoc(pca_full,sep_labels_full,'Learners',t,'FitPosterior',true,...
    'Verbose',2);  

    [label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);

end 

%% testing 

quadratic_fn = @(beta,x) ((x - beta(1).^2) / beta(2) + beta(3) * x + beta(4));   

% nonlinear model fitting
% first turn our response into a binary array so we can use mnpdf
binary_resps = zeros(size(sep_labels_full,1),length(unique(sep_labels_full)));
for response = 1:4
    binary_resps(double(sep_labels_full) == response,response) = 1;
end

%     nonlinear response function
T = size(pca_full,1);
nonlinear_fn = @(beta,x) sum(((x - beta(:,1)).^2 ./ beta(:,2) + beta(:,3) .* x + beta(:,4)));
model_fn = @(beta,x) [exp(nonlinear_fn(squeeze(beta(:,:,1)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,2)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,3)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,4)),x))] ./ sum([exp(nonlinear_fn(squeeze(beta(:,:,1)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,2)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,3)),x)); ...
    exp(nonlinear_fn(squeeze(beta(:,:,4)),x))]); % normalize over cols
multinomialNLL = @(beta) -sum(log(mnpdf(binary_resps',reshape(model_fn(beta,pca_full(:,1:nPCs)'),[],T))));

nPCs = 5; % use fewer PCs
% find MLE quadratic model
beta0 = ones(nPCs,4,4); % nPCs x 4 parameters, 4 classes
disp(size(model_fn(beta0,pca_full(:,1:nPCs)')))
opts = optimset('fminsearch');
opts.MaxFunEvals = Inf;
opts.MaxIter = 10000;
betaHatML = fminsearch(multinomialNLL,beta0,opts);

function x_softmax = softmax(x)  
    x_softmax = exp(x) ./ sum(exp(x));
end 

