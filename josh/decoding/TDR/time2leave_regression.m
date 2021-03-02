%% Choice axis derivation by finding time 2 leave following reward reception 
% how well does this agree w/ the stimuli axes? 

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

%% Load firing rate matrices, perform PCA
pca_trialed = cell(numel(sessions),1); 
mPFC_sessions = [1:8 10:13 15:18 23 25]; 
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

%% Generate "reward barcodes" to average firing rates  
rew_barcodes = cell(numel(sessions),1);
for i = 1:numel(mPFC_sessions) 
    sIdx = mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % Trial data
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3) - opt.patch_leave_buffer;
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

%% Now regress PCs vs time to leave on trials after last rew reception 

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
    
    % gather regression data
    r_ix_final = nan(nTrials,1);  
    pca_postLeave = cell(nTrials,1);
    time2leave = cell(nTrials,1);  
    timeSinceRew = cell(nTrials,1); 
    for iTrial = 1:nTrials 
        trial_len_ix = prts_ix(iTrial);
        time2leave{iTrial} = (1:trial_len_ix) * tbin_ms / 1000; 
        rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < (patchleave_sec(iTrial) - opt.patch_leave_buffer)) - patchstop_sec(iTrial)) + 1;
        r_ix_final(iTrial) = max([1 ; (rew_indices - 1) * 1000 / tbin_ms]);  
        
        pca_postLeave{iTrial} = TDR_struct(sIdx).pca_trials{iTrial}(:,r_ix_final(iTrial):end); 
        time2leave{iTrial} = fliplr(time2leave{iTrial}(r_ix_final(iTrial):end) - time2leave{iTrial}(r_ix_final(iTrial))); 
        timeSinceRew{iTrial} = fliplr(time2leave{iTrial}); % this looks wack but i think it's going to be important?
    end  
    TDR_struct(sIdx).pca_postLeave = pca_postLeave; 
    TDR_struct(sIdx).time2leave = time2leave;  
    TDR_struct(sIdx).timeSinceRew = timeSinceRew;  
    
    % now perform regression 
    time2leave_full = cat(2,TDR_struct(sIdx).time2leave{:}); 
    pca_postLeaveFull = cat(2,TDR_struct(sIdx).pca_postLeave{:});   
    nPCs = size(pca_postLeaveFull,1);   
    
    % fit time2leave linear model
    beta = zeros(nPCs,1); 
    lm = fitlm(pca_postLeaveFull',time2leave_full,'intercept',true);
    significant_coeffs = (lm.Coefficients.pValue(2:end) < .01); 
    coeffs = lm.Coefficients.Estimate(2:end);
    beta(significant_coeffs) = coeffs(significant_coeffs);  
    
    TDR_struct(sIdx).beta_time2leave = beta;   
    TDR_struct(sIdx).p_time2leave = anova(lm,'summary').pValue(2); 
    TDR_struct(sIdx).Rsquared_time2leave = lm.Rsquared.Adjusted;
end

%% Fit models... analyze evolution of coefficients 
close all
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);    
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    time2leave_full = cat(2,TDR_struct(sIdx).time2leave{:});  
%     time2leave_full = time2leave_full - mean(time2leave_full);
    timeSinceRew_full = cat(2,TDR_struct(sIdx).timeSinceRew{:});    
%     timeSinceRew_full = timeSinceRew_full - mean(timeSinceRew_full); 
    pca_postLeaveFull = cat(2,TDR_struct(sIdx).pca_postLeave{:});   
    pca_postLeaveFull = pca_postLeaveFull(1:end,:);
    nPCs = size(pca_postLeaveFull,1); 
    
    % shuffle control
    shuffled_trials = randperm(length(TDR_struct(sIdx).timeSinceRew));
    timeSinceRew_shuffled = cat(2,TDR_struct(sIdx).timeSinceRew(shuffled_trials));  
    timeSinceRew_shuffled = cat(2,timeSinceRew_shuffled{:});  
%     timeSinceRew_shuffled = timeSinceRew_shuffled(randperm(length(timeSinceRew_shuffled)));
    
    beta = zeros(nPCs,1); 
    % fit linear model
    lm = fitlm(pca_postLeaveFull',time2leave_full,'intercept',true);
    significant_coeffs = (lm.Coefficients.pValue(2:end) < .01); 
    coeffs = lm.Coefficients.Estimate(2:end);
    beta(significant_coeffs) = coeffs(significant_coeffs);  
    
    TDR_struct(sIdx).beta = beta;   
    TDR_struct(sIdx).p = anova(lm,'summary').pValue(2); 
    TDR_struct(sIdx).Rsquared = lm.Rsquared.Adjusted;
    
    % Now try a few different windows after last reward
    timesince_bins = linspace(0,2,41); 
    beta_timesince = zeros(nPCs,numel(timesince_bins)-1);   
    pvalue_timesince = zeros(numel(timesince_bins)-1,1);  
    Rsquared_timesince = zeros(numel(timesince_bins)-1,1);  
    [~,~,timesince_bin_labels] = histcounts(timeSinceRew_full,timesince_bins); 
    for bin = 1:max(timesince_bin_labels)  
        iLm = fitlm(pca_postLeaveFull(:,timesince_bin_labels == bin)',time2leave_full(timesince_bin_labels == bin),'intercept',true); 
        significant_coeffs = (iLm.Coefficients.pValue(2:end) < .01); 
        beta_timesince(significant_coeffs,bin) = iLm.Coefficients.Estimate(significant_coeffs);   
        pvalue_timesince(bin) = anova(iLm,'summary').pValue(2);  
        Rsquared_timesince(bin) = iLm.Rsquared.Adjusted;
    end  
    TDR_struct(sIdx).beta_timesince = beta_timesince;   
    TDR_struct(sIdx).pvalue_timesince = pvalue_timesince; 
    TDR_struct(sIdx).Rsquared_timesince = Rsquared_timesince;

    figure()  
    subplot(2,2,1)
    imagesc(beta_timesince(1:10,:))
    title("Time to leave decoders")  
    xticks(1:10:length(timesince_bins))
    xticklabels(timesince_bins(1:10:end))  
    ylabel("PC Decoding Coefficients") 
    xlabel("Time since reward bin")
    subplot(2,2,2)
    imagesc(corrcoef(beta_timesince))  
    title("Correlation between time to leave decoders")  
    suptitle(session_title)  
    xticks(1:10:length(timesince_bins))
    xticklabels(timesince_bins(1:10:end)) 
    yticks(1:10:length(timesince_bins))
    yticklabels(timesince_bins(1:10:end))  
    caxis([-1,1]) 
    colorbar()
    subplot(2,2,3) 
    plot(Rsquared_timesince,'linewidth',2) 
    hold on 
    plot(find(pvalue_timesince < .01),Rsquared_timesince(pvalue_timesince < .01) + .1,'*')  
    xticks(1:10:length(timesince_bins))
    xticklabels(timesince_bins(1:10:end)) 
    title("Time to Leave R^2 w.r.t. time since reward")
end

%% Heatmap R^2 for different timewindow decoders 



%% WAIT CAN WE ALSO USE THE REWARD WE JUST RECEIVED?? maybe? or maybe w/ same rew schedule? 
% think about how this works...




%% old code 

    % Now try a few different windows pre-leave
    preleave_bins = linspace(0,2,41); 
    beta_preleave = zeros(nPCs,numel(preleave_bins)-1);   
    pvalue_preleave = zeros(numel(preleave_bins)-1,1);  
    Rsquared_preleave = zeros(numel(preleave_bins)-1,1);  
    [~,~,time2leave_bin_labels] = histcounts(time2leave_full,preleave_bins); 
    for bin = 1:max(time2leave_bin_labels)  
        bin_mean = time2leave_full(time2leave_bin_labels == bin);
        iLm = fitlm(pca_postLeaveFull(:,time2leave_bin_labels == bin)',time2leave_full(time2leave_bin_labels == bin) - bin_mean,'intercept',true); 
        significant_coeffs = (iLm.Coefficients.pValue(2:end) < .01); 
        beta_preleave(significant_coeffs,bin) = iLm.Coefficients.Estimate(significant_coeffs);   
        pvalue_preleave(bin) = anova(iLm,'summary').pValue(2);  
        Rsquared_preleave(bin) = iLm.Rsquared.Adjusted;
    end    
    TDR_struct(sIdx).beta_preleave = beta_preleave;   
    TDR_struct(sIdx).pvalue_preleave = pvalue_preleave; 
    TDR_struct(sIdx).Rsquared_preleave = Rsquared_preleave;
    
    % compare to a random signal rotation to see what structure is not real
    preleave_bins = linspace(0,2,41); 
    beta_shuffle = zeros(nPCs,numel(preleave_bins)-1);   
    pvalue_shuffle = zeros(numel(preleave_bins)-1,1);  
    Rsquared_shuffle = zeros(numel(preleave_bins)-1,1);  
    [~,~,timeSinceRew_bin_labels_shuffle] = histcounts(timeSinceRew_shuffled,preleave_bins); 
    for bin = 1:max(timeSinceRew_bin_labels_shuffle)
%         bin_mean = mean(timeSinceRew_shuffled(timeSinceRew_bin_labels_shuffle == bin));
        iLm = fitlm(pca_postLeaveFull(:,timeSinceRew_bin_labels_shuffle == bin)',timeSinceRew_shuffled(timeSinceRew_bin_labels_shuffle == bin) - bin_mean,'intercept',true); 
        significant_coeffs = (iLm.Coefficients.pValue(2:end) < .01); 
        beta_shuffle(significant_coeffs,bin) = iLm.Coefficients.Estimate(significant_coeffs);   
        pvalue_shuffle(bin) = anova(iLm,'summary').pValue(2);  
        Rsquared_shuffle(bin) = iLm.Rsquared.Adjusted;
    end

