%% Targeted dimensionality reduction 
%  Driving questions: 
%  1. By looking at the low-dimensional components in neural
%     data that are correlated with task variables, can we predict leaving
%     with high fidelity?  
%  2. Does the trajectory of these low-dimensional components help us
%     understand a dynamic decision-making process?

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
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]};

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

%% Make task variables for regression
% - time on patch 
% - time since reward 
% - total rewards 
% - reward delivery event    
% - rewards above expected 

% avg CDF for reward above expected
tau = .125; 
N0 = .25; 
x = 0:tbin_ms/1000:150; 
avg_cdf = 1 - N0 * exp(-tau * x) / tau + N0 / tau;

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
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
    end
    
    % Make decision variables
    time_on_patch = cell(nTrials,1);
    time_since_rew = cell(nTrials,1);  
    rew_num = cell(nTrials,1); 
    total_uL = cell(nTrials,1); 
    rew_binary_early = cell(nTrials,1);  
    rew_binary_late = cell(nTrials,1);   
    rewcount_pe = cell(nTrials,1);
    X_trials = cell(nTrials,1); 
    for iTrial = 1:nTrials
        trial_len_ix = prts_ix(iTrial);
        time_on_patch{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        time_since_rew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;  
        rew_num{iTrial} = zeros(trial_len_ix,1);  
        total_uL{iTrial} = zeros(trial_len_ix,1);   
        rew_binary_early{iTrial} = zeros(trial_len_ix,1); 
        rew_binary_late{iTrial} = zeros(trial_len_ix,1); 
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            time_since_rew{iTrial}(rew_ix:end) =  (1:length(time_since_rew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
            rew_num{iTrial}(rew_ix:end) = r;  
            total_uL{iTrial}(rew_ix:end) = r * rewsize(iTrial); 
            rew_binary_early{iTrial}(rew_ix:min(trial_len_ix,rew_ix + round(500/tbin_ms))) = rewsize(iTrial);
            rew_binary_late{iTrial}(min(trial_len_ix,rew_ix + round(500/tbin_ms)):min(trial_len_ix,rew_ix + round(1000/tbin_ms))) = rewsize(iTrial);
        end  
        rewcount_pe{iTrial} = rew_num{iTrial}' - avg_cdf(1:length(rew_num{iTrial}));
        X_trials{iTrial} = [time_on_patch{iTrial}' time_since_rew{iTrial}' rew_num{iTrial} total_uL{iTrial} rew_binary_early{iTrial} rew_binary_late{iTrial} rewcount_pe{iTrial}']';
    end  
    
    TDR_struct(sIdx).X_trials = X_trials;
end  


%% Perform Regression against task events
%  would be nice to do this cross-validated for visualization
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);    
    dat = load(fullfile(paths.data,sessions{sIdx})); 
    nTrials = length(dat.patchCSL); 
    rewsize = mod(dat.patches(:,2),10);  
    
    % concatenate task variable and pca data
    X_full = cat(2,TDR_struct(sIdx).X_trials{:});
    pca_full = cat(2,TDR_struct(sIdx).pca_trials{:}); 
    nPCs = size(pca_full,1); 
    nTaskVars = size(X_full,1);   
    
    % Make folds for xval to evaluate coding fidelity per variable
    xval_opt = struct;
    xval_opt.numFolds = 5;
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

    % fit task variable regressions
    beta = zeros(nPCs,nTaskVars); 
    p_values = nan(nTaskVars,1); 
    Rsquared = nan(nTaskVars,1); 
    Rsquared_xval = nan(nTaskVars,xval_opt.numFolds);
    for vIdx = 1:nTaskVars
        lm = fitlm(pca_full',X_full(vIdx,:),'intercept',true);   
        significant_coeffs = (lm.Coefficients.pValue(2:end) < .01); 
        coeffs = lm.Coefficients.Estimate(2:end);
        beta(significant_coeffs,vIdx) = lm.Coefficients.Estimate(significant_coeffs);  
        p_values(vIdx) = anova(lm,'summary').pValue(2);  
        Rsquared(vIdx) = lm.Rsquared.Adjusted;  
        
        % now calculate cross-validated R^2 
        for f = 1:xval_opt.numFolds   
            % Fold training data
            pca_full_train = TDR_struct(sIdx).pca_trials(foldid~= f); 
            pca_full_train = cat(2,pca_full_train{:});  
            taskvars_train = TDR_struct(sIdx).X_trials(foldid ~= f);
            taskvars_train = cat(2,taskvars_train{:}); 
            % Fold test data
            pca_full_test = TDR_struct(sIdx).pca_trials(foldid == f); 
            pca_full_test = cat(2,pca_full_test{:});  
            taskvars_test = TDR_struct(sIdx).X_trials(foldid == f);
            taskvars_test = cat(2,taskvars_test{:});
            
            % Fit model on training data 
            fLm = fitlm(pca_full_train',taskvars_train(vIdx,:),'intercept',true);   
            fCoeffs = fLm.Coefficients.Estimate;  
            
            % Evaluate on test data
            pca_full_test_ones = [ones(size(pca_full_test',1),1) pca_full_test' ]; 
            taskvars_test_predictions = pca_full_test_ones * fCoeffs; 
            SS_tot = sum((taskvars_test(vIdx,:) - mean(taskvars_test(vIdx,:))).^2); 
            SS_res = sum((taskvars_test(vIdx,:) - taskvars_test_predictions').^2); 
            Rsquared_xval(vIdx,f) = 1 - SS_res / SS_tot;
        end
    end
    
    TDR_struct(sIdx).beta = beta;   
    TDR_struct(sIdx).beta_orth = orth(beta);
    TDR_struct(sIdx).p_values = p_values; 
    TDR_struct(sIdx).Rsquared = Rsquared;  
    TDR_struct(sIdx).Rsquared_xval = Rsquared_xval;
    TDR_struct(sIdx).Rsquared_xval_mean = mean(Rsquared_xval,2);
    TDR_struct(sIdx).Rsquared_xval_sd = std(Rsquared_xval,[],2);
end    

%% Visualize decoding fidelity of different variables across sessions   
regressors = ["Time on Patch","Time Since Reward","Reward Number","Total uL","0:500 msec Since Rew","500:1000 msec Since Rew","Reward Above Expected"]; 
session_titles = cell(numel(mPFC_sessions),1); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end
tdr_cell = squeeze(struct2cell(TDR_struct))';
Rsquared_xval_means = tdr_cell(mPFC_sessions,8);  
Rsquared_xval_means = cat(2,Rsquared_xval_means{:});
Rsquared_xval_sd = tdr_cell(mPFC_sessions,9);  
Rsquared_xval_sd = cat(2,Rsquared_xval_sd{:});
figure('Renderer', 'painters', 'Position', [300 300 1200 400]) 
b = bar(Rsquared_xval_means','Facecolor','Flat');   hold on 
taskVarColors = [0 0 1 ; .2 .4 1 ; 1 0 1 ;  .8 .2 .8 ; .7 0 0 ; 1 .3 .3 ; 1 0 0]; 
for i = 1:size(Rsquared_xval_means',2)
    x_pts = b(i).XEndPoints;  
    errorbar(x_pts(Rsquared_xval_means(i,:) > 0),Rsquared_xval_means(i,Rsquared_xval_means(i,:) > 0)',Rsquared_xval_sd(i,Rsquared_xval_means(i,:) > 0)','k.','linewidth',1)  
    b(i).CData = taskVarColors(i,:);
end 
legend(regressors) 

title("Mean Cross-Validated R^2 for PCA Task Variable Regressions Across Days") 

ylim([0,1])   
ylabel("Cross-Validated R^2")
xticks(1:length(Rsquared_xval_means))
xticklabels(session_titles) 
xtickangle(45)

%% Perform regression against time 2 leave
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
    pca_postRew = cell(nTrials,1);
    time2leave = cell(nTrials,1);  
    timeSinceRew = cell(nTrials,1); 
    for iTrial = 1:nTrials 
        trial_len_ix = prts_ix(iTrial);
        time2leave{iTrial} = (1:trial_len_ix) * tbin_ms / 1000; 
        rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < (patchleave_sec(iTrial) - opt.patch_leave_buffer)) - patchstop_sec(iTrial)) + 1;
        r_ix_final(iTrial) = max([1 ; (rew_indices - 1) * 1000 / tbin_ms]);  
        
        pca_postRew{iTrial} = TDR_struct(sIdx).pca_trials{iTrial}(:,r_ix_final(iTrial):end); 
        time2leave{iTrial} = fliplr(time2leave{iTrial}(r_ix_final(iTrial):end) - time2leave{iTrial}(r_ix_final(iTrial))); 
        timeSinceRew{iTrial} = fliplr(time2leave{iTrial}); % this looks wack but i think it's going to be important?
    end  
    TDR_struct(sIdx).pca_postRew = pca_postRew; 
    TDR_struct(sIdx).time2leave = time2leave;  
    TDR_struct(sIdx).timeSinceRew_postRew = timeSinceRew;  
    
    % Gather data
    time2leave_full = cat(2,TDR_struct(sIdx).time2leave{:}); 
    pca_postRewFull = cat(2,TDR_struct(sIdx).pca_postRew{:});   
    nPCs = size(pca_postRewFull,1);   
    
    % fit time2leave linear model
    beta_time2leave = zeros(nPCs,1); 
    lm = fitlm(pca_postRewFull',time2leave_full,'intercept',true);
    significant_coeffs = (lm.Coefficients.pValue(2:end) < .01); 
    coeffs = lm.Coefficients.Estimate(2:end);
    beta_time2leave(significant_coeffs) = coeffs(significant_coeffs);    
    
    % Now perform 
    % Make folds for xval to evaluate time2leave decoding fidelity
    xval_opt = struct;
    xval_opt.numFolds = 5;
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
    
    % fit task variable regressions
    Rsquared_time2leave_xval = nan(xval_opt.numFolds,1); 
    
    % now calculate cross-validated R^2
    for f = 1:xval_opt.numFolds
        % Fold training data
        pca_full_train = TDR_struct(sIdx).pca_postRew(foldid~= f);
        pca_full_train = cat(2,pca_full_train{:});
        time2leave_train = TDR_struct(sIdx).time2leave(foldid ~= f);
        time2leave_train = cat(2,time2leave_train{:});
        % Fold test data
        pca_full_test = TDR_struct(sIdx).pca_postRew(foldid == f);
        pca_full_test = cat(2,pca_full_test{:});
        time2leave_test = TDR_struct(sIdx).time2leave(foldid == f);
        time2leave_test = cat(2,time2leave_test{:});
        
        % Fit model on training data
        fLm = fitlm(pca_full_train',time2leave_train,'intercept',true);
        fCoeffs = fLm.Coefficients.Estimate;
        
        % Evaluate on test data
        pca_full_test_ones = [ones(size(pca_full_test',1),1) pca_full_test' ];
        time2leave_test_predictions = pca_full_test_ones * fCoeffs;
        SS_tot = sum((time2leave_test - mean(time2leave_test)).^2);
        SS_res = sum((time2leave_test - time2leave_test_predictions').^2);
        Rsquared_time2leave_xval(f) = 1 - SS_res / SS_tot;
    end
    
    TDR_struct(sIdx).beta_time2leave = beta_time2leave;   
    TDR_struct(sIdx).p_time2leave = anova(lm,'summary').pValue(2); 
    TDR_struct(sIdx).Rsquared_time2leave = lm.Rsquared.Adjusted; 
    TDR_struct(sIdx).Rsquared_time2leave_xval = Rsquared_time2leave_xval;
    TDR_struct(sIdx).Rsquared_time2leave_xval_mean = mean(Rsquared_time2leave_xval);
    TDR_struct(sIdx).Rsquared_time2leave_xval_sd = std(Rsquared_time2leave_xval);
end

%% visualize correlation between axes  
close all
regressors = ["Time on Patch","Time Since Reward","Reward Number","Total uL","0:500 msec Since Rew","500:1000 msec Since Rew","Reward Above Expected"]; 
RdBu = flipud(cbrewer('div','RdBu',100)); 
for i = 18
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    X_full = cat(2,TDR_struct(sIdx).X_trials{:});
    figure('Renderer', 'painters', 'Position', [300 300 1200 400])
    subplot(1,3,1)
    imagesc(corrcoef(TDR_struct(sIdx).beta)) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Correlations between Regression Axes",session_title))  
    subplot(1,3,2)
    imagesc(corrcoef(X_full')) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Correlations between Regressors",session_title))  
    subplot(1,3,3)
    imagesc(corrcoef(TDR_struct(sIdx).beta) - corrcoef(X_full')) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Corrcoef(Regressor Axes) - Corrcoef(Regressor) Correlations",session_title)) 
    
    TDR_struct(sIdx).betaCorrcoef = corrcoef(TDR_struct(sIdx).beta);
end 

%% Visualize correlation between taskvar coefficients and time2leave coefficient 
close all 
regressors = ["Time to Leave","Time on Patch","Time Since Reward","Reward Number","Total uL","0:500 msec Since Rew","500:1000 msec Since Rew","Reward Above Expected"]; 
RdBu = flipud(cbrewer('div','RdBu',100)); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    X_full = cat(2,TDR_struct(sIdx).X_trials{:});
    figure('Renderer', 'painters', 'Position', [350 350 500 500])
    imagesc(corrcoef([TDR_struct(sIdx).beta_time2leave TDR_struct(sIdx).beta])) 
    xticklabels(regressors) 
    xtickangle(45) 
    yticklabels(regressors) 
    ytickangle(45)   
    colormap(RdBu)  
    caxis([-1,1])
    colorbar() 
    title(sprintf("%s Correlations between Regression Axes",session_title))  
end 

%% Project PCs onto taskVar axes 
RX_means = cell(numel(mPFC_sessions),6); 
RXX_means = cell(numel(mPFC_sessions),12); 
rIdx = [1,5]; % orthogonalize visualization axes 
time2leave_projections = cell(numel(mPFC_sessions),1); 

for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session = sessions{sIdx}(1:end-4); 
    dat = load(fullfile(paths.data,session)); 
    patchstop_ms = dat.patchCSL(:,2);
    patchleave_ms = dat.patchCSL(:,3);  
    prts = patchleave_ms - patchstop_ms;  
    nTrials = length(prts);  
    rew_barcode = rew_barcodes{sIdx};
    
    TDR_struct(sIdx).projected_trials = cell(nTrials,1);  
    
%     beta_orth = orth(TDR_struct(sIdx).beta(:,rIdx)); 
    beta_orth = TDR_struct(sIdx).beta(:,rIdx); 
    
    for iTrial = 1:numel(TDR_struct(sIdx).X_trials) 
%         TDR_struct(sIdx).projected_trials{iTrial} = TDR_struct(sIdx).beta' * TDR_struct(sIdx).pca_trials{iTrial};
        TDR_struct(sIdx).projected_trials{iTrial} = beta_orth' * TDR_struct(sIdx).pca_trials{iTrial};
    end
    
    time2leave_projections{sIdx} = beta_orth' * TDR_struct(sIdx).beta_time2leave; % / norm(beta_orth' * TDR_struct(sIdx).beta_time2leave);
    
    % Now average projections over RX and RXX trials for visualization 
    sec2ix = round(2000 / tbin_ms); 
    sec3ix = round(3000 / tbin_ms);  
    rewsizes = [1,2,4];
    for r = 1:numel(rewsizes) 
        iRewsize = rewsizes(r);
        
        % RX trialtypes
        trialsR0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & prts > 2.55);
        trialsRRx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);  
        % RXX trialtypes
        trialsR00x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0 & prts > 3.55);
        trialsRR0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0 & prts > 3.55);
        trialsR0Rx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
        trialsRRRx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
    
        % average RX
        trialsR0x_cell = cellfun(@(x) x(:,1:sec2ix),TDR_struct(sIdx).projected_trials(trialsR0x),'UniformOutput',false);
        trialsRRx_cell = cellfun(@(x) x(:,1:sec2ix),TDR_struct(sIdx).projected_trials(trialsRRx),'UniformOutput',false);
        RX_means{sIdx,r} = mean(cat(3,trialsR0x_cell{:}),3); 
        RX_means{sIdx,r + 3} = mean(cat(3,trialsRRx_cell{:}),3); 
        
        % average RX
        trialsR00x_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsR00x),'UniformOutput',false);
        trialsRR0x_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsRR0x),'UniformOutput',false);
        trialsR0Rx_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsR0Rx),'UniformOutput',false);
        trialsRRRx_cell = cellfun(@(x) x(:,1:sec3ix),TDR_struct(sIdx).projected_trials(trialsRRRx),'UniformOutput',false); 
        RXX_means{sIdx,(r - 1) * 4 + 1} = mean(cat(3,trialsR00x_cell{:}),3); 
        RXX_means{sIdx,(r - 1) * 4 + 2} = mean(cat(3,trialsRR0x_cell{:}),3); 
        RXX_means{sIdx,(r - 1) * 4 + 3} = mean(cat(3,trialsR0Rx_cell{:}),3);  
        RXX_means{sIdx,(r - 1) * 4 + 4} = mean(cat(3,trialsRRRx_cell{:}),3); 
    end
end

%% Visualize projected axes RX trials  
colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};   
regressors = ["Time on Patch","Time Since Reward","Reward Number","Total uL","0:500 msec Since Rew","500:1000 msec Since Rew","Reward Above Expected"]; 
conds = 1:6;  
close all 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)]; 
    figure();hold on
    for condIdx = conds 
        if ~isempty(RX_means{sIdx,condIdx})
            plot(RX_means{sIdx,condIdx}(1,:),RX_means{sIdx,condIdx}(2,:),'color',colors{condIdx},'linewidth',1.5);hold on
            sec_ticks = 50;   
            plot(RX_means{sIdx,condIdx}(1,1),RX_means{sIdx,condIdx}(2,1), 'ko', 'markerSize', 6, 'markerFaceColor',colors{condIdx});
            plot(RX_means{sIdx,condIdx}(1,sec_ticks),RX_means{sIdx,condIdx}(2,sec_ticks), 'kd', 'markerSize', 6, 'markerFaceColor',colors{condIdx}); 
        end
    end  
    
    % get axis limits to draw arrows
    xl = xlim();
    yl = ylim();
    arrowSize = 5; 
    arrowGain = 0;
    arrowEdgeColor = 'k';
    for condIdx = conds  
        if ~isempty(RX_means{sIdx,condIdx})
            % for arrow, figure out last two points, and (if asked) supress the arrow if velocity is below a threshold.
            penultimatePoint = [RX_means{sIdx,condIdx}(1,end-1), RX_means{sIdx,condIdx}(2,end-1)];
            lastPoint = [RX_means{sIdx,condIdx}(1,end), RX_means{sIdx,condIdx}(2,end)];
            vel = norm(lastPoint - penultimatePoint);

            axLim = [xl yl];
            aSize = arrowSize + arrowGain * vel;  % if asked (e.g. for movies) arrow size may grow with vel
            arrowMMC(penultimatePoint, lastPoint, [], aSize, axLim, colors{condIdx}, arrowEdgeColor); 
        end
    end
    xlabel(sprintf("Projection onto %s Axis",regressors(rIdx(1))))
    ylabel(sprintf("Projection onto %s Axis",regressors(rIdx(2)))) 
    title(session_title) 
    
    plot(xlim,ylim * time2leave_projections{sIdx}(2) / time2leave_projections{sIdx}(1),'k--','linewidth',2) 
%     plot(xlim / time2leave_projections{sIdx}(1),ylim / time2leave_projections{sIdx}(2),'k--','linewidth',2)
end