%% Make Harvey 2012 style plots of averaged neural activity within bins of decision variables
%  sort by maximal firing rate to investigate single neuron vs population
%  hypotheses

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_matlab/neuroPixelsData/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_matlab/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = opt.tbin*1000; % for making index vectors
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

opt.maxtime_alignleave = 3000; %MB
opt.maxtime_alignstop = 3000; %MB

%MB *** need to incorporate this leaveBuffer in other plots aligned to patchStop, patchCue, etc
% can later do this trimming more precisely but looking directly at velocity per trial
opt.leaveBuffer_ms = 500; %MB 'buffer' time prior to 'patchLeave' being triggered to leave out of any plots aligned to patchStop or patchCue, to avoid corrupting on patch PSTHs etc w running just before patchLeave (since running will occur at different times relative to patchStop per patch)

opt.additionalBuffer = 200; %MB - ms to cut off from end of trials aligned to patchStop (temporary solution to reduce impact of slight misalignment of trials after running PCA on firing rate while onPatch), this is in addition to the leaveBuffer_ms


sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information

FR_decVar = struct;
trial_pc_traj = {{}};

for sIdx = 3:3 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    rew_size = mod(dat.patches(:,2),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    rew_ms = dat.rew_ts * 1000;
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
    
    p.patchstop_ms = patchstop_ms(keep);
    p.patchleave_ms = patchleave_ms(keep);
    
    % compute firing rate matrix
    tic
    [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
    toc
    
    % compute firing rate matrix off patch
    tic
    [fr_mat2,tbincent] = calcFRVsTime(good_cells, dat, opt); %MB includes only activity within patches
    toc
    
    %     fr_mat_zscore = my_zscore(fr_mat); % z-score our psth matrix
    
    patchstop_ms = p_out.patchstop_ms + 9; % + 9;
    patchleave_ms = p_out.patchleave_ms + 9; % + 9;
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round(patchleave_ms / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
    FR_decVar(sIdx).fr_mat = {length(dat.patchCSL)};
    for iTrial = 1:length(dat.patchCSL)
        FR_decVar(sIdx).fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
    end
    
    % perform PCA to use this in decision/integration variable estimation
    fr_mat_zscore = my_zscore(fr_mat);
    tic
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
    %     [u,s,v] = svd(fr_mat_zscore);
    toc
    
    pc_space_traj = score(:,1:6);
    
    % gather trajectories by trial using our new indices
    for iTrial = 1:numel(patchleave_ix)
        %         trial_pc_traj{sIdx}{iTrial} = pc_space_traj(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_pc_traj{sIdx}{iTrial} = pc_space_traj(patchstop_ix(iTrial):patchleave_ix(iTrial),1:3);
    end
    
    pc_reductions{sIdx} = score(:,1:6);
    
end

%% Define decision variables

for sIdx = 3:3 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices, maybe these will be useful, or at least this
    % code
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_sec_cell = {};
    for iTrial = 1:length(patchCSL)
        rew_sec = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_sec_cell{iTrial} = rew_sec(rew_sec > 1);
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_sec) = rewsize(iTrial);
    end
    
    % load these in from data structure
    
    m1fit = [1.6078 2.1854; 4.1571 0.7020; -1 -1; 9.5088 0.2480];
    m2fit = [1.1041 2.5000; 2.0799 1.8982; -1 -1; 2.5401 2.0451];
    m2fit_permute = [2.0799 1.8982; 2.5401 2.0451; -1 -1; 1.1041 2.5000;];
    m3fit = [1.6096 0.6942 2.5000; 1.6958  1.3746 2.3354; -1 -1 -1; 1.8815 1.7172 1.7627];
    m3fit_permute = [1.6958  1.3746 2.3354; 1.8815 1.7172 1.7627; -1 -1 -1; 1.6096 0.6942 2.5000;];
    m5fit = [8.86 0.93 0.68 2.29; 1.97 1.9 0.055 0.85; -1 -1 -1 -1; 516.5 0.12 0.76 0.65];
    
    FR_decVar(sIdx).decVar1 = {};
    FR_decVar(sIdx).decVar2 = {};
    FR_decVar(sIdx).decVar3 = {};
    FR_decVar(sIdx).decVar5 = {};
    FR_decVar(sIdx).decVarControl = {};
    FR_decVar(sIdx).decVarPerm2Control = {};
    FR_decVar(sIdx).decVarPerm3Control = {};
    FR_decVar(sIdx).decVarTime = {};
    for iTrial = 1:length(patchCSL)
        % PC 1
        FR_decVar(sIdx).decVarPC1{iTrial} = trial_pc_traj{sIdx}{iTrial}(:,1)';
        FR_decVar(sIdx).decVarPC3{iTrial} = trial_pc_traj{sIdx}{iTrial}(:,3)';
        trial_len_ix = size(FR_decVar(sIdx).fr_mat{iTrial},2);
        
        FR_decVar(sIdx).decVarTime{iTrial} = (1:trial_len_ix) * tbin_ms / 1000; % just plain time
        FR_decVar(sIdx).decVar1{iTrial} = m1fit(rewsize(iTrial),2) * (1:trial_len_ix) * tbin_ms/1000 - m1fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVar2{iTrial} = m2fit(rewsize(iTrial),2) * (1:trial_len_ix)* tbin_ms/1000 - m2fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarRawTimeSinceRew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        FR_decVar(sIdx).decVar3{iTrial} = m3fit(rewsize(iTrial),3) * (1:trial_len_ix) * tbin_ms/1000 - m3fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarPerm2Control{iTrial} = m2fit_permute(rewsize(iTrial),2) * (1:trial_len_ix)* tbin_ms/1000 - m2fit_permute(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarPerm3Control{iTrial} = m3fit_permute(rewsize(iTrial),3) * (1:trial_len_ix)* tbin_ms/1000 - m3fit_permute(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarRandControl{iTrial} = rand(1,trial_len_ix); % random vector
        
        % deal with resets
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end) = m2fit(rewsize(iTrial),2) * ((1:length(FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end)))* tbin_ms/1000 - m2fit(rewsize(iTrial),1));
            FR_decVar(sIdx).decVarRawTimeSinceRew{iTrial}(rew_ix:end) =  (1:length(FR_decVar(sIdx).decVarRawTimeSinceRew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
            FR_decVar(sIdx).decVar3{iTrial}(rew_ix:end) = (FR_decVar(sIdx).decVar3{iTrial}(rew_ix:end)) - r * m3fit(rewsize(iTrial),2);
            FR_decVar(sIdx).decVarPerm2Control{iTrial}(rew_ix:end) = m2fit_permute(rewsize(iTrial),2) * ((1:length(FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end)))* tbin_ms/1000 - m2fit_permute(rewsize(iTrial),1));
            FR_decVar(sIdx).decVarPerm3Control{iTrial}(rew_ix:end) = (FR_decVar(sIdx).decVar3{iTrial}(rew_ix:end) - m3fit_permute(rewsize(iTrial),1)) - r * m3fit_permute(rewsize(iTrial),2);
        end
        
        % these parameters are optimized for seconds timescale
        int_minusR = m5fit(rewsize(iTrial),1);
        beta = m5fit(rewsize(iTrial),2);
        decay = m5fit(rewsize(iTrial),3); % .^(1/tbin_ms);
        t0bonus_fractionMinusRew = m5fit(rewsize(iTrial),4);
        t0_bonus = int_minusR * t0bonus_fractionMinusRew;
        
        leakyRews = t0_bonus;
        r = 1;
        
        if ~isempty(rew_sec_cell{iTrial})
            rew_ix = [1; (rew_sec_cell{iTrial} - 1) * 1000/tbin_ms]; % add the reward at 0 s
        else
            rew_ix = 1;
        end
        
        next_rew_ix = rew_ix(r);
        
        for dIdx = 1:trial_len_ix
            
            % deliver reward
            if dIdx == next_rew_ix
                
                leakyRews = leakyRews + int_minusR/tbin_ms;
                r = r + 1;
                
                if length(rew_ix) >= r
                    next_rew_ix = rew_ix(r);
                end
            end
            
            leakyRews = leakyRews * (1 - decay)^(1/tbin_ms);
            int_currValue = dIdx / tbin_ms - leakyRews;
            
            FR_decVar(sIdx).decVar5{iTrial}(dIdx) = beta * int_currValue;
        end
    end
    % check out the decision variables on ex trial
    close all; figure();hold on;
    plot(FR_decVar(sIdx).decVar1{39})
    plot(FR_decVar(sIdx).decVar2{39})
    plot(FR_decVar(sIdx).decVar3{39})
    plot(FR_decVar(sIdx).decVar5{39})
    plot(FR_decVar(sIdx).decVarRawTimeSinceRew{39})
%     plot(FR_decVar(sIdx).decVarRandControl{39})
%     plot(FR_decVar(sIdx).decVarPC1{39})
%     plot(FR_decVar(sIdx).decVarPC3{39})
    %     plot(FR_decVar(sIdx).decVarPerm2Control{39})
    %     plot(FR_decVar(sIdx).decVarPerm3Control{39})
    legend("Model1","Model2","Model3","Model5","Raw time since last rew")%,"RandControl","PC1","PC3")
    title("Trial 39 decision variables")
    
end

%% Ridge plot all trials for a given decision variable, sorting and visualizing with the same order
% Compare to shuffle
% iterate through decision variables, making plots for all and distribution
% for shuffle control

% close all
for sIdx = 3:3
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    decVar_cell = FR_decVar(sIdx).decVarRawTimeSinceRew;
%     decVar_cell = FR_decVar(sIdx).decVarTime;
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,41);
    decVar_bins = quantile(all_decVar,p);
    nBins = length(decVar_bins);
    
    % collect FR matrices
    fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
    
    % shuffle data according to random rotation
    fr_mat_shuffle = zeros(size(fr_mat));
    shifts = randi(size(fr_mat,2),size(fr_mat,1),1);
    parfor neuron = 1:size(fr_mat,1)
        fr_mat_shuffle(neuron,:) = circshift(fr_mat(neuron,:),shifts(neuron));
    end
    
    decVar = cat(2,decVar_cell{:});
    
    %%% calculate FR averaged over decision variable bins for real data %%%
    avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if ~isempty(find(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
            avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
        elseif (dIdx > 1) & ~isempty(find(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),1))
            avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
        else
            avgFR_decVar(:,dIdx) = 0;
        end
    end
    
    avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);
%     avgFR_decVar = zscore(avgFR_decVar, [],2);
    [~,index] = max(avgFR_decVar');
    [~,index_sort] = sort(index);
    
    avgFR_decVar_sorted = avgFR_decVar(index_sort,:);
    
    %%% calculate FR averaged over decision variable bins for shuffled data %%%
    avgFR_decVar_shuffle = zeros(size(fr_mat_shuffle,1), numel(decVar_bins)-1);
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if ~isempty(find(fr_mat_shuffle(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
            avgFR_decVar_shuffle(:,dIdx) = mean(fr_mat_shuffle(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
        elseif (dIdx > 1) & ~isempty(find(fr_mat_shuffle(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),1))
            avgFR_decVar_shuffle(:,dIdx) = mean(fr_mat_shuffle(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
        end
    end
    avgFR_decVar_shuffle = avgFR_decVar_shuffle ./ max(avgFR_decVar_shuffle,[],2);
%     avgFR_decVar_shuffle = zscore(avgFR_decVar_shuffle,[],2);
    [~,index] = max(avgFR_decVar_shuffle');
    [~,index_sort] = sort(index);
    
    avgFR_shuffle_decVar_sorted = avgFR_decVar_shuffle(index_sort,:);
    
    figure()
%     subplot(1,2,2)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sorted))
    colorbar()
    title("FR normalized by max FR")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
%     xlim([0,91])
    xlim([0,41])
    xlabel("Time since last reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
%     subplot(1,2,1)
%     colormap('jet')
%     imagesc(flipud(avgFR_shuffle_decVar_sorted))
%     colorbar()
%     title("Shuffled FR normalized by max FR")
%     ylabel("Neuron")
%     xlabel("Time Since Last Reward")
% %     xlim([0,91])
%     xlim([0,41])
% %     xticks([0 23 45 68 90])
%     xticks([0 10 20 30 41])
% %     xticklabels({'5','25','50','75','95'})
%     xticklabels(decVar_bins([1,10,20,30,41]))
    
    % Now find ridge-to-background ratio for shuffled, unshuffled data
    nNeurons = size(avgFR_decVar_sorted,1);
    ridgeBaselineRatioUnshuffled = zeros(nNeurons,1);
    [~,max_unshuffled_ix] = max(avgFR_decVar_sorted,[],2);
    ridgeBaselineRatioShuffled = zeros(nNeurons,1);
    [~,max_shuffled_ix] = max(avgFR_shuffle_decVar_sorted,[],2);
    ridge_width = 10;
    
    backgroundUnshuffled = zeros(nNeurons,1);
    backgroundShuffled = zeros(nNeurons,1);
    
%     figure()
    %show area of ridge for quantification
%     subplot(1,2,1)
%     hold on
%     scatter(flipud(max_shuffled_ix-ridge_width),1:nNeurons,3,'w');
%     scatter(flipud(max_shuffled_ix+ridge_width),1:nNeurons,3,'w');
%     subplot(1,2,2)
%     hold on
%     scatter(flipud(max_unshuffled_ix-ridge_width),1:nNeurons,3,'w');
%     scatter(flipud(max_unshuffled_ix+ridge_width),1:nNeurons,3,'w');
    
    for neuron = 1:nNeurons
        backgroundUnshuffled(neuron) =  mean(avgFR_decVar_sorted(neuron,[1:max(1,max_unshuffled_ix(neuron)-ridge_width),min(nBins-1,(max_shuffled_ix(neuron)+ridge_width)):nBins-1]));
        ridgeUnshuffled(neuron) = mean(avgFR_decVar_sorted(neuron,max(1,max_unshuffled_ix(neuron)-ridge_width):min(nBins-1,max_unshuffled_ix(neuron)+ridge_width)));
        ridgeBaselineRatioUnshuffled(neuron) = ridgeUnshuffled(neuron) / backgroundUnshuffled(neuron);
        
        backgroundShuffled(neuron) =  mean(avgFR_shuffle_decVar_sorted(neuron,[1:max(1,max_shuffled_ix(neuron)-ridge_width), min(nBins-1,max_shuffled_ix(neuron)+ridge_width):nBins-1]));
        ridgeShuffled(neuron) = mean(avgFR_shuffle_decVar_sorted(neuron,max(1,max_shuffled_ix(neuron)-ridge_width):min(nBins-1,max_shuffled_ix(neuron)+ridge_width)));
        ridgeBaselineRatioShuffled(neuron) = ridgeShuffled(neuron) / backgroundShuffled(neuron);
    end
    
    ridgeBaselineRatioUnshuffled(ridgeBaselineRatioUnshuffled > 10) = 10;
    
    figure()
    bar([1 2],[mean(ridgeBaselineRatioShuffled(ridgeBaselineRatioShuffled > 0 & ~isinf(ridgeBaselineRatioShuffled)));mean(ridgeBaselineRatioUnshuffled(ridgeBaselineRatioUnshuffled > 0 & ~isinf(ridgeBaselineRatioUnshuffled)))])
    hold on
    non_inf_shuffled = length(find(ridgeBaselineRatioShuffled > 0 & ~isinf(ridgeBaselineRatioShuffled)));
    non_inf_unshuffled = length(find(ridgeBaselineRatioUnshuffled > 0 & ~isinf(ridgeBaselineRatioUnshuffled)));
    scatter([ones(non_inf_shuffled,1); 2.*ones(non_inf_unshuffled,1)],[ridgeBaselineRatioShuffled(ridgeBaselineRatioShuffled > 0 & ~isinf(ridgeBaselineRatioShuffled)); ridgeBaselineRatioUnshuffled(ridgeBaselineRatioUnshuffled > 0 & ~isinf(ridgeBaselineRatioUnshuffled))])
    xticks([1 2])
    xticklabels({'Shuffled','Unshuffled'})
    
%     [h,p] = ttest(ridgeBaselineRatioShuffled(ridgeBaselineRatioShuffled > 0 & ~isinf(ridgeBaselineRatioShuffled)),ridgeBaselineRatioUnshuffled(ridgeBaselineRatioUnshuffled > 0 & ~isinf(ridgeBaselineRatioUnshuffled)));

%     % Perform spectral clustering on unshuffled matrix 
%     [coeffs,score,~,~,expl] = pca(zscore(flipud(avgFR_decVar_sorted)));
%     figure();
%     subplot(1,3,1)
%     plot(expl(1:10) / sum(expl),'linewidth',2)
%     title("Variance explained per PC")
%     subplot(1,3,2)
%     plot(coeffs(:,1),'linewidth',2)
%     hold on
%     plot(coeffs(:,2),'linewidth',2)
%     plot(coeffs(:,3),'linewidth',2)
%     title("Principal Components")
%     legend("PC1","PC2","PC3") 
%     subplot(1,3,3)
%     colormap('jet')
%     scatter(score(:,1),score(:,2),[],1:nNeurons)
%     title("Cell position in PC space colored by cell number")
%     colorbar()
%     xlabel("PC1")
%     ylabel("PC2")
%     
    
end

%% Ridge plot all trials for multiple decision variables, sorting and visualizing with the same order
% Mult decision vars for bar graph
% takes a while to run

close all
for sIdx = 3:3
    decVar_cells = {};
%     decVar_cells{1} = FR_decVar(sIdx).decVar1;
%     decVar_cells{2} = FR_decVar(sIdx).decVar2;
%     decVar_cells{3} = FR_decVar(sIdx).decVar3;
%     decVar_cells{4} = FR_decVar(sIdx).decVar5;
    decVar_cells{1} = FR_decVar(sIdx).decVar1;
    decVar_cells{2} = FR_decVar(sIdx).decVarTime;
    decVar_cells{3} = FR_decVar(sIdx).decVarRawTimeSinceRew;
    labels = {};
    labels{1} = "Shuffled";
    labels{2} = "Model 1";
    labels{3} = "Raw Time";
    labels{4} = "Raw Time Since Last Reward";
%     labels{4} = "Model3";
%     labels{5} = "Model5";
%     labels{4} = "Raw Time Since Last Rew";
    
    rrbl_unshuffled = zeros(nNeurons,length(decVar_cells));
    
    for vIdx = 1:length(decVar_cells)
        decVar_cell = decVar_cells{vIdx};
        %%%% prep decision variable bins w/ all trials %%%%
        all_decVar = cat(2,decVar_cell{:});
        p = linspace(0.05,.95,41);
        decVar_bins = quantile(all_decVar,p);
        nBins = length(decVar_bins);
        
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
        decVar = cat(2,decVar_cell{:});
        
        %%% calculate FR averaged over decision variable bins for real data %%%
        avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if ~isempty(find(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            elseif dIdx > 1
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
            else
                avgFR_decVar(:,dIdx) = 0;
            end
        end
        
        avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);
        [~,index] = max(avgFR_decVar');
        [~,index_sort] = sort(index);
        
        avgFR_decVar_sorted = avgFR_decVar(index_sort,:);
        
        % Now find ridge-to-background ratio for shuffled, unshuffled data
        nNeurons = size(avgFR_decVar_sorted,1);
        ridgeBaselineRatioUnshuffled = zeros(nNeurons,1);
        [~,max_unshuffled_ix] = max(avgFR_decVar_sorted,[],2);
        
        backgroundUnshuffled = zeros(nNeurons,1);
        backgroundShuffled = zeros(nNeurons,1);

        for neuron = 1:nNeurons
            backgroundUnshuffled(neuron) =  mean(avgFR_decVar_sorted(neuron,[1:max(1,max_unshuffled_ix(neuron)-ridge_width),min(nBins-1,(max_shuffled_ix(neuron)+ridge_width)):nBins-1]));
            ridgeUnshuffled(neuron) = mean(avgFR_decVar_sorted(neuron,max(1,max_unshuffled_ix(neuron)-ridge_width):min(nBins-1,max_unshuffled_ix(neuron)+ridge_width)));
            ridgeBaselineRatioUnshuffled(neuron) = ridgeUnshuffled(neuron) / backgroundUnshuffled(neuron);
        end
        
        rrbl_unshuffled(:,vIdx) = ridgeBaselineRatioUnshuffled;
    end
    
    rrbl_unshuffled(rrbl_unshuffled > 15) = 15;
    
    % now perform shuffle control 1000x to test for significance per cell
    shuffRepeats = 100;
    
    decVar_cell = decVar_cells{2}; % arbitrarily use 2 here for now- not sure how to handle this
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,40);
    decVar_bins = quantile(all_decVar,p);
    nBins = length(decVar_bins);
    
    % collect FR matrices
    fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
    newShuffleControl = false;
    
    bad_neurons = [];
    
    if newShuffleControl == true
        rrbl_shuffled = zeros(nNeurons,shuffRepeats);
        for shuffIdx = 1:shuffRepeats
            
            if mod(shuffIdx,100) == 0
                display(shuffIdx)
            end
            
            %%% shuffle data according to random rotation %%%
            % figure out vectorized way to do this
            fr_mat_shuffle = zeros(size(fr_mat));
            shifts = randi(size(fr_mat,2),size(fr_mat,1),1);
            parfor neuron = 1:size(fr_mat,1)
                fr_mat_shuffle(neuron,:) = circshift(fr_mat(neuron,:),shifts(neuron));
            end

            %%% calculate FR averaged over decision variable bins for shuffled data %%%
            avgFR_decVar_shuffle = zeros(size(fr_mat_shuffle,1), numel(decVar_bins)-1);
            for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
                if ~isempty(find(fr_mat_shuffle(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
                    avgFR_decVar_shuffle(:,dIdx) = mean(fr_mat_shuffle(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
                elseif (dIdx > 1) & ~isempty(find(fr_mat_shuffle(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),1))
                    avgFR_decVar_shuffle(:,dIdx) = mean(fr_mat_shuffle(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
                end
            end
            avgFR_decVar_shuffle = avgFR_decVar_shuffle ./ max(avgFR_decVar_shuffle,[],2);
            [~,index] = max(avgFR_decVar_shuffle');
            [~,index_sort] = sort(index);
            
            avgFR_shuffle_decVar_sorted = avgFR_decVar_shuffle(index_sort,:);
            
%             figure()
%             colormap('jet')
%             imagesc(avgFR_shuffle_decVar_sorted)
            
            % Now find ridge-to-background ratio for shuffled, unshuffled data
            ridgeBaselineRatioShuffled = zeros(nNeurons,1);
            [~,max_shuffled_ix] = max(avgFR_shuffle_decVar_sorted,[],2);
            ridge_width = 15;
            
            backgroundShuffled = zeros(nNeurons,1);
            ridgeShuffled = zeros(nNeurons,1);
            
%             figure()
            hold on
            for neuron = 1:nNeurons
                backgroundShuffled(neuron) =  mean(avgFR_shuffle_decVar_sorted(neuron,[1:max(1,max_shuffled_ix(neuron)-ridge_width), min(nBins-1,max_shuffled_ix(neuron)+ridge_width):nBins-1]));
                ridgeShuffled(neuron) = mean(avgFR_shuffle_decVar_sorted(neuron,max(1,max_shuffled_ix(neuron)-ridge_width):min(nBins-1,max_shuffled_ix(neuron)+ridge_width)));
                ridgeBaselineRatioShuffled(neuron) = ridgeShuffled(neuron) / backgroundShuffled(neuron);
                if ridgeBaselineRatioShuffled(neuron) > 15
                    bad_neurons = [bad_neurons neuron];
                end
            end
            
            rrbl_shuffled(:,shuffIdx) = ridgeBaselineRatioShuffled;
        end
        
        rrbl_shuffled(isinf(rrbl_shuffled)) = 1;
        rrbl_shuffled(rrbl_shuffled > 15) = 15; % figure out how to deal w/ outliers better... 
    end
    
    figure()
    bar(1:(length(decVar_cells)+1),[mean(mean(rrbl_shuffled(200:300,:),'omitnan')) mean(rrbl_unshuffled(200:300,:),1)])
    hold on
    errorbar(1,mean(mean(rrbl_shuffled(200:300,:),'omitnan')),1.96 * std(mean(rrbl_shuffled(200:300,:),1,'omitnan')),'k')
    title("Ridge to background ratio across decision variables")
    xticks(1:(length(decVar_cells)+1))
    xticklabels(labels)
end


%% Now look at consistency across trials
%  WEIRD THINGS HAPPEN HERE
%  Display/quantify distribution of all neurons peak responsivity bins 

close all;

shuffle = false;
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    decVar_cell = FR_decVar(sIdx).decVar2;
    
    %%%% prep decision variable bins w/ all trials %%%%
%     all_decVar = cat(2,decVar_cell{:});
%     p = linspace(0.05,.95,40);
%     decVar_bins = quantile(all_decVar,p);
%     nBins = length(decVar_bins);
    
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    nTrials = size(FR_decVar(sIdx).fr_mat,2);
    
    maxDecVarBin = zeros(nNeurons,nTrials);
    cellIndexOrder = zeros(nNeurons,nTrials);

    for iTrial = 1:1
        fr_mat = FR_decVar(sIdx).fr_mat{iTrial};
        if shuffle == true
            shifts = randi(size(fr_mat,2),size(fr_mat,1),1);
            parfor neuron = 1:size(fr_mat,1)
                fr_mat(neuron,:) = circshift(fr_mat(neuron,:),shifts(neuron));
            end
        end
        decVar = decVar_cell{iTrial};
        
        % should this be outside the trial? 
        p = linspace(0.05,.95,40);
        decVar_bins = quantile(decVar,p);
        nBins = length(decVar_bins);
        
        %%% calculate FR averaged over decision variable bins for shuffled data %%%
        avgFR_decVar = nan(size(fr_mat,1), numel(decVar_bins)-1);
        
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if ~isempty(decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1))
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            end
        end
        
        %         length(find(isnan(avgFR_decVar)))
        avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);
        avgFR_decVar(isnan(avgFR_decVar)) = 0;
        figure()
        imagesc(avgFR_decVar);
        %         length(find(isnan(avgFR_decVar)))
        [~,index_maxDecVar] = max(avgFR_decVar');
        figure()
        imagesc(avgFR_decVar(index_neuronOrder,:));
        % index_neuronOrder is in some kind of order
        % might be the result of NaNs?
        [~,index_neuronOrder] = sort(index_maxDecVar);
        
        maxDecVarBin(:,iTrial) = index_maxDecVar;
        cellIndexOrder(:,iTrial) = index_neuronOrder;
        
    end
    
    histResolution = 50;
    rfs_manual = zeros(nNeurons,histResolution);
    for neuron = 1:nNeurons
        rfs_manual(neuron,:) = hist(cellIndexOrder(neuron,:),histResolution);
    end
    
%     figure();
%     imagesc(rfs_manual);
%     colormap('jet')
%     xlabel("Distribution of Neuron Position in sequence")
%     ylabel("Neuron")
    
    % this is super weird
    figure()
    plot(1:nNeurons,mean(cellIndexOrder,2),'linewidth',2)
    xlabel("Neuron")
    ylabel("Mean index in cell ordering")
    
%     figure() 
%     plot(hist(cellIndexOrder(50,:),50),'linewidth',2)
%     hold on
%     plot(hist(cellIndexOrder(90,:),50),'linewidth',2)
%     title("Example Decision Variable Receptive fields")
%     legend("50","90")
    
end

%% Re-doing consistency across trials
%  Display/quantify distribution of all neurons peak responsivity bins 

close all;
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    decVar_cell = FR_decVar(sIdx).decVarRawTimeSinceRew;
    shuffle = false;
    shifts = randi(size(FR_decVar(sIdx).fr_mat{1},2),size(FR_decVar(sIdx).fr_mat{1},1),1);
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,41);
    decVar_bins = quantile(all_decVar,p);
    nBins = length(decVar_bins);
    
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    nTrials = size(FR_decVar(sIdx).fr_mat,2);
    
    maxDecVarBin = nan(nNeurons,nTrials);
    cellIndexOrder = nan(nNeurons,nTrials);
    
    all0 = {nTrials};
    
    for iTrial = 1:nTrials
        fr_mat = FR_decVar(sIdx).fr_mat{iTrial};
        
        if shuffle == true
            parfor neuron = 1:size(fr_mat,1)
                fr_mat(neuron,:) = circshift(fr_mat(neuron,:),shifts(neuron));
            end
        end
        
        decVar = decVar_cell{iTrial};
        all0{iTrial} = find(all(fr_mat == 0,2));
        
%         % should this be outside the trial?
%         p = linspace(0.05,.9,41);
%         decVar_bins = quantile(decVar,p);
%         nBins = length(decVar_bins);
        
        %%% calculate FR averaged over decision variable bins for shuffled data %%%
        avgFR_decVar = nan(size(fr_mat,1), numel(decVar_bins)-1);
        
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if ~isempty(decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1))
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            end
        end

        avgFR_decVarNorm = avgFR_decVar ./ max(avgFR_decVar,[],2);
        nonfiring_ix = find(all(isnan(avgFR_decVarNorm),2));
        firing_ix = find(~all(isnan(avgFR_decVarNorm),2));
        maxDecVarBin(nonfiring_ix,iTrial) = 0; % nonfiring tag
        cellIndexOrder(nonfiring_ix,iTrial) = 0; % nonfiring tag
        
        avgFR_decVarNorm(nonfiring_ix,:) = [];
        
        [~,index_maxDecVar] = max(avgFR_decVarNorm');

        [~,index_neuronOrder] = sort(index_maxDecVar);
        avgFR_decVarSorted = avgFR_decVarNorm(index_neuronOrder,:);
       
        maxDecVarBin(firing_ix,iTrial) = index_maxDecVar';
        cellIndexOrder(firing_ix,iTrial) = index_neuronOrder ./max(index_neuronOrder); % norm to seq len
    end

%     xbins = linspace(1,max(max(cellIndexOrder)),40);
%     xbins = linspace(.05,1,40);
    xbins = 1:40;
    rfs_manual = zeros(nNeurons,length(xbins));
    for neuron = 1:nNeurons
%         [counts,centers] = hist(cellIndexOrder(neuron,cellIndexOrder(neuron,:) > 0),xbins);
        [counts,centers] = hist(maxDecVarBin(neuron,maxDecVarBin(neuron,:) > 0),xbins);
        rfs_manual(neuron,:) = counts;
%         rfs_manual(neuron,:) = rfs_manual(neuron,:) ./ max(rfs_manual(neuron,:));
%         rfs_manual(neuron,:) = zscore(rfs_manual(neuron,:));
        rfs_manual(neuron,:) = rfs_manual(neuron,:) / sum(rfs_manual(neuron,:));
    end
    
    [~,index_maxhist] = max(rfs_manual');
    [~,index_rfNeuronOrder] = sort(index_maxhist);
    
    figure();
    imagesc(flipud(rfs_manual(index_rfNeuronOrder,:)));

    % to show the neurons we're selecting for the next plot
    hold on     
    scatter(ones(3,1),nNeurons - find(index_rfNeuronOrder == 45 | index_rfNeuronOrder == 50| index_rfNeuronOrder == 90),75,'k*')
    colormap('jet')
    colorbar()
    xlabel("Distribution of Neuron Position in sequence")
    ylabel("Neuron")
    
    % perform spectral clustering on receptive field matrix
    [coeffs,score,~,~,expl] = pca(rfs_manual(index_rfNeuronOrder,:));
    figure();
    subplot(1,3,1)
    plot(cumsum(expl((1:10))) / sum(expl),'linewidth',2)
    title("Cumulative Variance explained per PC")
    xticks(1:10)
    xticklabels(1:10)
    grid()
    subplot(1,3,2)
    plot(coeffs(:,1) .* -1,'linewidth',2)
    hold on
    plot(coeffs(:,2),'linewidth',2)
    plot(coeffs(:,3),'linewidth',2)
    title("Principal Components")
    legend("PC1","PC2","PC3") 
    subplot(1,3,3)
    colormap('jet')
    scatter(score(:,1),score(:,2),[],1:nNeurons)
    title("Cell position in PC space colored by cell number")
    colorbar()
    xlabel("PC1")
    ylabel("PC2")
     
end

%% Based on this, look at individual neurons, single trials
% close all
for sIdx = 3:3
    decVar_cell = FR_decVar(sIdx).decVarRawTimeSinceRew;
    
    colors = [1 0 1;.75 .25 .75;0 1 1];
%     decision variable in bottom, neurons in 3 panes above
%     for iTrial = [21 27 39]
%         figure(iTrial)
%         neuronCounter = 1;
%         titles = {"Putative Early Responsive Neuron","Putative Middle Responsive Neuron","Putative Late Responsive Neuron"};
%         for neuron = [45 50 90]
%             fr_mat = FR_decVar(sIdx).fr_mat{iTrial};
%             subplot(4,1,neuronCounter);
%             plot(fr_mat(neuron,:) ./ max(fr_mat(neuron,:)),'linewidth',1.5,'color',colors(neuronCounter,:))
%             title(sprintf("Trial %i %s",iTrial,titles{neuronCounter}));
%             neuronCounter = neuronCounter + 1;
%         end
%         
%         subplot(4,1,4);
%         decVar = decVar_cell{iTrial};
%         plot(decVar,'linewidth',2)
%         title("Decision Variable 2")
%     end
    
    % for dec variable and neurons in same panes
    lims = [600 600 800];
    trialcounter = 1;
    for iTrial = [21 27 39]
        figure(iTrial)
        neuronCounter = 1;
        titles = {"Putative Early Responsive Neuron","Putative Middle Responsive Neuron","Putative Late Responsive Neuron"};
        fr_mat = FR_decVar(sIdx).fr_mat{iTrial};
        singleTrialHmap = nan(3,size(fr_mat,2));
        for neuron = [45 50 90]
            subplot(4,1,neuronCounter);
            plot(fr_mat(neuron,:) ./ max(fr_mat(neuron,:)),'linewidth',1.5,'color',colors(neuronCounter,:))
            singleTrialHmap(neuronCounter,:) = fr_mat(neuron,:) ./ max(fr_mat(neuron,:));
            title(sprintf("Trial %i %s",iTrial,titles{neuronCounter}));
            neuronCounter = neuronCounter + 1;
            decVar = decVar_cell{iTrial};
            hold on
            plot(decVar./max(decVar),'k--','linewidth',.75)
%             title("Decision Variable 2")
        end
        subplot(4,1,4)
        imagesc(singleTrialHmap);
        colormap('jet')
        xlim([0,lims(trialcounter)])
        trialcounter = trialcounter + 1;
    end
    
%     to display on same plot
%         for iTrial = [21 27 32 39]
%             figure(iTrial)
%             for neuron = [50 90]
%                 fr_mat = FR_decVar(sIdx).fr_mat{iTrial};
%                 decVar = decVar_cell{iTrial};
%                 hold on
%                 sp1 = subplot(2,1,1);
%                 plot(fr_mat(neuron,:) ./ max(fr_mat(neuron,:)),'linewidth',1.5)
%             end
%             sp2 = subplot(4,1,4);
%             title("Decision Variable 2")
%             sp2.Position = sp2.Position + [0 .2 0 0];
%             plot(decVar,'linewidth',2)
%             subplot(2,1,1)
%             legend("45","50","90")
%         title(sprintf("Trial %i Single Neuron Activity",iTrial))
%     end
    
end

%% Now performing quantifications w/ odd ordering, even visualization

% close all
for sIdx = 3:3
    decVar_cell = FR_decVar(sIdx).decVarRawTimeSinceRew;
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0,1,41);
    decVar_bins = quantile(all_decVar,p);
    
    shifts = randi(size(FR_decVar(sIdx).fr_mat{1},2),size(FR_decVar(sIdx).fr_mat{1},1),1);
    shuffle = false;
    
    %%%% First look at odd trials to get indices %%%%
    odd_frCell = FR_decVar(sIdx).fr_mat(1:2:end);
    odd_fr_mat = cat(2,odd_frCell{:});
    if shuffle == true
        parfor neuron = 1:size(fr_mat,1)
            odd_fr_mat(neuron,:) = circshift(odd_fr_mat(neuron,:),shifts(neuron));
        end
    end
    
    odd_decVarCell = decVar_cell(1:2:end);
    odd_decVar = cat(2,odd_decVarCell{:});
    
    odd_avgFR_decVar = zeros(size(FR_decVar(sIdx).fr_mat{1},1), numel(decVar_bins)-1);
    
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if length(find(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)))) > 0
            odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)),2);
        elseif dIdx > 1
            odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx-1) & odd_decVar < decVar_bins(dIdx)),2);
        else
            odd_avgFR_decVar(:,dIdx) = 0;
        end
    end
    
    odd_avgFR_decVar = zscore(odd_avgFR_decVar,[],2);
    [~,index] = max(odd_avgFR_decVar');
    [~,index_sort_odd] = sort(index);
    
    %%%% Next look at even trials for final visualization %%%%
    even_frCell = FR_decVar(sIdx).fr_mat(2:2:end);
    even_fr_mat = cat(2,even_frCell{:});
    if shuffle == true
        parfor neuron = 1:size(fr_mat,1)
            even_fr_mat(neuron,:) = circshift(even_fr_mat(neuron,:),shifts(neuron));
        end
    end
    even_decVarCell = decVar_cell(2:2:end);
    even_decVar = cat(2,even_decVarCell{:});
    
    p = linspace(0.05,.95,41);
    decVar_bins = quantile(even_decVar,p);
    
    even_avgFR_decVar = zeros(size(FR_decVar(sIdx).fr_mat{1},1), numel(decVar_bins)-1);
    
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if length(find(even_fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)))) > 0
            even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)),2);
        elseif dIdx > 1
            even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx-1) & even_decVar < decVar_bins(dIdx)),2);
        else
            even_avgFR_decVar(:,dIdx) = 0;
        end
    end
    
    [~,index] = max(even_avgFR_decVar');
    [~,index_sort_even] = sort(index);
    even_avgFR_decVar = zscore(even_avgFR_decVar,[],2);
%     even_avgFR_decVar = even_avgFR_decVar ./ max(even_avgFR_decVar,[],2);

    avgFR_decVar_sorted = even_avgFR_decVar(index_sort_odd,:);
%     avgFR_decVar_sorted = even_avgFR_decVar(index_sort,:);
    
    % Last, calculate P(leave|decVarBin)
    % for each trial, make a 1 x numel(decVar_bins) binary "on patch" vec
    nTrials = length(FR_decVar(sIdx).fr_mat);
    survival_mat = zeros(nTrials,numel(decVar_bins));
    
    for iTrial = 1:nTrials
        bins_greater = find(decVar_bins >= max(decVar_cell{iTrial}));
        if ~isempty(bins_greater)
            survival_mat(iTrial,1:bins_greater(1)) = 1;
        else
            survival_mat(iTrial,:) = 1;
        end
    end
    
    survival_curve = mean(survival_mat,1);
    
    % now regress out any linear components and see if we still get differentiation
    not_nan = any(~isnan(avgFR_decVar_sorted),2); % take out nans for fit
    avgFR_decVar_sorted = avgFR_decVar_sorted(not_nan,:);
    
    x = 1:size(avgFR_decVar_sorted,2);
    avgFR_decVar_sortedResidual = nan(size(avgFR_decVar_sorted));
    avgFR_decVar_sortedLinear = nan(size(avgFR_decVar_sorted));
    avgFR_decVar_sortedQuad = nan(size(avgFR_decVar_sorted));
    avgFR_decVar_sortedLinearResidual = nan(size(avgFR_decVar_sorted));
    avgFR_decVar_sortedQuadResidual = nan(size(avgFR_decVar_sorted));
    
    linear_r2 = nan(size(avgFR_decVar_sorted,1),1);
    quad_r2 = nan(size(avgFR_decVar_sorted,1),1);
    
    slope_fit = nan(size(avgFR_decVar_sorted,1),1);
    intercept_fit = nan(size(avgFR_decVar_sorted,1),1);
    
    for neuron = 1:size(avgFR_decVar_sorted,1)
        [linear_fit,linear_gof] = fit(x',avgFR_decVar_sorted(neuron,:)','poly1');
        slope_fit(neuron) = linear_fit.p1;
        intercept_fit(neuron) = linear_fit.p2;
        linear_fit = linear_fit.p1 * x' + linear_fit.p2;
        linear_r2(neuron) = linear_gof.rsquare;
        avgFR_decVar_sortedLinearResidual(neuron,:) = avgFR_decVar_sorted(neuron,:) - linear_fit';
        avgFR_decVar_sortedLinear(neuron,:) = linear_fit;
        
        [quad_fit,quad_gof] = fit(x',avgFR_decVar_sorted(neuron,:)','poly2');
        quad_fit = quad_fit.p1 * x'.^2 + quad_fit.p2 * x' + quad_fit.p3;
        quad_r2(neuron) = quad_gof.rsquare;
        avgFR_decVar_sortedQuadResidual(neuron,:) = avgFR_decVar_sorted(neuron,:) - quad_fit';
        avgFR_decVar_sortedQuad(neuron,:) = quad_fit;
    end
    
    %     avgFR_decVar_sortedQuad(avgFR_decVar_sortedQuad > 1) = 1;
    
    figure()
    subplot(1,2,1)
    colormap('jet')
    imagesc(flipud(even_avgFR_decVar(index_sort_even,:)))
    colorbar()
    title("Even Trials sorted by Even")
    xlabel("Time Since Last Rew Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    subplot(1,2,2)
    colormap('jet')
    imagesc(flipud(even_avgFR_decVar(index_sort_odd,:)))
    colorbar()
    title("Even Trials sorted by Odd")
    xlabel("Time Since Last Rew Percentile")
    ylabel("Neuron")
   %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    
    
    figure()
    subplot(1,3,1)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sorted))
    colorbar()
    title("Even Trials sorted by Odd")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
   %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    subplot(1,3,3)
    imagesc(flipud(avgFR_decVar_sortedLinearResidual))
    colorbar()
    title("Linear residuals")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    subplot(1,3,2)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sortedLinear))
    colorbar()
    title("Linear fit")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    
    figure()
    subplot(1,3,1)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sorted))
    colorbar()
    title("Even Trials sorted by Odd")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    colormap('jet')
    subplot(1,3,3)
    imagesc(flipud(avgFR_decVar_sortedQuadResidual))
    colorbar()
    title("Quadratic residuals")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    subplot(1,3,2)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sortedQuad))
    colorbar()
    title("Quadratic fit")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    
    % find the middle responsive neurons
    middle_neurons = index_sort_odd(304 - (100:250));
    subtr = quad_r2 - linear_r2;
    mid_responsive_neurons = middle_neurons(subtr(304 - (100:250)) > .3);
    
    % set coloring scheme
    labels = zeros(size(subtr,1),3);
%     labels(mid_responsive_neurons,:) = repmat([1,0,0],length(mid_responsive_neurons),1);
    labels(mid_responsive_neurons,1) = 1;
    
    adj_order = index_sort_odd; 
    adj_order(adj_order >= find(~not_nan)) = adj_order(adj_order >= find(~not_nan)) - 1;
    adj_order = adj_order(not_nan); 
    figure()
    scatter(flipud(subtr),1:length(subtr),[],flipud(labels(adj_order,:)))
    title("Quadratic fit improvement in variance explained by neuron")
    xlabel("Quadratic R^2 - Linear R^2")
    ylabel("Neuron")
    ylim([0,length(subtr)])
    set(gca, 'YDir','reverse')
    
    % check overall firing rate of ordered neurons
    whole_fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
    labels = zeros(size(whole_fr_mat,1),3);
    labels(mid_responsive_neurons,:) = repmat([1,0,0],length(mid_responsive_neurons),1);
    
    figure()
    %     labels(mid_responsive_neurons) = 3;
    means = mean(whole_fr_mat,2);
    scatter(means(index_sort_odd),1:length(means),[],labels(index_sort_odd,:))
    ylabel("Neuron")
    xlabel("Mean firing rate")
    ylim([0,305])
    set(gca, 'YDir','reverse')
    title("Firing rate of mid-responsive neurons")
    
    mid_responsive_neurons = index_sort_odd(50:150); % tough look, but hey
    figure()
    imagesc(even_avgFR_decVar(mid_responsive_neurons,:))
    colorbar()
    colormap('jet')
    xlabel("Time Since Last Rew Percentile")
    %     xlim([0,91])
    xlim([0,41])
    xlabel("Time Since Last Reward")
%     xticks([0 23 45 68 90])
    xticks([0 10 20 30 41])
%     xticklabels({'5','25','50','75','95'})
    xticklabels(decVar_bins([1,10,20,30,41]))
    ylabel("Neuron")
    title("Potential Sequence Participating Neurons")
end

%% Look at mid-responsive neurons on single trials
close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % Trial level features
    patches = data.patches;
    rew_size = mod(dat.patches(:,2),10);
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    decVar_cell = FR_decVar(sIdx).decVar2;
    
    colors = [1 0 1;.75 .25 .75;0 1 1];
    
    trials400 = find(rew_barcode(:,1) >0 & rew_barcode(:,2) == -1 & prts > 3.55);
    
    % for dec variable and neurons in same panes
    trialcounter = 1;
%     for j = 1:numel(trials400)
%         iTrial = trials400(j);
%         figure(iTrial)
    long_bois = find(prts > 5);
    all_trials = [];
    % get these guys side-by-side; single trials are not super interpretable
    for group = 0:4
        fr_mat_trials = [];
        decVar_trials = [];
        trial_lens = [];
        figure()
        subplot(2,1,1)
        hold on
        for j = 1:5
            iTrial = long_bois(j + 5 * group);
            fr_mat_iTrial = FR_decVar(sIdx).fr_mat{iTrial};
            fr_mat_trials = [fr_mat_trials fr_mat_iTrial];
            decVar_iTrial = decVar_cell{iTrial};
            decVar_trials = [decVar_trials decVar_iTrial];
            trial_lens = [trial_lens size(FR_decVar(sIdx).fr_mat{iTrial},2)];
        end
        all_trials = [all_trials fr_mat_trials];
        hold on
        imagesc(flipud(zscore(fr_mat_trials(mid_responsive_neurons,:))))
        plot([cumsum(trial_lens) ; cumsum(trial_lens)],[ones(1,length(trial_lens)) ; ones(1,length(trial_lens)) * length(mid_responsive_neurons)],'w--','linewidth',1.5)
        colormap('jet')
        title(sprintf("Putative Sequence Activity",iTrial))
        xlim([0,size(fr_mat_trials(mid_responsive_neurons,:),2)])
        ylim([1,length(mid_responsive_neurons)])
        sp2 = subplot(4,1,4);
        sp2.Position = sp2.Position + [0 .2 0 0];
        plot(decVar_trials,'linewidth',2)
        hold on 
        plot([cumsum(trial_lens) ; cumsum(trial_lens)],[ones(1,length(trial_lens)) * min(decVar_trials) ; ones(1,length(trial_lens)) * max(decVar_trials)],'k--','linewidth',1.5)
        title("Decision Variable 2")
        xlim([0,size(fr_mat_trials,2)])
    end
    
    % look at average heatmap across long trials
    long_frCell = FR_decVar(sIdx).fr_mat(long_bois);
    long_fr_mat = cat(2,long_frCell{:});
    long_decVarCell = decVar_cell(long_bois);
    long_decVar = cat(2,long_decVarCell{:});
    p = linspace(0.05,.95,40);
    decVar_bins = quantile(long_decVar,p);
    long_avgFR_decVar = zeros(size(FR_decVar(sIdx).fr_mat{1},1), numel(decVar_bins)-1);
    
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if length(find(long_fr_mat(:,long_decVar > decVar_bins(dIdx) & long_decVar < decVar_bins(dIdx+1)))) > 0
            long_avgFR_decVar(:,dIdx) = mean(long_fr_mat(:,long_decVar > decVar_bins(dIdx) & long_decVar < decVar_bins(dIdx+1)),2);
        else
            long_avgFR_decVar(:,dIdx) = mean(long_fr_mat(:,long_decVar > decVar_bins(dIdx-1) & long_decVar < decVar_bins(dIdx)),2);
        end
    end
    
%     long_avgFR_decVar = long_avgFR_decVar ./ max(long_avgFR_decVar,[],2);
    long_avgFR_decVar = zscore(long_avgFR_decVar,[],2);
    
    avgFR_decVar_sorted = long_avgFR_decVar(mid_responsive_neurons,:);
    figure()
    imagesc(avgFR_decVar_sorted)
    colormap('jet')
    colorbar()
    xlabel("Time Since Last Rew Percentile")
    xticks([0 10 20 30 40])
    xticklabels({'5','25','50','75','95'})
    ylabel("Neuron")
    title('Average activity for putative sequence neurons on long trials')
    
    % quantify with cross correlation
%     [cr,lgs] = xcorr(long_fr_mat(mid_responsive_neurons(25:30),:)','coeff');
%     for row = 1:5
%         for col = 1:5
%             nm = 5*(row-1)+col;
%             subplot(5,5,nm)
% %             stem(lgs,cr(:,nm),'.')
%             plot(lgs,cr(:,nm))
%             title(sprintf('c_{%d%d}',row,col))
%             ylim([0 1])
%         end
%     end
%     title("Sample Cross Correlations")

%     perform seqNMF
    figure()
%     K = 1;
%     L = 30;
    norm_fr_mat_trials = all_trials ./ max(all_trials,[],2);
    norm_fr_mat_trials(all(isnan(norm_fr_mat_trials),2),:) = 0;
%     [W,H] = seqNMF(norm_fr_mat_trials);

end

%% Now look at "eager" ramping population, quantify slope of linreg per trial
% check consistency of fit slope, potential corr w/ rewsize or prt

% this is going to take a lot of time %
close all;
for sIdx = 3:3
    
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % Trial level features
    patches = data.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    nTrials = size(FR_decVar(sIdx).fr_mat,2);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    ramp_idx = index_sort_odd(150:end);
    
    new_regs = true;
    
    % make the decision variable
    % probably shouldnt even use the decision variable here
    decVar_cell = FR_decVar(sIdx).decVarRawTimeSinceRew;
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,40);
    decVar_bins = quantile(all_decVar,p); % nonlinear
    decVar_bins = linspace(min(all_decVar),max(all_decVar),40); % linear 
    nBins = length(decVar_bins);
    
    if new_regs == true
        slopes = nan(nTrials,length(ramp_idx));
        intercepts = nan(nTrials,length(ramp_idx));
        for iTrial = 1:nTrials
            % subselect ramping portion of FR mat
            fr_mat_iTrial = FR_decVar(sIdx).fr_mat{iTrial}(ramp_idx,:);
            decVar_iTrial = decVar_cell{iTrial};
            
            % make avgFR_decVar matrix
            avgFR_decVar_iTrial = zeros(length(ramp_idx),numel(decVar_bins)-1);
            
            for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
                if ~isempty(find(fr_mat_iTrial(:,decVar_iTrial > decVar_bins(dIdx) & decVar_iTrial < decVar_bins(dIdx+1)),1))
                    avgFR_decVar_iTrial(:,dIdx) = mean(fr_mat_iTrial(:,decVar_iTrial > decVar_bins(dIdx) & decVar_iTrial < decVar_bins(dIdx+1)),2);
                elseif dIdx > 1 && ~isempty(find(fr_mat_iTrial(:,decVar_iTrial > decVar_bins(dIdx-1) & decVar_iTrial < decVar_bins(dIdx)),1))
                    avgFR_decVar_iTrial(:,dIdx) = mean(fr_mat_iTrial(:,decVar_iTrial > decVar_bins(dIdx-1) & decVar_iTrial < decVar_bins(dIdx)),2);
                end
            end
            
            % cut off columns of all 0s (unused decision variable)
            avgFR_decVar_iTrial(:,all(avgFR_decVar_iTrial == 0,1)) = [];
            
            avgFR_decVar_iTrialNorm = zscore(avgFR_decVar_iTrial,[],2);
%             avgFR_decVar_iTrialNorm = avgFR_decVar_iTrial ./ max(avgFR_decVar_iTrial,[],2);
            avgFR_decVar_iTrialNorm(isnan(avgFR_decVar_iTrialNorm)) = 0;
            avgFR_decVar_iTrialLinear = nan(size(avgFR_decVar_iTrialNorm));
            
            x = 1:size(avgFR_decVar_iTrialNorm,2); % just a line
            parfor neuron = 1:size(avgFR_decVar_iTrialNorm,1)
                [linear_fit,linear_gof] = fit(x',avgFR_decVar_iTrialNorm(neuron,:)','poly1');
                slopes(iTrial,neuron) = linear_fit.p1;
                intercepts(iTrial,neuron) = linear_fit.p2;
                linear_fit = linear_fit.p1 * x' + linear_fit.p2;
                avgFR_decVar_iTrialLinear(neuron,:) = linear_fit;
            end
            
            if mod(iTrial,10) == 0
                fprintf("Trial %i complete \n",iTrial)
            end
        end
    end
    
    figure()
    subplot(1,2,1)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_iTrialNorm));
    subplot(1,2,2)
    colormap('jet')
    imagesc(flipud(avgFR_decVar_iTrialLinear));
    
%     figure()  % artifact?
%     scatter(intercepts(:),slopes(:),.25) 
%     xlabel("Fitted Intercept")
%     ylabel("Fitted Slope")
%     title("Relationship between fitted intercept and slope")
    
    figure()
    subplot(1,2,1)
    imagesc(flipud(intercepts')); colormap('jet');colorbar();
    cl_int = caxis;
    xlabel("Trial")
    ylabel("Neuron")
    title("Fitted Intercept")
    subplot(1,2,2)
    imagesc(flipud(slopes')); colormap('jet');colorbar();
    cl_slope = caxis;
    xlabel("Trial")
    title("Fitted Slope")
    
    % now quantify consistency of cell responsivity between trials
    % rank order by mean, then add error bars 
    
    
    % analyze differences between reward sizes
    figure()
    subplot(1,3,1)
    imagesc(flipud(slopes(rewsize == 1,:)')); colormap('jet');colorbar();
    title('Fitted Slope 1 uL Trials')
    xlabel("Trial")
    ylabel("Neuron")
    caxis(cl_slope);
    subplot(1,3,2)
    imagesc(flipud(slopes(rewsize == 2,:)')); colormap('jet');colorbar();
    title('Fitted Slope 2 uL Trials')
    xlabel("Trial")
    caxis(cl_slope);
    subplot(1,3,3)
    imagesc(flipud(slopes(rewsize == 4,:)')); colormap('jet');colorbar();
    title('Fitted Slope 4 uL Trials')
    xlabel("Trial")
    caxis(cl_slope);
    
    figure()
    subplot(1,3,1)
    imagesc(flipud(intercepts(rewsize == 1,:)')); colormap('jet');colorbar();
    title('Fitted Intercept 1 uL Trials')
    xlabel("Trial")
    ylabel("Neuron")
    caxis(cl_int);
    subplot(1,3,2)
    imagesc(flipud(intercepts(rewsize == 2,:)')); colormap('jet');colorbar();
    title('Fitted Intercept 2 uL Trials')
    xlabel("Trial")
    caxis(cl_int);
    subplot(1,3,3)
    imagesc(flipud(intercepts(rewsize == 4,:)')); colormap('jet');colorbar();
    title('Fitted Intercept 4 uL Trials')
    xlabel("Trial")
    caxis(cl_int);
    
    [~,slope_sort] = sort(mean(slopes(rewsize == 4,:)));
    figure()
    bar(mean(slopes(rewsize == 4,slope_sort)))
    hold on
    er = errorbar(1:size(slopes(rewsize == 4,slope_sort),2),mean(slopes(rewsize == 4,slope_sort)),1.96 .* std(slopes(rewsize == 4,slope_sort)));
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    xlabel("Neuron")
    ylabel("Mean fitted slope")
    title("Distribution of Fitted Slope for 4 uL Trials")
    
    [~,int_sort] = sort(mean(intercepts(rewsize == 4,:)));
    figure()
    bar(mean(intercepts(rewsize == 4,int_sort)))
    hold on
    er = errorbar(1:size(intercepts(rewsize == 4,int_sort),2),mean(intercepts(rewsize == 4,int_sort)),1.96 .* std(intercepts(rewsize == 4,int_sort)));
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    xlabel("Neuron")
    ylabel("Mean fitted Intercept")
    title("Distribution of Fitted Intercept for 4 uL Trials")
    
    % perform ANOVA test for difference in means
    [p_slope,tbl_slope,stats_slope] = anova1(slopes(rewsize == 4,slope_sort));
    xticks(0:50:150)
    xticklabels(0:50:150)
    title("Distribution of Fitted Slope for 4 uL Trials")
    [p_int,tbl_int,stats_int] = anova1(intercepts(rewsize == 4,int_sort));
    xticks(0:50:150)
    xticklabels(0:50:150)
    title("Distribution of Fitted Intercept for 4 uL Trials")
    
    
    % Quantify differences betw reward sizes; will be easier once we have %
    % glm fits
    %
    %     slopes1 = slopes(rewsize == 1,:);
    %     slopes2 = slopes(rewsize == 2,:);
    %     slopes4 = slopes(rewsize == 4,:);
    %     ints1 = intercepts(rewsize == 1,:);
    %     ints2 = intercepts(rewsize == 2,:);
    %     ints4 = intercepts(rewsize == 4,:);
    %     %     figure()
    % %     subplot(1,2,2)
    % %     xticks([1,2,3])
    %     xticklabels({"1 uL","2 uL","4 uL"})
    %     bar([mean(slopes1(:)) mean(slopes2(:)) mean(slopes4(:))]); hold on;
    %     title("Mean slope across reward sizes")
    %     er = errorbar([1,2,3],[mean(slopes1(:)) mean(slopes2(:)) mean(slopes4(:))],[std(slopes1(:)) std(slopes2(:)) std(slopes4(:))].*1.96);
    %     er.Color = [0 0 0];
    %     er.LineStyle = 'none';
    %
    %     subplot(1,2,1)
    %     bar([mean(ints1(:)) mean(ints2(:)) mean(ints4(:))]); hold on;
    %     title("Mean intercept across reward sizes")
    %     xticks([1,2,3])
    %     xticklabels({"1 uL","2 uL","4 uL"})
    %         er = errorbar([1,2,3],[mean(ints1(:)) mean(ints2(:)) mean(ints4(:))],[std(ints1(:)) std(ints2(:)) std(ints4(:))].*1.96);
    %     er.Color = [0 0 0];
    %     er.LineStyle = 'none';
    % %
    
end