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

%% Look at integrator-like PCs to figure out some simple integrator decision variable

jPCA_data = {};

sec2ix = round(2000/tbin_ms);
sec3ix = round(3000/tbin_ms);

for sIdx = 3:3
    figcounter = 1;
    
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
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    rew_counter = 1;
    
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        cat10x = cellfun(@(x) x(1:sec2ix,:),trial_pc_traj{sIdx}(trials10x),'un',0);
        cat11x = cellfun(@(x) x(1:sec2ix,:),trial_pc_traj{sIdx}(trials11x),'un',0);
        
        mean10x = mean(cat(6,cat10x{:}),6);
        mean11x = mean(cat(6,cat11x{:}),6);
        
        % now define simple regression model for integrator based on PC1,
        % plot compared to activity
        mdl = fitlm(1:sec2ix,mean10x(:,1));
        bm = mdl.Coefficients.Estimate; b = bm(1); m = bm(2);
        
        figure()
        subplot(1,2,1)
        plot(mean10x(:,1))
        hold on
        plot(mean11x(:,1))
        plot(1:sec2ix, m * (1:sec2ix) +b);
        title(sprintf("%i uL, PC1",iRewsize))
        legend(sprintf("%i0",iRewsize),sprintf("%i%i",iRewsize,iRewsize),sprintf("%i0 linear model",iRewsize))
        
        subplot(1,2,2)
        plot(mean10x(:,3))
        hold on
        plot(mean11x(:,3))
        title(sprintf("%i uL, PC3",iRewsize))
        legend(sprintf("%i0",iRewsize),sprintf("%i%i",iRewsize,iRewsize))
        
    end
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
        
        FR_decVar(sIdx).decVarTime{iTrial} = 1:trial_len_ix; % just plain time 
        FR_decVar(sIdx).decVar1{iTrial} = m1fit(rewsize(iTrial),2) * (1:trial_len_ix)/tbin_ms - m1fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVar2{iTrial} = m2fit(rewsize(iTrial),2) * (1:trial_len_ix)/tbin_ms - m2fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVar3{iTrial} = m3fit(rewsize(iTrial),3) * (1:trial_len_ix)/tbin_ms - m3fit(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarPerm2Control{iTrial} = m2fit_permute(rewsize(iTrial),2) * (1:trial_len_ix)/tbin_ms - m2fit_permute(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarPerm3Control{iTrial} = m3fit_permute(rewsize(iTrial),3) * (1:trial_len_ix)/tbin_ms - m3fit_permute(rewsize(iTrial),1);
        FR_decVar(sIdx).decVarRandControl{iTrial} = rand(1,trial_len_ix); % random vector
        
        % deal with resets
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end) = m2fit(rewsize(iTrial),2) * ((1:length(FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end)))/tbin_ms - m2fit(rewsize(iTrial),1));
            FR_decVar(sIdx).decVar3{iTrial}(rew_ix:end) = (FR_decVar(sIdx).decVar3{iTrial}(rew_ix:end) - m3fit(rewsize(iTrial),1)) - r * m3fit(rewsize(iTrial),2);
            FR_decVar(sIdx).decVarPerm2Control{iTrial}(rew_ix:end) = m2fit_permute(rewsize(iTrial),2) * ((1:length(FR_decVar(sIdx).decVar2{iTrial}(rew_ix:end)))/tbin_ms - m2fit_permute(rewsize(iTrial),1));
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
    plot(FR_decVar(sIdx).decVarRandControl{39})
    plot(FR_decVar(sIdx).decVarPC1{39})
    plot(FR_decVar(sIdx).decVarPC3{39})
    %     plot(FR_decVar(sIdx).decVarPerm2Control{39})
    %     plot(FR_decVar(sIdx).decVarPerm3Control{39})
    legend("Model1","Model2","Model3","Model5","RandControl","PC1","PC3")
    title("Trial 39 decision variables")
    
end

%% Ridge plot all trials for a given decision variable, getting order from odd and visualization from even trials

close all
for sIdx = 3:3
    decVar_cell = FR_decVar(sIdx).decVar2;
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,91);
    decVar_bins = quantile(all_decVar,p);
    
    %%%% First look at odd trials to get indices %%%%
    odd_frCell = FR_decVar(sIdx).fr_mat(1:2:end);
    odd_fr_mat = cat(2,odd_frCell{:});
    odd_decVarCell = decVar_cell(1:2:end);
    odd_decVar = cat(2,odd_decVarCell{:});
    
    odd_avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
    
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if length(find(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)))) > 0
            odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)),2);
        else
            odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx-1) & odd_decVar < decVar_bins(dIdx)),2);
        end
    end
    
    odd_avgFR_decVar = zscore(odd_avgFR_decVar,[],2);
    [~,index] = max(odd_avgFR_decVar');
    [~,index_sort_odd] = sort(index);
    
    %%%% Next look at even trials for final visualization %%%%
    even_frCell = FR_decVar(sIdx).fr_mat(2:2:end);
    even_fr_mat = cat(2,even_frCell{:});
    even_decVarCell = decVar_cell(2:2:end);
    even_decVar = cat(2,even_decVarCell{:});
    
    p = linspace(0.05,.95,91);
    decVar_bins = quantile(even_decVar,p);
    
    even_avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
    
    for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
        if length(find(even_fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)))) > 0
            even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)),2);
        else
            even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx-1) & even_decVar < decVar_bins(dIdx)),2);
        end
    end
    
    [~,index] = max(even_avgFR_decVar');
    [~,index_sort_even] = sort(index);
    %     even_avgFR_decVar = zscore(even_avgFR_decVar,[],2);
    even_avgFR_decVar = even_avgFR_decVar ./ max(even_avgFR_decVar,[],2);

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
    
    figure()
    sp1 = subplot(2,1,1);
    colormap('jet')
    imagesc(flipud(avgFR_decVar_sorted))
    colorbar()
    title("Firing Rate Sorted by Argmax Decision Variable")
    %     xlabel("Decision Variable Percentile")
    ylabel("Neuron")
    xlim([0,90])
    xticks([])
    sp2 = subplot(3,1,3);
    sp2.Position = sp2.Position + [0 .2 0 0];
    plot(survival_curve,'linewidth',2)
    xlim([0,91])
    ylim([0,1.1])
    xticks([0 23 45 68 90])
    xticklabels({'5','25','50','75','95'})
    
    xlabel("Decision Variable Percentile")
    ylabel("Survival")
    
    avgFR_decVar_sorted(isnan(avgFR_decVar_sorted)) = 0;
    
    % Perform spectral clustering on unshuffled matrix
    [coeffs,score,~,~,expl] = pca(zscore(flipud(avgFR_decVar_sorted)));
    figure();
    subplot(1,3,1)
    plot(expl(1:10) / sum(expl),'linewidth',2)
    title("Variance explained per PC")
    xticks(1:10)
    xticklabels(1:10)
    grid()
    subplot(1,3,2)
    plot(coeffs(:,1),'linewidth',2)
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

%% divide by reward size, again dividing odd/even "train/test"
% Note we have some trial number problems here in getting enough decision
% variable observations

close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    patches = data.patches;
    rewsize = mod(patchType,10);
    decVar_cell = FR_decVar(sIdx).decVar2;
    
    for iRewsize = [1,2,4]
        
        %%%% prep decision variable bins w/ all trials %%%%
        all_decVar = cat(2,decVar_cell{:});
        p = linspace(0.05,.95,91);
        decVar_bins = quantile(all_decVar,p);
        
        %%%% First look at odd trials to get indices %%%%
        iRewsize_trials = find(rewsize == iRewsize);
        odd_frCell = FR_decVar(sIdx).fr_mat(iRewsize_trials(1:2:end));
        odd_fr_mat = cat(2,odd_frCell{:});
        odd_decVarCell = decVar_cell(iRewsize_trials(1:2:end));
        odd_decVar = cat(2,odd_decVarCell{:});
        
        odd_avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        
        for dIdx = 2:(numel(decVar_bins) - 1) % go up to 80th percentile
            if length(find(fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)))) > 0
                odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx) & odd_decVar < decVar_bins(dIdx+1)),2);
            else
                odd_avgFR_decVar(:,dIdx) = mean(odd_fr_mat(:,odd_decVar > decVar_bins(dIdx-1) & odd_decVar < decVar_bins(dIdx)),2);
            end
        end
        
        odd_avgFR_decVar = zscore(odd_avgFR_decVar,[],2);
        [~,index] = max(odd_avgFR_decVar');
        [~,index_sort_odd] = sort(index);
        
        %%%% Next look at even trials for final visualization %%%%
        even_frCell = FR_decVar(sIdx).fr_mat(iRewsize_trials(2:2:end));
        even_fr_mat = cat(2,even_frCell{:});
        even_decVarCell = decVar_cell(iRewsize_trials(2:2:end));
        even_decVar = cat(2,even_decVarCell{:});
        
        p = linspace(0.05,.95,91);
        decVar_bins = quantile(even_decVar,p);
        
        even_avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if length(find(fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)))) > 0
                even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx) & even_decVar < decVar_bins(dIdx+1)),2);
            else
                even_avgFR_decVar(:,dIdx) = mean(even_fr_mat(:,even_decVar > decVar_bins(dIdx-1) & even_decVar < decVar_bins(dIdx)),2);
            end
        end
        
        even_avgFR_decVar = zscore(even_avgFR_decVar,[],2);
        
        avgFR_decVar_sorted = even_avgFR_decVar(index_sort_odd,:);
        
        % Last, calculate P(leave|decVarBin)
        % for each trial, make a 1 x numel(decVar_bins) binary "on patch" vec
        nTrials = length(FR_decVar(sIdx).fr_mat);
        survival_mat = zeros(nTrials,numel(decVar_bins));
        
        for iTrial = 1:nTrials
            bins_greater = find(decVar_bins > max(decVar_cell{iTrial}));
            if ~isempty(bins_greater)
                survival_mat(iTrial,1:bins_greater(1)) = 1;
            else
                survival_mat(iTrial,:) = 1;
            end
        end
        
        survival_curve = mean(survival_mat,1);
        
        figure()
        sp1 = subplot(2,1,1);
        colormap('jet')
        imagesc(flipud(avgFR_decVar_sorted))
        title(sprintf("%i uL Trial Firing Rate Sorted by Argmax Decision Variable",iRewsize))
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
        xlim([0,90])
        xticks([])
        sp2 = subplot(3,1,3);
        sp2.Position = sp2.Position + [0 .2 0 0];
        plot(survival_curve,'linewidth',2)
        xlim([0,91])
        ylim([0,1.1])
        xticks([0 23 45 68 90])
        xticklabels({'5','25','50','75','95'})
        
        xlabel("Decision Variable Percentile")
        ylabel("Survival")
        
        figure()
        histogram(index,50)
        title("Distribution of Maximal FR Bin")
    end
end

%% Now look at division by reward size (cheating on ordering)
close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing

    patches = data.patches;
    rewsize = mod(patchType,10);
    decVar_cell = FR_decVar(sIdx).decVar2;
    
    for iRewsize = [1,2,4]
        % wanna concat here again... probably didn't need the cells earlier but
        % helps w/ comprehension
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{rewsize == iRewsize});
        decVar = cat(2,decVar_cell{rewsize == iRewsize});
        
        %     decVar_bins = 1:50:10000; % (max(decVar) - 5000); (for pure time)
        p = linspace(0.05,.95,40);
        %     p = logspace(0.05,.95,91) / max(logspace(0.05,.95,91));
        decVar_bins = quantile(decVar,p);
        
        % now calculate P(leave|decVarBin)
        % for each trial, make a 1 x numel(decVar_bins) binary "on patch" vec
        nTrials = length(FR_decVar(sIdx).fr_mat);
        survival_mat = zeros(nTrials,numel(decVar_bins));
        
        for iTrial = 1:nTrials
            bins_greater = find(decVar_bins > max(decVar_cell{iTrial}));
            if ~isempty(bins_greater)
                survival_mat(iTrial,1:bins_greater(1)) = 1;
            else
                survival_mat(iTrial,:) = 1;
            end
        end
        
        survival_curve = mean(survival_mat,1);
        
        avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if length(find(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)))) > 0
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            else
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
            end
        end
        
%         avgFR_decVar = zscore(avgFR_decVar);
        avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);

%         to look at indices from one reward size trials
%         if iRewsize == 2
%             [~,index] = max(avgFR_decVar');
%             [~,index_sort] = sort(index);
%         end

        [~,index] = max(avgFR_decVar');
        [~,index_sort] = sort(index);
        avgFR_decVar_sorted = avgFR_decVar(index_sort,:);
        
        figure()
        sp1 = subplot(2,1,1);
        colormap('jet')
        imagesc(flipud(avgFR_decVar_sorted))
        colorbar()
        title(sprintf("%i uL Trial Firing Rate Sorted by Argmax Decision Variable",iRewsize))
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
%         xlim([0,40])
        xticks([])
        sp2 = subplot(4,1,4);
        sp2.Position = sp2.Position + [0 .2 0 0];
        plot(survival_curve,'linewidth',2)
%         xlim([0,40])
        ylim([0,1.1])
        xticks([0 11 20 31 40])
        xticklabels({'5','25','50','75','95'})

        xlabel("Decision Variable Percentile")
        ylabel("Survival")
        
%         subplot(4,1,3)
%         histogram(index,50)
%         xticks([])
%         xlim([0,40])
%         ylabel("Maximal FR Bin Distn")
    end
end

%% Divide by 2 second reward history, see if separate population representations pull out
close all
sec1idx = 1000/tbin_ms;
sec2idx = 2000/tbin_ms;
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % make barcode mat
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    decVar_cell = FR_decVar(sIdx).decVarTime;
    
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        % wanna concat here again... probably didn't need the cells earlier but
        % helps w/ comprehension
        fr_mat10x = cat(2,FR_decVar(sIdx).fr_mat{trials10x});
        decVar10x = cat(2,decVar_cell{trials10x});
        fr_mat11x = cat(2,FR_decVar(sIdx).fr_mat{trials11x});
        decVar11x = cat(2,decVar_cell{trials11x});
        
        decVar_bins = linspace(1,sec2idx,50);
        
        % first R0X
        avgFR_10x = zeros(size(fr_mat10x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat10x(:,decVar10x > decVar_bins(dIdx) & decVar10x < decVar_bins(dIdx+1))))
                avgFR_10x(:,dIdx) = mean(fr_mat10x(:,decVar10x > decVar_bins(dIdx) & decVar10x < decVar_bins(dIdx+1)),2);
            else
                avgFR_10x(:,dIdx) = mean(fr_mat10x(:,decVar10x > decVar_bins(dIdx-1) & decVar10x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_10x = zscore(avgFR_10x,[],2);
%         avgFR_10x = avgFR_10x ./ max(avgFR_10x,[],2);
        [~,index] = max(avgFR_10x');
        [~,index_sort] = sort(index);
        avgFR_decVar10x_sorted = avgFR_10x(index_sort,:);
        
        % next RRX
        avgFR_11x = zeros(size(fr_mat11x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat11x(:,decVar11x > decVar_bins(dIdx) & decVar11x < decVar_bins(dIdx+1))))
                avgFR_11x(:,dIdx) = mean(fr_mat11x(:,decVar11x > decVar_bins(dIdx) & decVar11x < decVar_bins(dIdx+1)),2);
            else
                avgFR_11x(:,dIdx) = mean(fr_mat11x(:,decVar11x > decVar_bins(dIdx-1) & decVar11x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_11x = zscore(avgFR_11x,[],2);
%         avgFR_11x = avgFR_11x ./ max(avgFR_11x,[],2);
%         [~,index] = max(avgFR_11x');
%         [~,index_sort] = sort(index);
        avgFR_decVar11x_sorted = avgFR_11x(index_sort,:);
        
        figure() 
        subplot(2,1,1)
        colormap('jet')
        imagesc(flipud(avgFR_decVar10x_sorted))
        colorbar()
        title(sprintf("%i0X Trial Mean Activity",iRewsize))
        ylabel("Neuron (same order)")
        xticks([12 25 37 50])
        xticklabels({'500','1000','1500','2000'})
        
        subplot(2,1,2)
        colormap('jet')
        imagesc(flipud(avgFR_decVar11x_sorted))
        colorbar()
        title(sprintf("%i%iX Trial Mean Activity",iRewsize,iRewsize))
        xlabel("Time (ms)")
        ylabel("Neuron (same order)")
        xticks([12 25 37 50])
        xticklabels({'500','1000','1500','2000'})
        

    end
end

%% Divide by 3 second reward history, see if separate population representations pull out
close all

sec3idx = 3000/tbin_ms;

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    % make barcode mat
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    decVar_cell = FR_decVar(sIdx).decVarTime;
    
    for iRewsize = [2,4]
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.55);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.55);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
        
        % wanna concat here again... probably didn't need the cells earlier but
        % helps w/ comprehension
        fr_mat100x = cat(2,FR_decVar(sIdx).fr_mat{trials100x});
        decVar100x = cat(2,decVar_cell{trials100x});
        fr_mat110x = cat(2,FR_decVar(sIdx).fr_mat{trials110x});
        decVar110x = cat(2,decVar_cell{trials110x});
        fr_mat101x = cat(2,FR_decVar(sIdx).fr_mat{trials101x});
        decVar101x = cat(2,decVar_cell{trials101x});
        fr_mat111x = cat(2,FR_decVar(sIdx).fr_mat{trials111x});
        decVar111x = cat(2,decVar_cell{trials111x});
        
        decVar_bins = linspace(1,sec3idx,100);
        
        % first R00X
        avgFR_100x = zeros(size(fr_mat100x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat100x(:,decVar100x > decVar_bins(dIdx) & decVar100x < decVar_bins(dIdx+1))))
                avgFR_100x(:,dIdx) = mean(fr_mat100x(:,decVar100x > decVar_bins(dIdx) & decVar100x < decVar_bins(dIdx+1)),2);
            else
                avgFR_100x(:,dIdx) = mean(fr_mat10x(:,decVar100x > decVar_bins(dIdx-1) & decVar100x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_100x = zscore(avgFR_100x,[],2);
        [~,index] = max(avgFR_100x');
        [~,index_sort] = sort(index);
        avgFR_decVar100x_sorted = avgFR_100x(index_sort,:);
        
        % next RR0X
        avgFR_110x = zeros(size(fr_mat110x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat110x(:,decVar110x > decVar_bins(dIdx) & decVar110x < decVar_bins(dIdx+1))))
                avgFR_110x(:,dIdx) = mean(fr_mat110x(:,decVar110x > decVar_bins(dIdx) & decVar110x < decVar_bins(dIdx+1)),2);
            else
                avgFR_110x(:,dIdx) = mean(fr_mat110x(:,decVar110x > decVar_bins(dIdx-1) & decVar110x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_110x = zscore(avgFR_110x,[],2);
%         [~,index] = max(avgFR_110x');
%         [~,index_sort] = sort(index);
        avgFR_decVar110x_sorted = avgFR_110x(index_sort,:);
        
        % first R0RX
        avgFR_101x = zeros(size(fr_mat101x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat101x(:,decVar101x > decVar_bins(dIdx) & decVar101x < decVar_bins(dIdx+1))))
                avgFR_101x(:,dIdx) = mean(fr_mat101x(:,decVar101x > decVar_bins(dIdx) & decVar101x < decVar_bins(dIdx+1)),2);
            else
                avgFR_101x(:,dIdx) = mean(fr_mat101x(:,decVar101x > decVar_bins(dIdx-1) & decVar101x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_101x = zscore(avgFR_101x,[],2);
%         [~,index] = max(avgFR_101x');
%         [~,index_sort] = sort(index);
        avgFR_decVar101x_sorted = avgFR_101x(index_sort,:);
        
        % next RR0X
        avgFR_111x = zeros(size(fr_mat111x,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_mat111x(:,decVar111x > decVar_bins(dIdx) & decVar111x < decVar_bins(dIdx+1))))
                avgFR_111x(:,dIdx) = mean(fr_mat111x(:,decVar111x > decVar_bins(dIdx) & decVar111x < decVar_bins(dIdx+1)),2);
            else
                avgFR_111x(:,dIdx) = mean(fr_mat111x(:,decVar111x > decVar_bins(dIdx-1) & decVar111x < decVar_bins(dIdx)),2);
            end
        end
        avgFR_111x = zscore(avgFR_111x,[],2);
%         [~,index] = max(avgFR_111x');
%         [~,index_sort] = sort(index);
        avgFR_decVar111x_sorted = avgFR_111x(index_sort,:);
        
        figure() 
        subplot(4,1,1)
        colormap('jet')
        imagesc(flipud(avgFR_decVar100x_sorted))
        title(sprintf("%i00",iRewsize))
        
        subplot(4,1,2)
        colormap('jet')
        imagesc(flipud(avgFR_decVar110x_sorted))
        title(sprintf("%i%i0",iRewsize,iRewsize))
        
        subplot(4,1,3)
        colormap('jet')
        imagesc(flipud(avgFR_decVar101x_sorted))
        title(sprintf("%i0%i",iRewsize,iRewsize))
        
        subplot(4,1,4)
        colormap('jet')
        imagesc(flipud(avgFR_decVar111x_sorted))
        title(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize))

    end
end

%% Now just looking at trials where no extra rewards received (1... 2... etc)
close all

figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
    
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
    
    decVar_cell = FR_decVar(sIdx).decVarTime;
    p = linspace(0.05,.95,40);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);

    % iterate over reward sizes
    max_ix_line = zeros(nNeurons,3);
    rewsizecounter = 1;
    
    for iRewsize = [1,2,4]
        
        trialsR0 = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == -1);
        fr_matR0 = cat(2,FR_decVar(sIdx).fr_mat{trialsR0});
        decVarR0 = cat(2,decVar_cell{trialsR0});
        
        decVar_bins = quantile(decVarR0,p);
        
        % average FR
        avgFR_R0 = zeros(size(fr_matR0,1), numel(decVar_bins)-1);
        for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
            if ~isempty(find(fr_matR0(:,decVarR0 > decVar_bins(dIdx) & decVarR0 < decVar_bins(dIdx+1)),1))
                avgFR_R0(:,dIdx) = mean(fr_matR0(:,decVarR0 > decVar_bins(dIdx) & decVarR0 < decVar_bins(dIdx+1)),2);
            else
                avgFR_R0(:,dIdx) = mean(fr_matR0(:,decVarR0 > decVar_bins(dIdx-1) & decVarR0 < decVar_bins(dIdx)),2);
            end
        end
        avgFR_R0 = avgFR_R0 ./ max(avgFR_R0,[],2);
        
        [~,index] = max(avgFR_R0');
        [~,index_sort] = sort(index);
        
        avgFR_decVarR0_sorted = avgFR_R0(index_sort,:);
        [~,max_ix] = max(avgFR_decVarR0_sorted,[],2);
        max_ix_line(:,rewsizecounter) = max_ix;
        
        figure()
        colormap('jet')
        imagesc(flipud(avgFR_decVarR0_sorted))
        hold on
        plot(flipud(max_ix),1:nNeurons,'w','linewidth',3);
        title(sprintf("%i... Trials",iRewsize))
        
        rewsizecounter = rewsizecounter + 1;
    end
end

figure()
hold on
plot(max_ix_line(:,1),1:nNeurons,'linewidth',2,'color',[0,0,0])
plot(max_ix_line(:,2),1:nNeurons,'linewidth',2,'color',[0,1,1])
plot(max_ix_line(:,3),1:nNeurons,'linewidth',2,'color',[0,0,1])
legend("1 uL","2 uL","4 uL")

%% 3x3 400 plot:
% order by 1,2,then 3 seconds
% note that we only have 8 of these trials, 12 if we include all rewsizes
close all

sec1idx = 1000 / tbin_ms;
sec2idx = 2000 / tbin_ms;
sec3idx = 3000 / tbin_ms;

figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
    
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
    
    decVar_cell = FR_decVar(sIdx).decVarTime; % sort by time as decision variable is same for everyone
    p = linspace(0.05,.95,41);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    
    
    decVar_bins_cell = {};
    resolution = 40;
    decVar_bins_cell{1} = linspace(1,sec1idx,resolution);
    decVar_bins_cell{2} = linspace(1,sec2idx,resolution);
    decVar_bins_cell{3} = linspace(1,sec3idx,resolution);
    
    for iRewsize = [2,4]
        figure()
        spcounter = 0;
        index_sort_cell = {};
        fr_mat_cell = {};
        % get the diagonals and store matrices, indices
        for binIdx = 1:3
            decVar_bins = decVar_bins_cell{binIdx};
            
            trialsR0 = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == -1 & prts > 3.55);
            fr_matR0 = cat(2,FR_decVar(sIdx).fr_mat{trialsR0});
            decVarR0 = cat(2,decVar_cell{trialsR0});
            
            % average FR
            avgFR_R0 = zeros(size(fr_matR0,1), numel(decVar_bins)-1);
            for dIdx = 1:(length(decVar_bins)-1) % look up to the second second
                if ~isempty(find(fr_matR0(:,decVarR0 > decVar_bins(dIdx) & decVarR0 < decVar_bins(dIdx+1)),1))
                    avgFR_R0(:,dIdx) = mean(fr_matR0(:,decVarR0 > decVar_bins(dIdx) & decVarR0 < decVar_bins(dIdx+1)),2);
                else
                    avgFR_R0(:,dIdx) = mean(fr_matR0(:,decVarR0 > decVar_bins(dIdx-1) & decVarR0 < decVar_bins(dIdx)),2);
                end
            end
            
            % set only look at the neurons that fire in all 3 seconds. 
            % if they fire at least in the first timebin, they will fire in
            % all
            if binIdx == 1
                firing_ix = find(~all(avgFR_R0==0,2));
            end
            
            avgFR_R0 = avgFR_R0(firing_ix,:);
            
            avgFR_R0 = avgFR_R0 ./ max(avgFR_R0,[],2);

            [~,index] = max(avgFR_R0');
            [~,index_sort] = sort(index);
            
            fr_mat_cell{binIdx} = avgFR_R0;
            index_sort_cell{binIdx} = index_sort;
            
            avgFR_decVarR0_sorted = avgFR_R0(index_sort,:);
            
            subplot(3,3,1 + 4 * spcounter)
            
            colormap('jet')
            imagesc(flipud(avgFR_decVarR0_sorted))
            yticks([])
            if binIdx == 1
                xticks(40/2:40/2:40)
                xticklabels([500,1000])
                title(sprintf("1 Sec of %i00 sorted by 1 Sec",iRewsize))
            elseif binIdx == 2
                xticks(40/4:40/4:40)
                xticklabels([500,1000,1500,2000])
                title(sprintf("2 Sec of %i00 sorted by 2 Sec",iRewsize))
            elseif binIdx == 3
                xticks(40/6:40/6:40)
                xticklabels([500,1000,1500,2000,2500,3000])
                title(sprintf("3 Sec of %i00 sorted by 3 Sec",iRewsize))
            end
            spcounter = spcounter + 1;
        end
        
        % now go back and cover the diagonal 
        spcounter = 1;
        for binIdx = 1:3
            if binIdx == 1
                subplot(3,3,4)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{2},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("1 Sec of %i00 sorted by 2 Sec",iRewsize))
                yticks([])
                xticks(40/2:40/2:40)
                xticklabels([500,1000])
                subplot(3,3,7)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{3},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("1 Sec of %i00 sorted by 3 Sec",iRewsize))
                yticks([])
                xticks(40/2:40/2:40)
                xticklabels([500,1000])
            end
            if binIdx == 2
                subplot(3,3,2)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{1},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("2 Sec of %i00 sorted by 1 Sec",iRewsize))
                yticks([])
                xticks(40/4:40/4:40)
                xticklabels([500,1000,1500,2000])
                subplot(3,3,8)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{3},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("2 Sec of %i00 sorted by 3 Sec",iRewsize))
                yticks([])
                xticks(40/4:40/4:40)
                xticklabels([500,1000,1500,2000])
            end
            if binIdx == 3
                subplot(3,3,3)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{1},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("3 Sec of %i00 sorted by 1 Sec",iRewsize))
                yticks([])
                xticks(40/6:40/6:40)
                xticklabels([500,1000,1500,2000,2500,3000])
                subplot(3,3,6)
                avgFR_new_sort = fr_mat_cell{binIdx}(index_sort_cell{2},:);
                colormap('jet')
                imagesc(flipud(avgFR_new_sort))
                title(sprintf("3 Sec of %i00 sorted by 2 Sec",iRewsize))
                yticks([])
                xticks(40/6:40/6:40)
                xticklabels([500,1000,1500,2000,2500,3000])
            end    

        end
    end
end

