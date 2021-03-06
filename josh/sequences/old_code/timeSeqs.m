%% Building upon previous work w/ decision variables, now just looking at sequences through actual time
%  - Mainly comparing time vs time since last rewad
%  - Maybe use some seqNMF here
clear
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = opt.tbin*1000; % for making index vectors
opt.smoothSigma_time = 0.10; % gauss smoothing sigma for rate matrix (in sec)

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

for sIdx = 1:1 % 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL;
    
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    
    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        %         [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
        [fr_mat, tbincent] = calcFRVsTime(good_cells,dat,opt); % calc from full matrix
        toc
    end
    
    %     patchstop_ms = p_out.patchstop_ms + 9; % + 9;
    %     patchleave_ms = p_out.patchleave_ms + 9; % + 9;
    
    buffer = 500; % buffer before leave in ms
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round((patchleave_ms - buffer) / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
    % reinitialize ms vectors to make barcode matrix
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    rew_ms = dat.rew_ts;
    rew_size = mod(dat.patches(:,2),10);
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
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % make struct
    FR_decVar(sIdx).fr_mat = {length(dat.patchCSL)};
    for iTrial = 1:length(dat.patchCSL)
        FR_decVar(sIdx).fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_len_ix = size(FR_decVar(sIdx).fr_mat{iTrial},2);
        FR_decVar(sIdx).decVarTime{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        FR_decVar(sIdx).decVarTimeSinceRew{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
        
        for r = 1:numel(rew_sec_cell{iTrial})
            rew_ix = (rew_sec_cell{iTrial}(r) - 1) * 1000 / tbin_ms;
            FR_decVar(sIdx).decVarTimeSinceRew{iTrial}(rew_ix:end) =  (1:length(FR_decVar(sIdx).decVarTimeSinceRew{iTrial}(rew_ix:end))) * tbin_ms / 1000;
        end
    end
    
    close all; figure();hold on;
    plot(FR_decVar(sIdx).decVarTime{39})
    hold on
    plot(FR_decVar(sIdx).decVarTimeSinceRew{39})
    legend("Time","Time since last reward")
    title("Trial 39 decision variables")
end

%% First just make ridge plots w/ cheating on sorting
close all
index_sort_all = {sIdx};
for sIdx = 1:1
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end
%% make some datastructures
index_sort_all = {sIdx};
mid_responsive_neurons = {sIdx};
nMid = 100;
mid_responsive_neurons{1} = nan(nMid,2);
nBins = 40;
mid_responsive_peth = nan(2,nMid,nBins-1);

%% Use function to perform mid-responsive quantifications

close all
for sIdx = 1:1
    nTrials = length(FR_decVar(sIdx).fr_mat);
    
    % set decision variable
    dvar = "timesince";
    if dvar == "time"
        label = "Time";
        varIdx = 1;
    else
        label = "Time since last rew";
        varIdx = 2;
    end
    
    decVar_bins = linspace(0,2,nBins);
    
    all_opt = struct;
    all_opt.suppressVis = true;
    all_opt.norm = "zscore";
    all_opt.trials = 'all'; % 1:2:nTrials;
    [sorted_peth,index_sort_all{sIdx},~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,all_opt);
    
    % regress out linear, then nonlinear components to find neurons more defined by nonlinear
    % responsivity
    tic
    regressions = true;
    if regressions == true
        x = 1:size(sorted_peth,2);
        avgFR_decVar_sortedResidual = nan(size(sorted_peth));
        avgFR_decVar_sortedLinear = nan(size(sorted_peth));
        avgFR_decVar_sortedGauss = nan(size(sorted_peth));
        avgFR_decVar_sortedLinearResidual = nan(size(sorted_peth));
        avgFR_decVar_sortedGaussResidual = nan(size(sorted_peth));
        
        linear_r2 = nan(size(sorted_peth,1),1);
        gauss_r2 = nan(size(sorted_peth,1),1);
        
        slope_fit = nan(size(sorted_peth,1),1);
        intercept_fit = nan(size(sorted_peth,1),1);
        mu_fit = nan(size(sorted_peth,1),1);
        
        gaussModel = 'alpha * exp(-(x - mu).^2 / (2*sigma.^2))';
        
        for neuron = 1:size(sorted_peth,1)
            [linear_fit,linear_gof] = fit(x',sorted_peth(neuron,:)','poly1');
            slope_fit(neuron) = linear_fit.p1;
            intercept_fit(neuron) = linear_fit.p2;
            linear_fit = linear_fit.p1 * x' + linear_fit.p2;
            linear_r2(neuron) = linear_gof.rsquare;
            avgFR_decVar_sortedLinearResidual(neuron,:) = sorted_peth(neuron,:) - linear_fit';
            avgFR_decVar_sortedLinear(neuron,:) = linear_fit;
            
            % fit constrained s.t. mean between 250 and 1750 msec
            [gauss_fit,gauss_gof] = fit(x',sorted_peth(neuron,:)',gaussModel,'StartPoint',[1,20,20],'Lower',[.5,5,.5],'Upper',[20,35,10]);
            mu_fit(neuron) = gauss_fit.mu;
            gauss_fit = gauss_fit.alpha * exp(-(x - gauss_fit.mu).^2 / (2*gauss_fit.sigma.^2))';
            gauss_r2(neuron) = gauss_gof.rsquare;
            avgFR_decVar_sortedGaussResidual(neuron,:) = sorted_peth(neuron,:) - gauss_fit';
            avgFR_decVar_sortedGauss(neuron,:) = gauss_fit;
        end
    end
    toc
    
    % visualize fits
    threePaneFitPlot(sorted_peth,avgFR_decVar_sortedLinearResidual,avgFR_decVar_sortedLinear,decVar_bins,label,"Linear")
    threePaneFitPlot(sorted_peth,avgFR_decVar_sortedGaussResidual,avgFR_decVar_sortedGauss,decVar_bins,label,"Gaussian")
    % figure out who is mid-responsive
    subtr = gauss_r2 - linear_r2;
    starts = 1:100;
    mean_diff = nan(length(starts),1);
    for j = 1:numel(starts)
        neuron1 = starts(j);
        mean_diff(j) = mean(subtr(neuron1:neuron1 + 99));
    end
    [~,max_ix] = max(mean_diff);
    mid_responsive_neurons{sIdx}(:,varIdx) = starts(max_ix):(starts(max_ix) + 99);
    regLabels = ones(length(subtr),1);
    regLabels(mid_responsive_neurons{sIdx}(:,varIdx)) = 2;
    figure()
    gscatter(subtr,1:length(subtr),regLabels,[.4 .4 .4;1 0 0],'o')
    title("Gaussian fit improvement in variance explained by neuron")
    xlabel("Gaussian R^2 - Linear R^2")
    ylabel("Neuron") 
    legend(["Non Mid-responsive","Mid-responsive"]);
    
    figure();colormap('jet')
    %     [~,index] = max(sorted_peth(mid_responsive_neurons{sIdx}(:,varIdx),:),[],2);
    %     [~,resort_ix] = sort(index);
    imagesc(sorted_peth(mid_responsive_neurons{sIdx}(:,varIdx),:))
    mid_responsive_peth(varIdx,:,:) = reshape(sorted_peth(mid_responsive_neurons{sIdx}(:,varIdx),:),[1,nMid,nBins-1]);
end

%% Quantify ridge to background ratio for mid-responsive neurons vs shuffle
close all
for sIdx = 3:3
    decVar_cells = {};
    decVar_cells{1} = FR_decVar(sIdx).decVarTime;
    decVar_cells{2} = FR_decVar(sIdx).decVarTimeSinceRew;
    labels = {};
    labels{1} = "Shuffled";
    labels{2} = "Time";
    labels{3} = "Time Since Last Reward";
    
    r2b_ratio_unshuffled = nan(nMid,length(decVar_cells));
    ridge_width = 4; % .1 second window on either side of ridge
    
    figure()
    for vIdx = 1:length(decVar_cells)
        mid_fr_mat = squeeze(mid_responsive_peth(varIdx,:,:));
        [~,max_unshuffled_ix] = max(squeeze(mid_responsive_peth(vIdx,:,:)),[],2); % find the ridge
        
        backgroundUnshuffled = zeros(nMid,1);
        ridgeUnshuffled = zeros(nMid,1);
        
        figure(1);colormap('jet')
        subplot(1,2,vIdx)
        imagesc(flipud(squeeze(mid_responsive_peth(vIdx,:,:))));hold on
%         scatter(max_unshuffled_ix - ridge_width, 1:numel(max_unshuffled_ix),'w*')
%         scatter(max_unshuffled_ix + ridge_width, 1:numel(max_unshuffled_ix),'w*')
        for neuron = 1:nMid
            backgroundUnshuffled(neuron) =  mean(mid_fr_mat(neuron,[1:max(1,max_unshuffled_ix(neuron)-ridge_width),min(nBins-1,(max_unshuffled_ix(neuron)+ridge_width)):nBins-1]));
            ridgeUnshuffled(neuron) = mean(mid_fr_mat(neuron,max(1,max_unshuffled_ix(neuron)-ridge_width):min(nBins-1,max_unshuffled_ix(neuron)+ridge_width)));
            r2b_ratio_unshuffled(neuron,vIdx) = (ridgeUnshuffled(neuron)) / (backgroundUnshuffled(neuron)+3); % rough step
        end
    end
    
    %     r2b_ratio_unshuffled(r2b_ratio_unshuffled > 15) = 15;
    
    % now perform shuffle control 1000x to test for significance per cell
    shuffRepeats = 1000;
    
    % collect FR matrices
    fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
    fr_mat = fr_mat(index_sort_all{sIdx}(mid_responsive_neurons{sIdx}(:,1)),:);
    newShuffleControl = false;
    
    if newShuffleControl == true
        r2b_ratio_shuffled = zeros(nMid,shuffRepeats);
        for shuffIdx = 1:shuffRepeats
            if mod(shuffIdx,100) == 0
                display(shuffIdx)
            end
            
            % Now find ridge-to-background ratio for shuffled data
            shuffle_opt.shuffle = true;
            shuffle_opt.suppressVis = true; 
            shuffle_opt.norm = "peak";
            [shuffled_peth,~,~] = peakSortPETH(FR_decVar(sIdx),"time",decVar_bins,shuffle_opt);
            [~,max_shuffled_ix] = max(shuffled_peth,[],2);
            
            backgroundShuffled = zeros(nMid,1);
            ridgeShuffled = zeros(nMid,1);
            
            for neuron = 1:nMid
                backgroundShuffled(neuron) =  mean(shuffled_peth(neuron,[1:max(1,max_shuffled_ix(neuron)-ridge_width), min(nBins-1,max_shuffled_ix(neuron)+ridge_width):nBins-1]));
                ridgeShuffled(neuron) = mean(shuffled_peth(neuron,max(1,max_shuffled_ix(neuron)-ridge_width):min(nBins-1,max_shuffled_ix(neuron)+ridge_width)));
                r2b_ratio_shuffled(neuron,shuffIdx) = (ridgeShuffled(neuron)+3) / (backgroundShuffled(neuron)+3);
            end
        end
    end
    %
    figure()
    bar(1:(length(decVar_cells)+1),[mean(mean(r2b_ratio_shuffled,'omitnan')) mean(r2b_ratio_unshuffled,1)])
    hold on
    errorbar(1,mean(mean(r2b_ratio_shuffled,'omitnan')),1.96 * std(mean(r2b_ratio_shuffled,1,'omitnan')),'k')
    title("Ridge to background ratio across alignments")
    xticks(1:(length(decVar_cells)+1))
    xticklabels(labels)
    %
    % return p value
    mean_r2b_shuffled = mean(r2b_ratio_shuffled,1);
    mean_r2b_unshuffled = mean(r2b_ratio_unshuffled,1);
    for vIdx = 1:length(mean_r2b_unshuffled)
        p = length(find(mean_r2b_unshuffled(vIdx) < mean_r2b_shuffled)) / length(mean_r2b_shuffled);
        fprintf("\n p-value for %s: %f \n",labels{vIdx + 1},p);
    end
end
%% Ramping unit rew response heterogeneity analysis (temporary home)
close all;
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors to make barcode matrix
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    rew_ms = data.rew_ts;
    rew_size = mod(data.patches(:,2),10);
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_ix_cell{iTrial} = (rew_indices(rew_indices > 1) - 1) * 1000 / tbin_ms;
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    max_rew_ix = max(last_rew_ix);
    
    % Trial level features
    patches = data.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    nTrials = size(FR_decVar(sIdx).fr_mat,2);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    ramp_idx = index_sort_odd{sIdx}(150:end);
    %     ramp_idx = index_sort_odd{sIdx};
    seq_idx = index_sort_odd{sIdx}(1:150);
    
    new_regs = true;
    search_begin = round(250 / tbin_ms);
    search_end = round(750 / tbin_ms);
    
    % make the decision variable for regression
    decVar_cell = FR_decVar(sIdx).decVarTimeSinceRew;
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,40);
    decVar_bins = quantile(all_decVar,p); % nonlinear
    decVar_bins = linspace(min(all_decVar),max(all_decVar),40); % linear
    nBins = length(decVar_bins);
    
    if new_regs == true
        rew_responses_delta = {nTrials};
        rew_responses_value = {nTrials};
        pre_rew = {nTrials};
        rew_responses_delta_tensor = nan(nTrials,length(ramp_idx),max_rew_ix);
        rew_responses_value_tensor = nan(nTrials,length(ramp_idx),max_rew_ix);
        pre_rew_tensor = nan(nTrials,length(ramp_idx),max_rew_ix);
        slopes = nan(nTrials,length(ramp_idx));
        intercepts = nan(nTrials,length(ramp_idx));
        % for trial 39, 10 is a pretty active neuron to look at
        for iTrial = 39 % 1:nTrials
            %%% first investigate reward responses %%%
            % subselect ramping portion of FR mat
            fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(ramp_idx,:),[],2);
            
            % to visualize what's going on
            figure();colormap('jet')
            imagesc(flipud(fr_mat_iTrial))
            title(sprintf('Trial %i',iTrial))
            xlabel('Time')
            ylabel('All neurons, ordered by avg odd sort to timeSinceRew')
            %
            %             % draw lines to indicate start, end of reward response interval
            hold on
            plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w-','linewidth',1.5)
            rew_ix_cell{iTrial}(rew_ix_cell{iTrial} + search_end > size(fr_mat_iTrial,2)) = [];
            plot([rew_ix_cell{iTrial} + search_begin rew_ix_cell{iTrial} + search_begin]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w--','linewidth',1.5)
            plot([rew_ix_cell{iTrial} + search_end rew_ix_cell{iTrial} + search_end]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w--','linewidth',1.5)
            
            % for all neurons,
            % now find the mins forall neurons within these windows
            rew_responses_delta{iTrial} = nan(numel(ramp_idx),numel(rew_ix_cell{iTrial}));
            rew_responses_value{iTrial} = nan(numel(ramp_idx),numel(rew_ix_cell{iTrial}));
            pre_rew{iTrial} = nan(numel(ramp_idx),numel(rew_ix_cell{iTrial}));
            for iRew = 1:numel(rew_ix_cell{iTrial})
                rew_ix = round(rew_ix_cell{iTrial}(iRew));
                rew_sec = round(rew_ix_cell{iTrial}(iRew) * tbin_ms / 1000);
                
                [~,extrema_ix] = max(abs(fr_mat_iTrial(:,(rew_ix + search_begin):(rew_ix + search_end)) - fr_mat_iTrial(:,rew_ix)),[],2);
                %                 [~,extrema_ix] = min(fr_mat_iTrial(:,(rew_ix + search_begin):(rew_ix + search_end)),[],2);
                %                 hold on
                extrema_ix = extrema_ix + rew_ix + search_begin;
                %                 scatter(flipud(extrema_ix),1:numel(ramp_idx),1.5,'k*')
                
                % make sure this diag command is kosher
                rew_responses_delta{iTrial}(:,iRew) = diag(fr_mat_iTrial(:,extrema_ix)) - fr_mat_iTrial(:,rew_ix);
                rew_responses_value{iTrial}(:,iRew) = diag(fr_mat_iTrial(:,extrema_ix));
                pre_rew{iTrial}(:,iRew) = fr_mat_iTrial(:,rew_ix);
                rew_responses_delta_tensor(iTrial,:,rew_sec) = diag(fr_mat_iTrial(:,extrema_ix)) - fr_mat_iTrial(:,rew_ix);
                rew_responses_value_tensor(iTrial,:,rew_sec) = diag(fr_mat_iTrial(:,extrema_ix));
                pre_rew_tensor(iTrial,:,rew_sec) = fr_mat_iTrial(:,rew_ix);
            end
            
            %%% now perform linear fits %%%
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
                fprintf("Trial %i complete \n",iTrial);
            end
            
        end
    end
    
    %     iTrial = 160;
    %     for sample_neuron = 10 % 10 is a pretty nice neuron to look at
    %         figure()
    %         subplot(1,3,1)
    %         plot(fr_mat_iTrial(sample_neuron,:))
    %         hold on
    %         % the actual best way to do this would be to put a star on the
    %         % line
    %         scatter(rew_ix_cell{iTrial},fr_mat_iTrial(sample_neuron,rew_ix_cell{iTrial}),'k*')
    %         scatter(rew_ix_cell{iTrial} + search_begin,fr_mat_iTrial(sample_neuron,rew_ix_cell{iTrial} + search_begin),'g*')
    %         scatter(rew_ix_cell{iTrial} + search_end,fr_mat_iTrial(sample_neuron,rew_ix_cell{iTrial} + search_end),'r*')
    %         subplot(1,3,2);plot(rew_responses_delta{iTrial}(sample_neuron,:),'linewidth',1.5)
    %         subplot(1,3,3);plot(rew_responses_value{iTrial}(sample_neuron,:),'linewidth',1.5)
    %     end
    
    % look at 1st reward responses on diff rewsize trials as a first shot?
    delta_bySize = nan(length(ramp_idx),3);
    value_bySize = nan(length(ramp_idx),3);
    delta_RR0 = nan(length(ramp_idx),3);
    value_RR0 = nan(length(ramp_idx),3);
    delta_R0R = nan(length(ramp_idx),3);
    value_R0R = nan(length(ramp_idx),3);
    
    for iRewsize = [1,2,4]
        iRewsizeTrials = find(rewsize == iRewsize);
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.55);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.55);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
        
        delta_bySize(:,iRewsize) = mean(rew_responses_delta_tensor(trials11x,:,1),1,'omitnan');
        value_bySize(:,iRewsize) = mean(rew_responses_value_tensor(trials11x,:,1),1,'omitnan');
        
        delta_RR0(:,iRewsize) = mean(rew_responses_delta_tensor(trials110x,:,1),1,'omitnan');
        value_RR0(:,iRewsize) = mean(rew_responses_value_tensor(trials110x,:,1),1,'omitnan');
        
        delta_R0R(:,iRewsize) = mean(rew_responses_delta_tensor(trials101x,:,2),1,'omitnan');
        value_R0R(:,iRewsize) = mean(rew_responses_value_tensor(trials101x,:,2),1,'omitnan');
    end
    
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
    
    figure();
    subplot(2,2,1)
    scatter(delta_bySize(:,1),delta_bySize(:,4))
    xlabel("11 rew response")
    ylabel("44 rew response")
    subplot(2,2,2)
    scatter(delta_bySize(:,2),delta_bySize(:,4))
    xlabel("Mean 22 rew response")
    ylabel("Mean 44 rew response")
    subplot(2,2,3)
    scatter(delta_R0R(:,2),delta_R0R(:,4))
    xlabel("Mean 202 rew response")
    ylabel("Mean 404 rew response")
    subplot(2,2,4)
    scatter(delta_R0R(:,4),mean(slopes))
    xlabel("Mean 404 rew response")
    ylabel("Fitted slope to time since last rew")
    
    figure()
    subplot(1,2,1)
    means = mean(mean(rew_responses_value_tensor,1,'omitnan'),2,'omitnan');
    means = reshape(means,[size(means,3),1]);
    stds = std(mean(rew_responses_value_tensor,1,'omitnan'),[],2,'omitnan');
    stds = reshape(stds,[size(stds,3),1]);
    plot(means,'linewidth',1.5)
    title('Mean reward response (value of extrema) vs time of rew')
    xlabel("Reward delivery time")
    ylabel("Mean value")
    %     errorbar(means,stds)
    subplot(1,2,2)
    means = mean(mean(rew_responses_delta_tensor,1,'omitnan'),2,'omitnan');
    means = reshape(means,[size(means,3),1]);
    stds = std(mean(rew_responses_delta_tensor,1,'omitnan'),[],2,'omitnan');
    stds = reshape(stds,[size(stds,3),1]);
    plot(means,'linewidth',1.5)
    xlabel("Reward delivery time")
    ylabel("Mean delta")
    title('Mean reward response (delta) over time vs time of rew')
    %     errorbar(means,stds)
    
    [~,slope_sort] = sort(mean(slopes(rewsize == 4,:)));
    [p_slope,tbl_slope,stats_slope] = anova1(slopes(rewsize == 4,slope_sort));
    xticks(0:50:150)
    xticklabels(0:50:150)
    
    [~,int_sort] = sort(mean(intercepts(rewsize == 4,:)));
    [p_int,tbl_int,stats_int] = anova1(intercepts(rewsize == 4,int_sort));
    xticks(0:50:150)
    xticklabels(0:50:150)
    
    first_rew_resps = rew_responses_delta_tensor(:,:,2);
    [~,delta_sort] = sort(mean(first_rew_resps(rewsize == 4,:),'omitnan'));
    [p_delta,tbl_delta,stats_delta] = anova1(first_rew_resps(rewsize == 4,delta_sort));
    xticks(0:50:150)
    xticklabels(0:50:150)
    
    delta_bySize(:,3) = [];
    [~,deltaRew_sort] = sort(mean(delta_bySize,'omitnan'));
    [p_deltaBySize,tbl_deltaBySize,stats_deltaBySize] = anova1(delta_bySize(:,deltaRew_sort));
    
end

%% divide by PRT and look at sequence activity

close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    dvar = "timeSince";
    
    if dvar == "time"
        decVar_cell = FR_decVar(sIdx).decVarTime;
        label = "Time";
    else
        decVar_cell = FR_decVar(sIdx).decVarTimeSinceRew;
        label = "Time since last rew";
    end
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    % for axis by quartile
    p = linspace(0,.85,41);
    %     decVar_bins = quantile(all_decVar,p);
    decVar_bins = linspace(0,2,40);
    
    prt_bins = quantile(prts,[0,.25,.5,.75,1]);
    
    figure()
    
    nMid = length(mid_responsive_neurons{sIdx}{2});
    
    %     maxIx = nan(size(FR_decVar(sIdx).fr_mat{1},1),4);
    maxIx = nan(nMid,4);
    
    titles = {"Q1 PRT Trial PETH","Q2 PRT Trial PETH","Q3 PRT Trial PETH","Q4 PRT Trial PETH"};
    for prtIdx = 1:(length(prt_bins)-1)
        prt_trials = find(prts > prt_bins(prtIdx) & prts < prt_bins(prtIdx + 1));
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{prt_trials});
        fr_mat = fr_mat(mid_responsive_neurons{sIdx}{2},:);
        decVar = cat(2,decVar_cell{prt_trials});
        
        %%% calculate FR averaged over decision variable bins for real data %%%
        avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if ~isempty(find(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            elseif dIdx > 1
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
            end
        end
        
        avgFR_decVar = zscore(avgFR_decVar,[],2);
        %     avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);
        [~,index] = max(avgFR_decVar');
        
        %         if prtIdx == 1
        [~,index_sort] = sort(index);
        %         end
        
        avgFR_decVar_sorted = avgFR_decVar(index_sort,:);
        
        [~,iMax_ix] = max(avgFR_decVar_sorted,[],2);
        maxIx(:,prtIdx) = iMax_ix;
        
        % now making xticks at even seconds
        max_round = floor(max(decVar_bins));
        secs = 0:max_round;
        x_idx = [];
        for i = secs
            x_idx = [x_idx find(decVar_bins > i,1)];
        end
        
        figure(1)
        subplot(1,5,prtIdx)
        imagesc(flipud(avgFR_decVar_sorted));colormap('jet')
        if prtIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        colorbar()
        hold on
        %         scatter(flipud(iMax_ix),1:nNeurons,3,'w');
        scatter(flipud(iMax_ix),1:nMid,3,'w');
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        ylabel("Neurons")
        title(titles{prtIdx})
    end
    subplot(1,5,5)
    %     plot(maxIx,flipud(1:nNeurons),'linewidth',2)
    plot(maxIx,flipud(1:nMid),'linewidth',2)
    xticks(x_idx)
    xticklabels(secs)
    legend("Q1 PRT trials","Q2 PRT Trials","Q3 PRT Trials","Q4 PRT Trials")
    title("Activation progression")
    xlabel(label)
    %     ylim([0,nNeurons])
    %     yticks((0:50:nNeurons) + mod(nNeurons,100))
    %     yticklabels(nNeurons - mod(nNeurons,100) - (0:50:nNeurons))
    ylim([0,nMid])
    yticks((0:50:nMid) + mod(nMid,100))
    yticklabels(nMid - mod(nMid,100) - (0:50:nMid))
end

%% divide by reward size and look at sequence activity
close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    nMid = length(mid_responsive_neurons);
    nMid = 150;
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    
    dvar = "timeSince";
    
    if dvar == "time"
        decVar_cell = FR_decVar(sIdx).decVarTime;
        label = "Time";
    else
        decVar_cell = FR_decVar(sIdx).decVarTimeSinceRew;
        label = "Time since last rew";
    end
    
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    % for axis by quartile
    p = linspace(0,.70,41);
    decVar_bins = quantile(all_decVar,p);
    decVar_bins = linspace(0,2,40);
    
    figure()
    
    %     maxIx = nan(nNeurons,4);
    maxIx = nan(nMid,4);
    
    titles = {"1 uL Trials","2 uL Trials","4 uL Trials"};
    rewsizes = [1,2,4];
    
    for j = 1:3
        iRewsize = rewsizes(j);
        rewsizeTrials = find(rewsize == iRewsize);
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{rewsizeTrials});
        %         fr_mat = fr_mat(mid_responsive_neurons,:);
        fr_mat = fr_mat(index_sort_odd{sIdx}(1:150),:);
        decVar = cat(2,decVar_cell{rewsizeTrials});
        
        %%% calculate FR averaged over decision variable bins for real data %%%
        avgFR_decVar = zeros(size(fr_mat,1), numel(decVar_bins)-1);
        for dIdx = 1:(numel(decVar_bins) - 1) % go up to 80th percentile
            if ~isempty(find(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),1))
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx) & decVar < decVar_bins(dIdx+1)),2);
            elseif dIdx > 1
                avgFR_decVar(:,dIdx) = mean(fr_mat(:,decVar > decVar_bins(dIdx-1) & decVar < decVar_bins(dIdx)),2);
            end
        end
        
        avgFR_decVar = zscore(avgFR_decVar,[],2);
        %     avgFR_decVar = avgFR_decVar ./ max(avgFR_decVar,[],2);
        [~,index] = max(avgFR_decVar');
        
        %         if prtIdx == 1
        [~,index_sort] = sort(index);
        %         end
        
        avgFR_decVar_sorted = avgFR_decVar(index_sort,:);
        
        [~,iMax_ix] = max(avgFR_decVar_sorted,[],2);
        maxIx(:,j) = iMax_ix;
        
        % now making xticks at even seconds
        max_round = floor(max(decVar_bins));
        secs = 0:max_round;
        x_idx = [];
        for i = secs
            x_idx = [x_idx find(decVar_bins > i,1)];
        end
        
        figure(1)
        subplot(1,4,j)
        imagesc(flipud(avgFR_decVar_sorted));colormap('jet')
        if j == 1
            cl1 = caxis;
        end
        caxis(cl1)
        colorbar()
        hold on
        %         scatter(flipud(iMax_ix),1:nNeurons,3,'w');
        scatter(flipud(iMax_ix),1:nMid,3,'w');
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        ylabel("Neurons")
        title(titles{j})
    end
    
    subplot(1,4,4)
    plot(maxIx,flipud(1:nMid),'linewidth',2)
    %     plot(maxIx,flipud(1:nNeurons),'linewidth',2)
    xticks(x_idx)
    xticklabels(secs)
    legend(titles)
    title("Activation progression")
    xlabel(label)
    %     ylim([0,nNeurons])
    %     yticks((0:50:nNeurons) + mod(nNeurons,100))
    %     yticklabels(nNeurons - mod(nNeurons,100) - (0:50:nNeurons))
    ylim([0,nMid])
    yticks((0:50:nMid) + mod(nMid,100))
    yticklabels(nMid - mod(nMid,100) - (0:50:nMid))
end

%% Ay it's ur boi single trial quantification
close all

% define some data structures for cross day pooling
lgProgSlopes = [];
lgPRTs = [];
lgPRTs_dayVec = [];
mdProgSlopes = [];
mdPRTs = [];
mdPRTs_dayVec = [];
smProgSlopes = [];
smPRTs = [];
smPRTs_dayVec = [];
pooled_rampSlopes = [];
pooled_seqSlopes = [];
dayVec = [];

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors to make barcode matrix
    rew_ms = data.rew_ts;
    patchCSL = data.patchCSL;
    patches = data.patches;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(patchType);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_ix_cell{iTrial} = (rew_indices(rew_indices > 1) - 1) * 1000 / tbin_ms;
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % from barcode matrix, get the trials w/ only rew at t = 0
    one_rew_trials = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1);
    lg1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 4);
    md1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 2);
    sm1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 1);
    
    % outer loop to test over different population windows
    groupSize = 100;
    groupStarts = linspace(1,100,50);
    % datastructures for group testing
    seqPRTpValue = nan(4,numel(groupStarts));
    seqPRTpearsonr = nan(4,numel(groupStarts));
    seqRamppValue = nan(4,numel(groupStarts));
    seqRamppearsonr = nan(4,numel(groupStarts));
    
    shuffle = false;
    
    for gIdx = 1
        % select population for analysis
%         first = groupStarts(gIdx); 
        first = 45;
        last = first + groupSize;
        early_resp = index_sort_all{sIdx}(first:last); % fliplr(mid_responsive_neurons); %
        if shuffle == true
            early_resp = early_resp(randperm(length(early_resp)));
        end
        ramps = index_sort_all{sIdx}(151:end);
        
        % data structures
        prog_slopes = nan(nTrials,1);
        intercepts = nan(nTrials,1);
        slopes = nan(nTrials,1);
        intercepts_seq = nan(nTrials,1);
        slopes_seq = nan(nTrials,1);
        overall_mean_seq = nan(nTrials,1);
        
        for j = 1:numel(one_rew_trials)
            iTrial = one_rew_trials(j);
            norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(early_resp,:),[],2);
            [times,neurons] = find(norm_fr_mat_iTrial(:,1:50) > 0);
            activity = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,1:50) > 0);
            
            % weighted linear regression on first second sequence
            mdl = fitlm(neurons,times,'Intercept',false,'Weights',activity);
            prog_slopes(iTrial) = mdl.Coefficients.Estimate;
            
            % take slope of mean ramp
            ramp_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(ramps,:),[],2);
            mean_ramp = mean(ramp_iTrial);
            mdl2 = fitlm(1:size(ramp_iTrial,2),mean_ramp);
            intercepts(iTrial) = mdl2.Coefficients.Estimate(1);
            slopes(iTrial) = mdl2.Coefficients.Estimate(2);
            
            overall_mean_seq(iTrial) = mean(mean(norm_fr_mat_iTrial(:,1:50)));
            mean_seq = mean(norm_fr_mat_iTrial(:,1:50));
            mdl3 = fitlm(1:50,mean_seq);
            intercepts_seq(iTrial) = mdl3.Coefficients.Estimate(1);
            slopes_seq(iTrial) = mdl3.Coefficients.Estimate(2);
            
%             % visualize ramp
%             figure();
%             %             subplot(3,1,1)
%             %             colormap('jet')
%             %             imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_odd{sIdx},:),[],2)))
%             %             title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%             %             hold on
%             %             plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd{sIdx})],'w-','linewidth',1.5)
%             subplot(2,1,1)
%             colormap('jet')
%             imagesc(flipud(ramp_iTrial))
%             title(sprintf("%i uL Trial %i Ramping Neuron Activity",rewsize(iTrial),iTrial))
%             hold on
%             plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_all{sIdx})],'w-','linewidth',1.5)
%             if size(norm_fr_mat_iTrial,2) < 100
%                 xticks([0 25 50 75])
%                 xticklabels([0 500 1000 1500])
%             else
%                 xticks([0 25 50 75 100])
%                 xticklabels([0 500 1000 1500 2000])
%             end
%             xlabel("Time on patch (ms)")
%             % plot mean ramp and fit a linear regression
%             subplot(2,1,2);plot(mean_ramp,'linewidth',2)
%             title("Mean Ramping Activity and Linear Fit")
%             xlim([0,length(mean(ramp_iTrial))])
%             if size(norm_fr_mat_iTrial,2) < 100
%                 xticks([0 25 50 75])
%                 xticklabels([0 500 1000 1500])
%             else
%                 xticks([0 25 50 75 100])
%                 xticklabels([0 500 1000 1500 2000])
%             end
%             hold on; plot(mdl2.Fitted,'linewidth',2)
%             xlabel("Time on patch (ms)")
%             
%             %   visualize sequence
%             figure();
%             subplot(1,2,1)
%             colormap('jet')
%             [~,index] = max(zscore(FR_decVar(sIdx).fr_mat{iTrial},[],2),[],2);
%             %                         [~,trial_sort] = sort(index);
%             imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:),[],2)))
%             %                         imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(trial_sort,:),[],2)))
%             hold on
%             plot([50 50]',[305 - first 305 - last]','w-','linewidth',2)
%             plot([1 50]',[305 - first 305 - first]','w-','linewidth',2)
%             plot([1 50]',[305 - last 305 - last]','w-','linewidth',2)
%             title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%             xlabel("Time on Patch (ms)")
%             if size(norm_fr_mat_iTrial,2) < 100
%                 xticks([0 25 50 75])
%                 xticklabels([0 500 1000 1500])
%             else
%                 xticks([0 25 50 75 100])
%                 xticklabels([0 500 1000 1500 2000])
%             end
%             subplot(1,2,2)
%             colormap('jet')
%             imagesc(flipud(norm_fr_mat_iTrial(:,1:50)))
%             colorbar()
%             title(sprintf('Trial %i',iTrial))
%             xlabel('Time')
%             title("Area of interest")
%             %                         draw lines to indicate reward delivery
%             subplot(1,2,2); hold on
%             title("Data for Weighted Linear Regression")
%             scatter(neurons,times,activity,'kx')
%             xlabel("Time on Patch (ms)")
%             xticks([0 25 50])
%             xticklabels([0 500 1000])
%             plot(neurons,mdl.Fitted,'linewidth',2)

        end
        % Sequence-PRT correlation
        [r0,p0] = corrcoef(prog_slopes(one_rew_trials),prts(one_rew_trials));
        [r1,p1] = corrcoef(prog_slopes(sm1rew),prts(sm1rew));
        [r2,p2] = corrcoef(prog_slopes(md1rew),prts(md1rew));
        [r3,p3] = corrcoef(prog_slopes(lg1rew),prts(lg1rew));
        seqPRTpValue(1,gIdx) = p0(2);seqPRTpearsonr(1,gIdx) = r0(2);
        seqPRTpValue(2,gIdx) = p1(2);seqPRTpearsonr(2,gIdx) = r1(2);
        seqPRTpValue(3,gIdx) = p2(2);seqPRTpearsonr(3,gIdx) = r2(2);
        seqPRTpValue(4,gIdx) = p3(2);seqPRTpearsonr(4,gIdx) = r3(2); 
        
        % Sequence-ramp correlation
        [r0,p0] = corrcoef(prog_slopes(one_rew_trials),slopes(one_rew_trials));
        [r1,p1] = corrcoef(prog_slopes(sm1rew),slopes(sm1rew));
        [r2,p2] = corrcoef(prog_slopes(md1rew),slopes(md1rew));
        [r3,p3] = corrcoef(prog_slopes(lg1rew),slopes(lg1rew)); 
        seqRamppValue(1,gIdx) = p0(2);seqRamppearsonr(1,gIdx) = r0(2);
        seqRamppValue(2,gIdx) = p1(2);seqRamppearsonr(2,gIdx) = r1(2);
        seqRamppValue(3,gIdx) = p2(2);seqRamppearsonr(3,gIdx) = r2(2);
        seqRamppValue(4,gIdx) = p3(2);seqRamppearsonr(4,gIdx) = r3(2);

    end
    
    %         colors = [0 0 0; 0 1 1;0 0 1;0 0 1];
    % Seq-PRT
    [r0,p0] = corrcoef(prog_slopes(one_rew_trials),prts(one_rew_trials));
    [r1,p1] = corrcoef(prog_slopes(sm1rew),prts(sm1rew));
    [r2,p2] = corrcoef(prog_slopes(md1rew),prts(md1rew));
    [r3,p3] = corrcoef(prog_slopes(lg1rew),prts(lg1rew));
    colors = cool(3);
    figure();hold on
    gscatter(prog_slopes(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
    xlabel("Slope of sequence progression")
    ylabel("PRT")
    title(sprintf("Slope of sequence progression vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    legend("1 uL","2 uL","4 uL") 
    
    % Sequence-ramp correlation
    [r0,p0] = corrcoef(prog_slopes(one_rew_trials),slopes(one_rew_trials));
    [r1,p1] = corrcoef(prog_slopes(sm1rew),slopes(sm1rew));
    [r2,p2] = corrcoef(prog_slopes(md1rew),slopes(md1rew));
    [r3,p3] = corrcoef(prog_slopes(lg1rew),slopes(lg1rew));
    figure();hold on
    gscatter(prog_slopes(one_rew_trials),slopes(one_rew_trials),rewsize(one_rew_trials),colors,'.')
    title(sprintf("Slope of sequence progression vs slope of mean ramp (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    xlabel("Slope of sequence progression")
    ylabel("Slope of Mean Ramp")
    
    % Slope of mean sequence
%     [r0,p0] = corrcoef(slopes_seq(one_rew_trials),prts(one_rew_trials));
%     [r1,p1] = corrcoef(slopes_seq(sm1rew),prts(sm1rew));
%     [r2,p2] = corrcoef(slopes_seq(md1rew),prts(md1rew));
%     [r3,p3] = corrcoef(slopes_seq(lg1rew),prts(lg1rew));
%     figure();hold on
%     gscatter(slopes_seq(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'o')
%     title(sprintf("Slope of mean sequence vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
%     ylabel("PRT")
%     xlabel("Mean sequence slopes")
    
    % Mean value of sequence-PRT correlation
    [r0,p0] = corrcoef(overall_mean_seq(one_rew_trials),prts(one_rew_trials));
    [r1,p1] = corrcoef(overall_mean_seq(sm1rew),prts(sm1rew));
    [r2,p2] = corrcoef(overall_mean_seq(md1rew),prts(md1rew));
    [r3,p3] = corrcoef(overall_mean_seq(lg1rew),prts(lg1rew));
    figure();hold on
    gscatter(overall_mean_seq(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
    title(sprintf("Mean Sequence Activity vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    ylabel("PRT")
    xlabel("Mean Sequence Activity")

%     figure();hold on
%     gscatter(prog_slopes(one_rew_trials),slopes(one_rew_trials),rewsize(one_rew_trials),colors,'o')
%     title(sprintf("Slope of sequence progression vs slope of mean ramp (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
%     ylabel("Slope of mean ramp")
%     xlabel("Slope of sequence progression")
    
    %     end
    
    % update pooled datastructures
    lgProgSlopes = [lgProgSlopes; prog_slopes(lg1rew)];
    lgPRTs = [lgPRTs; prts(lg1rew)];
    lgPRTs_dayVec = [lgPRTs_dayVec; sIdx * ones(length(prts(lg1rew)),1)];
    mdProgSlopes = [mdProgSlopes; prog_slopes(md1rew)];
    mdPRTs = [mdPRTs; prts(md1rew)];
    mdPRTs_dayVec = [mdPRTs_dayVec; sIdx * ones(length(prts(md1rew)),1)];
    smProgSlopes = [smProgSlopes; prog_slopes(sm1rew)];
    smPRTs = [smPRTs; prts(sm1rew)];
    smPRTs_dayVec = [smPRTs_dayVec; sIdx * ones(length(prts(sm1rew)),1)];
    pooled_rampSlopes = [pooled_rampSlopes; slopes(one_rew_trials)];
    pooled_seqSlopes = [pooled_seqSlopes; prog_slopes(one_rew_trials)];
    dayVec = [dayVec; sIdx * ones(length(prog_slopes(one_rew_trials)),1)];
end

% fit to pooled data
mdl1 = fitlm(smPRTs,smProgSlopes);
p1 = mdl1.Coefficients.pValue(2);
mdl2 = fitlm(mdPRTs,mdProgSlopes);
p2 = mdl2.Coefficients.pValue(2);
mdl3 = fitlm(lgPRTs,lgProgSlopes);
p3 = mdl3.Coefficients.pValue(2);
%
figure()
% subplot(1,3,1)
gscatter(smProgSlopes,smPRTs,smPRTs_dayVec,[.25 1 1; 0 1 1; 0 .75 .75],'.');
title(sprintf("Pooled 1uL Sequence Progression Slope vs PRT (p = %f)",p1));
ylabel("PRT")
xlabel("Slope of sequence progression")
% subplot(1,3,2)
legend("3/15","3/16","3/17")
figure()
gscatter(mdProgSlopes(mdPRTs < 5),mdPRTs(mdPRTs < 5),mdPRTs_dayVec(mdPRTs < 5),[.75 .75 1; .5 .5 1; .25 .25 .75],'.');
title(sprintf("Pooled 2uL Sequence Progression Slope vs PRT (p = %f)",p2));
ylabel("PRT")
xlabel("Slope of sequence progression")
% subplot(1,3,3)
legend("3/15","3/16","3/17")
figure()
gscatter(lgProgSlopes,lgPRTs,lgPRTs_dayVec,[1 .25 1; 1 0 1; .75 0 .75],'.');
title(sprintf("Pooled 4 uL Sequence Progression Slope vs PRT (p = %f)",p3));
ylabel("PRT")
xlabel("Slope of sequence progression")
legend("3/15","3/16","3/17")

% now for seq-ramp connection
mdl = fitlm(pooled_seqSlopes,pooled_rampSlopes);
p = mdl.Coefficients.pValue(2);
figure()
gscatter(pooled_seqSlopes,pooled_rampSlopes,dayVec,[],'.')
title(sprintf("Pooled sequence progression slope vs mean ramp slope (p = %f)",p))
xlabel("Pooled sequence progression slope")
ylabel("Pooled mean ramp slope")

%% visualize differences across group choices
close all
% seq-PRT
figure()
subplot(2,1,1);hold on
plot(groupStarts,log(seqPRTpValue(1,:))','linewidth',2)
plot(groupStarts,log(seqRamppValue(1,:))','linewidth',2)
plot([0 max(groupStarts)],[log(.05) log(.05)],'k--')
title(sprintf("%i Neuron Sequence-PRT Correlation log-pValue across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)")
legend("Sequence-PRT","Sequence-Ramp") 
ylabel("Log pValue")
subplot(2,1,2);hold on
plot(groupStarts,seqPRTpearsonr(1,:)','linewidth',2)
plot(groupStarts,seqRamppearsonr(1,:)','linewidth',2)
title(sprintf("%i Neuron Sequence-PRT Pearson Correlation across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)")
legend("Sequence-PRT","Sequence-Ramp") 
ylabel("Pearson Correlation")

% seq-PRT by reward size
figure()
subplot(2,1,1);hold on
set(gca, 'ColorOrder', colors)
plot(groupStarts,log(seqPRTpValue(2:4,:))','linewidth',2)
plot([0 max(groupStarts)],[log(.05) log(.05)],'k--')
title(sprintf("%i Neuron Sequence-PRT Correlation log-pValue across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)")
subplot(2,1,2);hold on
set(gca, 'ColorOrder', colors)
plot(groupStarts,seqPRTpearsonr(2:4,:)','linewidth',2)
title(sprintf("%i Neuron Sequence-PRT Pearson Correlation across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)") 
ylabel("Log pValue")

% seq-Ramp by reward size
figure()
subplot(2,1,1);hold on
set(gca, 'ColorOrder', colors)
plot(groupStarts,log(seqRamppValue(2:4,:))','linewidth',2)
plot([0 max(groupStarts)],[log(.05) log(.05)],'k--')
title(sprintf("%i Neuron Sequence-Ramp Correlation log-pValue across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)")
subplot(2,1,2);hold on
set(gca, 'ColorOrder', colors)
plot(groupStarts,seqRamppearsonr(2:4,:)','linewidth',2)
title(sprintf("%i Neuron Sequence-Ramp Pearson Correlation across neuron selection",groupSize))
xlabel("Neuron group start (Mean PETH order)") 
ylabel("Pearson Correlation")


%% Now the same for 2-reward trials

close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors to make barcode matrix
    rew_ms = data.rew_ts;
    patchCSL = data.patchCSL;
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    prts = patchCSL(:,3) - patchCSL(:,2) - .55;
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(patchType);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_ix_cell{iTrial} = (rew_indices(rew_indices > 1) - 1) * 1000 / tbin_ms;
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    early_resp = index_sort_odd{sIdx}(40:140);
    
    firstTwo_reward_trials = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) > 0 & rew_barcode(:,3) == -1);
    % figure out this indexing when less high
    two_reward_trials = find((cellfun(@length,rew_ix_cell) == 1) & prts' > 2 & (prts - (last_rew_ix - 1))' > 1);
    sm2rew = find((rew_barcode(:,1) == 1)' & (cellfun(@length,rew_ix_cell) == 1) & prts' > 2 & (prts - (last_rew_ix - 1))' > 1);
    md2rew = find((rew_barcode(:,1) == 2)' & (cellfun(@length,rew_ix_cell) == 1) & prts' > 2 & (prts - (last_rew_ix - 1))' > 1);
    lg2rew = find((rew_barcode(:,1) == 4)' & (cellfun(@length,rew_ix_cell) == 1) & prts' > 2 & (prts - (last_rew_ix - 1))' > 1);
    
    rewLocs = cell2mat(rew_ix_cell(two_reward_trials)) ./ 50;
    rewLocsSm = cell2mat(rew_ix_cell(sm2rew)) ./ 50;
    rewLocsMd = cell2mat(rew_ix_cell(md2rew)) ./ 50;
    rewLocsLg = cell2mat(rew_ix_cell(lg2rew)) ./ 50;
    
    prog_slopes = nan(nTrials,2);
    
    for j = 1:numel(two_reward_trials)
        iTrial = two_reward_trials(j);
        
        % calculate and log sequence stuff
        norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(early_resp,:),[],2);
        [times,neurons] = find(norm_fr_mat_iTrial(:,1:100) > 0); % significant activations
        activity = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,1:100) > 0);
        [times1,neurons1] = find(norm_fr_mat_iTrial(:,1:50) > 0); % significant activations
        activity1 = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,1:50) > 0);
        [times2,neurons2] = find(norm_fr_mat_iTrial(:,rew_ix_cell{iTrial}:rew_ix_cell{iTrial} + 50) > 0); % significant activations
        activity2 = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,rew_ix_cell{iTrial}:rew_ix_cell{iTrial} + 50) > 0);
        % linear regression on pattern of significant activations
        mdl1 = fitlm(neurons1,times1,'Intercept',false);
        mdl2 = fitlm(neurons2,times2,'Intercept',false);
        prog_slopes(iTrial,:) = [mdl1.Coefficients.Estimate mdl2.Coefficients.Estimate];
        
        % log ramp stuff
        ramp_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(ramps,:),[],2);
        mean_ramp1 = mean(ramp_iTrial(:,1:(rew_ix_cell{iTrial})));
        mean_ramp2 = mean(ramp_iTrial(:,(rew_ix_cell{iTrial}):end));
        mdl1 = fitlm(1:length(mean_ramp1),mean_ramp1);
        mdl2 = fitlm(1:length(mean_ramp2),mean_ramp2);
        %         intercepts(iTrial) = mdl1.Coefficients.Estimate(1);
        %         slopes(iTrial) = mdl1.Coefficients.Estimate(2);
        
        %         % visualize ramp
        %         figure();
        %         subplot(3,1,1)
        %         colormap('jet')
        %         imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_odd{sIdx},:),[],2)))
        %         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
        %         hold on
        %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd{sIdx})],'w-','linewidth',1.5)
        %         subplot(3,1,2)
        %         colormap('jet')
        %         imagesc(flipud(ramp_iTrial))
        %         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
        %         hold on
        %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd{sIdx})],'w-','linewidth',1.5)
        %         % plot mean ramp and fit a linear regression up to its max
        %         subplot(3,1,3);plot(mean_ramp,'linewidth',2)
        %         xlim([0,length(mean(ramp_iTrial))])
        %         hold on;
        %         plot(1:length(mean_ramp1),mdl1.Fitted,'linewidth',2);
        %         plot((1:length(mean_ramp2)) + length(mean_ramp1),mdl2.Fitted,'linewidth',2)
        %         xlim([0,size(ramp_iTrial,2)])
        
        % visualize sequence
        %         figure();
        %         subplot(1,3,1)
        %         colormap('jet')
        %         imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_odd{sIdx},:),[],2)))
        %         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
        %         hold on
        %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd{sIdx})],'w-','linewidth',1.5)
        %         subplot(1,3,2)
        %         colormap('jet')
        %         imagesc(flipud(norm_fr_mat_iTrial(:,1:100)))
        %         colorbar()
        %         title(sprintf('Trial %i',iTrial))
        %         xlabel('Time')
        %         title("mean odd sort")
        %         % draw lines to indicate reward delivery
        %         hold on
        %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd{sIdx})],'w-','linewidth',1.5)
        %         subplot(1,3,3); hold on
        %         scatter(neurons,times,activity,'kx')
        %         plot([50 50],[0 100],'r--')
        %         plot(neurons1,mdl1.Fitted,'linewidth',2)
        % plot(neurons2 + 50,mdl2.Fitted,'linewidth',2)
        
    end
    
    % First Sequence-PRT correlation
    [r0,p0] = corrcoef(prog_slopes(two_reward_trials,1),prts(two_reward_trials));
    [r1,p1] = corrcoef(prog_slopes(sm2rew),prts(sm2rew));
    [r2,p2] = corrcoef(prog_slopes(md2rew),prts(md2rew));
    [r3,p3] = corrcoef(prog_slopes(lg2rew),prts(lg2rew));
    figure();hold on
    gscatter(prog_slopes(two_reward_trials,1),prts(two_reward_trials),rewsize(two_reward_trials),colors,'o')
    xlabel("Slope of sequence 1 progression")
    ylabel("PRT")
    title(sprintf("Sequence 1 Slope vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    legend("1 uL","2 uL","4 uL")
    
    % Second Sequence-PRT correlation
    [r0,p0] = corrcoef(prog_slopes(two_reward_trials,2),prts(two_reward_trials));
    display(r0)
    [r1,p1] = corrcoef(prog_slopes(sm2rew),prts(sm2rew));
    [r2,p2] = corrcoef(prog_slopes(md2rew),prts(md2rew));
    [r3,p3] = corrcoef(prog_slopes(lg2rew),prts(lg2rew));
    figure(); hold on
    gscatter(prog_slopes(two_reward_trials,2),prts(two_reward_trials),rewsize(two_reward_trials),colors,'o')
    xlabel("Slope of sequence 2 progression")
    ylabel("PRT")
    title(sprintf("Sequence 2 Slope vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    legend("1 uL","2 uL","4 uL")
    
    % Seq1-Seq2 correlation.. might not be appropriate to have mult
    % rewsizes
    [r0,p0] = corrcoef(prog_slopes(two_reward_trials,1),prog_slopes(two_reward_trials,2));
    [r1,p1] = corrcoef(prog_slopes(sm2rew,1),prog_slopes(sm2rew,2));
    [r2,p2] = corrcoef(prog_slopes(md2rew,1),prog_slopes(md2rew,2));
    [r3,p3] = corrcoef(prog_slopes(lg2rew,1),prog_slopes(lg2rew,2));
    figure(); hold on
    gscatter(prog_slopes(two_reward_trials,1),prog_slopes(two_reward_trials,2),rewsize(two_reward_trials),colors,'o')
    xlabel("Slope of sequence 1 progression")
    ylabel("Slope of sequence 2 progression")
    title(sprintf("Sequence 1 Slope vs Sequence 2 Slope (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",p0(2),p1(2),p2(2),p3(2)))
    legend("1 uL","2 uL","4 uL")
    
    figure();hold on
    % Second Sequence-PRT correlation
    [r0,p0] = corrcoef(rewLocs,prog_slopes(two_reward_trials,2));
    gscatter(rewLocs,prog_slopes(two_reward_trials,2),rewsize(two_reward_trials),colors,'o')
    xlabel("Reward 2 location")
    ylabel("Slope of sequence 2 progression")
    title(sprintf("Time of Second Reward Delivery vs Sequence 2 Slope (overall p = %f)",p0(2)))
    legend("1 uL","2 uL","4 uL")
    
end

%% Do cells along the order differentially contribute to PCs
close all

global gP
gP.cmap{1} = [0 0 0];
gP.cmap{3} = cool(3);
gP.cmap{4} = [0 0 0; winter(3)];

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    dat = load(fullfile(paths.data,session));
    patches = dat.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % pull out the on patch firing rate matrix
    fr_mat_on = cat(2,FR_decVar(sIdx).fr_mat{:});
    fr_mat_zscore = zscore(fr_mat_on,[],2);
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
    
    % get indices for PC visualization
    t_lens = cellfun(@size,FR_decVar(sIdx).fr_mat,'uni',false);
    t_lens = reshape(cell2mat(t_lens),[2,size(FR_decVar(sIdx).fr_mat,2)])';
    leave_ix = cumsum(t_lens(:,2));
    stop_ix = leave_ix - t_lens(:,2) + 1;
    
    psth_label = {'stop','leave'};
    t_align = cell(2,1);
    t_start = cell(2,1);
    t_end = cell(2,1);
    % stop-aligned
    t_align{1} = stop_ix;
    t_start{1} = stop_ix;
    t_end{1} = stop_ix + 100; % 2 seconds after
    % leave-aligned
    t_align{2} = leave_ix;
    t_start{2} = leave_ix - 100; % 2 seconds before
    t_end{2} = leave_ix;
    
    t_endmax = patchleave_ms;
    
    % group by rew size:
    grp = cell(4,1);
    grp{1} = rew_size;
    grp{2} = rew_size;
    
    % visualize PCs
    fig_counter = 1;
    for aIdx = 1:2 % currently just look at stop and leave alignments
        fig_counter = fig_counter+1;
        hfig(fig_counter) = figure('Position',[100 100 2300 700]);
        hfig(fig_counter).Name = sprintf('%s - pca on patch - task aligned - %s',session,psth_label{aIdx});
        for pIdx = 1:6 % plot for first 3 PCs
            subplot(2,6,pIdx);
            plot_timecourse('stream',score(:,pIdx),t_align{aIdx},t_start{aIdx},t_end{aIdx},[],'resample_bin',1);
            atitle(sprintf('PC%d/%s/ALL TRIALS/',pIdx,psth_label{aIdx}));
            subplot(2,6,pIdx+6);
            plot_timecourse('stream',score(:,pIdx),t_align{aIdx},t_start{aIdx},t_end{aIdx},grp{aIdx},'resample_bin',1);
            atitle(sprintf('PC%d/%s/SPLIT BY REW SIZE/',pIdx,psth_label{aIdx}));
        end
    end
    
    sorted_coeffs = coeffs(index_sort_odd{sIdx},:);
    figure()
    subplot(3,2,1)
    scatter(1:305,sorted_coeffs(1,:)')
    title("PC1")
    xlabel("PETH Sorted Neurons")
    ylabel("PC1 Loading")
    subplot(3,2,2)
    scatter(1:305,sorted_coeffs(2,:)')
    xlabel("PETH Sorted Neurons")
    title("PC2")
    ylabel("PC2 Loading")
    subplot(3,2,3)
    scatter(1:305,sorted_coeffs(3,:)')
    xlabel("PETH Sorted Neurons")
    title("PC3")
    ylabel("PC3 Loading")
    subplot(3,2,4)
    scatter(1:305,sorted_coeffs(4,:)')
    title("PC4")
    ylabel("PC4 Loading")
    xlabel("PETH Sorted Neurons")
    subplot(3,2,5)
    scatter(1:305,sorted_coeffs(5,:)')
    xlabel("PETH Sorted Neurons")
    title("PC5")
    ylabel("PC5 Loading")
    subplot(3,2,6)
    scatter(1:305,sorted_coeffs(6,:)')
    xlabel("PETH Sorted Neurons")
    title("PC6")
    ylabel("PC6 Loading")
    
    
end
%% old code

%         Sorting novelty- sequences aren't the same even in same trial
%         [~,ix] = max(norm_fr_mat_iTrial(:,1:50),[],2);
%         [~,ix_sort1] = sort(ix);
%         [~,ix] = max(norm_fr_mat_iTrial(:,50:100),[],2);
%         [~,ix_sort2] = sort(ix);
%
%         figure();
%         subplot(1,3,1);colormap('jet')
%         imagesc(flipud(zscore(FR_decVar(sIdx).fr_mat{iTrial}(early_resp,1:100),[],2)))
%         title("Sort by mean")
%         cl = caxis;
%         subplot(1,3,2);colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial(ix_sort1,1:100)))
%         caxis(cl);
%         title("Sort by 0-1000 ms")
%         subplot(1,3,3);colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial(ix_sort2,1:100)))
%         caxis(cl);
%         title("Sort by 1000-2000 ms")



%     catlg1 = cellfun(@(x) x(early_resp,1:75),FR_decVar(sIdx).fr_mat(lg1rew),'un',0);
%     meanlg1 = zscore(mean(cat(3,catlg1{:}),3),[],2);
%     catmd1 = cellfun(@(x) x(early_resp,1:75),FR_decVar(sIdx).fr_mat(md1rew),'un',0);
%     meanmd1 = zscore(mean(cat(3,catmd1{:}),3),[],2);
%     catsm1 = cellfun(@(x) x(early_resp,1:75),FR_decVar(sIdx).fr_mat(sm1rew),'un',0);
%     meansm1 = zscore(mean(cat(3,catsm1{:}),3),[],2);
%
% %     % fit per reward size
%     [times1,neurons1] = find(meansm1(:,1:50) > 0);
%     activity1 = meansm1(meansm1(:,1:50) > 0);
%     mdlsm = fitlm(neurons1,times1,'Intercept',false,'Weights',activity1);
%     [times2,neurons2] = find(meanmd1(:,1:50) > 0);
%     activity2 = meanmd1(meanmd1(:,1:50) > 0);
%     mdlmd = fitlm(neurons2,times2,'Intercept',false,'Weights',activity2);
%     [times3,neurons3] = find(meanlg1(:,1:50) > 0);
%     activity3 = meanlg1(meanlg1(:,1:50) > 0);
%     mdllg = fitlm(neurons3,times3,'Intercept',false,'Weights',activity3);
%
%  % code to visualize the two alternative approaches to sequence quantification:
%     figure();
%     subplot(2,4,1);colormap('jet')
%     imagesc(flipud(meansm1))
%     hold on;
%     plot([50 50]',[0 ; length(index_sort_odd{sIdx})],'k--','linewidth',1.5)
%     title("Mean 1uL activity progression")
%     cl1 = caxis;
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     ylabel("Neurons 25-150 from sorted PETH")
%     subplot(2,4,2);colormap('jet')
%     imagesc(flipud(meanmd1))
%     hold on;
%     plot([50 50]',[0 ; length(index_sort_odd{sIdx})],'k--','linewidth',1.5)
%     title("Mean 2uL activity progression")
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     caxis(cl1)
%     subplot(2,4,3);colormap('jet')
%     imagesc(flipud(meanlg1));
%     hold on;
%     plot([50 50]',[0 ; length(index_sort_odd{sIdx})],'k--','linewidth',1.5)
%     title("Average 4uL activity progression")
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     caxis(cl1)
%     subplot(1,4,4);hold on
%     plot(neurons1,mdlsm.Fitted,'linewidth',2);plot(neurons2,mdlmd.Fitted,'linewidth',2);plot(neurons3,mdllg.Fitted,'linewidth',2)
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     title("Fitted slopes")
%     legend("1 uL","2 uL","4 uL")
%     subplot(2,4,5);colormap('jet');hold on
%     scatter(neurons1,times1,activity1,'kx')
%     plot(neurons1,mdlsm.Fitted,'linewidth',2)
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     ylabel("Neurons 25-150 from sorted PETH")
%     title("Mean 1 uL Significant activity")
%     subplot(2,4,6);colormap('jet');hold on
%     scatter(neurons2,times2,activity2,'kx')
%     plot(neurons2,mdlmd.Fitted,'linewidth',2)
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     title("Mean 2 uL Significant activity")
%     subplot(2,4,7);colormap('jet');hold on
%     scatter(neurons3,times3,activity3,'kx')
%     plot(neurons3,mdllg.Fitted,'linewidth',2)
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     title("Fitted Slopes")
%     xlabel("Time (ms)")
%     title("Mean 4 uL Significant activity")
%
%
%     [~,ix] = max(meansm1,[],2);
%     [~,ix_sort1] = sort(ix);
%     [~,prog1] = max(meansm1(ix_sort1,:),[],1);
%     [~,ix] = max(meanmd1,[],2);
%     [~,ix_sort2] = sort(ix);
%     [~,prog2] = max(meanmd1(ix_sort2,:),[],1);
%     [~,ix] = max(meanlg1,[],2);
%     [~,ix_sort3] = sort(ix);
%     [~,prog3] = max(meanlg1(ix_sort3,:),[],1);

%     figure()
%     subplot(1,4,1)
%     imagesc(flipud(meansm1(ix_sort1,:)))
%     title("Re-sorted Mean 1uL")
%     cl1 = caxis;
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     ylabel("Neurons 25-150 from sorted PETH")
%     colormap('jet')
%     subplot(1,4,2)
%     imagesc(flipud(meanmd1(ix_sort2,:)))
%     caxis(cl1)
%     colormap('jet')
%     title("Re-sorted Mean 2uL")
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     subplot(1,4,3)
%     imagesc(flipud(meanlg1(ix_sort3,:)))
%     caxis(cl1)
%     colormap('jet')
%     title("Re-sorted Mean 4uL")
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")
%     subplot(1,4,4);hold on
%     plot(prog1,'linewidth',2); plot(prog2(2:end),'linewidth',2); plot(prog3,'linewidth',2)
%     legend("1 uL","2 uL","4 uL")
%     title("Sequence progression")
%     xlim([0,75])
%     xticks([0,25,50,75])
%     xticklabels([0,500,1000,1500])
%     xlabel("Time (ms)")