%% Building upon previous work w/ decision variables, now just looking at sequences through actual time 
%  - Mainly comparing time vs time since last rewad 
%  - Maybe use some seqNMF here
clear
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

for sIdx = 3:3 % 1:numel(sessions)
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
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
    
    p.patchstop_ms = patchstop_ms(keep);
    p.patchleave_ms = patchleave_ms(keep);
    
    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
        toc
    end
    
    patchstop_ms = p_out.patchstop_ms + 9; % + 9;
    patchleave_ms = p_out.patchleave_ms + 9; % + 9;
    
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round(patchleave_ms / tbin_ms) + 1,size(fr_mat,2)); % might not be good
    
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

%% Now performing quantifications w/ odd ordering, even visualization

% close all
for sIdx = 3:3 % 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    session = session([1:2,end-2:end]);
    
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
    p = linspace(0,.95,41);
    decVar_bins = quantile(all_decVar,p);
    % for linear axis
    decVar_bins2 = linspace(1,3.5,41);
    
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

%         even_avgFR_decVar = even_avgFR_decVar ./ max(even_avgFR_decVar,[],2);

    avgFR_decVar_sorted = even_avgFR_decVar(index_sort_odd,:);
    
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
    
    regressions = false;
    if regressions == true
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
    end
    
    %     avgFR_decVar_sortedQuad(avgFR_decVar_sortedQuad > 1) = 1;
    
    % now making xticks at even seconds
    max_round = floor(max(decVar_bins));
    secs = 0:max_round;
    x_idx = [];
    for i = secs
        x_idx = [x_idx find(decVar_bins > i,1)];
    end
    
    figure()
    subplot(1,2,1)
    colormap('jet')
    imagesc(flipud(even_avgFR_decVar(index_sort_even,:)))
    colorbar()
    title(sprintf("%s Even Trials sorted by Even",session))
    ylabel("Neuron")
    %     xlim([0,91])
    xlim([0,41])
    xlabel(label)
    xticks(x_idx)
    xticklabels(secs)
    subplot(1,2,2)
    colormap('jet')
    imagesc(flipud(even_avgFR_decVar(index_sort_odd,:)))
    colorbar()
    title(sprintf("%s Even Trials sorted by Odd",session))
    ylabel("Neuron")
   %     xlim([0,91])
    xlim([0,41])
    xlabel(label)
    xticks(x_idx)
    xticklabels(secs)
    
    if regressions == true
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
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        subplot(1,3,3)
        imagesc(flipud(avgFR_decVar_sortedLinearResidual))
        colorbar()
        title("Linear residuals")
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
        %     xlim([0,91])
        xlim([0,41])
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        subplot(1,3,2)
        colormap('jet')
        imagesc(flipud(avgFR_decVar_sortedLinear))
        colorbar()
        title("Linear fit")
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
        %     xlim([0,91])
        xlim([0,41])
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        
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
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        colormap('jet')
        subplot(1,3,3)
        imagesc(flipud(avgFR_decVar_sortedQuadResidual))
        colorbar()
        title("Quadratic residuals")
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
        %     xlim([0,91])
        xlim([0,41])
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        subplot(1,3,2)
        colormap('jet')
        imagesc(flipud(avgFR_decVar_sortedQuad))
        colorbar()
        title("Quadratic fit")
        %     xlabel("Decision Variable Percentile")
        ylabel("Neuron")
        %     xlim([0,91])
        xlim([0,41])
        xlabel(label)
        %     xticks([0 23 45 68 90])
        xticks([0 10 20 30 41])
        %     xticklabels({'5','25','50','75','95'})
        xticklabels(decVar_bins([1,10,20,30,41]))
        
        % find the middle responsive neurons
        middle_neurons = index_sort_odd(304 - (100:275));
        subtr = quad_r2 - linear_r2;
        mid_responsive_neurons = middle_neurons(subtr(304 - (100:275)) > .15);
        
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
        scatter(flipud(means(index_sort_odd)),1:length(means),[],flipud(labels(index_sort_odd,:)))
        ylabel("Neuron")
        xlabel("Mean firing rate")
        ylim([0,305])
        set(gca, 'YDir','reverse')
        title("Firing rate of mid-responsive neurons")
        
%         mid_responsive_neurons = index_sort_odd(50:150); % tough look, but hey
        figure()
        imagesc(even_avgFR_decVar(mid_responsive_neurons,:))
        colorbar()
        colormap('jet')
        xlim([0,41])
        xlabel(label)
        xticks(x_idx)
        xticklabels(secs)
        ylabel("Neuron")
        title("Potential Sequence Participating Neurons")
    end
end

%% Separate by reward size, sort by even/odd
close all;
for sIdx = 3:3
    
    session = sessions{sIdx}(1:end-4);
    % load data
    dat = load(fullfile(paths.data,session));
    % grab rewsize
    rew_size = mod(dat.patches(:,2),10);
    
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    session = session([1:2,end-2:end]);
    
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
    p = linspace(0,.95,41);
    decVar_bins = quantile(all_decVar,p);
    % for linear axis
    decVar_bins2 = linspace(1,3.5,41);
    
    for iRewsize = [2,4]
        
        rewsize_trials = find(rew_size == iRewsize);
        odd_rewsize_trials = rewsize_trials(mod(rewsize_trials,2) == 1);
        even_rewsize_trials = rewsize_trials(mod(rewsize_trials,2) == 0);
        
        shifts = randi(size(FR_decVar(sIdx).fr_mat{1},2),size(FR_decVar(sIdx).fr_mat{1},1),1);
        shuffle = false;
        
        %%%% First look at odd trials to get indices %%%%
        odd_frCell = FR_decVar(sIdx).fr_mat(odd_rewsize_trials);
        odd_fr_mat = cat(2,odd_frCell{:});
        
        if shuffle == true
            parfor neuron = 1:size(fr_mat,1)
                odd_fr_mat(neuron,:) = circshift(odd_fr_mat(neuron,:),shifts(neuron));
            end
        end
        
        odd_decVarCell = decVar_cell(odd_rewsize_trials);
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
        even_frCell = FR_decVar(sIdx).fr_mat(even_rewsize_trials);
        even_fr_mat = cat(2,even_frCell{:});
        if shuffle == true
            parfor neuron = 1:size(fr_mat,1)
                even_fr_mat(neuron,:) = circshift(even_fr_mat(neuron,:),shifts(neuron));
            end
        end
        even_decVarCell = decVar_cell(even_rewsize_trials);
        even_decVar = cat(2,even_decVarCell{:});
        
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
        
        % now making xticks at even seconds
        max_round = floor(max(decVar_bins));
        secs = 0:max_round;
        x_idx = [];
        for i = secs
            x_idx = [x_idx find(decVar_bins > i,1)];
        end
        
        figure()
        subplot(1,2,1)
        colormap('jet')
        imagesc(flipud(even_avgFR_decVar(index_sort_even,:)))
        if iRewsize == 2
            cl2 = caxis;
        end
        colorbar()
        caxis(cl2)
        title(sprintf("%s Even %i uL Trials sorted by Even",session,iRewsize))
        ylabel("Neuron")
        xlim([0,41])
        xlabel("Time since last reward")
        %         xticks([0 10 20 30 41])
        %         xticklabels(decVar_bins([1,10,20,30,41]))
        xticks(x_idx)
        xticklabels(secs)
        subplot(1,2,2)
        colormap('jet')
        imagesc(flipud(even_avgFR_decVar(index_sort_odd,:)))
        colorbar()
        caxis(cl2)
        title(sprintf("%s Even %i uL Trials sorted by Odd",session,iRewsize))
        ylabel("Neuron")
        xlim([0,41])
        xlabel("Time since last reward")
        xticks(x_idx)
        xticklabels(secs)
    end
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
    
    nMid = length(mid_responsive_neurons);
    ridge_width = 10;
    
    rrbl_shuffled = nan(nMid,length(decVar_cells));
    rrbl_unshuffled = nan(nMid,length(decVar_cells));
    
    for vIdx = 1:length(decVar_cells) 
        decVar_cell = decVar_cells{vIdx};
        %%%% prep decision variable bins w/ all trials %%%%
        all_decVar = cat(2,decVar_cell{:});
        p = linspace(0.05,.95,50);
        decVar_bins = quantile(all_decVar,p);
        nBins = length(decVar_bins);
        
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
        fr_mat = fr_mat(index_sort_odd(mid_responsive_neurons),:);
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
        nMid = size(avgFR_decVar_sorted,1);
        ridgeBaselineRatioUnshuffled = zeros(nMid,1);
        [~,max_unshuffled_ix] = max(avgFR_decVar_sorted,[],2);
        
        backgroundUnshuffled = zeros(nMid,1);
        backgroundShuffled = zeros(nMid,1);

        for neuron = 1:nMid
            backgroundUnshuffled(neuron) =  mean(avgFR_decVar_sorted(neuron,[1:max(1,max_unshuffled_ix(neuron)-ridge_width),min(nBins-1,(max_unshuffled_ix(neuron)+ridge_width)):nBins-1]));
            ridgeUnshuffled(neuron) = mean(avgFR_decVar_sorted(neuron,max(1,max_unshuffled_ix(neuron)-ridge_width):min(nBins-1,max_unshuffled_ix(neuron)+ridge_width)));
            ridgeBaselineRatioUnshuffled(neuron) = ridgeUnshuffled(neuron) / backgroundUnshuffled(neuron);
        end
        
        rrbl_unshuffled(:,vIdx) = ridgeBaselineRatioUnshuffled;
    end
    
    rrbl_unshuffled(rrbl_unshuffled > 15) = 15;
    
    % now perform shuffle control 1000x to test for significance per cell
    shuffRepeats = 100;
    
    decVar_cell = decVar_cells{1}; % arbitrarily use 1 here for now- not sure how to handle this
    %%%% prep decision variable bins w/ all trials %%%%
    all_decVar = cat(2,decVar_cell{:});
    p = linspace(0.05,.95,40);
    decVar_bins = quantile(all_decVar,p);
    nBins = length(decVar_bins);
    
    % collect FR matrices
    fr_mat = cat(2,FR_decVar(sIdx).fr_mat{:});
    fr_mat = fr_mat(index_sort_odd(mid_responsive_neurons),:);
    newShuffleControl = false;
    
    bad_neurons = [];
    
    if newShuffleControl == true
        rrbl_shuffled = zeros(nMid,shuffRepeats);
        for shuffIdx = 1:shuffRepeats
            
            if mod(shuffIdx,100) == 0
                display(shuffIdx)
            end
            
            %%% shuffle data according to random rotation %%%
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
            
            % Now find ridge-to-background ratio for shuffled, unshuffled data
            ridgeBaselineRatioShuffled = zeros(nMid,1);
            [~,max_shuffled_ix] = max(avgFR_shuffle_decVar_sorted,[],2);
            
            backgroundShuffled = zeros(nMid,1);
            ridgeShuffled = zeros(nMid,1);
            
            for neuron = 1:nMid
                backgroundShuffled(neuron) =  mean(avgFR_shuffle_decVar_sorted(neuron,[1:max(1,max_shuffled_ix(neuron)-ridge_width), min(nBins-1,max_shuffled_ix(neuron)+ridge_width):nBins-1]));
                ridgeShuffled(neuron) = mean(avgFR_shuffle_decVar_sorted(neuron,max(1,max_shuffled_ix(neuron)-ridge_width):min(nBins-1,max_shuffled_ix(neuron)+ridge_width)));
                ridgeBaselineRatioShuffled(neuron) = ridgeShuffled(neuron) / backgroundShuffled(neuron);
                if ridgeBaselineRatioShuffled(neuron) > 15
                    bad_neurons = [bad_neurons neuron]; % outlier neurons... low fr cells?
                end
            end
            
            rrbl_shuffled(:,shuffIdx) = ridgeBaselineRatioShuffled;
        end
        
        rrbl_shuffled(isinf(rrbl_shuffled)) = 1;
        rrbl_shuffled(rrbl_shuffled > 15) = 15; % figure out how to deal w/ outliers better... 
    end
    
    figure()
    bar(1:(length(decVar_cells)+1),[mean(mean(rrbl_shuffled,'omitnan')) mean(rrbl_unshuffled,1)])
    hold on
    errorbar(1,mean(mean(rrbl_shuffled,'omitnan')),1.96 * std(mean(rrbl_shuffled,1,'omitnan')),'k')
    title("Ridge to background ratio across decision variables")
    xticks(1:(length(decVar_cells)+1))
    xticklabels(labels)
    
    % return p value
    mean_rrbl_shuffled = mean(rrbl_shuffled,1);
    mean_rrbl_unshuffled = mean(rrbl_unshuffled,1);
    for vIdx = 1:length(mean_rrbl_unshuffled) 
        p = length(find(mean_rrbl_unshuffled < mean_rrbl_shuffled(vIdx))) / length(mean_rrbl_shuffled);
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
    ramp_idx = index_sort_odd(150:end);
%     ramp_idx = index_sort_odd;
    seq_idx = index_sort_odd(1:150);
    
    new_regs = false;
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
        for iTrial = 1:nTrials
            %%% first investigate reward responses %%%
            % subselect ramping portion of FR mat
            fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(ramp_idx,:),[],2);
            
            % to visualize what's going on
%             figure();colormap('jet')
%             imagesc(flipud(fr_mat_iTrial))
%             title(sprintf('Trial %i',iTrial))
%             xlabel('Time')
%             ylabel('All neurons, ordered by avg odd sort to timeSinceRew')
%             
%             % draw lines to indicate start, end of reward response interval
%             hold on
%             plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w-','linewidth',1.5)
            rew_ix_cell{iTrial}(rew_ix_cell{iTrial} + search_end > size(fr_mat_iTrial,2)) = [];
%             plot([rew_ix_cell{iTrial} + search_begin rew_ix_cell{iTrial} + search_begin]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w--','linewidth',1.5)
%             plot([rew_ix_cell{iTrial} + search_end rew_ix_cell{iTrial} + search_end]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(ramp_idx)],'w--','linewidth',1.5)
%             
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
    
    nMid = length(mid_responsive_neurons);
    
%     maxIx = nan(size(FR_decVar(sIdx).fr_mat{1},1),4);
    maxIx = nan(nMid,4);
    
    titles = {"Q1 PRT Trial PETH","Q2 PRT Trial PETH","Q3 PRT Trial PETH","Q4 PRT Trial PETH"};
    for prtIdx = 1:(length(prt_bins)-1)
        prt_trials = find(prts > prt_bins(prtIdx) & prts < prt_bins(prtIdx + 1));
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{prt_trials});
        fr_mat = fr_mat(mid_responsive_neurons,:);
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
    maxIx = nan(length(mid_responsive_neurons),4);
    
    titles = {"1 uL Trials","2 uL Trials","4 uL Trials"};
    rewsizes = [1,2,4];
    
    for j = 1:3
        iRewsize = rewsizes(j);
        rewsizeTrials = find(rewsize == iRewsize);
        % collect FR matrices
        fr_mat = cat(2,FR_decVar(sIdx).fr_mat{rewsizeTrials});
        fr_mat = fr_mat(mid_responsive_neurons,:);
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
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors to make barcode matrix
    rew_ms = data.rew_ts;
    prts = patchCSL(:,3) - patchCSL(:,2);
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
    
    early_resp = index_sort_odd(1:150);
    
    one_rew_trials = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1);
    two_rew_trials = find(rew_barcode(:,1) > 1 & rew_barcode(:,2) > 1 & rew_barcode(:,3) == -1);
    
    prog_slopes = nan(nTrials,1);
    
    for j = 1:numel(one_rew_trials)
        iTrial = one_rew_trials(j);
        norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(early_resp,:),[],2);
        [times,neurons] = find(norm_fr_mat_iTrial(:,1:50) > 2); % significant activations
        % visualize what we're up to
%         figure();
%         subplot(1,2,1)
%         colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial))
%         colorbar()
%         title(sprintf('Trial %i',iTrial))
%         xlabel('Time')
%         title("mean odd sort")
%         % draw lines to indicate reward delivery
%         hold on
%         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * length(index_sort_odd)],'w-','linewidth',1.5)
%         subplot(1,2,2); hold on
%         scatter(neurons,times,1.5,'kx')

        % linear regression on pattern of significant activations
        x = 1:max(times);
        mdl = fitlm(neurons,times,'Intercept',false);
%         plot(neurons,mdl.Fitted)
        
        prog_slopes(iTrial) = mdl.Coefficients.Estimate;
    end
    
    colors = [0 0 0; 0 1 1;0 0 1;0 0 1];
    
    labels = nan(nTrials,3);
    labels(rewsize == 1,:) = repmat(colors(1,:),[length(find(rewsize == 1)),1]);
    labels(rewsize == 2,:) = repmat(colors(2,:),[length(find(rewsize == 2)),1]);
    labels(rewsize == 4,:) = repmat(colors(4,:),[length(find(rewsize == 4)),1]);
    
    figure();hold on
    gscatter(prts(one_rew_trials),prog_slopes(one_rew_trials),rewsize(one_rew_trials),colors,'o')
    ylabel("Slope of activation progression")
    xlabel("PRT")
    title("PRT vs fitted slope of first second activation progression")
    legend("1 uL","2 uL","4 uL")
    mdl = fitlm(prts(one_rew_trials),prog_slopes(one_rew_trials));
    

end