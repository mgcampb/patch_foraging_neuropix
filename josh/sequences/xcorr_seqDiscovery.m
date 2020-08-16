%% The goal of this script is to determine whether there exist subsequences within population activity 
%  - End result will be an NxN heatmap where the elements represent mean
%    xcorr between neurons 
%  - If there are blocks along the diagonal, that means we have found
%    consistent subsequences 
%  Method: forall pairs of neurons: (or just 100 mid resp)
%           forall trials: (or just 400 trials) 
%             - take cross correlogram for latency up to 1 sec
%             - pull out peak xcorr, add to heatmap 
%             - also pull out the latency of peak xcorr  
% time within trial: use 2 seconds of R00 or 1 second of all trials

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

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};
%% Extract FR matrices and timing information

FR_decVar = struct;
FRandTimes = struct;

for sIdx = 1:3 % 1:numel(sessions)
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

    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
        [fr_mat, tbincent] = calcFRVsTime(good_cells,dat,opt); % calc from full matrix
        toc
    end

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
    
    FRandTimes(sIdx).fr_mat = fr_mat;
    FRandTimes(sIdx).stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes(sIdx).stop_leave_ix = [patchstop_ix patchleave_ix];
end 

%% Sort by all trials to get ordering

index_sort_all = {sIdx};
for sIdx = 1:3
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "time";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Practice and visualize process w/ single trials
% start w/ first 1 second of activity for all trials 
% then might want to check out first 2 seconds of R00X trials 
close all
for sIdx = 3:3 
    nTrials = length(FR_decVar(sIdx).fr_mat);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);  
    maxlag = 1000 / tbin_ms; % look maximally 1 second away
    for iTrial = 1:5
        % sort to make this stuff more meaningful
        sorted_fr_mat = FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:);
        r = xcorr(sorted_fr_mat',maxlag,'normalized'); % normalized xcorr 
        r = reshape(r,2*maxlag+1,nNeurons,nNeurons);
        [max_corr,ix_max] = max(r,[],1); % max along latency dimension
        figure();colormap('jet')
        subplot(1,3,1) 
        imagesc(squeeze(r(round(maxlag/2),:,:))) % no lag correlation 
        title("0 Lag Correlation")
        colorbar()
        subplot(1,3,2) 
        imagesc(squeeze(max_corr))
        title("Max Correlation")
        colorbar()
        subplot(1,3,3) 
        imagesc(squeeze(ix_max)) 
        title("Index of Max Correlation")
        colorbar()
    end
end

%% Now iterate over all neurons and trials to make tensors for analysis
for sIdx = 3:3
    % get sorted matrix so this stuff is a bit more meaningful 
    clc
    nTrials = length(FR_decVar(sIdx).fr_mat);
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1); 
    maxlag = round(500 / tbin_ms); % look maximally 1 second away 
    max_time = 1000 / tbin_ms; % so rewards don't mess things up
    
    % make datastructure
    corr_tensor = nan(2*maxlag+1,nNeurons,nNeurons,nTrials); 
    
    % iterate over trials
    for iTrial = 1:nTrials % figure out broadcast variable to use parfor 
        t_len = size(FR_decVar(sIdx).fr_mat{iTrial},2);
        sorted_fr_mat = zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},1:min(max_time,t_len)),[],2);
        r = xcorr(sorted_fr_mat',maxlag,'coeff'); % normalized xcorr 
        r(isnan(r)) = 0; % account for xcorr betw rows of 0s
        r = reshape(r,2*maxlag+1,nNeurons,nNeurons);
        corr_tensor(:,:,:,iTrial) = r;
        
        if mod(iTrial,10) == 0 
            fprintf("Trial %i Complete \n",iTrial)
        end
    end 
end 

%% visualize results   
close all
mean_corr_tensor = mean(corr_tensor,4); % avg over trials 
[max_mean_corr,max_mean_latency] = max(squeeze(mean_corr_tensor),[],1); % max over latency
mean_lagless = mean_corr_tensor(maxlag,:,:);  

% get rid of extra dimension
max_mean_corr = squeeze(max_mean_corr); 
max_mean_latency = squeeze(max_mean_latency); 
mean_lagless = squeeze(mean_lagless);

% max_mean_corr = max_mean_corr - diag(diag(max_mean_corr));
% mean_lagless = mean_lagless - diag(diag(mean_lagless));

close all
figure();colormap('jet')
imagesc(max_mean_corr(60:160,60:160))
title("Mean max correlation")  
colorbar()
figure() ;colormap('jet')
imagesc(mean_lagless(60:160,60:160))
title("Mean lagless correlation") 
colorbar()
figure() ;colormap('jet')
imagesc(max_mean_latency(60:160,60:160))
title("Mean latency to max correlation")
colorbar()


%% dumb tests 
% how do we make an upper triangular matrix so we don't redo computation 
n = 10000;
test_hmap = zeros(n,n);  

% tic
% parfor i = 1:n  
%     for j = 1:n 
%         if j > i 
%             test_hmap(i,j) = 0;
%         else
%             test_hmap(i,j) = i * 2 + j; % just some dummy computation 
%         end
%     end 
% end
% toc  
    
tic
for i = 1:n
    for j = 1:n 
        if j > i 
            test_hmap(i,j) = 0;
        else
            test_hmap(i,j) = i * 2 + j; % just some dummy computation 
        end
    end 
end
toc 

% Winner!
tic
for i = 1:n
    for j = 1:n 
        if j > i 
            break
        else
            test_hmap(i,j) = i * 2 + j; % just some dummy computation 
        end
    end 
end
toc

figure() 
imagesc(test_hmap)

