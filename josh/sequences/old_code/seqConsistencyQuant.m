%% Quantify the distribution of peak firing rate time to assess consistency
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

for sIdx = 1:3
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

    buffer = 0; % buffer before leave in ms
    
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

%% Get order from avg PETH
close all
index_sort_all = {numel(sessions)};
for sIdx = 1:3 
    decVar_bins = linspace(0,2,41); 
    nTrials = length(FR_decVar(sIdx).fr_mat); 
    opt.norm = "zscore";
    opt.trials = 'all';
    dvar = "time";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Sort by odd, visualize by even 
close all 
for sIdx = 3:3 
    decVar_bins = linspace(0,2,41); 
    nTrials = length(FR_decVar(sIdx).fr_mat); 
    session = sessions{sIdx}(1:end-4);
    session_name = ['m' session(1:2) ' ' session(end-2:end)];
    
    dvar = "timesince";
    if dvar == "time"
        label = "Time on patch";
    else
        label = "Time since last rew";
    end
    
    odd_opt.norm = "zscore";
    odd_opt.trials = 1:2:nTrials; 
    odd_opt.suppressVis = true;
    [~,odd_sort,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,odd_opt);
    even_opt.norm = "zscore";
    even_opt.trials = 2:2:nTrials; 
    even_opt.suppressVis = true;
    [~,~,even_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,even_opt); 
    figure();colormap('jet')
    imagesc(flipud(even_peth(odd_sort,:))) 
    max_round = floor(max(decVar_bins));
    secs = 0:max_round;
    
    x_idx = [];
    for i = secs
        x_idx = [x_idx find(decVar_bins > i,1)];
    end
    xlim([1,40])
    xlabel([label; " (ms)"])
    title(sprintf("%s Even Trials Sorted by Odd %s",session_name,label))
    xticks(x_idx)
    xticklabels(secs * 1000)
    ylabel("Neurons")
end

%% Iterate over single trials, find max for all rewards
% close all
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
    nNeurons = length(index_sort_all{sIdx});
    
    % parameters for response finding
    search_begin = round(250 / tbin_ms);
    search_end = round(750 / tbin_ms); % deprecated
    threshold = 0;

    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_ix_cell{iTrial} = (rew_indices(rew_indices > 1) - 1) * 1000 / tbin_ms;
        rew_ix_cell{iTrial}(rew_ix_cell{iTrial} > prts(iTrial) * 1000 / tbin_ms - search_begin) = [];
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    nRews = nTrials + sum(cellfun(@length,rew_ix_cell)); % add nTrials to account for t = 0
    after_rew_ix = nan(nNeurons,nRews);
    total_rew_counter = 1;
    for iTrial = 1:nTrials
        norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:),[],2);
        extrema_ix = nan(nNeurons,length(rew_ix_cell{iTrial}));
        t_len = size(norm_fr_mat_iTrial,2);
        rew_ix_iTrial = [1; rew_ix_cell{iTrial}; t_len];
        for j = 1:(length(rew_ix_iTrial)-1)
            rIdx = rew_ix_iTrial(j);
            next_rIdx = rew_ix_iTrial(j + 1);
%             [~,i_extrema_ix] = max(abs(norm_fr_mat_iTrial(:,(rIdx + search_begin):min(t_len,(rIdx + search_end))) - norm_fr_mat_iTrial(:,rIdx)),[],2);
            [~,i_extrema_ix] = max(norm_fr_mat_iTrial(:,(rIdx + search_begin):next_rIdx) - norm_fr_mat_iTrial(:,rIdx),[],2); % look for max
            extrema_ix(:,j) = min(t_len,i_extrema_ix + rIdx + search_begin);
            trivial_ix = min(t_len,rIdx + search_begin + 1);
            activations = diag(norm_fr_mat_iTrial(:,extrema_ix(:,j)));
            extrema_ix(activations < threshold,j) = nan(1,length(find(activations < threshold))); % threshold by mag of activation
            extrema_ix(extrema_ix(:,j) == trivial_ix,j) = nan(length(extrema_ix(extrema_ix(:,j) == trivial_ix,j)),1);
            extrema_ix(extrema_ix(:,j) == (next_rIdx+1),j) = nan(length(extrema_ix(extrema_ix(:,j) == (next_rIdx+1),j)),1); % get rid of artifact
            after_rew_ix(:,total_rew_counter) = extrema_ix(:,j) - rIdx;
            total_rew_counter = total_rew_counter + 1;
        end
        
        % visualize process
%         figure();colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial));
%         hold on
%         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * nNeurons],'w--','linewidth',2)
%         extrema_ix = flipud(extrema_ix);
%         neurons = repmat((1:nNeurons)',[1 size(flipud(extrema_ix),2)]);
%         scatter(extrema_ix(:),neurons(:),2,'k*') % just a technicality
%         xticks(0:1000/tbin_ms:t_len)
%         xticklabels((0:1000/tbin_ms:t_len) * tbin_ms)
%         xlabel("Time (msec)")
%         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%         caxis([-3 3])
%         colorbar()
    end
end 

%% Now visualize distributions of receptivity for all rewards
bins = (search_begin+1):2:(3500 / tbin_ms);
% build probability density matrix
activity_density = nan(size(after_rew_ix,1),length(bins));
for neuron = 1:nNeurons
    activity_density(neuron,:) = hist(after_rew_ix(neuron,:),bins) / sum(hist(after_rew_ix(neuron,:),bins));
end
%     % line plot
%     figure();hold on
%     set(0,'DefaultAxesColorOrder',cool(10))
%     for neuron = 1:30:300
%         plot(activity_density(neuron,:),'linewidth',1.5)
%     end
%     xlabel("Time (msec)")
%     xticks(bins)
%     xticklabels(bins * 1000 / tbin_ms)
% heatmap
figure()
imagesc(flipud(activity_density(:,1:(end-1))))
xlabel("Time after reward reception (msec)")
xticks(1:10:numel(bins))
xticklabels(bins(1:10:end) * tbin_ms)
colorbar() 
session_name = ['m' session(1:2) ' ' session(end-2:end)];
title(sprintf("%s Distn of post-rew activation latency",session_name))
colormap('hot')

%% Iterate over single trials, find max ix for just t = 0:1000 ms
% close all
for sIdx = 1:1
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
    nNeurons = length(index_sort_all{sIdx});
    
    % parameters for response finding
    search_begin = round(0 / tbin_ms) + 1;
    search_end = round(1000 / tbin_ms); % deprecated
    threshold = 0; % no threshold!
    
    after_rew_ix = nan(nNeurons,nTrials);
    total_rew_counter = 1;
    for iTrial = 1:nTrials
        norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:),[],2);
        
        [~,i_extrema_ix] = max(norm_fr_mat_iTrial(:,search_begin:search_end) - norm_fr_mat_iTrial(:,search_begin),[],2); % look for max
        extrema_ix = min(t_len,i_extrema_ix + search_begin);
        %         activations = diag(norm_fr_mat_iTrial(:,extrema_ix(:,j)));
        %         extrema_ix(activations < threshold,j) = nan(1,length(find(activations < threshold))); % threshold by mag of activation
        extrema_ix(extrema_ix == (search_begin+1)) = nan(length(extrema_ix(extrema_ix == (search_begin+1))),1);
        extrema_ix(extrema_ix == (search_end+1)) = nan(length(extrema_ix(extrema_ix == (search_end+1))),1); % get rid of artifact
        after_rew_ix(:,iTrial) = extrema_ix;
        
        % visualize process
%         figure();colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial));
%         hold on
% %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * nNeurons],'w--','linewidth',2)
%         extrema_ix = flipud(extrema_ix);
%         neurons = repmat((1:nNeurons)',[1 size(flipud(extrema_ix),2)]);
%         scatter(extrema_ix(:),neurons(:),2,'k*') % just a technicality
%         xticks(0:1000/tbin_ms:t_len)
%         xticklabels((0:1000/tbin_ms:t_len) * tbin_ms)
%         xlabel("Time (msec)")
%         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%         caxis([-3 3])
%         colorbar()
    end
end 

%% Now visualize distributions of receptivity for just first rew
bins = (search_begin+1):2:(1000 / tbin_ms);
% build probability density matrix
activity_density = nan(size(after_rew_ix,1),length(bins));
for neuron = 1:nNeurons
    activity_density(neuron,:) = hist(after_rew_ix(neuron,:),bins) / sum(hist(after_rew_ix(neuron,:),bins));
end
%     % line plot
%     figure();hold on
%     set(0,'DefaultAxesColorOrder',cool(10))
%     for neuron = 1:30:300
%         plot(activity_density(neuron,:),'linewidth',1.5)
%     end
%     xlabel("Time (msec)")
%     xticks(bins)
%     xticklabels(bins * 1000 / tbin_ms)
% heatmap
figure()
imagesc(flipud(activity_density(:,1:(end-1))))
xlabel("Time after reward reception (msec)")
xticks([1,6,12,18,24])
xticklabels([0,250,500,750,1000])
colorbar()
session_name = ['m' session(1:2) ' ' session(end-2:end)];
title(sprintf("%s Distn of post-rew activation latency",session_name))
colormap('hot') 

%% Iterate over single trials, find max ix for just t = 0:2000 ms on R0 trials
% close all
for sIdx = 1:3
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
    nNeurons = length(index_sort_all{sIdx}); 

    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end 
    
    R0_trials = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == 0 & prts > 2.55);
    
    % parameters for response finding
    search_begin = round(0 / tbin_ms) + 1;
    search_end = round(2000 / tbin_ms); % deprecated
    threshold = 0; % no threshold!
    
    after_rew_ix = nan(nNeurons,numel(R0_trials));
    total_rew_counter = 1;
    for j = 1:numel(R0_trials)
        iTrial = R0_trials(j);
        norm_fr_mat_iTrial = zscore(FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:),[],2);
        
        [~,i_extrema_ix] = max(norm_fr_mat_iTrial(:,search_begin:search_end) - norm_fr_mat_iTrial(:,search_begin),[],2); % look for max
        extrema_ix = i_extrema_ix + search_begin; % min(t_len,i_extrema_ix + search_begin);
        %         activations = diag(norm_fr_mat_iTrial(:,extrema_ix(:,j)));
        %         extrema_ix(activations < threshold,j) = nan(1,length(find(activations < threshold))); % threshold by mag of activation
        extrema_ix(extrema_ix == (search_begin+1)) = nan(length(extrema_ix(extrema_ix == (search_begin+1))),1);
        extrema_ix(extrema_ix == (search_end+1)) = nan(length(extrema_ix(extrema_ix == (search_end+1))),1); % get rid of artifact
        after_rew_ix(:,j) = extrema_ix;
        
%         % visualize process
%         figure();colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial));
%         hold on
% %         plot([rew_ix_cell{iTrial} rew_ix_cell{iTrial}]',[ones(1,length(rew_ix_cell{iTrial})) ; ones(1,length(rew_ix_cell{iTrial})) * nNeurons],'w--','linewidth',2)
%         extrema_ix = flipud(extrema_ix);
%         neurons = repmat((1:nNeurons)',[1 size(flipud(extrema_ix),2)]);
%         scatter(extrema_ix(:),neurons(:),2,'k*') % just a technicality
%         xticks(0:1000/tbin_ms:t_len)
%         xticklabels((0:1000/tbin_ms:t_len) * tbin_ms)
%         xlabel("Time (msec)")
%         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%         caxis([-3 3])
%         colorbar()
    end
    
    % visualize
    bins = (search_begin+1):2:(2000 / tbin_ms);
    % build probability density matrix
    activity_density = nan(size(after_rew_ix,1),length(bins));
    for neuron = 1:nNeurons
        activity_density(neuron,:) = hist(after_rew_ix(neuron,:),bins) / sum(hist(after_rew_ix(neuron,:),bins));
    end
    %     % line plot
    %     figure();hold on
    %     set(0,'DefaultAxesColorOrder',cool(10))
    %     for neuron = 1:30:300
    %         plot(activity_density(neuron,:),'linewidth',1.5)
    %     end
    %     xlabel("Time (msec)")
    %     xticks(bins)
    %     xticklabels(bins * 1000 / tbin_ms)
    % heatmap
    figure()
    imagesc(flipud(activity_density(:,1:(end-1))))
    xlabel("Time after reward reception (msec)")
    xticks([1,3,6,9,12,15,18,21,24] * 2)
    xticklabels([0,250,500,750,1000,1250,1500,1750,2000])
    colorbar()
    session_name = ['m' session(1:2) ' ' session(end-2:end)];
    title(sprintf("%s Distribution of first reward response latency across R0 Trials",session_name))
    colormap('hot')
end

