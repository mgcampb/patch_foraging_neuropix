%% Assess ramping diversity
% 1. make time averaged PETH 
% 2. pull off ramping neurons 
% 3. average over RXX conditions
% 4. sort by slope of ramp/dip after reward
% 5. profit

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information 
FR_decVar = struct; 
FRandTimes = struct;
for sIdx = 24:24
    buffer = 500;
    [FR_decVar_tmp,FRandTimes_tmp] = genSeqStructs(paths,sessions,opt,sIdx,buffer);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
    FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
    FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix; 
end

%% Sort by all trials to get ordering

index_sort_all = {sIdx};
for sIdx = 24:24
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Now average over RX conditions 
RX_data = {};
for sIdx = 24:24
    RX_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms;
    times = -1000:tbin_ms:1000;
    
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
  
    % get that money
    rew_counter = 1;
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        temp_fr_mat = {length(trials10x)};
        for j = 1:numel(trials10x)
            iTrial = trials10x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        RX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2);
        
        temp_fr_mat = {length(trials11x)};
        for j = 1:numel(trials11x)
            iTrial = trials11x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:(stop_ix + sec2ix));
        end
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        RX_data{sIdx}(rew_counter+3).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
        rew_counter = rew_counter + 1;
    end
end

%% Now visualize RX PETHs, sort by peak responsivity
close all
conditions = {"10","20","40","11","22","44"};
for sIdx = 24:24
    figure();colormap('jet') 
    spcounter = 1;
    for cIdx = [1 4 2 5 3 6]
        subplot(3,2,spcounter)
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("Z-Scored Mean %s Trial Activity",conditions{cIdx}))
        xlabel("Time (msec)")
        xticks([0 50 100])
        xticklabels([0 1000 2000]) 
        spcounter = spcounter + 1;
    end
    
    figure();colormap('jet')
    index_sorts = {3};
    for cIdx = 1:6
        subplot(2,3,cIdx)
        % re-sort if on an omission trial
        sortIdx = cIdx;
        if sortIdx > 3
            sortIdx = cIdx - 3;
        else
            [~,index] = max(RX_data{sIdx}(cIdx).fr_mat,[],2);
            [~,index_sort] = sort(index);
            index_sorts{sortIdx} = index_sort;
        end
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat(index_sorts{sortIdx},:)))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Sort by %s",conditions{cIdx},conditions{sortIdx}))
        xlabel("Time (msec)")
        xticks([0 50 100])
        xticklabels([0 1000 2000])
    end
end

%% Now visualize RX ramp PETHs w/ various sorts
close all
conditions = {"10","20","40","11","22","44"};
for sIdx = 1:1
    ramp_idx = 150:300;
    % iterate over ramp-like neurons and take slope of ramp linreg
    slopes = nan(numel(ramp_idx),1);
    intercepts = nan(numel(ramp_idx),1);
    for j = 1:numel(ramp_idx)
        neuron = ramp_idx(j);
        mdl = fitlm(1:size(avgFR_decVar(neuron,:),2),avgFR_decVar(neuron,:));
        intercepts(j) = mdl.Coefficients.Estimate(1);
        slopes(j) = mdl.Coefficients.Estimate(2);
    end
    
    [~,slope_sort] = sort(slopes);
    slope_sort = ramp_idx(slope_sort);
    [~,int_sort] = sort(intercepts);
    int_sort = ramp_idx(int_sort);
    
    figure();colormap('jet')
    for cIdx = 1:6
        subplot(2,3,cIdx)
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat(ramp_idx,:)))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Ramps Sort by Session",conditions{cIdx}))
    end
    
    figure();colormap('jet')
    for cIdx = 1:6
        subplot(2,3,cIdx)
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat(slope_sort,:)))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Ramps Sort by Slope",conditions{cIdx}))
    end
    
    figure();colormap('jet')
    for cIdx = 1:6
        subplot(2,3,cIdx)
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat(int_sort,:)))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Ramps Sort by Intercept",conditions{cIdx}))
    end
end

%% Now same over RXX conditions 

RXX_data = {};
for sIdx = 1:24
    RXX_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;

    sec3ix = 3000/tbin_ms;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(prts);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
  
    rew_counter = 1;
    
    dvar = "timesince";
    decVar_bins = linspace(0,2,41);
    opt = struct;
    opt.suppressVis = true;
    for iRewsize = [2,4] 
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.5);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.5);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.5);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.5);
        
        if ~isempty(trials100x) > 0
            temp_fr_mat = {length(trials100x)};
            for j = 1:numel(trials100x)
                iTrial = trials100x(j);
                stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
                temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
            end
            
            mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
            % exclude trials of this exact type
            opt.trials = setdiff(1:nTrials,trials100x);
            [~,exclude100order,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
            RXX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2);
        else
            RXX_data{sIdx}(rew_counter).fr_mat = zeros(length(index_sort_all{sIdx}),sec3ix);
        end
        %         RXX_data{sIdx}(rew_counter).fr_mat = zscore(mean_condition_fr(exclude100order,:),[],2);
        
        if ~isempty(trials110x)
            temp_fr_mat = {length(trials110x)};
            for j = 1:numel(trials110x)
                iTrial = trials110x(j);
                stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
                temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
            end
            
            mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
            % exclude trials of this exact type
            opt.trials = setdiff(1:nTrials,trials110x);
            [~,exclude110order,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
            RXX_data{sIdx}(rew_counter+1).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
        else
            RXX_data{sIdx}(rew_counter+1).fr_mat = zeros(length(index_sort_all{sIdx}),sec3ix);
        end
        %         RXX_data{sIdx}(rew_counter+1).fr_mat = zscore(mean_condition_fr(exclude110order,:),[],2); % 3 reward sizes
        
        if ~isempty(trials101x)
            temp_fr_mat = {length(trials101x)};
            for j = 1:numel(trials101x)
                iTrial = trials101x(j);
                stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
                temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
            end
            
            mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
            % exclude trials of this exact type
            opt.trials = setdiff(1:nTrials,trials101x);
            [~,exclude101order,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
            RXX_data{sIdx}(rew_counter+2).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
            %         RXX_data{sIdx}(rew_counter+2).fr_mat = zscore(mean_condition_fr(exclude101order,:),[],2); % 3 reward sizes
        else
            RXX_data{sIdx}(rew_counter+2).fr_mat = zeros(length(index_sort_all{sIdx}),sec3ix);
        end
        
        if ~isempty(trials111x)
        temp_fr_mat = {length(trials111x)};
        for j = 1:numel(trials111x)
            iTrial = trials111x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        % exclude trials of this exact type
        opt.trials = setdiff(1:nTrials,trials111x);
        [~,exclude111order,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
        RXX_data{sIdx}(rew_counter+3).fr_mat = zscore(mean_condition_fr(index_sort_all{sIdx},:),[],2); % 3 reward sizes
%         RXX_data{sIdx}(rew_counter+3).fr_mat =
%         zscore(mean_condition_fr(exclude111order,:),[],2); % 3 reward
%         sizes  
        else 
            RXX_data{sIdx}(rew_counter+3).fr_mat = zeros(length(index_sort_all{sIdx}),sec3ix);
        end
        
        rew_counter = rew_counter + 4;
    end
end

%% Now visualize RXX PETHs, sort by peak responsivity
close all
conditions = {"200","220","202","222","400","440","404","444"};
sorts = {"10","20","40"};
for sIdx = 11:24
    session = sessions{sIdx}; 
    session_title = session([1:2 end-2:end]);
    figure();colormap('jet')
    for cIdx = 1:8
        subplot(2,4,cIdx)
        imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat))
        if cIdx == 1
            cl1 = caxis;
        end
        caxis(cl1)
%         title(sprintf("%s Sort Excluding %s",conditions{cIdx},conditions{cIdx}))
        title(sprintf("%s %s Sort by Avg PETH",session_title,conditions{cIdx}))
        xticks([0 50 100 150])
        xticklabels([0 1 2 3]) 
        yticks([0,100,200,300,400])
        xlabel("Time on Patch (sec)")
    end
    
%     figure();colormap('jet')
%     for cIdx = 1:8
%         subplot(2,4,cIdx)
%         if cIdx < 4
%             sort_idx = 2;
%         else
%             sort_idx = 3;
%         end
%         imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat(index_sorts{sort_idx},:)))
%         if cIdx == 1
%             cl1 = caxis;
%         end
%         caxis(cl1)
%         title(sprintf("%s Sort by %s",conditions{cIdx},sorts{sort_idx}))
%         xticks([0 50 100 150])
%         xticklabels([0 1 2 3])
%         yticks([0,100,200,300])
%         xlabel("Time on Patch (sec)")
%     end

end

%% Now visualize RXX ramp PETHs, sort by ramp characteristics

close all
conditions = {"200","220","202","222","400","440","404","444"};
sorts = {"10","20","40"};

for sIdx = 3:3
    ramp_idx = 200:300;
    
    % now sort by dip after reward in one-reward trials
    % reward response search window
    search_begin = round(250 / tbin_ms);
    search_end = round(750 / tbin_ms);
    % first find the location of the reward response
    rew_ix = [0 50 100 0 0 50 100 0];
    rew_resps = nan(numel(ramp_idx),8);
    one_rew_conditions = [2 3 6 7];
    for cIdx = one_rew_conditions
        cond_fr_mat = RXX_data{sIdx}(cIdx).fr_mat(ramp_idx,:);
        [~,extrema_ix] = max(abs(cond_fr_mat(:,(rew_ix(cIdx) + search_begin):(rew_ix(cIdx) + search_end)) - cond_fr_mat(:,rew_ix(cIdx))),[],2);
        extrema_ix = extrema_ix + rew_ix(cIdx) + search_begin;
        rew_resps(:,cIdx) = diag(cond_fr_mat(:,extrema_ix)) - cond_fr_mat(:,rew_ix(cIdx));
        
        %         figure();colormap('jet')
        %         subplot(2,1,1)
        %         imagesc(flipud(cond_fr_mat));hold on
        %         scatter(flipud(extrema_ix),1:numel(ramp_idx),1.5,'k*')
        %         xlim([0 size(cond_fr_mat,2)])
        %         ylim([0 size(cond_fr_mat,1)])
        %         title(sprintf("%s Rew Responses",conditions{cIdx}))
        %         subplot(2,1,2)
        %         scatter(1:numel(ramp_idx),rew_resps(:,cIdx))
%         xticks([0 1000 2000 3000] / tbin_ms)
%         xticklabels([0 1000 2000 3000])
    end
    
    % All neurons, sort by peak responsivity
    figure();colormap('jet')
    for j = 1:numel(one_rew_conditions)
        cIdx = one_rew_conditions(j);
        [~,rew_sort] = sort(rew_resps(:,cIdx));
        subplot(2,2,j)
        imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat))
        if j == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Sort by Session",conditions{cIdx}))
        xticks([0 1000 2000 3000] / tbin_ms)
        xticklabels([0 1000 2000 3000])
        xlabel("Time (ms)")
    end
    
    % Ramp neurons, sort by own reward response
    figure();colormap('jet')
    for j = 1:numel(one_rew_conditions)
        cIdx = one_rew_conditions(j);
        [~,rew_sort] = sort(rew_resps(:,cIdx));
        rew_sort = ramp_idx(rew_sort);
        subplot(2,2,j)
        imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat(rew_sort,:)))
        if j == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Sort by Reward Response",conditions{cIdx}))
        xticks([0 1000 2000 3000] / tbin_ms)
        xticklabels([0 1000 2000 3000])
        xlabel("Time (ms)")
    end
    
    % Ramp neurons, sort by R0 reward response
    figure();colormap('jet')
    for j = 1:numel(one_rew_conditions)
        cIdx = one_rew_conditions(j);
        if mod(cIdx,2) == 0
            sort_ix = cIdx + 1;
        else
            sort_ix = cIdx;
        end
        [~,rew_sort] = sort(rew_resps(:,sort_ix));
        rew_sort = ramp_idx(rew_sort);
        subplot(2,2,j)
        imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat(rew_sort,:)))
        if j == 1
            cl1 = caxis;
        end
        caxis(cl1)
        title(sprintf("%s Sort by %s Reward Response",conditions{cIdx},conditions{sort_ix}))
        xticks([0 1000 2000 3000] / tbin_ms)
        xticklabels([0 1000 2000 3000])
        xlabel("Time (ms)")
    end
end
