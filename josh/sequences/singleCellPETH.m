%% Make single cell figures ala Plitt Preprint

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
for sIdx = 3:3
    decVar_bins = linspace(0,2,50);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "time";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Now pull out single neurons and plot activity over trials
% maybe also cosine similarity or correlation coefficient
close all
for sIdx = 3:3 
    nTrials = length(FR_decVar(sIdx).fr_mat);
    
    session = sessions{sIdx}(1:end-4);
    % load data
    dat = load(fullfile(paths.data,session));
    patchCSL = dat.patchCSL; 
    prts = patchCSL(:,3) - patchCSL(:,2);
    patches = dat.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10); 
    
    colors3rew = [.5 1 1 ; .75 .75 1 ; 1 .5 1 ; 0 1 1 ;.5 .5 1;1 0 1];
    colors2rew = [.75 .75 1 ; 1 .5 1 ;.5 .5 1;1 0 1];
    
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
    
    % make rew2 vector that just says whether there was a reward at t = 1
    % in ones place, size of reward in 10ths place
    trials10x = find(rew_barcode(:,1) > 1 & rew_barcode(:,2) < 0 & prts > 2.55);
    trials11x = find(rew_barcode(:,1) > 1 & rew_barcode(:,2) > 1 & prts > 2.55); 
    rew2 = nan(nTrials,1); % second reward?
    rew2(trials10x) = 1; 
    rew2(trials11x) = 2; 
    rew2 = rew2 + .1 * rewsize;

        % make some sorts for the array
    [~,prt_sort] = sort(prts);
    [~,rewsize_sort] = sort(rewsize);

% sort by PRT within 10,11,20,22,40,44, order [10,20,40,11,22,44]
    prt_R0_sort = []; 
    prt_RR_sort = []; 
    for iRewsize = [2,4] 
        trialsr0x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & prts > 2.55);
        trialsrrx = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55); 

        % sort by PRTs within reward history condition
        [~,prtsr0_sort] = sort(prts(trialsr0x));
        prtsr0_sort = trialsr0x(prtsr0_sort);
        [~,prtsrr_sort] = sort(prts(trialsrrx)); 
        prtsrr_sort = trialsrrx(prtsrr_sort);
        prt_rew2_sort = [prtsr0_sort ; prtsrr_sort]; 
        
        prt_R0_sort = [prt_R0_sort;prtsr0_sort];
        prt_RR_sort = [prt_RR_sort;prtsrr_sort];
    end 
    
    t0_prt_size_sort = [prt_R0_sort ; prt_RR_sort];
    rew_2_sort = rew2(t0_prt_size_sort); 
    prt_2_sort = prts(t0_prt_size_sort); 
    
    max_round = floor(max(decVar_bins));
    secs = 0:max_round;
    x_idx = [];
    for i = secs
        x_idx = [x_idx find(decVar_bins > i,1)];
    end
    
    for iNeuron = 75:85
        neuron = index_sort_all{sIdx}(iNeuron);
        start = (0 / tbin_ms) + 1; % beginning of trial
        stop = (2000 / tbin_ms) + 1;
        cellPETH = zeros(nTrials,(stop-start+1)); % might want option to not include all trials?
        
        for j = 1:numel(t0_prt_size_sort) % numel(prt_rew2_sort) %  numel(trials10x)
%             iTrial = prt_rew2_sort(j); % prt_rew2_sort(j);
            iTrial = t0_prt_size_sort(j);
%             cellPETH(j,:) = zscore(FR_decVar(sIdx).fr_mat{iTrial}(neuron,start:stop),[],2);
            cellPETH(j,:) = FR_decVar(sIdx).fr_mat{iTrial}(neuron,start:stop);
        end
        
        active_trials = find(any(cellPETH,2));
        cellPETH = cellPETH(active_trials,:);
        figure()
        subplot(2,1,1)
        imagesc(cellPETH); colormap('jet')
        title(sprintf("Cell %i",iNeuron))
        hold on; 
%         gscatter(zeros(length(active_trials),1),1:size(cellPETH,1),rew_2_sort(active_trials))
        gscatter(zeros(length(active_trials),1),1:size(cellPETH,1),rew_2_sort(active_trials),colors2rew)
        xlabel("Time (msec)")
        ylabel("PRT-sorted Trials")
        xticks((start:50:stop))
        xticklabels((start-1:50:stop-1) * tbin_ms)
        xlim([start-1,stop]) 
        
        subplot(2,1,2)   
        active_rew_2_sort = rew_2_sort(active_trials); 
        active_prt_2_sort = prt_2_sort(active_trials); 
        trials20xPETH = cellPETH(1:length(find(active_rew_2_sort < 2)),:);
        [~,ix] = max(trials20xPETH,[],2);
%         active_prts = prts(active_trials);
%         scatter(ix(active_prts < 10),active_prts(active_prts < 10),'.')  
        active_rewsizes = mod(active_rew_2_sort(1:numel(ix)) * 10,10);
        gscatter(ix,active_prt_2_sort(1:numel(ix)),active_rewsizes,colors2rew(1:2,:),'.')
        xticks((start:50:stop))
        xticklabels((start-1:50:stop-1) * tbin_ms)
%         [r,p] = corrcoef(ix(active_prts < 10),active_prts(active_prts < 10)); 
        [~,p] = corrcoef(ix,active_prt_2_sort(1:numel(ix)));
        title(sprintf("R0 Peak Response Time vs PRT (p = %f )",p(2)))
        xlabel("Maximal response time")
        ylabel("PRT-sorted Trials")
        
%         subplot(2,1,2)   
%         active_rew_2_sort = rew_2_sort(active_trials); 
%         active_prt_2_sort = prt_2_sort(active_trials); 
%         trials20xPETH = cellPETH(1:length(find(active_rew_2_sort < 2)),:);
%         [~,ix] = max(trials20xPETH,[],2);
% %         active_prts = prts(active_trials);
% %         scatter(ix(active_prts < 10),active_prts(active_prts < 10),'.')  
%         active_rewsizes = mod(active_rew_2_sort(1:numel(ix)) * 10,10);
%         gscatter(ix,1:numel(ix),active_rewsizes,colors2rew(1:2,:),'.')
%         xticks((start:50:stop))
%         xticklabels((start-1:50:stop-1) * tbin_ms)
% %         [r,p] = corrcoef(ix(active_prts < 10),active_prts(active_prts < 10)); 
%         [~,p] = corrcoef(ix,1:numel(ix));
%         title(sprintf("R0 Peak Response Time vs PRT (p = %f )",p(2)))
%         xlabel("Maximal response time")
%         ylabel("PRT")        
        
%         subplot(2,2,2) 
%         D = squareform(pdist(cellPETH,'cosine')); 
%         imagesc(D) 
%         title(sprintf("Cell %i Cosine distance between trial activity",iNeuron)) 
%         colorbar() 
%         xlabel("PRT-sorted Trials")
%         ylabel("PRT-sorted Trials")
%         subplot(2,2,3)
%         plot(unsorted_peth(neuron,:),'linewidth',2)
%         title(sprintf("Cell %i Mean Response",iNeuron))
%         xlabel("Time (msec)")
%         xticks(x_idx)
%         xticklabels(secs * 1000)  

%         % To get reward sizes in separate panes
%         subplot(2,2,3)   
%         active_rew_2_sort = rew_2_sort(active_trials);
%         trials20xPETH = cellPETH(1:length(find(active_rew_2_sort == 1.2)),:);
%         [~,ix] = max(trials20xPETH,[],2);
% %         active_prts = prts(active_trials);
% %         scatter(ix(active_prts < 10),active_prts(active_prts < 10),'.')  
%         active_rewsizes = mod(active_rew_2_sort(1:numel(ix)) * 10,10);
%         gscatter(ix,1:numel(ix),active_rewsizes,colors2rew(1,:),'.')
%         xticks((start:50:stop))
%         xticklabels((start-1:50:stop-1) * tbin_ms)
% %         [r,p] = corrcoef(ix(active_prts < 10),active_prts(active_prts < 10)); 
%         [~,p] = corrcoef(ix,1:numel(ix)); 
%         title(sprintf("20 Peak response time ordered by PRT (p = %f )",p(2)))
%         xlabel("Maximal response time")
%         ylabel("PRT-sorted Trials")
%         subplot(2,2,4)   
%         active_rew_2_sort = rew_2_sort(active_trials);
%         trials40xPETH = cellPETH(1:length(find(active_rew_2_sort == 1.4)),:);
%         [~,ix] = max(trials40xPETH,[],2);
% %         active_prts = prts(active_trials);
% %         scatter(ix(active_prts < 10),active_prts(active_prts < 10),'.')  
%         active_rewsizes = mod(active_rew_2_sort(1:numel(ix)) * 10,10);
%         gscatter(ix,1:numel(ix),active_rewsizes,colors2rew(2,:),'.')
%         xticks((start:50:stop))
%         xticklabels((start-1:50:stop-1) * tbin_ms)
% %         [r,p] = corrcoef(ix(active_prts < 10),active_prts(active_prts < 10)); 
%         [~,p] = corrcoef(ix,1:numel(ix)); 
%         title(sprintf("40 Peak response time ordered by PRT (p = %f )",p(2)))
%         xlabel("Maximal response time")
%         ylabel("PRT-sorted Trials")
        
        
        hFig=findall(0,'type','figure');
        hLeg=findobj(hFig(1,1),'type','legend');
        set(hLeg,'visible','off')
    end
end

%% Quantify cosine distance across trials, across neurons to investigate differences in consistency
close all
for sIdx = 3:3 
    nNeurons = numel(index_sort_all{sIdx});
    med_dists = nan(nNeurons,1); 
    prtRewsize_corr = nan(100,1); 
    prtRewsize_p = zeros(100,1);
    
    for iNeuron = 40:140
        neuron = index_sort_all{sIdx}(iNeuron);
        start = (0 / tbin_ms) + 1; % beginning of trial
        stop = (2000 / tbin_ms) + 1;
        cellPETH = zeros(nTrials,(stop-start+1)); % might want option to not include all trials?
        
        for j = 1:numel(trials10x) %  numel(t0_prt_size_sort)
            iTrial = t0_prt_size_sort(j); % prt_rew2_sort(j);
            if size(FR_decVar(sIdx).fr_mat{iTrial},2) >= (stop - start + 1)
%                 cellPETH(j,:) = zscore(FR_decVar(sIdx).fr_mat{iTrial}(neuron,start:stop),[],2);
                cellPETH(j,:) = FR_decVar(sIdx).fr_mat{iTrial}(neuron,start:stop);
            end 
        end 
        active_trials = find(any(cellPETH,2));
        cellPETH = cellPETH(active_trials,:);
        D = pdist(cellPETH,'cosine');  
        med_dists(iNeuron) = median(D(:)); 
        
        trials10xPETH = cellPETH(1:length(find(rew_2_sort(active_trials) < 2)),:); 
        [~,ix] = max(trials10xPETH,[],2);  
        
        active_prt_2_sort = prt_2_sort(active_trials); 
        if numel(ix) > 2
            [r,p] = corrcoef(ix,active_prt_2_sort(1:numel(ix)));  
            prtRewsize_corr(iNeuron) = r(2); 
            prtRewsize_p(iNeuron) = p(2) < .05;
        end
    end 
    
    figure()
    scatter(1:numel(med_dists),med_dists) 
    title("Median cosine distance ordered by peak responsivity") 
    xlabel("Peak-response sorted neurons")
    ylabel("Median cosine distance between trials") 
    figure() 
    gscatter(1:numel(prtRewsize_corr),prtRewsize_corr,prtRewsize_p,[.4 .4 .4;1 0 0],'.')  
    xlim([40,140])
    title("Pearson Correlation between peak firing location and PRT ordering within Rewsize") 
    xlabel("Peak-response sorted neurons")
    ylabel("Pearson Correlation") 
    legend("Non-significant","Significant")
end

