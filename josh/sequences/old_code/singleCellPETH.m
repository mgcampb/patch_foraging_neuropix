%% Make single cell figures ala Plitt Preprint

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData')); 
load('midresp_struct.mat')

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

mid_resp_mPFC_sessions = [2:4 6:12 14:18 22:24]; 
for i = 1:numel(mid_resp_mPFC_sessions) % 1:numel(sessions) 
    sIdx = mid_resp_mPFC_sessions(i);
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
    
    FRandTimes(sIdx).fr_mat = fr_mat;
    FRandTimes(sIdx).stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes(sIdx).stop_leave_ix = [patchstop_ix patchleave_ix];
end

%% Sort by all trials to get ordering
index_sort_all = {sIdx};
mid_resp_mPFC_sessions = [2:4 6:12 14:18 22:24]; 
for i = 1:numel(mid_resp_mPFC_sessions) % 1:numel(sessions)  
    sIdx = mid_resp_mPFC_sessions(i);
    decVar_bins = linspace(0,2,50);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = true;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order;
end

%% Now pull out single neurons and plot activity over trials
% maybe also cosine similarity or correlation coefficient
close all
for sIdx = 24:24
    nTrials = length(FR_decVar(sIdx).fr_mat);
    
    session = sessions{sIdx}(1:end-4);
    % load data
    dat = load(fullfile(paths.data,session));
    patchCSL = dat.patchCSL;  
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    prts = patchCSL(:,3) - patchCSL(:,2); 
    floor_prts = floor(prts);
    patches = dat.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    rew_ms = dat.rew_ts;
    
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
    
    midresp_neurons = midresp_struct(sIdx).mid_resp_ix; 
    
    for iNeuron = 61:70
%         neuron = index_sort_all{sIdx}(midresp_neurons(sigs(iNeuron))); 
        neuron = index_sort_all{sIdx}(midresp_neurons(iNeuron));
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
        
        active_rew_2_sort = rew_2_sort(active_trials); 
        active_prt_2_sort = prt_2_sort(active_trials); 
        trials20xPETH = cellPETH(1:length(find(active_rew_2_sort < 2)),:);
        [max_resp,ix] = max(trials20xPETH,[],2);
%         active_prts = prts(active_trials);
%         scatter(ix(active_prts < 10),active_prts(active_prts < 10),'.')  
        active_rewsizes = mod(active_rew_2_sort(1:numel(ix)) * 10,10); 
        subplot(2,3,4)   
        gscatter(ix,active_prt_2_sort(1:numel(ix)),active_rewsizes,colors2rew(1:2,:),'.')
        xticks((start:50:stop))
        xticklabels((start-1:50:stop-1) * tbin_ms)
%         [r,p] = corrcoef(ix(active_prts < 10),active_prts(active_prts < 10)); 
        [~,p_ix] = corrcoef(ix,active_prt_2_sort(1:numel(ix)));
        title(sprintf("PeakRespTime vs PRT (p = %f )",p_ix(2)))
        xlabel("PeakRespTime")
        ylabel("PRT")
        subplot(2,3,5)   
        gscatter(max_resp,active_prt_2_sort(1:numel(max_resp)),active_rewsizes,colors2rew(1:2,:),'.')
        xticks((start:50:stop))
        xticklabels((start-1:50:stop-1) * tbin_ms)
        [~,p_max] = corrcoef(max_resp,active_prt_2_sort(1:numel(max_resp)));
        title(sprintf("PeakRespMag vs PRT (p = %f )",p_max(2)))
        xlabel("PeakRespMag")
        ylabel("PRT") 
        subplot(2,3,6)
        gscatter(mean(trials20xPETH,2),active_prt_2_sort(1:numel(max_resp)),active_rewsizes,colors2rew(1:2,:),'.')
        xticks((start:50:stop))
        xticklabels((start-1:50:stop-1) * tbin_ms)
        [~,p_max] = corrcoef(mean(trials20xPETH,2),active_prt_2_sort(1:numel(max_resp)));
        title(sprintf("MeanResp vs PRT (p = %f )",p_max(2)))
        xlabel("MeanResp")
        ylabel("PRT") 
        
        hFig=findall(0,'type','figure');
        hLeg=findobj(hFig(1,1),'type','legend');
        set(hLeg,'visible','off')
    end
end

%% Visualize cell by cell results across population
close all 
prop_corr = nan(numel(mid_resp_mPFC_sessions),8); 

mid_resp_mPFC_sessions = [2:4 6:12 14:18 22:24]; 
for i = 1:numel(mid_resp_mPFC_sessions) % 1:numel(sessions)  
    sIdx = mid_resp_mPFC_sessions(i);
    % reload session data 
    session = sessions{sIdx}(1:end-4); 
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    dat = load(fullfile(paths.data,session));
    patchCSL = dat.patchCSL; 
    prts = patchCSL(:,3) - patchCSL(:,2); 
    floor_prts = floor(prts);
    patches = dat.patches; 
    patchType = patches(:,2);
    rewsize = mod(patchType,10);  
    rew_ms = dat.rew_ts;
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3); 
    nTrials = length(prts);
    
    midresp_neurons = midresp_struct(sIdx).mid_resp_ix;  
    nNeurons = numel(midresp_neurons);
    med_dists = nan(nNeurons,1); 
    prtIx_corr = nan(nNeurons,1); 
    prtIx_p = nan(nNeurons,1);  
    prtMaxresp_corr = nan(nNeurons,1);
    prtMaxresp_p = nan(nNeurons,1); 
    prtMeanresp_corr = nan(nNeurons,1); 
    prtMeanresp_p = nan(nNeurons,1);
    start = (0 / tbin_ms) + 1; % beginning of trial
    stop = (2000 / tbin_ms) + 1; 
    
    % make barcode matrix
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:nTrials
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_sec_cell{iTrial} = rew_indices(rew_indices > 1);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end  
    % 2 and 4 uL trials with no rewards after t = 0
    trials10x = find(rew_barcode(:,1) > 1 & rew_barcode(:,2) < 0 & prts > 2.55); 
    prts10x = prts(trials10x);
    
    for iNeuron = 1:numel(midresp_neurons)
        neuron = index_sort_all{sIdx}(midresp_neurons(iNeuron)); 
        cellPETH = zeros(numel(trials10x),(stop-start+1)); % might want option to not include all trials?
        
        for j = 1:numel(trials10x) %  numel(t0_prt_size_sort)
            iTrial = trials10x(j);
            if size(FR_decVar(sIdx).fr_mat{iTrial},2) >= (stop - start + 1)
                cellPETH(j,:) = FR_decVar(sIdx).fr_mat{iTrial}(neuron,start:stop);
            end 
        end 
        active_trials = find(any(cellPETH,2)); 
%         active_trials = 1:numel(trials10x);
        [max_resp,ix] = max(cellPETH(active_trials,:),[],2); % only look at active trials for max finding
        
        if numel(ix) > 3
            [r_ix,p_ix] = corrcoef(ix,prts10x(active_trials));  
            prtIx_corr(iNeuron) = r_ix(2); 
            if p_ix(2) < .001
                prtIx_p(iNeuron) = 3;  
            elseif p_ix(2) < .01 
                prtIx_p(iNeuron) = 2; 
            elseif p_ix(2) < .05 
                prtIx_p(iNeuron) = 1; 
            else
                prtIx_p(iNeuron) = 0; 
            end
            
            [r_max,p_max] = corrcoef(max_resp,prts10x(active_trials));  
            prtMaxresp_corr(iNeuron) = r_max(2); 
            if p_max(2) < .001
                prtMaxresp_p(iNeuron) = 3;  
            elseif p_max(2) < .01 
                prtMaxresp_p(iNeuron) = 2; 
            elseif p_max(2) < .05 
                prtMaxresp_p(iNeuron) = 1; 
            else
                prtMaxresp_p(iNeuron) = 0; 
            end
            
            [r_mean,p_mean] = corrcoef(mean(cellPETH,2),prts10x); 
            prtMeanresp_corr(iNeuron) = r_mean(2); 
            if p_mean(2) < .001
                prtMeanresp_p(iNeuron) = 3;  
            elseif p_mean(2) < .01 
                prtMeanresp_p(iNeuron) = 2; 
            elseif p_mean(2) < .05 
                prtMeanresp_p(iNeuron) = 1; 
            else
                prtMeanresp_p(iNeuron) = 0; 
            end
        end   
    end 
    
    visualize = false;
    if visualize == true
        figure()  
        subplot(1,3,1)
        gscatter(1:numel(prtIx_corr),prtIx_corr,prtIx_p,[.4 .4 .4;1 .4 .4; 1 .2 .2; 1 0 0],'.')  
        title("PeakLoc - R0 PRT corr") 
        xlabel("Mid-responsive Neurons")
        ylabel("Pearson Correlation") 
        legend("Non-significant","p < .05","p < .01","p < .001")
        subplot(1,3,2)
        gscatter(1:numel(prtMaxresp_corr),prtMaxresp_corr,prtMaxresp_p,[.4 .4 .4;1 .4 .4; 1 .2 .2; 1 0 0],'.')  
        title("PeakMag - R0 PRT corr") 
        xlabel("Mid-responsive Neurons")
        ylabel("Pearson Correlation") 
        legend("Non-significant","p < .05","p < .01","p < .001")
        subplot(1,3,3)
        gscatter(1:numel(prtIx_corr),prtMeanresp_corr,prtMeanresp_p,[.4 .4 .4;1 .4 .4; 1 .2 .2; 1 0 0],'.')  
        title("MeanFR - R0 PRT corr") 
        xlabel("Mid-responsive Neurons")
        ylabel("Pearson Correlation") 
        legend("Non-significant","p < .05","p < .01","p < .001") 
        suptitle(sprintf("%s Mid-responsive Behavior Correlations",session_title))  
    end
    
    % now assign proportion to pie chart for x-session visualization 
    proportions = nan(8,1); 
    proportions(1) = length(find(prtIx_p == 0 & prtMaxresp_p == 0 & prtMeanresp_p == 0)); % no corrs
    proportions(2) = length(find(prtIx_p > 0 & prtMaxresp_p == 0 & prtMeanresp_p == 0)); % just ix 
    proportions(3) = length(find(prtIx_p == 0 & prtMaxresp_p > 0 & prtMeanresp_p == 0)); % just maxresp 
    proportions(4) = length(find(prtIx_p == 0 & prtMaxresp_p == 0 & prtMeanresp_p > 0)); % just meanresp 
    proportions(5) = length(find(prtIx_p > 0 & prtMaxresp_p > 0 & prtMeanresp_p == 0)); % maxresp and prtIx 
    proportions(6) = length(find(prtIx_p > 0 & prtMaxresp_p == 0 & prtMeanresp_p > 0)); % meanresp and prtIx 
    proportions(7) = length(find(prtIx_p == 0 & prtMaxresp_p > 0 & prtMeanresp_p > 0)); % meanresp and maxresp 
    proportions(8) = length(find((prtIx_p > 0 & prtMaxresp_p > 0 & prtMeanresp_p > 0))); % all
    prop_corr(i,:) = proportions;
end

%% Visualize x session results with pie chart 
close all
t = tiledlayout(4,5);  
labels = {'No PRT Corr','Sig PeakLoc-PRT Corr','Sig MaxResp-PRT Corr','Sig Meanresp-PRT Corr',...
            'Sig PeakLoc+MaxResp Corr','Sig PeakLoc+MeanResp Corr','Sig MaxResp+MeanResp Corr',...
            'Sig PeakLoc+MaxResp+MeanResp Corr'};
        
mid_resp_mPFC_sessions = [2:4 6:12 14:18 22:24];  
colors = [0 0 0; 1 0 0; 0 .5 .8; .5 0 .8; ... 
        .5 .3 .5; .5 .3 1; 0 0 1; 1 0 1]; 
ax = nexttile ;
ax = nexttile;
for i = 1:3 % numel(mid_resp_mPFC_sessions) % 1:numel(sessions)  
    sIdx = mid_resp_mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4); 
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    ax = nexttile; 
    sPie = pie(ax,prop_corr(i,:),[0,1,1,1,1,1,1,1]);  
    
    for k = 1:size(prop_corr,2)
      % Create a color for this sector of the pie
      pieColorMap = colors(k,:);  % Color for this segment.
      % Apply the colors we just generated to the pie chart.
      set(sPie(k*2-1), 'FaceColor', pieColorMap);
    end
    
    iTitle = title(session_title);
    set(iTitle, 'Position', [-1.6, -.9], ...
      'VerticalAlignment', 'bottom', ...
      'HorizontalAlignment', 'center')  
    percentages = prop_corr(i,:) / sum(prop_corr(i,:)) * 100;
    lowIx = find(percentages < 2); 
    TextChildren = findobj(ax,'Type','text');
    for j = 1:numel(TextChildren)
        if strcmp(TextChildren(j).String,'0%') == true || strcmp(TextChildren(j).String,'< 1%') == true ... 
                || strcmp(TextChildren(j).String,'1%') == true || strcmp(TextChildren(j).String,'2%') || strcmp(TextChildren(j).String,'3%')
            set(TextChildren(j),'visible','off')
        end
    end  
    if i == 1
        legend(labels) 
    end
end 
suptitle("Mid responsive neuron behavioral correlations across days")

%% pie chart war Room 

%%sample:
figure(1)
%3 pie charts to maximize area they can be placed would be in a triangle
%position first subplot top middle

%generate some dummy data with 90 entries
    x=randi(100,1,90);
    pieH1=pie(x);
%since there are so many entries it may be hard to visualize especially
%with the text possibly overlapping
%adjust slice and text position (here we've offset every other entry)
%examine the pieH1 data and you'll see that pie slice data is odd indexes
%text is even entries.  
    for ind = 2:4:numel(pieH1)
       pieH1(ind).Position = pieH1(ind).Position*1.2; 
       pieH1(ind-1).Vertices = pieH1(ind-1).Vertices*1.1; 
    end


