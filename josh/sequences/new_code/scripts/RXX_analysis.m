%% Assess ramping diversity
% 1. make time averaged PETH 
% 2. pull off ramping neurons 
% 3. average over RXX conditions
% 4. sort by slope of ramp/dip after reward
% 5. profit

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
opt.preLeave_buffer = 500; 
opt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information 
FR_decVar = struct; 
for sIdx = 23:25
    FR_decVar_tmp = genSeqStructs(paths,sessions,opt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;
end  

%% Generate "reward barcodes" to average firing rates  
rew_barcodes = cell(numel(sessions),1);
for sIdx = 23:25
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session)); 
    
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
    rewsize = mod(patches(:,2),10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end 
    rew_barcodes{sIdx} = rew_barcode;
end
%% Get mid-responsive neurons from driscoll selection  
midresp = cell(numel(sessions),1); 
decVar_bins = linspace(0,2,41);
for sIdx = 23:25  
    nTrials = length(FR_decVar(sIdx).fr_mat);
    transient_opt = struct; 
    transient_opt.visualization = false; 
    transient_opt.nShuffles = 500;
    transient_opt.preRew_buffer = round(opt.smoothSigma_time * 3 * 1000 / tbin_ms); 
    transient_opt.postStop_buffer = NaN; % allow first reward 
    trial_selection = 1:nTrials;

    transients_struct_tmp = driscoll_transient_discovery(FR_decVar(sIdx),trial_selection,decVar_bins,tbin_ms,transient_opt); 
    midresp{sIdx} = transients_struct_tmp.midresp;
end

%% Sort by all trials to get ordering
peak_sort_all = cell(numel(sessions),1);  
peak_sort_midresp = cell(numel(sessions),1);
for sIdx = 23:25 
    nNeurons = length(FR_decVar(sIdx).cell_depths);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = true; 
    opt.neurons = 'all';
    dvar = "timesince"; 
    [~,neuron_order,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    peak_sort_all{sIdx} = neuron_order;  
    
    opt.neurons = midresp{sIdx};
    [~,neuron_order_midresp,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    peak_sort_midresp{sIdx} = neuron_order_midresp; 
    
    %     figure()
    %     binscatter(1:numel(neuron_order),FR_decVar(sIdx).cell_depths(neuron_order),10)
    %     hold on
    %     yline(mean(FR_decVar(sIdx).cell_depths),'k--','linewidth',2) 
end 

%% Now average over RX conditions 
RX_data = cell(numel(sessions),1);
for sIdx = 23:25 
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);  
    nTrials = length(prts);
    
    RX_data{sIdx} = struct;
    rew_barcode = rew_barcodes{sIdx};  
    sec2ix = 2000 / tbin_ms;
  
    % Average over reward histories
    rew_counter = 1;
    opt = struct;
    opt.dvar = "timesince";
    opt.decVar_bins = linspace(0,2,41);
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);  
        
        RX_data{sIdx}(rew_counter).fr_mat = avgPETH(FR_decVar(sIdx),trials10x,setdiff(1:nTrials,trials10x),sec2ix,opt);
        RX_data{sIdx}(rew_counter+3).fr_mat = avgPETH(FR_decVar(sIdx),trials11x,setdiff(1:nTrials,trials11x),sec2ix,opt);

        rew_counter = rew_counter + 1;
    end 
end

%% Now visualize RX PETHs, sort by peak responsivity
close all
conditions = {"10","20","40","11","22","44"};
for sIdx = 25
    session = sessions{sIdx}; 
    session_title = session([1:2 end-6:end-4]); 
    nNeurons = length(peak_sort_all{sIdx});
    figure(); % colormap('jet') 
    spcounter = 1;
    for cIdx = 1:6
        subplot(2,3,spcounter)
        imagesc(flipud(RX_data{sIdx}(cIdx).fr_mat))
        title(sprintf("%s %s (All)",session_title,conditions{cIdx}))
        xlabel("Time (msec)")
        xticks([0 50 100])
        xticklabels([0 1000 2000]) 
        spcounter = spcounter + 1; 
        caxis([-3,3]) 
        colorbar()
    end 
end

%% Now same over RXX conditions 

RXX_data = cell(numel(sessions),1);
for sIdx = 23:25
    RXX_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    prts = data.patchCSL(:,3) - data.patchCSL(:,2);  
    nTrials = length(prts);
    
    % reinitialize ms vectors
    rew_barcode = rew_barcodes{sIdx};  
    sec3ix = 3000 / tbin_ms;
  
    rew_counter = 1;

    opt = struct;
    opt.dvar = "timesince";
    opt.decVar_bins = linspace(0,2,41);
    for iRewsize = [2,4] 
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.5);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.5);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.5);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.5);
        
        RXX_data{sIdx}(rew_counter).fr_mat = avgPETH(FR_decVar(sIdx),trials100x,setdiff(1:nTrials,trials100x),sec3ix,opt);
        RXX_data{sIdx}(rew_counter+1).fr_mat = avgPETH(FR_decVar(sIdx),trials110x,setdiff(1:nTrials,trials110x),sec3ix,opt);
        RXX_data{sIdx}(rew_counter+2).fr_mat = avgPETH(FR_decVar(sIdx),trials101x,setdiff(1:nTrials,trials101x),sec3ix,opt);
        RXX_data{sIdx}(rew_counter+3).fr_mat = avgPETH(FR_decVar(sIdx),trials111x,setdiff(1:nTrials,trials111x),sec3ix,opt);
        
        rew_counter = rew_counter + 4;
    end
end

%% Now visualize RXX PETHs, sort by peak responsivity
close all
conditions = {"200","220","202","222","400","440","404","444"};
for sIdx = 23
    session = sessions{sIdx}; 
    session_title = session([1:2 end-6:end-4]);
    nNeurons = size(RXX_data{sIdx}(1).fr_mat,2);  
    
    figure();
    for cIdx = 1:8
        subplot(2,4,cIdx);colormap('parula')
        imagesc(flipud(RXX_data{sIdx}(cIdx).fr_mat));colormap('parula')
        %         title(sprintf("%s %s Sort Excluding %s",session_title,conditions{cIdx},conditions{cIdx}))
        title(sprintf("%s %s (All)",session_title,conditions{cIdx}))
        xticks([0 50 100 150])
        xticklabels([0 1 2 3]) 
        yticks([0,100,200,300,400])
        xlabel("Time on Patch (sec)")  
        caxis([-3,3])
        colorbar()
    end
    
end

%% Old code to look at ramping neurons

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
    
    figure(); % colormap('jet')
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
