%% Stack RXX PETHs across days  
%  Create struct w/ avg FR per RXX trialtype and neuron ordering
%  ~Cross-validate~ order by using peaks from other trials

%% Set paths and basic analysis options 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mgc';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs 
paths.rampIDs = 'Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/ramping_neurons';

addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
FR_calcOpt = struct;
FR_calcOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = FR_calcOpt.tbin * 1000;
FR_calcOpt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
FR_calcOpt.preLeave_buffer = 500; 
FR_calcOpt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};  
mPFC_sessions = [1:8 10:13 15:18 23 25]; 
mgcPFC_sessions = [1:2 5 7]; 
mgcSTR_sessions = [3:4 6 8:9];

%% Extract FR matrices and timing information 
FR_decVar = struct;  
driscoll_struct = struct; 
decVar_bins = linspace(0,2,41);  
bin_tbin = decVar_bins(2) - decVar_bins(1);
for sIdx = 1:numel(sessions) 
%     sIdx = mPFC_sessions(i); 

    if ismember(sIdx,mgcPFC_sessions) 
        FR_calcOpt.region_selection = "PFC";  
        FR_calcOpt.cortex_only = false;
    elseif ismember(sIdx,mgcSTR_sessions)  
        FR_calcOpt.region_selection = "STR";   
        FR_calcOpt.cortex_only = false;
    else 
        disp("Warning: no region for this session")
    end
    
    FR_decVar_tmp = genSeqStructs(paths,sessions,FR_calcOpt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths; 
    
    if ~exist('/structs/driscoll_struct_mgc.mat','file')
    %         analyze whether we have significant transient activity
        transient_opt = struct;
        transient_opt.visualization = false;
        transient_opt.nShuffles = 500;
        transient_opt.preRew_buffer = round(FR_calcOpt.smoothSigma_time * 3 * 1000 / tbin_ms);
        transient_opt.postStop_buffer = NaN; % allow first reward 
        nTrials = length(FR_decVar(sIdx).fr_mat);
        trial_selection = 1:nTrials;
        transients_struct_tmp = driscoll_transient_discovery(FR_decVar(sIdx),trial_selection,decVar_bins,tbin_ms,transient_opt);
        driscoll_struct(sIdx).peak_ix = transients_struct_tmp.peak_ix;
        driscoll_struct(sIdx).midresp = transients_struct_tmp.midresp; 
        driscoll_struct(sIdx).midresp_peak_ix = transients_struct_tmp.midresp_peak_ix;  
    end
end  

%% Generate "reward barcodes" to average firing rates  
rew_barcodes = cell(numel(sessions),1);
for sIdx = 1:numel(sessions)
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

%% Generate cross-days cell array with RXX averages and peak indices per day  
% column 1:: trial types: 100, 110, 101, 111, etc... 
% column 2:: avg PETHs, peak indices (from unvisualized trials)
% column 3:: sessions
RXX_data = cell(8,2,numel(sessions)); 
for sIdx = 1:numel(sessions)
%     sIdx = mPFC_sessions(i);
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
    opt.sort = false; % don't sort yet... just get the peak indices 
    opt.preRew_buffer = round(FR_calcOpt.smoothSigma_time * 3 * 1000 / tbin_ms); 
    opt.postStop_buffer = NaN; % allow first reward 
    opt.tbin_ms = FR_calcOpt.tbin * 1000; 
    opt.neurons = find(~isnan(driscoll_struct(sIdx).peak_ix)); % only use neurons w/ significant peak
    for iRewsize = [2,4] 
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.5);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.5);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.5);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.5); 
        disp(trials100x)
        
        [RXX_data{rew_counter,1,sIdx},RXX_data{rew_counter,2,sIdx}] = avgPETH(FR_decVar(sIdx),trials100x,setdiff(1:nTrials,trials100x),sec3ix,opt);
        [RXX_data{rew_counter+1,1,sIdx},RXX_data{rew_counter+1,2,sIdx}] = avgPETH(FR_decVar(sIdx),trials110x,setdiff(1:nTrials,trials110x),sec3ix,opt);
        [RXX_data{rew_counter+2,1,sIdx},RXX_data{rew_counter+2,2,sIdx}] = avgPETH(FR_decVar(sIdx),trials101x,setdiff(1:nTrials,trials101x),sec3ix,opt);
        [RXX_data{rew_counter+3,1,sIdx},RXX_data{rew_counter+3,2,sIdx}] = avgPETH(FR_decVar(sIdx),trials111x,setdiff(1:nTrials,trials111x),sec3ix,opt);
        
        rew_counter = rew_counter + 4;
    end
end 

%% Now collect PETH across days with cross validated sort  
% session_grps = {(2:9),(10:12),(14:18),[23 25],[1:18 23 25]};   
mouse_grps = {1:2,3,[5 7],[4 6 8:9]};
RXX_avgPETHs = cell(12,numel(mouse_grps));
for iMouse = 1:numel(mouse_grps)
    these_sessions = mouse_grps{iMouse};
    for cond = 1:8 
        cond_frmat_cell = squeeze(RXX_data(cond,1,these_sessions));
        cond_frmat = cat(1,cond_frmat_cell{:});  
        cond_frmat(all(isnan(cond_frmat),2),:) = []; % get rid of filler
        cond_peakIx_cell = squeeze(RXX_data(cond,2,these_sessions));
        cond_peakIx = cat(1,cond_peakIx_cell{:});   
        cond_peakIx(isnan(cond_peakIx)) = []; % get rid of filler
        [~,peak_sort] = sort(cond_peakIx); 
        cond_fr_mat_sorted = cond_frmat(peak_sort,:);  
        RXX_avgPETHs{cond,iMouse} = cond_fr_mat_sorted;
    end
end 

%% Now visualize  
close all

session_grp_titles = ["mc2 PFC","mc2 STR","mc4 PFC","mc4 STR"];  
conditions = ["200","220","202","222","400","440","404","444"];

for iMouse = 1:4
    session_grp_title = session_grp_titles{iMouse};
    figure();
    for cIdx = 1:8
        subplot(2,4,cIdx);colormap('parula')
        imagesc(flipud(RXX_avgPETHs{cIdx,iMouse}));colormap('parula')
        title(sprintf("%s %s",session_grp_title,conditions{cIdx}))
        xticks([0 50 100 150])
        xticklabels([0 1 2 3]) 
        xlabel("Time on Patch (sec)")  
        caxis([-3,3])
        colorbar()
    end 
end
