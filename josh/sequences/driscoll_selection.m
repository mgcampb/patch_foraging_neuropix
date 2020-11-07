%% Find transiently responsive neurons w/ laura driscoll's method 

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
FRandTimes = struct;
for sIdx = 25:25
    [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,opt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;
%     FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
%     FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
%     FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix; 
end  

%% Bin activity and compare to shuffle (un-normalized) 
index_sort_all = cell(numel(sessions),1); 
decVar_bins = linspace(0,2,41);  
bin_tbin = decVar_bins(2) - decVar_bins(1);
peak_ix_cell = cell(numel(sessions),4); 
driscoll_midresp_struct = struct();

for sIdx = 25:25
    session = sessions{sIdx}; 
    data = load(fullfile(paths.data,session));   
    rewsize = mod(data.patches(:,2),10);   
    nTrials = length(rewsize); 
    
    transient_opt = struct; 
    transient_opt.visualization = false; 
    transient_opt.nShuffles = 500;
    transient_opt.preRew_buffer = round(opt.smoothSigma_time * 3 * 1000 / tbin_ms); 
    transient_opt.postStop_buffer = NaN; % allow first reward
    
    for iRewsize = [1,2,4]
        trial_selection = find(rewsize == iRewsize);  
        transients_struct_tmp = driscoll_transient_discovery(FR_decVar(sIdx),trial_selection,decVar_bins,tbin_ms,transient_opt);
        driscoll_midresp_struct(sIdx,iRewsize).peak_ix = transients_struct_tmp.peak_ix;
        driscoll_midresp_struct(sIdx,iRewsize).midresp = transients_struct_tmp.midresp; 
        driscoll_midresp_struct(sIdx,iRewsize).midresp_peak_ix = transients_struct_tmp.midresp_peak_ix;  
    end  
    
    % now perform similar procedure for even/odd trials
    for odd_even = 1:2
        trial_selection = odd_even:2:nTrials;  
        transients_struct_tmp = driscoll_transient_discovery(FR_decVar(sIdx),trial_selection,decVar_bins,tbin_ms,transient_opt);
        driscoll_midresp_struct(sIdx,odd_even + 4).peak_ix = transients_struct_tmp.peak_ix;
        driscoll_midresp_struct(sIdx,odd_even + 4).midresp = transients_struct_tmp.midresp; 
        driscoll_midresp_struct(sIdx,odd_even + 4).midresp_peak_ix = transients_struct_tmp.midresp_peak_ix; 
    end 
end 

%% Visualize results 
close all
for sIdx = 25
    % get common sort order
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = true;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt); 
    nNeurons = numel(driscoll_midresp_struct(sIdx,1).peak_ix); 
    
    bins = -2:.05:2;
    
    % plot distribution of peak shifts between reward sizes
    figure();hold on 
    histogram(driscoll_midresp_struct(sIdx,2).peak_ix - driscoll_midresp_struct(sIdx,1).peak_ix,bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix - driscoll_midresp_struct(sIdx,2).peak_ix,bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix - driscoll_midresp_struct(sIdx,1).peak_ix,bins) 
    legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak")  
    title("Distribution of peak location differences between reward sizes (All cells)") 
    xlabel("Peak time difference") 
    xline(0,'--','linewidth',2) 
    xlim([-2,2]) 
    
    % need this to look at midresp shifts
    midresp_all = intersect( ...
                            intersect( ...
                                driscoll_midresp_struct(sIdx,1).midresp,driscoll_midresp_struct(sIdx,2).midresp), ...
                                driscoll_midresp_struct(sIdx,4).midresp); 
                            
    % plot distribution of peak shifts between reward sizes for midresp
    figure();hold on 
    histogram(driscoll_midresp_struct(sIdx,2).peak_ix(midresp_all) - driscoll_midresp_struct(sIdx,1).peak_ix(midresp_all),bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix(midresp_all) - driscoll_midresp_struct(sIdx,2).peak_ix(midresp_all),bins)
    histogram(driscoll_midresp_struct(sIdx,4).peak_ix(midresp_all) - driscoll_midresp_struct(sIdx,1).peak_ix(midresp_all),bins) 
    legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak")  
    title("Distribution of peak location differences between reward sizes (Midresp only)") 
    xlabel("Peak time difference") 
    xline(0,'--','linewidth',2) 
    xlim([-2,2]) 
    
    % as a conntrol
    figure()
    histogram(driscoll_midresp_struct(sIdx,6).peak_ix - driscoll_midresp_struct(sIdx,5).peak_ix) 
    title("Distribution of peak location differences on Odd/Even Trials") 
    xlabel("Peak time difference") 
    xlim([-2,2])
    
    figure()
    histogram(driscoll_midresp_struct(sIdx,6).peak_ix(midresp_all) - driscoll_midresp_struct(sIdx,5).peak_ix(midresp_all)) 
    title("Distribution of peak location differences on Odd/Even Trials (Midresp only)") 
    xlabel("Peak time difference") 
    xlim([-2,2])
end