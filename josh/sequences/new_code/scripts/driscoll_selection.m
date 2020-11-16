%% Find transiently responsive neurons w/ laura driscoll's method 

paths = struct;
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mgc';
paths.rampIDs = 'Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/ramping_neurons';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
opt.preLeave_buffer = 500; 
opt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};  
mPFC_sessions = [1:8 10:13 15:18 23 25]; 
mgcPFC_sessions = [1:2 5 7]; 
mgcSTR_sessions = [3:4 6 8:9];

%% Extract FR matrices and timing information 
FR_decVar = struct; 
for sIdx = 1:numel(sessions)
    %     sIdx = mPFC_sessions(i);
    if ismember(sIdx,mgcPFC_sessions)
        opt.region_selection = "PFC";
        opt.cortex_only = false;
    elseif ismember(sIdx,mgcSTR_sessions)
        opt.region_selection = "STR";
        opt.cortex_only = false;
    else
        disp("Warning: no region for this session")
    end
    
    [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,opt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;

    % Bin activity and compare to shuffle (un-normalized)
    index_sort_all = cell(numel(sessions),1);
    decVar_bins = linspace(0,2,41);
    bin_tbin = decVar_bins(2) - decVar_bins(1);
    peak_ix_cell = cell(numel(sessions),4);
    driscoll_midresp_struct = struct();
    
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
mouse_names = ["m75","m76","m78","m79","m80"];
mouse_groups = {1:2,3:8,10:13,15:18,[23 25]}; % group mPFC sessions by animal  

for m = 1:5
    ul1_peak_ix = []; 
    ul2_peak_ix = []; 
    ul4_peak_ix = []; 
    odd_peak_ix = []; 
    even_peak_ix = []; 
    ul1_peak_ix_midresp = []; 
    ul2_peak_ix_midresp = []; 
    ul4_peak_ix_midresp = []; 
    odd_peak_ix_midresp = []; 
    even_peak_ix_midresp = [];
    for i = mouse_groups{m} 
        ul1_peak_ix = [ul1_peak_ix ; driscoll_midresp_struct(i,1).peak_ix];
        ul2_peak_ix = [ul2_peak_ix ; driscoll_midresp_struct(i,2).peak_ix];
        ul4_peak_ix = [ul4_peak_ix ; driscoll_midresp_struct(i,4).peak_ix]; 
        odd_peak_ix = [odd_peak_ix ; driscoll_midresp_struct(i,5).peak_ix]; 
        even_peak_ix = [even_peak_ix ; driscoll_midresp_struct(i,6).peak_ix];  
        
        midresp_2uL = driscoll_midresp_struct(i,2).midresp; % compromise? 
        ul1_peak_ix_midresp = [ul1_peak_ix_midresp ; driscoll_midresp_struct(i,1).peak_ix(midresp_2uL)];
        ul2_peak_ix_midresp = [ul2_peak_ix_midresp ; driscoll_midresp_struct(i,2).peak_ix(midresp_2uL)];
        ul4_peak_ix_midresp = [ul4_peak_ix_midresp ; driscoll_midresp_struct(i,4).peak_ix(midresp_2uL)]; 
        odd_peak_ix_midresp = [odd_peak_ix_midresp ; driscoll_midresp_struct(i,5).peak_ix(midresp_2uL)]; 
        even_peak_ix_midresp = [even_peak_ix_midresp ; driscoll_midresp_struct(i,6).peak_ix(midresp_2uL)];  
    end
    
    % plot distribution of peak shifts between reward sizes
%     figure();hold on 
%     histogram(ul2_peak_ix - ul1_peak_ix,bins)
%     histogram(ul4_peak_ix - ul2_peak_ix,bins)
%     histogram(ul4_peak_ix - ul1_peak_ix,bins) 
%     legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak")  
%     title(sprintf("%s Distribution of peak location differences between reward sizes (All cells)",mouse_names{m})) 
%     xlabel("Peak time difference") 
% %     xline(0,'--','linewidth',2) 
%     xlim([-2,2]) 
%     % plot distribution of peak shifts between odd/even trials as a control
%     figure()
%     histogram(even_peak_ix - odd_peak_ix,bins,'FaceColor','k') 
%     title(sprintf("%s Distribution of peak location differences on Odd/Even Trials",mouse_names{m})) 
%     xlabel("Peak time difference") 
%     xlim([-2,2]) 
    
    % Visualize again using ECDF
    figure();hold on 
    h21 = cdfplot(ul2_peak_ix - ul1_peak_ix); 
    h21.LineWidth = 1.5;
    h42 = cdfplot(ul4_peak_ix - ul2_peak_ix); 
    h42.LineWidth = 1.5;
    h41 = cdfplot(ul4_peak_ix - ul1_peak_ix);  
    h41.LineWidth = 1.5;
    xlabel("Peak time difference") 
%     xline(0,'--','linewidth',2) 
    xlim([-2,2]) 
    % plot distribution of peak shifts between odd/even trials as a control
    heo = cdfplot(even_peak_ix - odd_peak_ix); 
    heo.LineWidth = 1.5; 
    heo.Color = 'k';
    legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak","Even-Odd Peak") 
%     title("Distribution of peak location differences on Odd/Even Trials") 
    xlabel("Peak time difference") 
    xlim([-2,2]) 
    title(sprintf("%s Distribution of peak location differences between reward sizes (All cells)",mouse_names{m})) 

    % Now just for mid-responsive neurons %%%%
%     % plot distribution of peak shifts between reward sizes
%     figure();hold on 
%     histogram(ul2_peak_ix_midresp - ul1_peak_ix_midresp,bins)
%     histogram(ul4_peak_ix_midresp - ul2_peak_ix_midresp,bins)
%     histogram(ul4_peak_ix_midresp - ul1_peak_ix_midresp,bins) 
%     legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak")  
%     title(sprintf("%s Distribution of peak location differences between reward sizes (Midresponsive cells)",mouse_names{m})) 
%     xlabel("Peak time difference") 
% %     xline(0,'--','linewidth',2) 
%     xlim([-2,2]) 
%     % plot distribution of peak shifts between odd/even trials as a control
%     figure()
%     histogram(even_peak_ix_midresp - odd_peak_ix_midresp,bins,'FaceColor','k') 
%     title("Distribution of peak location differences on Odd/Even Trials") 
%     xlabel("Peak time difference") 
%     xlim([-2,2]) 
    
    % Visualize again using ECDF
    figure();hold on 
    h21 = cdfplot(ul2_peak_ix_midresp - ul1_peak_ix_midresp); 
    h21.LineWidth = 1.5;
    h42 = cdfplot(ul4_peak_ix_midresp - ul2_peak_ix_midresp); 
    h42.LineWidth = 1.5;
    h41 = cdfplot(ul4_peak_ix_midresp - ul1_peak_ix_midresp);  
    h41.LineWidth = 1.5;
    xlabel("Peak time difference") 
%     xline(0,'--','linewidth',2) 
    xlim([-2,2]) 
    % plot distribution of peak shifts between odd/even trials as a control
    heo = cdfplot(even_peak_ix_midresp - odd_peak_ix_midresp); 
    heo.LineWidth = 1.5; 
    heo.Color = 'k';
    legend("2 uL Peak - 1 uL Peak","4 uL Peak - 2 uL Peak","4 uL Peak - 1 uL Peak","Even-Odd Peak") 
%     title("Distribution of peak location differences on Odd/Even Trials") 
    xlabel("Peak time difference") 
    xlim([-2,2]) 
    title(sprintf("%s Distribution of peak location differences between reward sizes (Midresponsive cells)",mouse_names{m})) 
end