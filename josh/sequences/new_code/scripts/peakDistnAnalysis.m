%% Analyze distribution of peak indices w.r.t. time since rew across days 
%  use peak indices from driscoll transient discovery 

%% Set paths and basic analysis options 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mgc';
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
% mPFC_sessions = [1:8 10:13 15:18 23 25]; 
% mouse_grps = {1:2,3:8,10:13,15:18,[23 25]}; 
% mouse_names = ["m75","m76","m78","m79","m80"];  
mouse_grps = {1:2,3,[5 7],[4 6 8:9]}; 
mouse_names = ["mc2 PFC","mc2 STR","mc4 PFC","mc4 STR"];  

%% Extract FR matrices and timing information (here for all trials)
FR_decVar = struct;  
driscoll_struct = struct; 
decVar_bins = linspace(0,2,41);  
bin_tbin = decVar_bins(2) - decVar_bins(1);
for sIdx = 1:numel(sessions) 
%     sIdx = mPFC_sessions(i);
    FR_decVar_tmp = genSeqStructs(paths,sessions,FR_calcOpt,sIdx);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat; 
    FR_decVar(sIdx).goodcell_IDs = FR_decVar_tmp.goodcell_IDs; 
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths; 
    
    if ~exist('/structs/driscoll_struct.mat','file')
        % analyze whether we have significant transient activity
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
    else 
        load('/structs/driscoll_struct.mat');
    end
end  

%% Visualize distribution of significant peaks across mice  
close all
peak_ix_mousePooled = []; 
driscoll_cell = squeeze(struct2cell(driscoll_struct))'; 
figure()
for m = 1:numel(mouse_grps) 
    mouse_peak_ix_cell = driscoll_cell(mouse_grps{m},1); 
    peak_ix_mouse = cat(1,mouse_peak_ix_cell{:});    
    midresp_label = (peak_ix_mouse > .25) & (peak_ix_mouse < 1.75);
    jitter_y =  .5 * (rand(length(peak_ix_mouse),1) - .5);
    jitter_x =  .1 * (rand(length(peak_ix_mouse),1) - .5);  
    subplot(1,2,1);hold on
    gscatter(peak_ix_mouse + jitter_x,m + zeros(length(peak_ix_mouse),1) + jitter_y,midresp_label,[0 0 0;1 0 0]) 
    legend("Early/Late Label","Mid-Responsive Label")
    subplot(4,2,2 + 8-2*m) 
    histogram(peak_ix_mouse,0:.05:1,'Normalization','probability')  
    ylabel(mouse_names{m}) 
    yticks([]) 
    if m > 1
        xticks([]) 
    end 
end  
subplot(1,2,1)
ylim([0,4]) ;yticks(1:4); yticklabels(mouse_names) 
title("Distribution of Significant Peak Indices")   
xlabel("Time Since Reward")
subplot(4,2,2) 
title("Distribution of Significant Peak Indices")  
subplot(4,2,8)  
xlabel("Time Since Reward")
% subplot(1,2,2)  

%% pie charts of proportion mid vs early vs late vs nonselective 
driscoll_cell = squeeze(struct2cell(driscoll_struct))'; 
figure()   
t = tiledlayout(numel(mouse_grps),1);
colors = [0 0 0 ;cool(3)];
for m = 1:numel(mouse_grps) 
    subplot(5,1,6 - m)
    mouse_peak_ix_cell = driscoll_cell(mouse_grps{m},1); 
    peak_ix_mouse = cat(1,mouse_peak_ix_cell{:});  
    proportions = [length(find(isnan(peak_ix_mouse))); ... 
                   length(find(peak_ix_mouse < .25)); ... 
                   length(find(peak_ix_mouse > .25 & peak_ix_mouse < 1.75 )); ... 
                   length(find(peak_ix_mouse > 1.75))]; 
    p = pie(proportions) ;
    for k = 1:size(colors,1)
        set(p(k*2-1), 'FaceColor', colors(k,:));
    end  
    title(sprintf("%s                   ",mouse_names{m}))
    if m == 5 
        legend("No Significant Peak","Significant Peak < .25 sec After Rew","Significant Peak .25-1.75 sec After Rew","Significant Peak > 1.75 sec After Rew")
    end
    
end
