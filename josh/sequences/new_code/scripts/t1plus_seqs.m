%% Analyze consistency of neurons for t = 0 vs t > 1 rewards 
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
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
    FR_decVar(sIdx).rew_sec = FR_decVar_tmp.rew_sec;
    FR_decVar(sIdx).rew_num = FR_decVar_tmp.rew_num;
    FR_decVar(sIdx).cell_depths = FR_decVar_tmp.spike_depths;

end  
