%% Visualization to prepare for dynamical systems analysis 
%  We are making a 3 x 2 plot showing 10, 20, 40 trials
%  Row 1: Binscatter to show density of vistation in PC space 
%  Row 2: Heatmap of PC1/3 gradient magnitude + vector field  

%% Generic setup
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
tbin_ms = opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Load firing rate matrices, perform PCA

for sIdx = 24:24 
    session = sessions{sIdx}(1:end-4);
    
    % analysis options
    opt = struct;
    opt.session = session; % session to analyze
    opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
    opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
    opt.patch_leave_buffer = 0.5; % in seconds; only takes within patch times up to this amount before patch leave

    % minimum firing rate (on patch, excluding buffer) to keep neurons
    opt.min_fr = 1; 
    
    [coeffs,fr_mat,good_cells,score] = standard_pca_fn(paths,opt);
    
end