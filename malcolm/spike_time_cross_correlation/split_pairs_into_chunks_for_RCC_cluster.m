% splits all pairs into chunks for parallel computation on RCC cluster
% MGC 4/14/2021

% Also saves a table of spike times for each GLM cell

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = fullfile('C:\data\patch_foraging_neuropix\spike_time_cross_correlation\data_organized_for_RCC_cluster');
paths.spikes = 'C:\code\spikes'; % spikes repo
addpath(genpath(paths.spikes));
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.sig_cells = 'C:\data\patch_foraging_neuropix\sig_cells';

opt = struct;
opt.max_dist = 200; % in microns
opt.sig_cells = 'sig_cells_table_20210413'; % which sig_cells table to use
opt.num_chunks = 100;

%% load sig_cells table
load(fullfile(paths.sig_cells,opt.sig_cells));

session_all = unique(sig_cells.Session);

%% iterate over sessions

mouse_all_pairs = {};
session_all_pairs = {};
uid1 = {};
uid2 = {};
cid1 = [];
cid2 = [];
dist_all_pairs = [];
gmm_cluster1 = [];
gmm_cluster2 = [];

spike_t_all_glm_cells = {}; 

maxT = nan(numel(session_all),1); 

for sesh_idx = 1:numel(session_all)
    
    opt.session = session_all{sesh_idx};
    fprintf('Session %d/%d: %s\n',sesh_idx,numel(session_all),opt.session);
    
    % only keep cells from the sig cells table for this session
    keep_cell = strcmp(sig_cells.Session,opt.session);
    good_cells = sig_cells.CellID(keep_cell);
    gmm_cluster = sig_cells.GMM_cluster(keep_cell);

    %% load data
    dat = load(fullfile(paths.data,opt.session));
    
    maxT(sesh_idx) = max(dat.sp.st); % need this for spike time crosscorr later
    
    %% get spike times for each cell
    spike_t_this = cell(numel(good_cells),1);
    for cIdx = 1:numel(good_cells)
        spike_t_this{cIdx} = dat.sp.st(dat.sp.clu==good_cells(cIdx));
    end
    % add to list
    spike_t_all_glm_cells = [spike_t_all_glm_cells; spike_t_this];

    %% get spike depths
    [~,spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps,dat.sp.winv,dat.sp.ycoords,dat.sp.spikeTemplates,dat.sp.tempScalingAmps);

    spike_depths = nan(numel(good_cells),1);
    for cIdx = 1:numel(good_cells)
        spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
    end

    %% get pairs

    % all possible pairs
    pairs_idx = nchoosek(1:numel(good_cells),2);

    % only keep pairs within the specified distance
    spike_dist = abs(diff(spike_depths(pairs_idx),[],2));
    keep_pair = spike_dist<opt.max_dist;
    pairs_idx = pairs_idx(keep_pair,:);
    N_pairs = size(pairs_idx,1);
    spike_dist = spike_dist(keep_pair);

    %% add to list
    
    cid1_this = good_cells(pairs_idx(:,1));
    cid2_this = good_cells(pairs_idx(:,2));
    uid1_this = cell(N_pairs,1);
    uid2_this = cell(N_pairs,1);    
    for pIdx = 1:N_pairs
        uid1_this{pIdx} = sprintf('%s_c%d',opt.session,cid1_this(pIdx));
        uid2_this{pIdx} = sprintf('%s_c%d',opt.session,cid2_this(pIdx));
    end 
    
    mouse_all_pairs = [mouse_all_pairs; repmat({opt.session(1:2)},N_pairs,1);];
    session_all_pairs = [session_all_pairs; repmat({opt.session},N_pairs,1)];
    uid1 = [uid1; uid1_this];
    uid2 = [uid2; uid2_this];
    cid1 = [cid1; cid1_this];
    cid2 = [cid2; cid2_this];
    dist_all_pairs = [dist_all_pairs; spike_dist];
    gmm_cluster1 = [gmm_cluster1; gmm_cluster(pairs_idx(:,1))];
    gmm_cluster2 = [gmm_cluster2; gmm_cluster(pairs_idx(:,2))];
    
end

%% make table of all pairs
pairs_table = table;
pairs_table.Mouse = mouse_all_pairs;
pairs_table.Session = session_all_pairs;
pairs_table.UniqueID1 = uid1;
pairs_table.UniqueID2 = uid2;
pairs_table.CellID1 = cid1;
pairs_table.CellID2 = cid2;
pairs_table.Dist = dist_all_pairs;
pairs_table.GMM_Cluster1 = gmm_cluster1;
pairs_table.GMM_Cluster2 = gmm_cluster2;

%% add spike times and unique IDs to sig_cells table
uid = cell(size(sig_cells,1),1);
for cIdx = 1:size(sig_cells,1)
    uid{cIdx} = sprintf('%s_c%d',sig_cells.Session{cIdx},sig_cells.CellID(cIdx));
end
sig_cells.UniqueID = uid;
sig_cells.SpikeTimes = spike_t_all_glm_cells;

%% split data into chunks
chunks = cell(opt.num_chunks,1);
tmp = round(linspace(1,size(pairs_table,1)+1,opt.num_chunks+1));
for i = 1:opt.num_chunks
    chunks{i} = tmp(i):tmp(i+1)-1;
end

%% maxT for each session
xcorr_time_window = table;
xcorr_time_window.Session = session_all;
xcorr_time_window.MinT = zeros(numel(session_all),1);
xcorr_time_window.MaxT = maxT;

%% save data
if ~isfolder(paths.results)
    mkdir(paths.results);
end
save(fullfile(paths.results,'data_organized_for_spike_t_crosscorr_RCC_cluster.mat'),'pairs_table','sig_cells','chunks','xcorr_time_window');