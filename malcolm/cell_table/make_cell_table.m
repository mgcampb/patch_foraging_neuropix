% Script to make big table of all cells in the MB dataset
% including anatomy info and waveform.
% Can add more info to it later.
% MGC 6/27/2021

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = 'C:\data\patch_foraging_neuropix\cell_table';
if ~isfolder(paths.results)
    mkdir(paths.results);
end
paths.waveforms = 'C:\data\patch_foraging_neuropix\waveforms\waveform_cluster';

%% sessions to analyze
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'));

%% iterate over sessions
mouse = [];
session = [];
cellid = [];
uniqid = [];
br_rough = [];
anat = [];

for session_idx = 1:numel(session_all)
    
    opt.session = session_all{session_idx};
    fprintf('Loading session %d/%d: %s\n',session_idx,numel(session_all),opt.session);

    % load data    
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);    
    N = numel(good_cells_all);

    if ~(isfield(dat,'anatomy') && isfield(dat,'brain_region_rough'))
        continue;
    end
    
    strspl_this = strsplit(opt.session,'_');
    mouse_this = repmat(strspl_this(1),N,1);
    sesh_this = repmat({opt.session},N,1);
    uniqid_this = cell(size(sesh_this));
    for cidx = 1:N
        uniqid_this{cidx} = sprintf('%s_c%d',opt.session,good_cells_all(cidx));
    end
    
    mouse = [mouse; mouse_this];
    session = [session; sesh_this];
    cellid = [cellid; good_cells_all'];
    uniqid = [uniqid; uniqid_this];
    anat = [anat; dat.anatomy.cell_labels];
    br_rough = [br_rough; dat.brain_region_rough];
    
end

%% make table
cell_table = table;
cell_table.Mouse = mouse;
cell_table.Session = session;
cell_table.CellID = cellid;
cell_table.UniqueID = uniqid;
cell_table.BrainRegionRough = br_rough;

%% add anatomy info

assert(all(cell_table.CellID==anat.CellID));
assert(all(strcmp(cell_table.BrainRegionRough,'PFC')==anat.Cortex));
cell_table.BrainRegion = anat.BrainRegion;
cell_table.DepthFromSurface = anat.DepthFromSurface;
cell_table.DepthFromTip = anat.DepthFromTip;

%% add waveform to table
load(fullfile(paths.waveforms,'waveform_cluster.mat'));
assert(all(strcmp(cell_table.UniqueID,waveform_cluster.UniqueID)));
cell_table.WaveformType = waveform_cluster.WaveformType;

%% save
save(fullfile(paths.results,'cell_table'),'cell_table');