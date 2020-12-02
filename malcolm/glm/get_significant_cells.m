% Script to identify significant cells from GLM model comparison analysis
% and save them in a table

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.results_save = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison\sig_cells';
if ~isfolder(paths.results_save)
    mkdir(paths.results_save)
end

opt = struct;
opt.brain_region = 'MOs';
opt.data_set = 'mc';
opt.pval_thresh = 0.05;

%%
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

% get mouse name for each session
mouse = cell(size(session_all));
if strcmp(opt.data_set,'mb')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:2);
    end
elseif strcmp(opt.data_set,'mc')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:3);
    end
end
uniq_mouse = unique(mouse);

%%

sig_cells = table;
n_start = 10000;
sig_cells.Mouse = cell(n_start,1);
sig_cells.Session = cell(n_start,1);
sig_cells.CellID = nan(n_start,1);
sig_cells.BrainRegionRough = cell(n_start,1);
sig_cells.BrainRegion = cell(n_start,1);
sig_cells.DepthFromSurface = nan(n_start,1);
sig_cells.DepthFromTip = nan(n_start,1);

counter = 1;
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    
    sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    N = sum(sig);
    
    keep_idx = find(keep_cell);
    keep_idx = keep_idx(sig);
    
    sig_cells.Mouse(counter:counter+N-1) = repmat(mouse(i),N,1);
    sig_cells.Session(counter:counter+N-1) = repmat(session_all(i),N,1);
    sig_cells.CellID(counter:counter+N-1) = fit.good_cells(keep_idx);
    sig_cells.BrainRegionRough(counter:counter+N-1) = fit.brain_region_rough(keep_idx);
    if isfield(fit.anatomy,'cell_labels')
        keep_idx2 = ismember(fit.good_cells_all,fit.good_cells(keep_idx));
        sig_cells.BrainRegion(counter:counter+N-1) = fit.anatomy.cell_labels.BrainRegion(keep_idx2);
        sig_cells.DepthFromSurface(counter:counter+N-1) = fit.anatomy.cell_labels.DepthFromSurface(keep_idx2);
        sig_cells.DepthFromTip(counter:counter+N-1) = fit.anatomy.cell_labels.DepthFromTip(keep_idx2);
    end
    
    counter = counter+N;

end
sig_cells = sig_cells(1:counter-1,:);

%% save results
save(fullfile(paths.results_save,sprintf('sig_cells_%s_cohort_%s.mat',opt.data_set,opt.brain_region)),'sig_cells');