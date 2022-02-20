%%

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

run_name = '20210526_full';
run_name_base = '20210517_SessionTime';

paths = struct;
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_output',run_name);
paths.results_base = fullfile('C:\data\patch_foraging_neuropix\GLM_output',run_name_base);
paths.figs = fullfile('C:\figs\patch_foraging_neuropix\glm_pca_on_model_coefficients',run_name);
paths.waveforms = 'C:\data\patch_foraging_neuropix\waveforms\waveform_cluster';

paths.sig_cells = 'C:\data\patch_foraging_neuropix\sig_cells';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.data_set = 'mb';

opt.save_figs = false;

%% get sessions
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

%% get significant cells from all sessions
sesh_all = [];
beta_all = [];
cellID_uniq = [];
num_cells_glm_total = 0;
num_cells_total = 0;
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    num_cells_glm_total = num_cells_glm_total + numel(fit.good_cells);
    num_cells_total = num_cells_total + numel(fit.good_cells_all);
    
    good_cells_this = fit.good_cells;
    
    beta_this = fit.beta_all;
    beta_all = [beta_all fit.beta_all];
    sesh_all = [sesh_all; i*ones(numel(good_cells_this),1)];
    
    cellID_uniq_this = cell(numel(good_cells_this),1);
    for j = 1:numel(good_cells_this)
        cellID_uniq_this{j} = sprintf('%s_c%d',fit.opt.session,good_cells_this(j));
    end
    cellID_uniq = [cellID_uniq; cellID_uniq_this];

end
var_name = fit.var_name';