addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201110_all_sessions';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pie_charts\20201110_2uL_and_4uL_all_sessions';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';
opt.variable = {'TimeOnPatch','TotalRew','TimeSinceRew'};

%%
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

%%

total_neurons = 0;
num_sig_neurons = 0;
for i = 1:numel(session_all)
    
    dat = load(fullfile(paths.results,session_all{i}));
    
    keep_cell = strcmp(dat.brain_region_rough(ismember(dat.good_cells_all,dat.good_cells)),opt.brain_region);

    sig_neurons = nan(numel(opt.variable),sum(keep_cell));
    for j = 1:numel(opt.variable)
        idx1 = contains(dat.var_name,opt.variable{j}) & contains(dat.var_name,'2uL');
        idx2 = contains(dat.var_name,opt.variable{j}) & contains(dat.var_name,'4uL');
    
        sig_neurons(j,:) = sum(abs(dat.beta_all(idx1,keep_cell)),1)>0 & sum(abs(dat.beta_all(idx2,keep_cell)),1)>0;
    end
    sig_neurons = sum(sig_neurons,1)>0;
    
    total_neurons = total_neurons + numel(sig_neurons);
    num_sig_neurons = num_sig_neurons + sum(sig_neurons);
end