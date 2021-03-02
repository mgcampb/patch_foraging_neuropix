paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output';

run_name = 'run_20210208';

chunks = dir(fullfile(paths.results,run_name,'*.mat'));
chunks = {chunks.name}';


session = {};
cellID = [];
pval = [];
beta = [];
job_time = [];
brain_region_rough = {};
for i = 1:numel(chunks)
    fprintf('Processing chunk %d/%d\n',i,numel(chunks));
    dat = load(fullfile(paths.results,run_name,chunks{i}));
    pval = [pval; dat.pval_full_vs_base];
    beta = [beta; dat.beta_all'];
    job_time = [job_time; dat.run_times];
    session = [session; repmat({dat.opt.session},numel(dat.good_cells),1)];
    cellID = [cellID; dat.good_cells];
    brain_region_rough = [brain_region_rough; dat.brain_region_rough];
end

%%

var_name = dat.var_name;

cell_info = table;
cell_info.Session = session;
cell_info.CellID = cellID;
cell_info.BrainRegionRough = brain_region_rough;
cell_info.pval = pval;
cell_info.beta = beta;
cell_info.job_time = job_time;

%%
% save(fullfile(paths.results,run_name,'output_combined.mat'),'cell_info','beta','job_time');

%%
session_uniq = unique(cell_info.Session);
job_time_by_session = nan(numel(session_uniq),1);
for i = 1:numel(session_uniq)
    job_time_by_session(i) = mean(cell_info.job_time(strcmp(cell_info.Session,session_uniq{i})));
end