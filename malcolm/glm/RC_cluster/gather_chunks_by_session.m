%FIRST: copy all input files (one per session) from GLM_input folder into paths.save folder using Unix command cp

paths = struct;
% paths.base = '/n/holystore01/LABS/uchida_users/Users/mcampbell/run_GLM_on_cluster';
paths.base = 'C:\data\patch_foraging_neuropix';
paths.GLM_input = fullfile(paths.base,'GLM_input');
paths.GLM_output = fullfile(paths.base,'GLM_output');
run_name = '20210813_squared_terms';

paths.chunks = fullfile(paths.GLM_output,run_name,'chunks');
chunks = dir(fullfile(paths.chunks,'*.mat'));
chunks = {chunks.name}';

paths.save = fullfile(paths.GLM_output,run_name);

%%
session = cell(numel(chunks),1);
for i = 1:numel(chunks)
    % fprintf('Processing chunk %d/%d\n',i,numel(chunks));
    dat = load(fullfile(paths.chunks,chunks{i}));
    session{i} = dat.session;
end

%% now gather chunks by session
session_uniq = unique(session);

% for combining sessions
beta_concat = [];
dev_all = [];
nulldev_all = [];
cellID_all = [];
brain_region_rough_all = [];
depth_from_surface_all = [];

for i = 1:numel(session_uniq)
    fprintf('Processing session %d/%d: %s\n',i,numel(session_uniq),session_uniq{i});
    
    % get GLM input for this session
    dat_input = load(fullfile(paths.save,session_uniq{i}));
    chunks_this = chunks(strcmp(session,session_uniq{i}));
    beta_all = [];
    good_cells_beta = [];
    dev = [];
    for j = 1:numel(chunks_this)
        dat = load(fullfile(paths.chunks,chunks_this{j}));
        cellID_this = reshape(dat.cellID,numel(dat.cellID),1);
        beta_all = [beta_all dat.beta'];
        good_cells_beta = [good_cells_beta; cellID_this];
        dev = [dev; dat.dev];
    end
    
    [good_cells_beta,sort_idx] = sort(good_cells_beta);
    beta_all = beta_all(:,sort_idx);
    dev = dev(sort_idx);
    keep = ismember(dat_input.good_cells,good_cells_beta);
    good_cells = dat_input.good_cells(keep);
    brain_region_rough = dat_input.brain_region_rough(keep);
    depth_from_surface = dat_input.depth_from_surface(keep);
    spikecounts = dat_input.spikecounts(:,keep);
    assert(all(good_cells==good_cells_beta'),'cell IDs do not match');
    save(fullfile(paths.save,session_uniq{i}),'beta_all','dev',...
        'good_cells','brain_region_rough','depth_from_surface','spikecounts','-append');
    
    % concat over sessions
    beta_concat = [beta_concat beta_all];
    dev_all = [dev_all; dev];
    nulldev_all = [nulldev_all; dat_input.nulldev(keep)];
    brain_region_rough_all = [brain_region_rough_all; brain_region_rough];
    depth_from_surface_all = [depth_from_surface_all; depth_from_surface];
    
    cellID_this = cell(numel(good_cells),1);
    for j = 1:numel(good_cells)
        cellID_this{j} = sprintf('%s_c%d',dat_input.opt.session,good_cells(j));
    end
    cellID_all = [cellID_all; cellID_this];
end

%% save concatenated data

beta_all = beta_concat;
dev = dev_all;
nulldev = nulldev_all;
dev_expl = 100*(1-dev./nulldev);
cellID = cellID_all;
brain_region_rough = brain_region_rough_all;
depth_from_surface = depth_from_surface_all;

if ~isfolder(fullfile(paths.save,'combined'))
    mkdir(fullfile(paths.save,'combined'));
end

save(fullfile(paths.save,'combined','GLM_coeffs'),'beta_all','dev','nulldev','dev_expl','cellID','brain_region_rough','depth_from_surface');