paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output';
run_name = 'run_20210208';

chunks = dir(fullfile(paths.results,run_name,'*.mat'));
chunks = {chunks.name}';

paths.save = fullfile(paths.results,run_name,'sessions');
if ~isfolder(paths.save)
    mkdir(paths.save);
end

%%
session = cell(numel(chunks),1);
for i = 1:numel(chunks)
    % fprintf('Processing chunk %d/%d\n',i,numel(chunks));
    dat = load(fullfile(paths.results,run_name,chunks{i}));
    session{i} = dat.opt.session;
end

%% now gather chunks by session
session_uniq = unique(session);
for i = 1:numel(session_uniq)
    fprintf('Processing session %d/%d: %s\n',i,numel(session_uniq),session_uniq{i});
    chunks_this = chunks(strcmp(session,session_uniq{i}));
    beta_all = [];
    brain_region_rough = [];
    depth_from_surface = [];
    fr = [];
    good_cells = [];
    good_cells_all = [];
    pval_full_vs_base = [];
    run_times = [];
    spikecounts = [];
    for j = 1:numel(chunks_this)
        dat = load(fullfile(paths.results,run_name,chunks_this{j}));
        X_full = dat.X_full;
        Xmean = dat.Xmean;
        Xstd = dat.Xstd;
        anatomy = dat.anatomy;
        bas_patch_stop = dat.bas_patch_stop;
        bas_rew = dat.bas_rew;
        base_var = dat.base_var;
        beta_all = [beta_all dat.beta_all];
        brain_region_rough = [brain_region_rough; dat.brain_region_rough];
        depth_from_surface = [depth_from_surface; dat.depth_from_surface];
        foldid = dat.foldid;
        fr = [fr dat.fr];
        good_cells = [good_cells; dat.good_cells];
        good_cells_all = [good_cells_all; dat.good_cells_all];
        opt = dat.opt;
        pval_full_vs_base = [pval_full_vs_base; dat.pval_full_vs_base];
        run_times = [run_times; dat.run_times];
        spikecounts = [spikecounts dat.spikecounts];
        t_basis_patch_stop = dat.t_basis_patch_stop;
        t_basis_rew = dat.t_basis_rew;
        trial_grp = dat.trial_grp;
        var_name = dat.var_name;
    end
    save(fullfile(paths.save,session_uniq{i}),'opt','beta_all','pval_full_vs_base',...
        'var_name','base_var','X_full','Xmean','Xstd','spikecounts','good_cells',...
        'trial_grp','foldid','bas_patch_stop','t_basis_patch_stop','bas_rew','t_basis_rew',...
        'anatomy','run_times','depth_from_surface','fr','good_cells_all','brain_region_rough');
end
