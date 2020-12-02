paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));

opt = struct;
opt.brain_region = 'STR';
opt.data_set = 'mc';
opt.tbin = 0.02;
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)


%% sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end


%%
pb = ParforProgressbar(numel(session_all));
speed_score_all = cell(numel(session_all),1);
parfor sesh_idx = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',sesh_idx,numel(session_all),session_all{sesh_idx});

    dat = load(fullfile(paths.data,session_all{sesh_idx}));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    if ~isfield(dat,'brain_region_rough')
        pb.increment();
        continue
    end
    
    good_cells = good_cells(strcmp(dat.brain_region_rough,opt.brain_region));
    
    if isempty(good_cells)
        pb.increment();
        continue
    end

    % time bins
    tbinedge = dat.velt;
    tbincent = tbinedge(1:end-1)+opt.tbin/2;

    % extract in-patch times (including patch cue to patch stop)

    in_patch = false(size(tbincent));
    for i = 1:size(dat.patchCSL,1)
        in_patch(tbincent>=dat.patchCSL(i,1) & tbincent<=dat.patchCSL(i,3)) = true;
    end

    % compute firing rate matrix
    opt_this = opt;
    opt_this.tstart = 0;
    opt_this.tend = max(dat.velt);
    fr_mat = calcFRVsTime(good_cells,dat,opt_this);
    
    % compute out-of-patch speed score
    fr_out_patch = fr_mat(:,~in_patch);
    vel_out_patch = dat.vel(~in_patch);
    speed_score_all{sesh_idx} = corr(fr_out_patch',vel_out_patch');
    
    pb.increment();
end

%%

speed_score_concat = [];
for i = 1:numel(speed_score_all)
    speed_score_concat = [speed_score_concat; speed_score_all{i}];
end