% Standard PCA for on patch data
% MGC 10/9/2020, edited by MB to use whole session instead of in-patch for PCA, and selecting for cortex units only

% paths
paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';

% analysis options
opt = struct;
opt.session = '80_20200317'; % session to analyze
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = 0.5; % in seconds; only takes within patch times up to this amount before patch leave

opt.inPatchPCA = 0;
opt.cortexOnly = 1;

% minimum firing rate (on patch, excluding buffer) to keep neurons
opt.min_fr = 1;

%%
session_all = {opt.session};

for i = 1:numel(session_all)
    fprintf('Computing PCs for session %d/%d: %s\n',i,numel(session_all),session_all{i});
    opt.session = session_all{i};
    
    %% load data
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);

    % load anatomy table
    if opt.cortexOnly ==1
        keep_cells = table2array(dat.anatomy.cell_labels(:, 5));
        good_cells_all = good_cells_all(keep_cells);
    end

    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    tbinedge = opt.tstart:opt.tbin:opt.tend;
    tbincent = tbinedge(1:end-1)+opt.tbin/2;


    %% extract in-patch times
    %if opt.inPatchPCA==1
    in_patch = false(size(tbincent));
    in_patch_buff = false(size(tbincent)); % add buffer for pca
    for i = 1:size(dat.patchCSL,1)
        in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
        in_patch_buff(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)-opt.patch_leave_buffer) = true;
    end
    tbincent = tbincent(in_patch);
    %end


    %% remove cells that don't pass minimum firing rate cutoff

    % compute binned spikecounts for each cell
    t = dat.velt;
    spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
    for cIdx = 1:numel(good_cells_all)
        spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
        spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
    end


    % filter spikecounts to only include in patch times (excluding buffer)
    spikecounts = spikecounts_whole_session(in_patch_buff,:);

    % apply firing rate cutoff
    T = size(spikecounts,1)*opt.tbin;
    N = sum(spikecounts);
    fr = N/T;
    good_cells = good_cells_all(fr>opt.min_fr);


    %% compute PCA

    % compute firing rate mat
    fr_mat = calcFRVsTime(good_cells,dat,opt);

    % take zscore
    fr_mat_zscore = zscore(fr_mat,[],2); % z-score is across whole session including out-of-patch times - is this weird??

    if opt.inPatchPCA==1
        % pca on firing rate matrix, only in patches with buffer before patch leave
        [coeffs, ~,~,~,expl] = pca(fr_mat_zscore(:,in_patch_buff)');
    else % use whole session
        [coeffs, ~,~,~,expl] = pca(fr_mat_zscore(:,:)');

    end

    % project full session onto these PCs
    score = coeffs'*fr_mat_zscore;

    if opt.inPatchPCA==1
        % only take on-patch times (including buffer)
        score = score(:,in_patch)';
    end
    
    %% save PCs to file
end