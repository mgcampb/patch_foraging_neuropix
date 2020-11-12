function [coeffs,fr_mat,good_cells,score,score_full,expl] = standard_pca_fn(paths,opt)
% Standard function to calculate PCA and perform smoothing 

    %% Handle optional arguments
    % include off patch data?
    onPatchOnly = true;
    if exist('opt', 'var') && isfield(opt,'onPatchOnly')
        onPatchOnly = opt.onPatchOnly;
    end 
    
    cortex_only = true; 
    if exist('opt', 'var') && isfield(opt,'cortex_only')
        cortex_only = opt.cortex_only;
    end  

    %% Load in data
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);  
    if cortex_only == true
        good_cells_all = good_cells_all(dat.anatomy.cell_labels.Cortex); 
    end

    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    tbinedge = opt.tstart:opt.tbin:opt.tend;
    tbincent = tbinedge(1:end-1)+opt.tbin/2; 

    %% extract in-patch times
    in_patch = false(size(tbincent));
    in_patch_buff = false(size(tbincent)); % add buffer for pca
    for i = 1:size(dat.patchCSL,1)
        in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
        in_patch_buff(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)-opt.patch_leave_buffer) = true;
    end
    tbincent = tbincent(in_patch);

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
    good_cells = good_cells_all(fr>=opt.min_fr);


    %% compute PCA

    % compute firing rate mat
    fr_mat = calcFRVsTime(good_cells,dat,opt);

    % take zscore
    fr_mat_zscore = zscore(fr_mat,[],2); % z-score is across whole session including out-of-patch times - is this weird??

    % pca on firing rate matrix, only in patches with buffer before patch leave
    if onPatchOnly == true
%         coeffs = pca(fr_mat_zscore(:,in_patch_buff)'); 
        [coeffs,~,~,~,expl] = pca(fr_mat_zscore(:,in_patch_buff)');  
    else 
%         coeffs = pca(fr_mat_zscore');  
        [coeffs,~,~,~,expl] = pca(fr_mat_zscore');  
    end

    % project full session onto these PCs
    score_full = coeffs'*fr_mat_zscore;

    % only take on-patch times (including buffer)
    score = score_full(:,in_patch)';
end

