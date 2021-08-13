% script to fit GLM to PCs
% MGC 6/26/2021

run_name = '20210626';

restoredefaultpath;
paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.glmnet = 'C:\code\glmnet_matlab_from_cluster';
addpath(genpath(paths.glmnet));
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes = 'C:\code\spikes';
addpath(genpath(paths.spikes));
paths.results = fullfile('C:\data\patch_foraging_neuropix\GLM_on_PCs\',run_name);
if ~isfolder(paths.results)
    mkdir(paths.results);
end

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'));

% analysis opts
opt = struct;

opt.rew_size = [1 2 4];

opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)
opt.smooth_sigma_vel = 0.1; % in seconds, for smoothing velocity before computing accel (vel is fed unsmoothed into GLM)

% basis functions for patch stop
opt.nbasis_patch_stop = 6;
opt.basis_length_patch_stop = 1;

% basis functions for time since reward
opt.nbasis_rew = 11; % number of raised cosine basis functions to use
opt.basis_length_rew = 2; % in seconds; make sure basis functions divide evenly into 1 second intervals (makes analysis easier)
opt.cut_off_kernel_sequence_after_each_reward = true; % whether to cut off the sequence of kernels when a new reward arrives
opt.include_first_reward = true; % whether or not to include first reward in reward kernels

% whether or not to zscore predictors
% glmnet handles this for us, so turning this off for now. 
% NOTE: glmnet standardizes X for the fit, then returns the coefficients on
% the original scale. So, to compare coefficients across variables within a
% single fit, re-standardize them by multiplying each coefficient by its
% corresponding variable's standard deviation
opt.zscore_predictors = true; 

% minimum firing rate to keep neurons
opt.min_fr = 1;

% regularization
opt.alpha = 0.9; % weighting of L1 and L2 penalties in elastic net regularization

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized fold, assigning (roughly) equal numbers of each reward size to each fold

% which groups of variables to include
opt.var_grps_to_include = {'SessionTime','Behav','PatchKern','RewKern','DVs'}; % any subset of {'SessionTime','Behav','PatchKern','RewKern','DVs'}

% smooth sigma in sec for computing firing rate matrix
opt.smoothSigma_time = 0.1;

% whether or not to only use cortex cells for PCA
opt.cortexOnly = 1;

% number of PCs to fit GLM to 
opt.num_pcs = 10;


%% raised cosine basis for time since patch stop

t_basis_patch_stop = 0:opt.tbin:opt.basis_length_patch_stop;
db = (max(t_basis_patch_stop) - min(t_basis_patch_stop))/(opt.nbasis_patch_stop-1);
c = min(t_basis_patch_stop):db:max(t_basis_patch_stop);
bas_patch_stop = nan(opt.nbasis_patch_stop,length(t_basis_patch_stop));
for k = 1:opt.nbasis_patch_stop
  bas_patch_stop(k,:) = (cos(max(-pi, min(pi,pi*(t_basis_patch_stop - c(k))/(db))) ) + 1) / 2;
end

%% raised cosine basis for time since reward

t_basis_rew = 0:opt.tbin:opt.basis_length_rew;
db = (max(t_basis_rew) - min(t_basis_rew))/(opt.nbasis_rew-1);
c = min(t_basis_rew):db:max(t_basis_rew);
bas_rew = nan(opt.nbasis_rew,length(t_basis_rew));
for k = 1:opt.nbasis_rew
  bas_rew(k,:) = (cos(max(-pi, min(pi,pi*(t_basis_rew - c(k))/(db))) ) + 1) / 2;
end

%%
tic
for session_idx = 1:numel(session_all)
    
    opt.session = session_all{session_idx};
    fprintf('Analyzing session %d/%d: %s\n',session_idx,numel(session_all),opt.session);

    %% load data    
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);    

    if isfield(dat,'anatomy') && isfield(dat,'brain_region_rough')

        anatomy = dat.anatomy;
        brain_region_rough = dat.brain_region_rough;

        % get depth
        [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps,dat.sp.winv,dat.sp.ycoords,dat.sp.spikeTemplates,dat.sp.tempScalingAmps);
        spike_depths = nan(numel(good_cells_all),1);
        for cidx = 1:numel(good_cells_all)
            spike_depths(cidx) = median(spike_depths_all(dat.sp.clu==good_cells_all(cidx)));
        end
        
        if isfield(anatomy,'insertion_depth')
            depth_from_surface = spike_depths-anatomy.insertion_depth;
        else
            depth_from_surface = nan;
        end

    else
        continue;
    end

    %% compute binned spikecounts for each cell
    
    t = dat.velt;
    spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
    for cIdx = 1:numel(good_cells_all)
        spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
        spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
    end
    
    %% get patch num for each patch
    
    patch_num = nan(size(t));
    for i = 1:size(dat.patchCSL,1)
        % include one time bin before patch stop to catch the first reward
        patch_num(t>=(dat.patchCSL(i,2)-opt.tbin) & t<=dat.patchCSL(i,3)) = i;
    end
    in_patch = ~isnan(patch_num);

    %% get size of each reward
    
    dat.rew_size = nan(size(dat.rew_ts));
    rew_size_patch = mod(dat.patches(:,2),10);
    for i = 1:numel(dat.rew_size)
        which_patch = find(dat.patchCSL(:,1)<dat.rew_ts(i) & dat.patchCSL(:,3)>dat.rew_ts(i));
        if ~isempty(which_patch)
            dat.rew_size(i) = rew_size_patch(which_patch);
        end
    end

    %% make predictors for this session

    X = []; % predictor matrix
    
    % intercept
    var_name = {'Intercept'};
    base_var = [1]; % whether or not to use each variable in the "base" model

    % SESSION TIME
    if any(strcmp(opt.var_grps_to_include,'SessionTime'))
        X_this = [t' t.^2'];
        X = [X X_this];
        var_name = [var_name,'SessionTime','SessionTime^2'];
        base_var = [base_var 1 1];
    end
    
    % BEHAVIORAL VARIABLES
    if any(strcmp(opt.var_grps_to_include,'Behav'))
        % PATCH POSITION
        if isfield(dat,'patch_pos')
            dat.patch_pos(isnan(dat.patch_pos)) = -4;
            X_this = [dat.patch_pos' dat.patch_pos'.^2];
            X = [X X_this];
            var_name = [var_name,'Position','Position^2'];
            base_var = [base_var 1 1];
        end

        % RUNNING SPEED
        % running speed and running speed squared
        X_this = [dat.vel' dat.vel'.^2];
        X = [X X_this];
        var_name = [var_name,'Speed','Speed^2'];
        base_var = [base_var 1 1];

        % ACCELERATION
        % smooth velocity first before taking derivative; otherwise super noisy
        vel_smooth = gauss_smoothing(dat.vel,opt.smooth_sigma_vel/opt.tbin);
        accel = diff(vel_smooth)/opt.tbin;
        accel = [accel accel(end)];
        X_this = [accel' accel'.^2];
        X = [X X_this];
        var_name = [var_name,'Accel','Accel^2'];
        base_var = [base_var 1 1];

        % LICK RATE
        if ~all(isnan(dat.lick_ts))
            lickcounts = histc(dat.lick_ts,t)/opt.tbin;
            lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate/opt.tbin);
            X_this = lickrate;
            X = [X X_this];
            var_name = [var_name,'LickRate'];
            base_var = [base_var 1];
        end
    end

    % iterate over reward sizes
    for rIdx = 1:numel(opt.rew_size)

        rew_size_this = opt.rew_size(rIdx);
        
        % TIME SINCE PATCH STOP (patch onset)
        if any(strcmp(opt.var_grps_to_include,'PatchKern'))
            patch_stop_binary = histc(dat.patchCSL(rew_size_patch==rew_size_this,2),t);
            patch_stop_conv = nan(numel(patch_stop_binary),opt.nbasis_patch_stop);
            for i = 1:opt.nbasis_patch_stop
                conv_this = conv(patch_stop_binary,bas_patch_stop(i,:));
                patch_stop_conv(:,i) = conv_this(1:numel(patch_stop_binary));
            end
            X_this = patch_stop_conv;
            X = [X X_this];
            for i = 1:opt.nbasis_patch_stop
                var_name = [var_name,sprintf('PatchStopKern%d_%duL',i,rew_size_this)];
            end
            base_var = [base_var ones(1,opt.nbasis_patch_stop)];
        end

        % TIME SINCE REWARD KERNELS
        if any(strcmp(opt.var_grps_to_include,'RewKern'))
            rew_ts_this = dat.rew_ts(dat.rew_size==rew_size_this);
            if ~opt.include_first_reward
                rew_ts_this = rew_ts_this(min(abs(rew_ts_this-dat.patchCSL(:,2)'),[],2)>0.01);
            end
            rew_binary = histc(rew_ts_this,t);
            rew_conv = nan(numel(rew_binary),opt.nbasis_rew);
            % convolve with basis functions
            for i = 1:opt.nbasis_rew
                conv_this = conv(rew_binary,bas_rew(i,:));
                rew_conv(:,i) = conv_this(1:numel(rew_binary));
            end
            if opt.cut_off_kernel_sequence_after_each_reward
                if opt.basis_length_rew>1
                    % cut off kernels when new reward comes
                    [~,idx]=max(rew_conv>0,[],2);
                    for i = 1:size(rew_conv,1)
                        rew_conv(i,idx(i)+2:end) = 0;
                    end
                end
            end
            X_this = rew_conv;
            X = [X X_this];
            for i = 1:opt.nbasis_rew
                var_name = [var_name,sprintf('RewKern%d_%duL',i,rew_size_this)];
            end
            base_var = [base_var zeros(1,opt.nbasis_rew)];
        end


        % DECISION VARIABLES (DVs)
        if any(strcmp(opt.var_grps_to_include,'DVs'))
            % time on patch
            t_on_patch = zeros(size(t));
            t_on_patch(in_patch) = t(in_patch) - dat.patchCSL(patch_num(in_patch),2)';
            t_on_patch(~ismember(patch_num,find(rew_size_patch==rew_size_this)))=0;
            % total rewards on patch so far
            tot_rew = zeros(size(t));
            for i = 1:size(dat.patchCSL,1)
                tot_rew(patch_num==i) = cumsum(rew_binary(patch_num==i));
            end
            % time since reward
            t_since_rew = zeros(size(t));
            for i = 2:numel(t)
                if rew_binary(i)
                    t_since_rew(i) = 0;
                else
                    t_since_rew(i) = t_since_rew(i-1)+opt.tbin;
                end
            end
            t_since_rew(~in_patch)=0;
            t_since_rew(~ismember(patch_num,find(rew_size_patch==rew_size_this)))=0;
            X_this = [t_on_patch' tot_rew' t_since_rew'];
            X = [X X_this];
            var_name = [var_name,sprintf('TimeOnPatch_%duL',rew_size_this),sprintf('TotalRew_%duL',rew_size_this),sprintf('TimeSinceRew_%duL',rew_size_this)];
            base_var = [base_var 0 0 0];
        end
    end
    
    %% subselect data to fit GLM to (in patch times)

    % final predictor matrix
    % take zscore to be able to compare coefficients across predictors
    if opt.zscore_predictors
        [X_full,Xmean,Xstd] = zscore(X(in_patch,:));
    else
        X_full = X(in_patch,:);
        Xmean = mean(X_full);
        Xstd = std(X_full);
    end

    % filter spikecounts to only include in patch times
    spikecounts = spikecounts_whole_session(in_patch,:);
    patch_num = patch_num(in_patch);

    %% remove cells that don't pass minimum firing rate cutoff

    T = size(spikecounts,1)*opt.tbin;
    N = sum(spikecounts);
    fr = N/T;
    keep_cell = fr>opt.min_fr;
    if opt.cortexOnly == 1
        keep_cell = keep_cell & strcmp(brain_region_rough,'PFC')';
    end
    spikecounts = spikecounts(:,keep_cell);
    good_cells = good_cells_all(keep_cell);
    brain_region_rough = brain_region_rough(keep_cell);
    depth_from_surface = depth_from_surface(keep_cell);

    %% Create fold indices (for cross validation)

    % split trials into groups (num groups = opt.numFolds)
    [trial,~,IC] = unique(patch_num);
    trial_grp = nan(size(trial));
    shift_by = 0; % to make sure equal numbers of trials end up in each fold
    for i = 1:numel(opt.rew_size)
        keep_this = rew_size_patch==opt.rew_size(i);
        trial_grp_this = repmat(circshift(1:opt.numFolds,shift_by),1,ceil(sum(keep_this)/opt.numFolds)*opt.numFolds);
        trial_grp(keep_this) = trial_grp_this(1:sum(keep_this));
        shift_by = shift_by-mod(sum(keep_this),opt.numFolds);
    end
    foldid = trial_grp(IC);
    
    %% compute PCA

    opt.tstart = 0;
    opt.tend = max(dat.sp.st)+opt.tbin;
    
    % compute firing rate mat
    fr_mat = calcFRVsTime(good_cells,dat,opt);

    % take zscore
    fr_mat_zscore = zscore(fr_mat,[],2); % z-score is across whole session including out-of-patch times - is this weird??

    % use whole session
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
    
    score_in_patch = score(in_patch,:);
    
    %% regress PCs on X
    
    beta_all = nan(size(X,2)+1,opt.num_pcs);
    
    % options for glmnet
    opt_glmnet = glmnetSet;
    opt_glmnet.alpha = opt.alpha; % alpha for elastic net
    
    for pcIdx = 1:opt.num_pcs
    
        y = score_in_patch(:,pcIdx);

        fit = cvglmnet(X_full,y,'gaussian',opt_glmnet,[],[],foldid);
        beta_all(:,pcIdx) = cvglmnetCoef(fit);
    end
    
    %% save results
    save(fullfile(paths.results,sprintf('%s',opt.session)),'opt',...
        'var_name','base_var','X_full','Xmean','Xstd','spikecounts','good_cells',...
        'trial_grp','foldid','bas_patch_stop','t_basis_patch_stop','bas_rew','t_basis_rew',...
        'anatomy','depth_from_surface','fr','good_cells_all','brain_region_rough',...
        'patch_num','rew_size_patch',...
        'beta_all','opt_glmnet','score_in_patch');

end
toc