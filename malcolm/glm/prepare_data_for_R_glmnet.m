% script to prepare data to fit GLMs on cluster with glmnet (R version)
% MGC 2/12/21

% based on fit_glm_all_sessions.m

restoredefaultpath;
paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes = 'C:\code\spikes';
addpath(genpath(paths.spikes));
paths.results = 'C:\data\patch_foraging_neuropix\GLM_input\20210212_original_vars';
if ~isfolder(paths.results)
    mkdir(paths.results);
end
paths.results_for_R = fullfile(paths.results,'for_R');
if ~isfolder(paths.results_for_R)
    mkdir(paths.results_for_R);
end

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
%%
% analysis opts
opt = struct;

opt.rew_size = [1 2 4];

opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)

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
for k = 5 % :opt.nbasis_rew
  bas_rew(k,:) = (cos(max(-pi, min(pi,pi*(t_basis_rew - c(k))/(db))  ) ) + 1) / 2;
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
    rew_size_all = mod(dat.patches(:,2),10);
    for i = 1:numel(dat.rew_size)
        which_patch = find(dat.patchCSL(:,1)<dat.rew_ts(i) & dat.patchCSL(:,3)>dat.rew_ts(i));
        if ~isempty(which_patch)
            dat.rew_size(i) = rew_size_all(which_patch);
        end
    end

    %% make predictors for this session
    % group by type (in A) for forward search and dropout analysis
    
    A = {}; % grouped by type for forward search and dropout
    grp_name = {'Intercept'};
    var_name = {'Intercept'};
    base_var = [1]; % whether or not to use each variable in the "base" model

    % SESSION TIME
    X_this = [t' t.^2'];
    A = [A, {X_this}];
    grp_name = [grp_name,'SessionTime'];
    var_name = [var_name,'SessionTime','SessionTime^2'];
    base_var = [base_var 1 1];

    % RUNNING SPEED
    % running speed and running speed squared
    X_this = [dat.vel' dat.vel'.^2];
    A = [A, {X_this}];
    grp_name = [grp_name, 'Speed'];
    var_name = [var_name,'Speed','Speed^2'];
    base_var = [base_var 1 1];

    % LICK RATE
    if ~all(isnan(dat.lick_ts))
        lickcounts = histc(dat.lick_ts,t)/opt.tbin;
        lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate/opt.tbin);
        X_this = lickrate;
        A = [A, {X_this}];
        grp_name = [grp_name,'LickRate'];
        var_name = [var_name,'LickRate'];
        base_var = [base_var 1];
    end
    
    % PATCH POSITION
    if isfield(dat,'patch_pos')
        dat.patch_pos(isnan(dat.patch_pos)) = -4;
        X_this = [dat.patch_pos' dat.patch_pos'.^2];
        A = [A, {X_this}];
        grp_name = [grp_name,'Position'];
        var_name = [var_name,'Position','Position^2'];
        base_var = [base_var 1 1];
    end

    % iterate over reward sizes
    for rIdx = 1:numel(opt.rew_size)

        rew_size_this = opt.rew_size(rIdx);
        
        % TIME SINCE PATCH STOP (patch onset)
        patch_stop_binary = histc(dat.patchCSL(rew_size_all==rew_size_this,2),t);
        patch_stop_conv = nan(numel(patch_stop_binary),opt.nbasis_patch_stop);
        for i = 1:opt.nbasis_patch_stop
            conv_this = conv(patch_stop_binary,bas_patch_stop(i,:));
            patch_stop_conv(:,i) = conv_this(1:numel(patch_stop_binary));
        end
        X_this = patch_stop_conv;
        A = [A, {X_this}];
        grp_name = [grp_name,sprintf('TimeSincePatchStop_%uL',rew_size_this)];
        for i = 1:opt.nbasis_patch_stop
            var_name = [var_name,sprintf('PatchStopKern%d_%duL',i,rew_size_this)];
        end
        base_var = [base_var ones(1,opt.nbasis_patch_stop)];

        % TIME SINCE REWARD KERNELS
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
        A = [A, {X_this}];
        grp_name = [grp_name,sprintf('TimeSinceRewardKernels_%uL',rew_size_this)];
        for i = 1:opt.nbasis_rew
            var_name = [var_name,sprintf('RewKern%d_%duL',i,rew_size_this)];
        end
        base_var = [base_var zeros(1,opt.nbasis_rew)];


        % DECISION VARIABLES (DVs)
        % time on patch
        t_on_patch = zeros(size(t));
        t_on_patch(in_patch) = t(in_patch) - dat.patchCSL(patch_num(in_patch),2)';
        t_on_patch(~ismember(patch_num,find(rew_size_all==rew_size_this)))=0;
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
        t_since_rew(~ismember(patch_num,find(rew_size_all==rew_size_this)))=0;
        X_this = [t_on_patch' tot_rew' t_since_rew'];
        A = [A, {X_this}];
        grp_name = [grp_name,sprintf('DVs_%uL',rew_size_this)];
        var_name = [var_name,sprintf('TimeOnPatch_%duL',rew_size_this),sprintf('TotalRew_%duL',rew_size_this),sprintf('TimeSinceRew_%duL',rew_size_this)];
        base_var = [base_var 0 0 0];
    end

    % CONCATENATE ALL PREDICTORS
    X = []; % predictor matrix
    for i = 1:numel(A)
        X = [X A{i}];
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

    %% remove cells that don't pass minimum firing rate cutoff

    T = size(spikecounts,1)*opt.tbin;
    %N = sum(spikecounts);
    Nbin = size(spikecounts,1);
    N = nan(3,size(spikecounts,2));
    N(1,:) = sum(spikecounts(1:floor(Nbin/3),:));
    N(2,:) = sum(spikecounts(floor(Nbin/3):floor(2*Nbin/3),:));
    N(3,:) = sum(spikecounts(floor(2*Nbin/3):end,:));
    fr = N/T;
    keep_cell = mean(fr)>opt.min_fr & all(fr>opt.min_fr/2);
    spikecounts = spikecounts(:,keep_cell);
    good_cells = good_cells_all(keep_cell);
    brain_region_rough = brain_region_rough(keep_cell);
    depth_from_surface = depth_from_surface(keep_cell);
    % anatomy = anatomy_all(keep_cell,:));
    Ncells = numel(good_cells);

    %% Create fold indices (for cross validation)

    % split trials into groups (num groups = opt.numFolds)
    [trial,~,IC] = unique(patch_num(in_patch));
    trial_grp = nan(size(trial));
    shift_by = 0; % to make sure equal numbers of trials end up in each fold
    for i = 1:numel(opt.rew_size)
        keep_this = rew_size_all==opt.rew_size(i);
        trial_grp_this = repmat(circshift(1:opt.numFolds,shift_by),1,ceil(sum(keep_this)/opt.numFolds)*opt.numFolds);
        trial_grp(keep_this) = trial_grp_this(1:sum(keep_this));
        shift_by = shift_by-mod(sum(keep_this),opt.numFolds);
    end
    foldid = trial_grp(IC);
    
    %% save results
    save(fullfile(paths.results,sprintf('%s',opt.session)),'opt',...
        'var_name','base_var','X_full','Xmean','Xstd','spikecounts','good_cells',...
        'trial_grp','foldid','bas_patch_stop','t_basis_patch_stop','bas_rew','t_basis_rew',...
        'anatomy','depth_from_surface','fr','good_cells_all','brain_region_rough');
    alpha = opt.alpha;
    save(fullfile(paths.results_for_R,sprintf('%s',opt.session)),'alpha','X_full','spikecounts','good_cells','foldid','base_var');

end
toc