% script to fit GLM to all cells in a given list of sessions
% using the glmnet package
% MGC 8/27/2020

% 24 Aug 2020: Extended reward kernels to 2 sec, cut off when new reward
% arrives
% QUESTION: should we do this cutoff or not?

% 25 Aug 2020: All reward sizes

% NOTE: currently only analyzes sessions with histology

% NOTE: There is no patch leave buffer here

paths = struct;
paths.data = 'H:\My Drive\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.glmnet = 'C:\code\glmnet_matlab';
addpath(genpath(paths.glmnet));
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_1Sep2020_PCA_MOs';
if exist(paths.results,'dir')~=7
    mkdir(paths.results);
end

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

% analysis opts
opt = struct;

opt.rew_size = [1 2 4];

opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)
opt.smoothSigma_time  = 0.1; % for smoothing firing rate (prior to computing PCA)

% basis functions for time since reward
opt.nbasis = 11; % number of raised cosine basis functions to use
opt.basis_length = 2; % in seconds; make sure basis functions divide evenly into 1 second intervals (makes analysis easier)
opt.cut_off_kernel_sequence_after_each_reward = true; % whether to cut off the sequence of kernels when a new reward arrives

% whether or not to zscore predictors
% NOTE: glmnet standardizes X for the fit, then returns the coefficients on
% the original scale. So, if this option is FALSE, to compare coefficients across variables within a
% single fit, re-standardize them by multiplying each coefficient by its
% corresponding variable's standard deviation
opt.zscore_predictors = true; 

% minimum firing rate to keep neurons
opt.min_fr = 1;

% regularization
opt.alpha = 0.9; % weighting of L1 and L2 penalties in elastic net regularization

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized fold, assigning (roughly) equal numbers of each reward size to each fold

% which brain region to analyze
opt.brain_region = 'MOs';

% how many PCs to analyze
opt.num_pcs = 10;
opt.min_num_cells_for_pca = 10;

% only takes within patch times up to this amount before patch leave
opt.patch_leave_buffer = 0.5; % in seconds

%% raised cosine basis for time since reward

t_basis = 0:opt.tbin:opt.basis_length;
db = (max(t_basis) - min(t_basis))/(opt.nbasis-1);
c = min(t_basis):db:max(t_basis);
bas = nan(opt.nbasis,length(t_basis));
for k = 1:opt.nbasis
  bas(k,:) = (cos(max(-pi, min(pi,pi*(t_basis - c(k))/(db))) ) + 1) / 2;
end

%%
for session_idx = 1:numel(session_all)
    
    opt.session = session_all{session_idx};
    fprintf('Analyzing session %d/%d: %s\n',session_idx,numel(session_all),opt.session);

    %% load data    
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);

    if isfield(dat,'anatomy')
        anatomy_all = dat.anatomy.cell_labels;
    elseif ~strcmp(opt.brain_region,'all')
        continue;
    end
    
    % only keep cells from given brain region
    if ~strcmp(opt.brain_region,'all')
        keep = contains(anatomy_all{:,2},opt.brain_region);
        good_cells_all = good_cells_all(keep);
        anatomy_all = anatomy_all(keep,:);
        if numel(good_cells_all)<opt.min_num_cells_for_pca
            continue
        end
    end

    %% compute binned spikecounts for each cell
    
    t = dat.velt;
    spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
    for cIdx = 1:numel(good_cells_all)
        spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
        spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
    end

    %% get size of each reward
    
    dat.rew_size = nan(size(dat.rew_ts));
    rew_size_all = mod(dat.patches(:,2),10);
    for i = 1:numel(dat.rew_size)
        which_patch = find(dat.patchCSL(:,1)<dat.rew_ts(i) & dat.patchCSL(:,3)>dat.rew_ts(i));
        if ~isempty(which_patch)
            dat.rew_size(i) = rew_size_all(which_patch);
        end
    end
    
    %% get in-patch times (with buffer)
    
    in_patch = false(size(t));
    for i = 1:size(dat.patchCSL,1)
        in_patch(t>=dat.patchCSL(i,2) & t<=dat.patchCSL(i,3)-opt.patch_leave_buffer) = true;
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
    lickcounts = histc(dat.lick_ts,t)/opt.tbin;
    lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate/opt.tbin);
    X_this = lickrate;
    A = [A, {X_this}];
    grp_name = [grp_name,'LickRate'];
    var_name = [var_name,'LickRate'];
    base_var = [base_var 1];

    % iterate over reward sizes
    for rIdx = 1:numel(opt.rew_size)

        rew_size_this = opt.rew_size(rIdx);

        % TIME SINCE REWARD KERNELS
        rew_binary = histc(dat.rew_ts(dat.rew_size==rew_size_this),t);
        rew_conv = nan(numel(rew_binary),opt.nbasis);
        % convolve with basis functions
        for i = 1:opt.nbasis
            conv_this = conv(rew_binary,bas(i,:));
            rew_conv(:,i) = conv_this(1:numel(rew_binary));
        end
        if opt.cut_off_kernel_sequence_after_each_reward
            if opt.basis_length>1
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
        for i = 1:opt.nbasis
            var_name = [var_name,sprintf('Kern%d_%duL',i,rew_size_this)];
        end
        base_var = [base_var zeros(1,opt.nbasis)];


        % DECISION VARIABLES (DVs)
        % get patch num for each patch
        patch_num = nan(size(t));
        for i = 1:size(dat.patchCSL,1)
            % include one time bin before patch stop to catch the first reward
            patch_num(t>=(dat.patchCSL(i,2)-opt.tbin) & t<=dat.patchCSL(i,3)) = i;
        end
        patch_num(~in_patch) = nan;
        % time on patch
        t_on_patch = zeros(size(t));
        t_on_patch(in_patch) = t(in_patch) - dat.patchCSL(patch_num(in_patch),2)';
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
    N = sum(spikecounts);
    fr = N/T;
    spikecounts = spikecounts(:,fr>opt.min_fr);
    good_cells = good_cells_all(fr>opt.min_fr);
    anatomy = anatomy_all(fr>opt.min_fr,:);
    Ncells = numel(good_cells);
    
    %% compute PCA

    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);

    % compute firing rate mat
    fr_mat = calcFRVsTime(good_cells,dat,opt);
    fr_mat = [fr_mat fr_mat(:,end)];

    % only keep "in-patch" times
    fr_mat_in_patch = fr_mat(:,in_patch);

    % take zscore
    fr_mat_zscore = zscore(fr_mat_in_patch,[],2);

    % pca on firing rate matrix
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');

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

    %% Fit GLM to each cell
    
    beta_all = nan(size(X,2)+1,opt.num_pcs);

    % options for glmnet
    opt_glmnet = glmnetSet;
    opt_glmnet.alpha = opt.alpha; % alpha for elastic net
    
    for pIdx = 1:opt.num_pcs    
        % use cross validation to find optimal lambda for this PC
        fit = cvglmnet(X_full,score(:,pIdx),'gaussian',opt_glmnet,[],[],foldid);
        beta_all(:,pIdx) = cvglmnetCoef(fit);
    end
    
    % save results
    save(fullfile(paths.results,sprintf('%s',opt.session)),'opt','beta_all',...
        'var_name','X_full','Xmean','Xstd','score','expl','good_cells',...
        'trial_grp','foldid','bas','t_basis','anatomy');

end