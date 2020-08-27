% script to fit GLM to all cells in a given list of sessions
% using the glmnet package
% MGC 8/27/2020

% 24 Aug 2020: Extended reward kernels to 2 sec, cut off when new reward
% arrives
% QUESTION: should we do this cutoff or not?

% 25 Aug 2020: All reward sizes

paths = struct;
paths.data = 'H:\My Drive\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.glmnet = 'C:\code\glmnet_matlab';
addpath(genpath(paths.glmnet));
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_27Aug2020';
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

% basis functions for time since reward
opt.nbasis = 11; % number of raised cosine basis functions to use
opt.basis_length = 2; % in seconds; make sure basis functions divide evenly into 1 second intervals (makes analysis easier)
opt.cut_off_kernel_sequence_after_each_reward = true; % whether to cut off the sequence of kernels when a new reward arrives

% whether or not to zscore predictors
% glmnet handles this for us, so turning this off for now. 
% NOTE: glmnet standardizes X for the fit, then returns the coefficients on
% the original scale. So, to compare coefficients across variables within a
% single fit, re-standardize them by multiplying each coefficient by its
% corresponding variable's standard deviation
opt.zscore_predictors = false; 

% minimum firing rate to keep neurons
opt.min_fr = 1;

% regularization
opt.alpha = 0.9; % weighting of L1 and L2 penalties in elastic net regularization

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized fold, assigning (roughly) equal numbers of each reward size to each fold

opt.compute_full_vs_base_pval = false; % whether to run model comparison between full and base model using nested cross validation


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
        in_patch = ~isnan(patch_num);
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

    %% Fit GLM to each cell
    
    pb = ParforProgressbar(Ncells);
    beta_all = nan(size(X,2)+1,Ncells);
    pval_full_vs_base = nan(Ncells,1);

    % options for glmnet
    opt_glmnet = glmnetSet;
    opt_glmnet.alpha = opt.alpha; % alpha for elastic net
    parfor cIdx = 1:Ncells

        y = spikecounts(:,cIdx);

        try
            if opt.compute_full_vs_base_pval
                % iterate over cross-validation folds
                log_llh_diff = nan(opt.numFolds,1);
                for fIdx = 1:opt.numFolds

                    y_train = y(foldid~=fIdx);
                    y_test = y(foldid==fIdx);

                    if sum(y_train)>0 && sum(y_test)>0

                        % FULL MODEL
                        X_train = X_full(foldid~=fIdx,:);
                        X_test = X_full(foldid==fIdx,:);
                        fit = cvglmnet(X_train,y_train,'poisson',opt_glmnet,[],5);
                        beta = cvglmnetCoef(fit);     
                        r_test = exp([ones(size(X_test,1),1) X_test] * beta);
                        log_llh_full_model = nansum(r_test-y_test.*log(r_test)+log(factorial(y_test)))/sum(y_test);

                        % BASE MODEL
                        X_train = X_full(foldid~=fIdx,base_var==1);
                        X_test = X_full(foldid==fIdx,base_var==1);
                        fit = cvglmnet(X_train,y_train,'poisson',opt_glmnet,[],5);
                        beta = cvglmnetCoef(fit);    
                        r_test = exp([ones(size(X_test,1),1) X_test] * beta);
                        log_llh_base_model = nansum(r_test-y_test.*log(r_test)+log(factorial(y_test)))/sum(y_test);

                        log_llh_diff(fIdx) = log_llh_full_model-log_llh_base_model;
                    end
                end
                % statistical test to see if full model does better than base model
                [~,pval_full_vs_base(cIdx)] = ttest(log_llh_diff);
            end

            % fit parameters to full data  
            fit = cvglmnet(X_full,y,'poisson',opt_glmnet,[],[],foldid);
            beta_all(:,cIdx) = cvglmnetCoef(fit);
        catch
            fprintf('error: cell %d\n',cIdx);
        end

        pb.increment();
    end
    
    % save results
    save(fullfile(paths.results,sprintf('%s',opt.session)),'opt','beta_all','pval_full_vs_base',...
        'var_name','X_full','Xmean','Xstd','spikecounts','good_cells',...
        'trial_grp','foldid','bas','t_basis');

end