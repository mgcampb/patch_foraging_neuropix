% script to fit Poisson GLM to the all cells in a session using glmnet

% TO DO:
% add more decision variables
% percent deviance explained for each variable

paths = struct;

paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes_repo = 'C:\code\spikes';
addpath(genpath(paths.spikes_repo));
paths.glmnet = 'C:\code\glmnet_matlab';
addpath(genpath(paths.glmnet));

% analysis opts
opt = struct;
opt.session = '78_20200311'; % session to analyze
paths.figs = fullfile(paths.figs,opt.session);
opt.tbin = 0.02; % in seconds
opt.min_fr = 1; % cells must have at least this avg firing rate within patches to be kept
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)
opt.smoothSigma_time = 0.1;
opt.pval_thresh = 0.05; % for selecting "significant" GLM cells

% basis functions for rewards
opt.nbasis = 5; % number of raised cosine basis functions to use
opt.basis_length = 1; % in seconds

% whether or not to zscore predictors
opt.zscore_predictors = true;

% regularization
% *** NOTE: figure out how to get lambda for each session *** 
opt.alpha = 0; % weighting of L1 (LASSO) and L2 (Ridge) penalties in elastic net regularization. alpha=1 => 100% LASSO (L1), which emphasizes sparseness
% setting alpha to 0 here (100% Ridge) because we are not trying to perform
% variable selection, just get the best model fit

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized chunks

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);
t = dat.velt;

% binned spikecounts for each cell
spikecounts = nan(numel(t),numel(good_cells));
for cIdx = 1:numel(good_cells)
    % spikecounts
    spike_t = dat.sp.st(dat.sp.clu==good_cells(cIdx));
    spikecounts(:,cIdx) = histc(spike_t,t);
end

%% raised cosine basis for discrete events (rewards)

t_basis = 0:opt.tbin:opt.basis_length; % 1 second long
db = (max(t_basis) - min(t_basis))/(opt.nbasis-1);
c = min(t_basis):db:max(t_basis);
bas = nan(opt.nbasis,length(t_basis));
for k = 1:opt.nbasis
  bas(k,:) = (cos(max(-pi, min(pi,pi*(t_basis - c(k))/(db))) ) + 1) / 2;
end

%% make predictors for this session
% group by type (in A) for forward search and dropout analysis

% Q: Where to put reward size information?

A = {}; % grouped by type for forward search and dropout
grp_name = {};
var_name = {};

% RUNNING SPEED
% running speed and running speed squared
X_this = [dat.vel' dat.vel'.^2];
A = [A, {X_this}];
grp_name = [grp_name, 'Speed'];
var_name = [var_name,'Speed','Speed^2'];

% LICK RATE
lickcounts = histc(dat.lick_ts,t)/opt.tbin;
lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate/opt.tbin);
X_this = lickrate;
A = [A, {X_this}];
grp_name = [grp_name,'LickRate'];
var_name = [var_name,'LickRate'];

% % REWARDS
rew_binary = histc(dat.rew_ts,t);
rew_conv = nan(numel(rew_binary),opt.nbasis);
% convolve with basis functions
for i = 1:opt.nbasis
    conv_this = conv(rew_binary,bas(i,:));
    rew_conv(:,i) = conv_this(1:numel(rew_binary));
end
X_this = rew_conv;
A = [A, {X_this}];
grp_name = [grp_name,'Rewards'];
for i = 1:opt.nbasis
    var_name = [var_name,sprintf('Rewards%d',i)];
end

% SESSION TIME
X_this = [t' t.^2'];
A = [A, {X_this}];
grp_name = [grp_name,'SessionTime'];
var_name = [var_name,'SessionTime','SessionTime^2'];

% DECISION VARIABLES
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
% total rewards on patch
tot_rew = zeros(size(t));
for i = 1:size(dat.patchCSL,1)
    tot_rew(patch_num==i) = cumsum(rew_binary(patch_num==i));
end
% DV from model 3
DV3 = t_on_patch-1.7172*tot_rew; % this is the fit for m80 0317
% time since last reward
t_since_last_rew = zeros(size(t));
for i = 2:numel(t)
    if rew_binary(i)
        t_since_last_rew(i) = 0;
    else
        t_since_last_rew(i) = t_since_last_rew(i-1)+opt.tbin;
    end
end
X_this = [t_on_patch' tot_rew' t_since_last_rew'];
A = [A, {X_this}];
grp_name = [grp_name,'DVs'];
var_name = [var_name,'TimeOnPatch','TotalRew','TimeSinceLastRew'];

%% create predictors for models 1, 2, and 3

% full model
X_all = []; % predictor matrix
for i = 1:numel(A)
    X_all = [X_all A{i}];
end

% regressors for the three models (1,2,3); plus full model with all regressors
X_mdl = cell(5,1);

% base model
X_mdl{1} = X_all(:,~strcmp(var_name,'TimeOnPatch') & ~strcmp(var_name,'TotalRew') & ...
    ~strcmp(var_name,'TimeSinceLastRew'));

% model 1
X_mdl{2} = X_all(:,~strcmp(var_name,'TotalRew') & ~strcmp(var_name,'TimeSinceLastRew'));

% model 2
X_mdl{3} = X_all(:,~strcmp(var_name,'TimeOnPatch') & ~strcmp(var_name,'TotalRew'));

% model 3
X_mdl{4} = X_all(:,~strcmp(var_name,'TimeSinceLastRew'));

% full model
X_mdl{5} = X_all;

%% subselect data to fit GLM to

% large reward patches only:
rew_size = mod(dat.patches(:,2),10);
keep = false(size(t));
keep(in_patch) = rew_size(patch_num(in_patch))==4;
patch_num_final = patch_num(keep);

% final predictor matrix
% take zscore to be able to compare coefficients across predictors
if opt.zscore_predictors
    for mdl_idx = 1:numel(X_mdl)
        X_mdl{mdl_idx} = zscore(X_mdl{mdl_idx}(keep,:));
    end
else
    for mdl_idx = 1:numel(X_mdl)
        X_mdl{mdl_idx} = X_mdl{mdl_idx}(keep,:);
    end
end

% final spikecounts matrix
spikecounts_final = spikecounts(keep,:);

% filter out cells with low firing rate
mean_fr = sum(spikecounts_final)/(size(spikecounts_final,1)*opt.tbin);
spikecounts_final = spikecounts_final(:,mean_fr>=opt.min_fr);
Ncells = size(spikecounts_final,2);

%% Create fold indices (for cross validation)

% split trials into groups (num groups = opt.numFolds)
[trial,~,IC] = unique(patch_num_final);
trial_grp = repmat(1:opt.numFolds,1,ceil(numel(trial)/opt.numFolds)*opt.numFolds);
trial_grp = trial_grp(1:numel(trial));
foldid = trial_grp(IC);

%% Fit GLM for each PC

tic
pb = ParforProgressbar(Ncells);

% options for glmnet (taken from opt structure)
opt_glmnet = struct;
opt_glmnet.alpha = opt.alpha;

llh_all = nan(Ncells,numel(X_mdl),opt.numFolds); % mean MSE (mean squared error) over test folds
parfor cIdx = 1:Ncells

    % EVENTUALLY: NESTED CROSS VALIDATION
    y = spikecounts_final(:,cIdx);
    llh_this = nan(numel(X_mdl),opt.numFolds);
    for mdl_idx = 1:numel(X_mdl)
        X = X_mdl{mdl_idx};
        for fIdx = 1:opt.numFolds
            % split into train and test data
            Xtrain = X(foldid~=fIdx,:);
            ytrain = y(foldid~=fIdx);
            Xtest = X(foldid==fIdx,:);
            ytest = y(foldid==fIdx);
            
            % fit model on train data (using CV within test data to select
            % regularization parameter)
            fit = cvglmnet(Xtrain,ytrain,'poisson',opt_glmnet,[],5);
            lambda_idx = find(fit.lambda==fit.lambda_1se);
            beta = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];
            
            % get log likelihoods on test data
            meanFR_train = nanmean(ytrain);
            r_test = exp([ones(size(Xtest,1),1) Xtest] * beta);
            log_llh_model = -nansum(r_test-ytest.*log(r_test)+log(factorial(ytest)))/sum(ytest);
            log_llh_mean = -nansum(meanFR_train-ytest.*log(meanFR_train)+log(factorial(ytest)))/sum(ytest);
            llh_this(mdl_idx,fIdx) = log_llh_model-log_llh_mean;
        end
    end    
    llh_all(cIdx,:,:) = llh_this;
    pb.increment();
end
toc

%% find significant cells and take mean llh_diff

[~,pval] = ttest(squeeze(llh_all(:,end,:))');
lbar = mean(llh_all(pval<=opt.pval_thresh,:,:),3);

hfig = figure('Position',[50 50 1200 1200]);
hfig.Name = sprintf('%s single cell model comparison (LLH_model - LLH_mean)',opt.session);
counter = 1;
ax_labels = {'base model','model1','model2','model3','full model'};
for i = 1:numel(X_mdl)-1
    for j = i+1:numel(X_mdl)
        subplot(numel(X_mdl)-1,numel(X_mdl)-1,(i-1)*(numel(X_mdl)-1)+j-1);
        my_scatter(lbar(:,i),lbar(:,j),'k',0.2);
        pval_this = signrank(lbar(:,i),lbar(:,j));
        title(sprintf('p = %0.3f',pval_this));
        xlabel(ax_labels{i});
        ylabel(ax_labels{j});
        axis equal
        grid on;
        refline(1,0);
        counter = counter+1;
    end
end
save_figs(paths.figs,hfig,'png')