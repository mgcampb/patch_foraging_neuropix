% script to fit GLM to the PCs of the firing rate matrix
% using MATLAB's lassoglm
% MGC 7/1/2020

% 7/16/2020 modified to use glmnet, way faster

% TO DO:
% add more decision variables
% cross validation
% dropout analysis
% forward search
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
opt.session = '80_20200317'; % session to analyze
paths.figs = fullfile(paths.figs,opt.session,'PCs','sig_cells_only');
opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)
opt.smoothSigma_time = 0.1;

% num PCs to fit GLM to
opt.num_pcs = 10;

% basis functions for rewards
opt.nbasis = 5; % number of raised cosine basis functions to use
opt.basis_length = 1; % in seconds

% whether or not to zscore predictors
opt.zscore_predictors = true;

% regularization
% *** NOTE: figure out how to get lambda for each session *** 
opt.regularize = true;
opt.alpha = 0.9; % weighting of L1 (LASSO) and L2 (Ridge) penalties in elastic net regularization. alpha=1 => 100% LASSO (L1), which emphasizes sparseness
opt.cv_for_lambda = 10; % number of cv folds for estimating optimal lambda (regularization parameter)

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized chunks

% threshold for calling a cell "significant" in encoding regressors
% (t-test full model vs mean over GLM cross validation folds)
opt.pval_thresh = 0.01;

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);
t = dat.velt;

% get depth on probe for each cell
depth = nan(numel(good_cells),1);
[~, spikeDepths] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);
for cIdx = 1:numel(good_cells)
    depth(cIdx) = median(spikeDepths(dat.sp.clu==good_cells(cIdx)));
end

%% raised cosine basis for discrete events (rewards)

t_basis = 0:opt.tbin:opt.basis_length; % 1 second long
db = (max(t_basis) - min(t_basis))/(opt.nbasis-1);
c = min(t_basis):db:max(t_basis);
bas = nan(opt.nbasis,length(t_basis));
for k = 1:opt.nbasis
  bas(k,:) = (cos(max(-pi, min(pi,pi*(t_basis - c(k))/(db))) ) + 1) / 2;
end

hfig = figure; hold on;
hfig.Name = 'discrete event basis functions';
for i = 1:opt.nbasis
    plot(t_basis,bas(i,:));
end
xlabel('time (sec)');
title('discrete event basis functions');
save_figs(paths.figs,hfig,'png');

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

% model 1


keyboard

%% compute PCA

% time bins
opt.tstart = 0;
opt.tend = max(dat.sp.st);
tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

% compute firing rate mat
fr_mat = calcFRVsTime(good_cells,dat,opt);

% only keep "in-patch" times
fr_mat_in_patch = fr_mat(:,in_patch);
tbincent = tbincent(in_patch);

% take zscore
fr_mat_zscore = my_zscore(fr_mat_in_patch);

% pca on firing rate matrix
[coeffs,score,~,~,expl] = pca(fr_mat_zscore');

%% subselect data to fit GLM to

% large reward patches only:
rew_size = mod(dat.patches(:,2),10);
keep = false(size(t));
keep(in_patch) = rew_size(patch_num(in_patch))==4;
patch_num_final = patch_num(keep);

% final predictor matrix
% take zscore to be able to compare coefficients across predictors
if opt.zscore_predictors
    X_final = zscore(X(keep,:));
else
    X_final = X(keep,:);
end
X_dropout_final = {};
for i = 1:numel(X_dropout)
    if opt.zscore_predictors
        X_dropout_final{i} = zscore(X_dropout{i}(keep,:));
    else
        X_dropout_final{i} = X_dropout{i}(keep,:);
    end
end

% final score matrix (zscore each PC)
score_final = zscore(score(keep(in_patch),1:opt.num_pcs));

%% Create fold indices (for cross validation)

% split trials into groups (num groups = opt.numFolds)
[trial,~,IC] = unique(patch_num_final);
trial_grp = repmat(1:opt.numFolds,1,ceil(numel(trial)/opt.numFolds)*opt.numFolds);
trial_grp = trial_grp(1:numel(trial));
foldid = trial_grp(IC);

%% Fit GLM for each PC

tic
pb = ParforProgressbar(opt.num_pcs);

% options for glmnet (taken from opt structure)
opt_glmnet = struct;
opt_glmnet.alpha = opt.alpha;

beta_all = nan(size(X,2)+1,opt.num_pcs);
se_all = nan(size(beta_all));
R2 = nan(opt.num_pcs,1);
parfor pIdx = 1:opt.num_pcs
    
    if opt.regularize
        % use cross validation to find optimal lambda for this PC
        fit = cvglmnet(X_final,score_final(:,pIdx),'gaussian',opt_glmnet,[],[],foldid);
        lambda_idx = find(fit.lambda==fit.lambda_1se);
        beta_this = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];
    else
        beta_this = [ones(size(X_final,1),1) X_final]\score_final(:,pIdx);
    end
    beta_all(:,pIdx) = beta_this;
    
    % get standard errors
    X_this = [ones(size(X_final,1),1) X_final];    
    mu = X_this*beta_this;    
    y_this = score_final(:,pIdx);
    sigma2 = mean((y_this-mu).^2);
    W = diag(mu.^2/sigma2);   
    V = inv(X_this'*W*X_this);
    se_all(:,pIdx) = sqrt(diag(V));
    R2(pIdx) = 1-sigma2/mean((y_this-mean(y_this)).^2);

    pb.increment();
end
toc

%% plot: regressor weights: all pcs

hfig = figure('Position',[100 100 1200 1200]);
hfig.Name = sprintf('%s regressor weights_top %d PCs_4 uL patches_alpha=%0.1f',opt.session,opt.num_pcs,opt.alpha);
for i = 1:opt.num_pcs
    subplot(ceil(opt.num_pcs/2),2,i); hold on;
    errorbar(1:numel(var_name),beta_all(2:end,i)',se_all(2:end,i)','k.');
    title(sprintf('PC%d',i));
    xticks(1:numel(var_name));
    xticklabels(var_name);
    xtickangle(90);
    ylabel('beta');
    plot(xlim,[0 0],'k-');
    grid on;
end
% save_figs(paths.figs,hfig,'png');

%% plot: R2

hfig = figure('Position',[500 500 400 300]);
hfig.Name = sprintf('%s R2 for top %d PCs_4 uL patches_alpha=%0.1f',opt.session,opt.num_pcs,opt.alpha);
bar(1:opt.num_pcs,R2);
ylim([0 1]);
xlabel('PC');
ylabel('R^2');
title(opt.session,'Interpreter','none');
% save_figs(paths.figs,hfig,'png');
