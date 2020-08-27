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
opt.session = '80_20200315'; % session to analyze
paths.figs = fullfile(paths.figs,opt.session,'PCs');
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
opt.alpha = 0.9; % weighting of L1 (LASSO) and L2 (Ridge) penalties in elastic net regularization. alpha=1 => 100% LASSO (L1), which emphasizes sparseness

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized chunks

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

% hfig = figure; hold on;
% hfig.Name = 'discrete event basis functions';
% for i = 1:opt.nbasis
%     plot(t_basis,bas(i,:));
% end
% xlabel('time (sec)');
% title('discrete event basis functions');
% save_figs(paths.figs,hfig,'png');

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

% regressors for the three models (1,2,3); plus full model with all regressors
X_mdl = cell(4,1);

% model 1
X_mdl{1} = X_all(:,~strcmp(var_name,'TotalRew') & ~strcmp(var_name,'TimeSinceLastRew'));

% model 2
X_mdl{2} = X_all(:,~strcmp(var_name,'TotalRew') & ~strcmp(var_name,'TimeOnPatch'));

% model 3
X_mdl{3} = X_all(:,~strcmp(var_name,'TimeSinceLastRew'));

% full model
X_mdl{4} = X_all;


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
    for mdl_idx = 1:numel(X_mdl)
        X_mdl{mdl_idx} = zscore(X_mdl{mdl_idx}(keep,:));
    end
else
    for mdl_idx = 1:numel(X_mdl)
        X_mdl{mdl_idx} = X_mdl{mdl_idx}(keep,:);
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
% pb = ParforProgressbar(opt.num_pcs);

% options for glmnet (taken from opt structure)
opt_glmnet = struct;
opt_glmnet.alpha = opt.alpha;

err_mean = nan(opt.num_pcs,numel(X_mdl)); % mean MSE (mean squared error) over test folds
err_se = nan(opt.num_pcs,numel(X_mdl)); % estimate of standard error of MSE over test folds
for pIdx = 1:opt.num_pcs

    % EVENTUALLY: NESTED CROSS VALIDATION

    for mdl_idx = 1:numel(X_mdl)
        fit = cvglmnet(X_mdl{mdl_idx},score_final(:,pIdx),'gaussian',opt_glmnet,[],[],foldid);
        lambda_idx = find(fit.lambda==fit.lambda_1se);
        err_mean(pIdx,mdl_idx) = fit.cvm(lambda_idx);
        err_se(pIdx,mdl_idx) = fit.cvsd(lambda_idx);
%         beta = X_mdl{mdl_idx}\score_final(:,pIdx);
%         y_pred = X_mdl{mdl_idx}*beta;
%         err_mean(pIdx,mdl_idx) = mean((score_final(:,pIdx)-y_pred).^2);
    end
%    pb.increment();
end
toc

%% plot: MSE all models all PCs

hfig = figure('Position',[100 100 1200 1200]);
hfig.Name = sprintf('%s MSE_top %d PCs_4 uL patches_alpha=%0.1f',opt.session,opt.num_pcs,opt.alpha);
for i = 1:opt.num_pcs
    subplot(ceil(opt.num_pcs/2),2,i); hold on;
    errorbar(1:size(err_mean,2),err_mean(i,:),err_se(i,:),'k.');
    title(sprintf('PC%d',i));
    xticks(1:numel(var_name));
    xticklabels({'Mdl1','Mdl2','Mdl3','All'});
    xtickangle(90);
    ylabel('MSE');
    xlim([0.5 4.5]);
    plot(xlim,[0 0],'k-');
    grid on;
end
save_figs(paths.figs,hfig,'png');