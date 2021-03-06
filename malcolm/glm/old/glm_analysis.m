% script to fit GLM to all cells in an example session
% using MATLAB's lassoglm
% MGC 7/1/2020

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
paths.figs = fullfile(paths.figs,opt.session);
opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)

% basis functions for rewards
opt.nbasis = 5; % number of raised cosine basis functions to use
opt.basis_length = 1; % in seconds

% whether or not to zscore predictors
opt.zscore_predictors = true;

% minimum firing rate to keep neurons
opt.min_fr = 1;

% regularization
% *** NOTE: figure out how to get lambda for each session *** 
% opt.lambda = 0.0206; % regularization parameter; came from trying a bunch of values on an example cell (m80 0317 c368)
opt.alpha = 0.9; % weighting of L1 and L2 penalties in elastic net regularization

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized chunks

% threshold for calling a cell "significant" in encoding regressors
% (t-test full model vs mean over GLM cross validation folds)
opt.pval_thresh = 0.01;

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);

% binned spikecounts for each cell
t = dat.velt;
spikecounts = nan(numel(t),numel(good_cells));
for cIdx = 1:numel(good_cells)
    % spikecounts
    spike_t = dat.sp.st(dat.sp.clu==good_cells(cIdx));
    spikecounts(:,cIdx) = histc(spike_t,t);
end

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

% REWARDS
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
% time since last reward
t_since_last_rew = zeros(size(t));
for i = 2:numel(t)
    if rew_binary(i)
        t_since_last_rew(i) = 0;
    else
        t_since_last_rew(i) = t_since_last_rew(i-1)+opt.tbin;
    end
end
% DV from model 3
% DV3 = t_on_patch-1.7172*tot_rew; % this the fit for m80 0317
X_this = [t_on_patch' tot_rew' t_since_last_rew'];
A = [A, {X_this}];
grp_name = [grp_name,'DVs'];
var_name = [var_name,'TimeOnPatch','TotalRew','TimeSinceLastRew'];

% previous patch reward size
rew_size = mod(dat.patches(:,2),10);
rew_size_last = zeros(size(t));
rew_size_last(patch_num>1) = rew_size(patch_num(patch_num>1)-1);
X_this = rew_size_last';
A = [A,{X_this}];
grp_name = [grp_name,'RewSizeLast'];
var_name = [var_name,'RewSizeLast'];
    
%% all predictors
X = []; % predictor matrix
for i = 1:numel(A)
    X = [X A{i}];
end

% dropout each predictor individually
X_dropout = {};
for i = 1:numel(A)
    X_dropout{i} = [];
    for j = 1:numel(A)
        if j~=i
            X_dropout{i} = [X_dropout{i} A{j}];
        end
    end
end

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

% final spikecounts matrix
spikecounts_final = spikecounts(keep,:);

% filter out cells with low firing rate
mean_fr = sum(spikecounts_final)/(size(spikecounts_final,1)*opt.tbin);
spikecounts_final = spikecounts_final(:,mean_fr>=opt.min_fr);
Ncells = size(spikecounts_final,2);
depth = depth(mean_fr>=opt.min_fr);

%% Create fold indices (for cross validation)

% split trials into groups (num groups = opt.numFolds)
[trial,~,IC] = unique(patch_num_final);
trial_grp = repmat(1:opt.numFolds,1,ceil(numel(trial)/opt.numFolds)*opt.numFolds);
trial_grp = trial_grp(1:numel(trial));
foldid = trial_grp(IC);

%% Fit GLM to each cell
pb = ParforProgressbar(Ncells);
beta_all = nan(size(X,2)+1,Ncells);
pval = nan(Ncells,1);

% options for glmnet (taken from opt structure)
opt_glmnet = struct;
opt_glmnet.alpha = opt.alpha;
parfor cIdx = 1:Ncells
    
    X_this = X_final;
    y = spikecounts_final(:,cIdx);
    
    try
        % iterate over cross-validation folds
        log_llh_diff = nan(opt.numFolds,1);
        for fIdx = 1:opt.numFolds

            X_train = X_this(foldid~=fIdx,:);
            y_train = y(foldid~=fIdx);
            X_test = X_this(foldid==fIdx,:);
            y_test = y(foldid==fIdx);

            % fit params on train data
            fit = cvglmnet(X_train,y_train,'poisson',opt_glmnet,[],5);
            lambda_idx = find(fit.lambda==fit.lambda_1se);
            beta = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];       
            meanFR_train = nanmean(y_train);        

            % get log likelihoods on test data
            r_test = exp([ones(size(X_test,1),1) X_test] * beta);
            log_llh_model = nansum(r_test-y_test.*log(r_test)+log(factorial(y_test)))/sum(y_test);
            log_llh_mean = nansum(meanFR_train-y_test.*log(meanFR_train)+log(factorial(y_test)))/sum(y_test);

            log_llh_diff(fIdx) = log_llh_model-log_llh_mean;
        end
        % statistical test to see if model does better than mean
        [~,pval(cIdx)] = ttest(log_llh_diff);

        % fit parameters to full data  
        fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],5);
        lambda_idx = find(fit.lambda==fit.lambda_1se);
        beta_all(:,cIdx) = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];     
    catch
        fprintf('error: cell %d',cIdx);
    end
    
    pb.increment();
end

%% plot: histogram of p values

hfig = figure;
hfig.Name = sprintf('%s histogram of cross-validation pvalues',opt.session);
histogram(pval);
xlabel('cross-validation p-value: full model vs just mean');
ylabel('num cells');
title(opt.session,'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% plot: absolute regressor weights: all cells

hfig = figure;
hfig.Name = sprintf('%s boxplot of absolute regressor weights_all cells',opt.session);
boxplot(abs(beta_all(2:end,:)'));
xticklabels(var_name);
xtickangle(90);
ylabel('|beta|');
title(sprintf('%s 4uL patches: All cells',opt.session),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% plot: absolute regressor weights: significant cells only

hfig = figure;
hfig.Name = sprintf('%s boxplot of absolute regressor weights_significant only',opt.session);
boxplot(abs(beta_all(2:end,pval<=opt.pval_thresh)'));
xticklabels(var_name);
xtickangle(90);
ylabel('|beta|');
title(sprintf('%s 4uL patches: Significant cells',opt.session),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% plot: heatmap of absolute regressor weights, sorted by t_since_last_rew

z = beta_all(:,pval<=opt.pval_thresh);
absz = abs(z);
[~,sort_idx] = sort(absz(find(strcmp(var_name,'TimeSinceLastRew'))+1,:)); % sort by time since last reward regressor weight

hfig = figure('Position',[200 200 1000 500]);
hfig.Name = sprintf('%s heatmap of beta weights sorted by time since last rew',opt.session);
subplot(6,1,1:5);
imagesc(absz(2:end,sort_idx))
xticks([]);
yticks(1:15)
yticklabels(var_name)
title(opt.session,'Interpreter','none')
h = colorbar;
ylabel(h,'|beta|');

subplot(6,1,6);
imagesc(absz(1,sort_idx));
h = colorbar;
ylabel(h,'|beta|');
yticks(1);
yticklabels({'Intercept'});
xlabel('cells')

save_figs(paths.figs,hfig,'png');

%% plot: GLM pvalue and beta weights vs depth
hfig = figure('Position',[200 200 1600 800]);
hfig.Name = sprintf('%s beta weight vs depth (significant cells only)',opt.session);
sig = pval<=opt.pval_thresh;
counter = 0;

counter = counter+1;
subplot(3,6,counter); hold on;
my_scatter(depth,-log10(pval),'k',0.2)
plot(xlim,[-log10(opt.pval_thresh) -log10(opt.pval_thresh)],'r--','LineWidth',2);
xlabel('depth (from tip of probe) (um)');
ylabel('-log10(pvalue)');
nsig = sum(pval<=opt.pval_thresh);
title(sprintf('%d/%d significant (%0.1f%%)',nsig,Ncells,100*nsig/Ncells));

counter = counter+1;
subplot(3,6,counter); hold on;
y = beta_all(1,pval<=opt.pval_thresh);
my_scatter(depth(sig),y,'k',0.2);
plot(xlim,[0 0],'k-');
xlabel('depth (from tip of probe) (um)');
ylabel('beta');
title('Intercept');

for i = 1:numel(var_name)
    counter = counter+1;
    subplot(3,6,counter); hold on;
    y = beta_all(i+1,pval<=opt.pval_thresh);
    my_scatter(depth(sig),y,'k',0.2);
    plot(xlim,[0 0],'k-');
    xlabel('depth (from tip of probe) (um)');
    ylabel('beta');
    title(var_name{i});
end

save_figs(paths.figs,hfig,'png');

%% save pvalues

save(fullfile('GLM_pvalues',opt.session),'pval');
