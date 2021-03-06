% script to fit GLM to all cells in an example session
% using MATLAB's lassoglm
% MGC 7/1/2020

% TO DO:
% add more decision variables
% dropout analysis
% percent deviance explained for each variable

% 24 Aug 2020: Extended reward kernels to 2 sec, cut off when new reward
% arrives

% 25 Aug 2020: All reward sizes

paths = struct;
paths.data = 'H:\My Drive\processed_neuropix_data';
paths.figs_root = 'C:\figs\patch_foraging_neuropix\glm_time_since_reward_kernels_allRewSize';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes_repo = 'C:\code\spikes';
addpath(genpath(paths.spikes_repo));
paths.glmnet = 'C:\code\glmnet_matlab';
addpath(genpath(paths.glmnet));
paths.hgrk = 'C:\code\HGRK_analysis_tools'; % Hyunggoo's code
addpath(genpath(paths.hgrk));
paths.results = 'GLM_output\allRewSize';

% analysis opts
opt = struct;

% what session to analyze:
opt.session = '80_20200317';
opt.rew_size = [1 2 4];
paths.figs = fullfile(paths.figs_root,opt.session);

opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)
opt.smooth_sigma_fr = 0.1; % for smoothing firing rate traces

% basis functions for time since reward
opt.nbasis = 11; % number of raised cosine basis functions to use
opt.basis_length = 2; % in seconds; make sure basis functions divide evenly into 1 second intervals (makes analysis easier)

% whether or not to zscore predictors
opt.zscore_predictors = true;

% minimum firing rate to keep neurons
opt.min_fr = 1;

% regularization
opt.alpha = 0.9; % weighting of L1 and L2 penalties in elastic net regularization

% cross validation over trials
opt.numFolds = 5; % split up trials into (roughly) equally sized chunks

% threshold for calling a cell "significant" in encoding regressors
% (t-test full model vs mean over GLM cross validation folds)
opt.pval_thresh = 0.05; % 0.01;

opt.compute_cross_val_pval = false;

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

%% get size of each reward
dat.rew_size = nan(size(dat.rew_ts));
rew_size_all = mod(dat.patches(:,2),10);
for i = 1:numel(dat.rew_size)
    which_patch = find(dat.patchCSL(:,1)<dat.rew_ts(i) & dat.patchCSL(:,3)>dat.rew_ts(i));
    if ~isempty(which_patch)
        dat.rew_size(i) = rew_size_all(which_patch);
    end
end

%% filter out cells with low firing rate

mean_fr = sum(spikecounts)/(size(spikecounts,1)*opt.tbin);
spikecounts_filt = spikecounts(:,mean_fr>=opt.min_fr);
good_cells_filt = good_cells(mean_fr>=opt.min_fr);
Ncells = numel(good_cells_filt);

hfig = figure; hold on;
hfig.Name = sprintf('%s mean FR histogram',opt.session);
histogram(mean_fr,0:1:max(mean_fr));
plot([opt.min_fr opt.min_fr],ylim,'r--');
xlabel('Mean firing rate (Hz)');
ylabel('Num cells');
title(sprintf('%s: %d/%d neurons > %0.1f Hz',opt.session,Ncells,numel(mean_fr),opt.min_fr),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% raised cosine basis for discrete events (rewards)

t_basis = 0:opt.tbin:opt.basis_length;
db = (max(t_basis) - min(t_basis))/(opt.nbasis-1);
c = min(t_basis):db:max(t_basis);
bas = nan(opt.nbasis,length(t_basis));
for k = 1:opt.nbasis
  bas(k,:) = (cos(max(-pi, min(pi,pi*(t_basis - c(k))/(db))) ) + 1) / 2;
end

hfig = figure; hold on;
hfig.Name = 'reward basis functions';
for i = 1:opt.nbasis
    plot(t_basis,bas(i,:));
end
xlabel('time (sec)');
title('reward basis functions');
save_figs(paths.figs,hfig,'png');

%% make predictors for this session
% group by type (in A) for forward search and dropout analysis

% Q: Where to put reward size information?

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
    if opt.basis_length>1
        % cut off kernels when new reward comes
        [~,idx]=max(rew_conv>0,[],2);
        for i = 1:size(rew_conv,1)
            rew_conv(i,idx(i)+2:end) = 0;
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

%% subselect data to fit GLM to

% final predictor matrix
% take zscore to be able to compare coefficients across predictors
if opt.zscore_predictors
    X_full = zscore(X(in_patch,:));
else
    X_full = X(in_patch,:);
end

% final spikecounts matrix
spikecounts_final = spikecounts_filt(in_patch,:);

% further filter by firing rate
T = size(spikecounts_final,1)*opt.tbin;
N = sum(spikecounts_final);
fr = N/T;
spikecounts_filt = spikecounts_filt(:,fr>opt.min_fr);
spikecounts_final = spikecounts_final(:,fr>opt.min_fr);
good_cells_filt = good_cells_filt(fr>opt.min_fr);
Ncells = numel(good_cells_filt);

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
pval = nan(Ncells,1);

% options for glmnet (taken from opt structure)
opt_glmnet = struct;
opt_glmnet.alpha = opt.alpha;
parfor cIdx = 1:Ncells
    
    y = spikecounts_final(:,cIdx);
    
    
    try
        if opt.compute_cross_val_pval
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
                    lambda_idx = find(fit.lambda==fit.lambda_1se);
                    beta = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];       
                    r_test = exp([ones(size(X_test,1),1) X_test] * beta);
                    log_llh_full_model = nansum(r_test-y_test.*log(r_test)+log(factorial(y_test)))/sum(y_test);

                    % BASE MODEL
                    X_train = X_full(foldid~=fIdx,base_var==1);
                    X_test = X_full(foldid==fIdx,base_var==1);
                    fit = cvglmnet(X_train,y_train,'poisson',opt_glmnet,[],5);
                    lambda_idx = find(fit.lambda==fit.lambda_1se);
                    beta = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];       
                    r_test = exp([ones(size(X_test,1),1) X_test] * beta);
                    log_llh_base_model = nansum(r_test-y_test.*log(r_test)+log(factorial(y_test)))/sum(y_test);

                    log_llh_diff(fIdx) = log_llh_full_model-log_llh_base_model;
                end
            end
            % statistical test to see if model does better than mean
            [~,pval(cIdx)] = ttest(log_llh_diff);
        end

        % fit parameters to full data  
        fit = cvglmnet(X_full,y,'poisson',opt_glmnet,[],[],foldid);
        lambda_idx = find(fit.lambda==fit.lambda_1se);
        beta_all(:,cIdx) = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];     
    catch
        fprintf('error: cell %d\n',cIdx);
    end
    
    pb.increment();
end
% save results
save(fullfile(paths.results,sprintf('%s',opt.session)),'beta_all','pval');

%% plot: histogram of p values

%sig_cells = pval<=opt.pval_thresh;
sig_cells = true(size(pval));
good_cells_sig = good_cells_filt(sig_cells);
beta_sig = beta_all(:,sig_cells);

hfig = figure;
hfig.Name = sprintf('%s histogram of cross-validation pvalues',opt.session);
histogram(pval,0:0.01:1); hold on;
plot([opt.pval_thresh opt.pval_thresh],ylim,'r--');
xlabel('cross-validation p-value: full model vs base model');
ylabel('num cells');
title(sprintf('%s: %d/%d significant cells',opt.session,sum(sig_cells),Ncells),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% plot: regressor weights (significant cells only)

hfig = figure;
hfig.Name = sprintf('%s boxplot of regressor weights_pval_thresh=%0.2f',opt.session,opt.pval_thresh);
boxplot(beta_sig(2:end,:)'); hold on;
plot(xlim,[0 0],'k-');
xticklabels(var_name);
xtickangle(90);
ylabel('beta');
ylim([-1 1]);
title(sprintf('%s: %d significant cells',opt.session,sum(sig_cells)),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

% %% Characterize activity patterns in time since reward
% 
% hfig = figure('Position',[300 300 1600 500]);
% hfig.Name = sprintf('%s %duL time since reward coding',opt.session,opt.rew_size);
% colormap(jet);
% 
% % Kernels only
% tkern = 0:opt.tbin:opt.basis_length;
% rew_kern = contains(var_name,'Kern');
% beta_sig_kern = beta_sig(1+find(rew_kern),:);
% ypred = exp(bas'*beta_sig_kern);
% [~,max_idx] = max(abs(ypred-1));
% [~,sort_idx] = sort(max_idx);
% 
% subplot(1,4,1);
% imagesc(tkern,1:size(ypred,2),ypred(:,sort_idx)');
% cb = colorbar;
% xlabel('Time since reward');
% ylabel('Neuron, sorted by time of peak absolute signal change');
% title('A. Model fit (kernels only)')
% ylabel(cb,'Multiplicative firing rate change');
% 
% % Kernels plus linear component
% idx = [find(rew_kern) find(strcmp(var_name,'TimeSinceRew'))];
% beta_sig_kern_and_ramp = beta_sig(1+idx,:);
% ypred = exp([bas' tkern'] * beta_sig_kern_and_ramp);
% 
% subplot(1,4,2);
% imagesc(tkern,1:size(ypred,2),ypred(:,sort_idx)');
% cb = colorbar;
% xlabel('Time since reward');
% ylabel('Neuron, same sort as A');
% title('B. Model fit (kernels and ramp)')
% ylabel(cb,'Multiplicative firing rate change');
% 
% % Data
% subplot(1,4,3);
% fr_mat = spikecounts_final(:,sig_cells)/opt.tbin;
% for i = 1:size(fr_mat,2)
%     fr_mat(:,i) = gauss_smoothing(fr_mat(:,i),opt.smooth_sigma_fr/opt.tbin);
% end
% fr_mat = zscore(fr_mat);
% keep = false(size(t));
% keep(in_patch) = rew_size(patch_num(in_patch))==opt.rew_size;
% x = t_since_rew(keep);
% tkernedge = -opt.tbin/2:opt.tbin:opt.basis_length+opt.tbin/2;
% [~,~,bin]=histcounts(x,tkernedge);
% psth = nan(max(bin),size(fr_mat,2));
% for i = 1:max(bin)
%     psth(i,:) = mean(fr_mat(bin==i,:));
% end
% imagesc(tkern,1:sum(sig_cells),psth(:,sort_idx)');
% cb = colorbar;
% xlabel('Time since reward');
% ylabel('Neuron, same sort as A');
% title('C. Data')
% ylabel(cb,'z-scored firing rate');
% 
% % Full model
% subplot(1,4,4);
% ypred = exp([ones(size(X_full,1),1) X_full] * beta_sig);
% ypred = zscore(ypred);
% psth = nan(max(bin),sum(sig_cells));
% for i = 1:max(bin)
%     psth(i,:) = mean(ypred(bin==i,:));
% end
% imagesc(tkern,1:sum(sig_cells),psth(:,sort_idx)');
% cb = colorbar;
% xlabel('Time since reward');
% ylabel('Neuron, same sort as A');
% title('D. Full model')
% ylabel(cb,'z-scored firing rate (model prediciton)');
% 
% save_figs(paths.figs,hfig,'png');

%% 
num_var = numel(var_name)-sum(base_var);

hfig = figure('Position',[200 200 2000 600]);
for i = 1:num_var/3
    subplot(3,num_var/3,i);
    idx1 = sum(base_var)+i;
    idx2 = sum(base_var)+i+num_var/3;
    plot(beta_all(idx1,:),beta_all(idx2,:),'k.');
    xlabel(var_name{idx1},'Interpreter','none');
    ylabel(var_name{idx2},'Interpreter','none');
    axis square
    grid on
    refline(1,0);
end
for i = 1:num_var/3
    subplot(3,num_var/3,i+num_var/3);
    idx1 = sum(base_var)+i;
    idx2 = sum(base_var)+i+2*num_var/3;
    plot(beta_all(idx1,:),beta_all(idx2,:),'k.');
    xlabel(var_name{idx1},'Interpreter','none');
    ylabel(var_name{idx2},'Interpreter','none');
    axis square
    grid on
    refline(1,0);
end
for i = 1:num_var/3
    subplot(3,num_var/3,i+2*num_var/3);
    idx1 = sum(base_var)+i+num_var/3;
    idx2 = sum(base_var)+i+2*num_var/3;
    plot(beta_all(idx1,:),beta_all(idx2,:),'k.');
    xlabel(var_name{idx1},'Interpreter','none');
    ylabel(var_name{idx2},'Interpreter','none');
    axis square
    grid on
    refline(1,0);
end