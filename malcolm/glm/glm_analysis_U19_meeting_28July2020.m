% script to fit GLM to all cells in an example session
% using MATLAB's lassoglm
% MGC 7/1/2020

% TO DO:
% add more decision variables
% dropout analysis
% percent deviance explained for each variable

paths = struct;

paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_U19_meeting_28July2020';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes_repo = 'C:\code\spikes';
addpath(genpath(paths.spikes_repo));
paths.glmnet = 'C:\code\glmnet_matlab';
addpath(genpath(paths.glmnet));
paths.hgrk = 'C:\code\HGRK_analysis_tools'; % hyunggoo's code
addpath(genpath(paths.hgrk));

% analysis opts
opt = struct;
opt.session = '80_20200317'; % session to analyze
paths.figs = fullfile(paths.figs,opt.session);
opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)

% which reward size to analyze
opt.rew_size = 4;

% basis functions for rewards
opt.nbasis = 10; % number of raised cosine basis functions to use
opt.basis_length = 2; % in seconds

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

% example neuron to plot
opt.example_cell_id = 368;
opt.smooth_sigma_fr = 0.05; % in seconds, for smoothing firing rate trace of example cell

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

t_basis = 0:opt.tbin:opt.basis_length; % 1 second long
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
grp_name = {};
var_name = {};

% SESSION TIME
X_this = [t' t.^2'];
A = [A, {X_this}];
grp_name = [grp_name,'SessionTime'];
var_name = [var_name,'SessionTime','SessionTime^2'];

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
    
% CONCATENATE ALL PREDICTORS
X = []; % predictor matrix
for i = 1:numel(A)
    X = [X A{i}];
end

%% subselect data to fit GLM to

% large reward patches only:
rew_size = mod(dat.patches(:,2),10);
keep = false(size(t));
keep(in_patch) = rew_size(patch_num(in_patch))==opt.rew_size;
patch_num_final = patch_num(keep);

% final predictor matrix
% take zscore to be able to compare coefficients across predictors
if opt.zscore_predictors
    X_final = zscore(X(keep,:));
else
    X_final = X(keep,:);
end

% final spikecounts matrix
spikecounts_final = spikecounts_filt(keep,:);

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
        fit = cvglmnet(X_this,y,'poisson',opt_glmnet,[],[],foldid);
        lambda_idx = find(fit.lambda==fit.lambda_1se);
        beta_all(:,cIdx) = [fit.glmnet_fit.a0(lambda_idx); fit.glmnet_fit.beta(:,lambda_idx)];     
    catch
        fprintf('error: cell %d\n',cIdx);
    end
    
    pb.increment();
end

%% plot: histogram of p values

sig_cells = pval<=opt.pval_thresh;
good_cells_sig = good_cells_filt(sig_cells);
hfig = figure;
hfig.Name = sprintf('%s histogram of cross-validation pvalues',opt.session);
histogram(pval,0:0.01:1); hold on;
plot([opt.pval_thresh opt.pval_thresh],ylim,'r--');
xlabel('cross-validation p-value: full model vs mean model');
ylabel('num cells');
title(sprintf('%s %duL patches: %d/%d significant cells',opt.session,opt.rew_size,sum(sig_cells),Ncells),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% plot: regressor weights (significant cells only)

beta_sig = beta_all(:,sig_cells);
hfig = figure;
hfig.Name = sprintf('%s boxplot of regressor weights_significant only',opt.session);
boxplot(beta_sig(2:end,:)'); hold on;
plot(xlim,[0 0],'k-');
xticklabels(var_name);
xtickangle(90);
ylabel('beta');
ylim([-1 1]);
title(sprintf('%s %duL patches: %d significant cells',opt.session,opt.rew_size, sum(sig_cells)),'Interpreter','none');
save_figs(paths.figs,hfig,'png');

%% EXAMPLE NEURON: regressor trace + spike activity + predicted firing rate + smoothed firing rate, model coefficients

cellidx = find(good_cells_filt==opt.example_cell_id);
y = spikecounts_final(:,cellidx);

keep = false(size(t));
keep(in_patch) = rew_size(patch_num(in_patch))==opt.rew_size;
ysmooth = gauss_smoothing(spikecounts_filt(:,cellidx),opt.smooth_sigma_fr/opt.tbin);
ysmooth = ysmooth(keep);
ypred = exp([ones(size(X_final,1),1) X_final] * beta_all(:,cellidx)); % predict based on GLM

patch_times = [1; find(diff(X_final(:,1))>median(diff(X_final(:,1)))*10)+1];
snippet = [4000 5000];
patch_times = patch_times(patch_times>=snippet(1) & patch_times<=snippet(2));
rew_times = [1; find(diff(X_final(:,strcmp(var_name,'Rewards1')))>0)+1];
rew_times = rew_times(rew_times>=snippet(1) & rew_times<=snippet(2));

% plot regressor matrix
hfig = figure('Position',[100 100 2200 800]);
hfig.Name = sprintf('%s c%d regressor matrix snippet',opt.session,opt.example_cell_id);
ha = tight_subplot(size(X_final,2),1);
for i = 1:size(X_final,2)
    axes(ha(i)); hold on;
    plot(opt.tbin*(snippet(1):snippet(2)),X_final(snippet(1):snippet(2),i));
    for j = 1:numel(patch_times)
        plot([patch_times(j) patch_times(j)]*opt.tbin,ylim,'b-');
    end
    xticks([]);
    yticks([]);
    ylh = ylabel(var_name{i});
    set(ylh,'Rotation',0,'HorizontalAlignment','right');
end
axes(ha(end));
xticks([snippet(1) snippet(2)]*opt.tbin);
xticklabels([snippet(1) snippet(2)]*opt.tbin);
xlabel('time (sec)');
save_figs(paths.figs,hfig,'png');

% plot real and predicted neural activity
hfig = figure('Position',[100 100 2200 400]);
hfig.Name = sprintf('%s c%d firing rate true and predicted snippet',opt.session,opt.example_cell_id);

ha = tight_subplot(2,1);
axes(ha(1)); hold on;
plot(opt.tbin*(snippet(1):snippet(2)),y(snippet(1):snippet(2)),'k-');
for j = 1:numel(patch_times)
    plot([patch_times(j) patch_times(j)]*opt.tbin,ylim,'b-');
end
xticks([]);
yticks(0:3);
ylh = ylabel('spike counts');
set(ylh,'Rotation',0,'HorizontalAlignment','right');

axes(ha(2)); hold on;
plot(opt.tbin*(snippet(1):snippet(2)),ysmooth(snippet(1):snippet(2)),'k-');
plot(opt.tbin*(snippet(1):snippet(2)),ypred(snippet(1):snippet(2)),'r-');
for j = 1:numel(patch_times)
    plot([patch_times(j) patch_times(j)]*opt.tbin,ylim,'b-');
end
for k = 1:numel(rew_times)
    plot(rew_times(k)*opt.tbin,max(ylim),'bv','MarkerFaceColor','b');
end
ylh = ylabel('firing rate');
set(ylh,'Rotation',0,'HorizontalAlignment','right');
xlim([snippet(1) snippet(2)]*opt.tbin);
xticks([snippet(1) snippet(2)]*opt.tbin);
yticks([]);
xticklabels([snippet(1) snippet(2)]*opt.tbin);
xlh=xlabel('time (sec)');
legend('smoothed firing rate','predicted firing rate');
set(xlh,'Position',[mean(xlim),min(ylim)]);
save_figs(paths.figs,hfig,'png');

% model fit coefficients
beta_this = beta_all(2:end,cellidx);
hfig = figure; hold on;
hfig.Name = sprintf('%s c%d GLM coefficients',opt.session,opt.example_cell_id);
my_scatter(1:numel(beta_this),beta_this,'k',0.5);
plot(xlim,[0 0],'k--');
xticks(1:numel(beta_this));
xticklabels(var_name);
xtickangle(90);
ylabel('beta');
title(sprintf('%s c%d',opt.session,opt.example_cell_id),'Interpreter','none');
save_figs(paths.figs,hfig,'png');


%% speed trace

hfig = figure('Position',[100 100 1500 1200]);
hfig.Name = sprintf('%s_behavior',opt.session);

% behavioral events to align to
rew_size = mod(dat.patches(:,2),10);
N0 = mod(round(dat.patches(:,2)/10),10);
patchcue_ms = dat.patchCSL(:,1)*1000;
patchstop_ms = dat.patchCSL(:,2)*1000;
patchleave_ms = dat.patchCSL(:,3)*1000;
rew_ms = dat.rew_ts*1000;
lickt_ms = dat.lick_ts*1000;
speed_ms = interp1(dat.velt,dat.vel,0:0.001:max(dat.velt));

% exclude patchstop rewards and rewards that preceded patchleave by <1s    
keep = true(size(rew_ms));    
for rIdx = 1:numel(rew_ms)
    if sum(patchstop_ms<rew_ms(rIdx) & patchleave_ms>rew_ms(rIdx))==0 % only keep rewards in patches
        keep(rIdx) = false;
    end
    if min(abs(rew_ms(rIdx)-patchstop_ms))<500 || min(abs(rew_ms(rIdx)-patchleave_ms))<1000
        keep(rIdx) = false;
    end
end
rew_ms = rew_ms(keep);

% get size of each reward
rew_size_indiv = nan(size(rew_ms));
for rIdx = 1:numel(rew_ms)
    patch_id = find(patchstop_ms<rew_ms(rIdx) & patchleave_ms>rew_ms(rIdx));
    rew_size_indiv(rIdx) = rew_size(patch_id);
end

% patch cue
t_align = patchcue_ms;
t_start = patchcue_ms-1000;
t_end = patchstop_ms;
subplot(4,4,1);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
atitle('CUE/SPEED/ALL TRIALS');
subplot(4,4,5);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
atitle('CUE/SPEED/SPLIT BY REW SIZE');
subplot(4,4,9);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
atitle('CUE/LICK/ALL TRIALS');
subplot(4,4,13);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
atitle('CUE/LICK/SPLIT BY REW SIZE');

% patch stop
t_align = patchstop_ms;
t_start = patchcue_ms; 
t_end = min(patchleave_ms,patchstop_ms+5000); % maximum of 5 seconds after patch stop
subplot(4,4,2);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
atitle('STOP/SPEED/ALL TRIALS');
subplot(4,4,6);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
atitle('STOP/SPEED/SPLIT BY REW SIZE');
subplot(4,4,10);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
atitle('STOP/LICK/ALL TRIALS');
subplot(4,4,14);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
atitle('STOP/LICK/SPLIT BY REW SIZE');   

% patch leave
t_align = patchleave_ms;
t_start = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
t_end = patchleave_ms+2000;
subplot(4,4,3);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
atitle('LEAVE/SPEED/ALL TRIALS');
subplot(4,4,7);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
atitle('LEAVE/SPEED/SPLIT BY REW SIZE');
subplot(4,4,11);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
atitle('LEAVE/LICK/ALL TRIALS');
subplot(4,4,15);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
atitle('LEAVE/LICK/SPLIT BY REW SIZE');

% reward
t_align = rew_ms;
t_start = rew_ms-1000; % -1 to +1 sec rel. to reward
t_end = rew_ms+1000;
subplot(4,4,4);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end);   
atitle('REW/SPEED/ALL TRIALS');
subplot(4,4,8);
plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size_indiv);
atitle('REW/SPEED/SPLIT BY REW SIZE');
subplot(4,4,12);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
atitle('REW/LICK/ALL TRIALS');
subplot(4,4,16);
plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size_indiv);
atitle('REW/LICK/SPLIT BY REW SIZE');

% fig: just running speed, not split by reward size
hfig = figure;
hfig.Name = sprintf('%s - running speed aligned to rewards',opt.session);
% reward
t_align = rew_ms;
t_start = rew_ms-1000; % -1 to +1 sec rel. to reward
t_end = rew_ms+1000;
plot_timecourse('stream',speed_ms,t_align,t_start,t_end);   
title('Mouse running speed (a.u.) aligned to rewards');
save_figs(paths.figs,hfig,'png');

%% pca on coefficients

beta_filt = beta_sig(4:end,:);
var_name_filt = var_name(3:end);
[coeff, score, ~, ~, expl] = pca(beta_filt');

hfig = figure('Position',[200 200 1000 800]);
hfig.Name = sprintf('%s: PCA on GLM coefficients',opt.session);

subplot(2,2,1);
my_scatter(1:numel(expl),expl,'k',0.5);
xlabel('PC');
ylabel('Variance Explained (%)');
xticks(1:numel(expl));
title(sprintf('Session: %s',opt.session),'Interpreter','none');

subplot(2,2,2); hold on;
my_scatter(score(:,1),score(:,2),'k',0.2);
xlabel('PC1 score');
ylabel('PC2 score');
cellidx = find(good_cells_sig==opt.example_cell_id);
my_scatter(score(cellidx,1),score(cellidx,2),'m',0.5);
title(sprintf('%d significant cells',sum(sig_cells)));
plot(xlim,[0 0],'k:');
ymax = max(ylim);
ymin = min(ylim);
plot([0 0],[ymin ymax],'k:');
ylim([ymin ymax]);

subplot(2,2,3);
my_scatter(1:numel(expl),coeff(:,1),'k',0.5);
xticks(1:numel(var_name_filt))
xticklabels(var_name_filt)
xtickangle(90)
hold on;
xlim([0 numel(var_name_filt)+1]);
plot(xlim,[0 0],'k--')
ylabel('coeff');
title('PC1');

subplot(2,2,4);
my_scatter(1:numel(expl),coeff(:,2),'k',0.5);
xticks(1:numel(var_name_filt))
xticklabels(var_name_filt)
xtickangle(90)
hold on;
xlim([0 numel(var_name_filt)+1]);
plot(xlim,[0 0],'k--')
ylabel('coeff');
title('PC2');

save_figs(paths.figs,hfig,'png');

%% plot example cell coefficients, then with PC reconstruction

hfig = figure; hold on;
hfig.Name = sprintf('%s c%d: Model fit coefficients plus PCA reconstruction',opt.session,opt.example_cell_id);
cellidx = find(good_cells_sig==opt.example_cell_id);
pca_reconstruction = coeff(:,1:2)*score(cellidx,1:2)';
my_scatter(1:size(beta_filt,1),beta_filt(:,cellidx),'k',0.5);
my_scatter(1:size(beta_filt,1),pca_reconstruction,'m',0.5);
plot(xlim,[0 0],'k--','HandleVisibility','off')
xticks(1:numel(var_name_filt))
xticklabels(var_name_filt)
xtickangle(90)
ylabel('coeff');
title(sprintf('%s c%d',opt.session,opt.example_cell_id),'Interpreter','none');
legend('true coeff','pc1,2 reconstruction','Location','northeastoutside');

save_figs(paths.figs,hfig,'png');
