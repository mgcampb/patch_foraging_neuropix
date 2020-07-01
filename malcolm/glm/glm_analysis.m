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

% analysis opts
opt = struct;
opt.session = '80_20200317'; % session to analyze
opt.tbin = 0.02; % in seconds
opt.smooth_sigma_lickrate = 0.1; % in seconds (for smoothing lickrate trace)

% basis functions for rewards
opt.nbasis = 5; % number of raised cosine basis functions to use
opt.basis_length = 1; % in seconds

% whether or not to zscore predictors
opt.zscore_predictors = true;

% regularization
opt.lambda = 0.0206; % came from trying a bunch of values on an example cell (m80 0317 c368)
opt.alpha = 0.5;

% cross validation
% opt.numFolds = 10;
% opt.crossvar_chunk = 3; % number of seconds in each cross validation chunk. Kiah tried #'s from 1-10 seconds.. not sure what is best

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

%% raised cosine basis for discrete events (rewards)

t_basis = 0:opt.tbin:opt.basis_length; % 1 second long
db = (max(t_basis) - min(t_basis))/(opt.nbasis-1);
c = min(t_basis):db:max(t_basis);
bas = nan(opt.nbasis,length(t_basis));
for k = 1:opt.nbasis
  bas(k,:) = (cos(max(-pi, min(pi,pi*(t_basis - c(k))/(db))) ) + 1) / 2;
end

figure; hold on;
for i = 1:opt.nbasis
    plot(t_basis,bas(i,:));
end
xlabel('time (sec)');
title('discrete event basis functions');

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

% % PATCH CUE 1: patch cue to (patch stop or patch leave)
% patch_cue1 = zeros(numel(t),1);
% trigs = [dat.patchCL; dat.patchCSL(:,1:2)];
% for i = 1:size(trigs,1)
%     patch_cue1(t-trigs(i,1)>=0 & t<trigs(i,2)) = 1;
% end
% X_this = patch_cue1;
% A = [A, {X_this}];
% grp_name = [grp_name,'PatchCue1'];
% var_name = [var_name,'PatchCue1'];
% 
% % PATCH CUE 2: patch stop to patch leave
% patch_cue2 = zeros(numel(t),1);
% trigs = dat.patchCSL(:,2:3);
% for i = 1:size(trigs,1)
%     patch_cue2(t-trigs(i,1)>=0 & t<trigs(i,2)) = 1;
% end
% X_this = patch_cue2;
% A = [A,{X_this}];
% grp_name = [grp_name,'PatchCue2'];
% var_name = [var_name,'PatchCue2'];

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
% time since most recent reward
% 
X_this = [t_on_patch' tot_rew'];
A = [A, {X_this}];
grp_name = [grp_name,'DVs'];
var_name = [var_name,'TimeOnPatch','TotalRew'];

% previous patch type
%

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

%% Create fold indices (for cross validation)
% T = numel(t(keep));
% numPts = opt.crossvar_chunk*round(1/opt.tbin); 
% [train_ind,test_ind] = compute_test_train_ind(opt.numFolds,numPts,T);

%% Fit GLM to each cell
pb = ParforProgressbar(numel(good_cells));
beta = nan(size(X,2)+1,numel(good_cells));
FitInfo = cell(numel(good_cells),1);
parfor cIdx = 1:numel(good_cells)
    
    [B, FitInfo{cIdx}] = lassoglm(X_final,spikecounts_final(:,cIdx),'poisson','Lambda',opt.lambda,'Alpha',opt.alpha);        
    beta(:,cIdx) = [FitInfo{cIdx}.Intercept; B];
    pb.increment();

end

%% plot

hfig = figure;
hfig.Name = sprintf('%s boxplot of absolute regressor weights',opt.session);
boxplot(abs(beta(2:end,:)'));
xticklabels(var_name);
xtickangle(90);
ylabel('|beta|');
title(sprintf('%s 4uL patches',opt.session),'Interpreter','none');
save_figs(paths.figs,hfig,'png');