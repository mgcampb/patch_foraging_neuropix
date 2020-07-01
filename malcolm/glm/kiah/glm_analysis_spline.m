% example_cell.m
% script to fit GLM to all cells in an example session
% using MATLAB's glmfit
% MGC 6/16/2020

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
addpath(genpath('kiah'));

opt = struct;
opt.session = '80_20200315.mat'; % session to analyze
opt.TimeBin = 0.02; % in seconds
opt.spline_s = 0.5; % tension parameter for splines

% opts for running speed predictors
opt.num_ctrl_pts_speed = 5; % number of control points for speed spline
opt.speed_bin = 0.5; % for speed tuning and computing max and min speed
opt.lower_pctile_speed = 0.01; % for computing min_speed (eliminates outliers)
opt.upper_pctile_speed = 0.99; % for computing max_speed (eliminates outliers)

% opts for lick rate predictors
opt.smooth_sigma_lickrate = 3; % gauss SD in bins (20 ms bins - opt.TimeBin)

% opts for cross validation
opt.numFolds = 10;
opt.crossvar_chunk = 3; % number of seconds in each cross validation chunk. Kiah tried #'s from 1-10 seconds.. not sure what is best

%% load data
dat = load(fullfile(paths.data,opt.session));

%% make predictors, A, for this session

A = {}; % initialize predictor matrix; separate in this way for forward search and/or dropout

% RUNNING SPEED
% % spline
speed = dat.vel;
sorted_speed = sort(speed);
min_speed = floor(sorted_speed(round(numel(speed)*opt.lower_pctile_speed))/opt.speed_bin)*opt.speed_bin;
max_speed = ceil(sorted_speed(round(numel(speed)*opt.upper_pctile_speed))/opt.speed_bin)*opt.speed_bin;
speed(speed < min_speed) = min_speed; % send everything below min to min
speed(speed > max_speed) = max_speed; % send everything over max to max
ctrl_pts_speed_init = linspace(min_speed,max_speed,opt.num_ctrl_pts_speed); % speed control points  
ctrl_pts_speed_init(1) = ctrl_pts_speed_init(1)-0.01; % spline doesn't work otherwise
[speedgrid,ctrl_pts_speed] = spline_1d(speed,ctrl_pts_speed_init,opt.spline_s); % create speedgrid (spline)
% A{1} = speedgrid;
A{1} = dat.vel';

% LICK RATE
lickcounts = histc(dat.lick_ts,dat.velt)/opt.TimeBin;
lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate);
A{2} = lickrate;

% REWARDS
rew_binary = zeros(numel(dat.velt),1);
for i = 1:numel(dat.rew_ts)
    rew_binary(dat.velt-dat.rew_ts(i)>=0 & dat.velt-dat.rew_ts(i)<=0.5) = 1;
end
A{3} = rew_binary;

% PATCH CUE 1: patch cue to (patch stop or patch leave)
patch_cue1 = zeros(numel(dat.velt),1);
trigs = [dat.patchCL; dat.patchCSL(:,1:2)];
for i = 1:size(trigs,1)
    patch_cue1(dat.velt-trigs(i,1)>=0 & dat.velt<trigs(i,2)) = 1;
end
A{4} = patch_cue1;

% PATCH CUE 2: patch stop to patch leave
patch_cue2 = zeros(numel(dat.velt),1);
trigs = dat.patchCSL(:,2:3);
for i = 1:size(trigs,1)
    patch_cue2(dat.velt-trigs(i,1)>=0 & dat.velt<trigs(i,2)) = 1;
end
A{5} = patch_cue2;

% Session time
A{6} = [dat.velt' dat.velt.^2'];

% "DECISION VARIABLES"
t_on_patch = dat.velt - dat.patchCSL(patch_num,2)';
% time on patch
% total rewards on patch so far
% time since last reward

% all predictors
X = [];
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

%% Create fold indices (for cross validation)
T = numel(dat.velt); 
numPts = opt.crossvar_chunk*round(1/opt.TimeBin); 
[train_ind,test_ind] = compute_test_train_ind(opt.numFolds,numPts,T);

%% Fit GLM to each cell
good_cells = dat.sp.cids(dat.sp.cgs==2);
encoded_vars = cell(numel(good_cells),1);
R2_full = nan(numel(good_cells),1);
R2_dropout = nan(numel(good_cells),numel(X_dropout));
pb = ParforProgressbar(numel(good_cells));
parfor cIdx = 1:numel(good_cells)

    % fprintf('Fitting GLM for cell %d of %d\n',cIdx,numel(good_cells));
    
    % spiketrain
    spike_t = dat.sp.st(dat.sp.clu==good_cells(cIdx));
    spikecounts = histc(spike_t,dat.velt); % spiketrain (no smoothing)
    
    try
        
%         % forward search procedure:
%         [allModelFits, allModelTrainFits, bestModels, bestModelFits, parameters, pvals, final_pval] = forward_search_kfold(A,y,train_ind,test_ind);
%         encoded_vars{cIdx} = bestModels;
%         
%         X = [];
%         for mIdx = 1:numel(bestModels)
%             X = [X A{bestModels(mIdx)}];
%         end
        
        [beta,~,stats] = glmfit(X,spikecounts,'poisson');
        % yfit = glmval(beta,X,'log');
        % yresid = y-yfit;       
        R2_full(cIdx) = 1 - var(stats.resid)/var(spikecounts);
        
        % *** NOTE *** : MAYBE THESE R2's SHOULD BE CROSS VALIDATED 
        
        tmp = nan(1,numel(X_dropout));
        for varIdx = 1:numel(X_dropout)
            [beta,~,stats] = glmfit(X_dropout{varIdx},spikecounts,'poisson');
            tmp(varIdx) = 1 - var(stats.resid)/var(spikecounts);
        end
        R2_dropout(cIdx,:) = tmp;
    catch
    end
        
%     [speed_y,speed_x] = spline_1d_plot(beta(2:end)',ctrl_pts_speed,s);
%     %scale(1) = mean(exp([ones(size(speedgrid,1),1) speedgrid]*beta));
%     %scale_factor_ind = setdiff(variables,var_k);
%     %scale_factor = scale(scale_factor_ind);
% 
%     pred_tc = exp(speed_y)*exp(beta(1))/opt.TimeBin;
%     figure; hold on;
%     plot(spdVec_tuning,speed_tuning_curve/opt.TimeBin);
%     plot(speed_x,pred_tc);
% 
%     [speed_tuning_curve,speed_occupancy] = compute_1d_tuning_curve(speed,spiketrain,numbins_tuning_speed,min_speed,max_speed);

    pb.increment();

end