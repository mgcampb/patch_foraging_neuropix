function [output, A, ctl_pts_all] = create_glm_coupling(posx,speed,spiketrain_all,params)
% edited by malcolm 2/12/2019
% for compatibility with 1D VR data
% spike_idx is for plotting rasters
%
% added spike history cell-cell coupling terms.
% spiketrain_all has all the spiketrains from all the cells.
% TO DO: theta/gamma coupling terms
% MGC 3/4/2019

% Edited coupling filter to just use the spike history filter
% MGC 3/6/2019

%% Params
xbinedges = params.TrackStart:params.SpatialBin:params.TrackEnd;
xbincent = xbinedges+params.SpatialBin/2; xbincent = xbincent(1:end-1);
numposbin = numel(xbincent);
numctrlpoints_pos = 30;
s = 0.5; % spline parameter
numcells = size(spiketrain_all,1);
if ~contains(fields(params),'single_models_only')
    params.single_models_only=false;
end

%% Compute the input matrices for position and speed

%%%%%%%%% POSITION %%%%%%%%%
    fprintf('Making position matrix\n');
    x_vec = linspace(params.TrackStart,params.TrackEnd,numctrlpoints_pos);
    x_vec(1) = x_vec(1)-0.01;
    [posgrid,ctl_pts_pos] = spline_1d(posx,x_vec,s);
    A{1} = posgrid;
    ctl_pts_all{1} = ctl_pts_pos;

%%%%%%%%% SPEED %%%%%%%%%
    fprintf('Making speed matrix\n');
    sorted_speed = sort(speed);
    max_speed = ceil(sorted_speed(round(numel(speed)*0.99))/10)*10;
    spdVec = 0:5:max_speed; spdVec(end) = max_speed; spdVec(1) = -0.1;
    speed(speed > max_speed) = max_speed; %send everything over max to max
    [speedgrid,ctl_pts_speed] = spline_1d(speed,spdVec,s);
    A{2} = speedgrid;
    ctl_pts_all{2} = ctl_pts_speed;

%% Compute the input matrix for neuron-specific variables: spike history and cell-cell coupling
% ADD IN THETA/GAMMA LOCKING LATER

% set control points for spike history (or coupling)
spikeHistoryVec = [0:3 5 10 20 35 50]; %in bins
spikeCouplingVec = [-fliplr(spikeHistoryVec) spikeHistoryVec(2:end)];
shgrid_all = cell(numcells,1);
coupgrid_all = cell(numcells,1);

for n = 1:numcells
    % SPIKE HISTORY / COUPLING
    fprintf('Making spike history spline for cell %d/%d\n', n, numcells);
    spiketrain = spiketrain_all(n,:);
    [shgrid,~] = spline_spike_hist(spiketrain,spikeHistoryVec,s);
    shgrid_all{n} = sparse(shgrid); % make it sparse to save memory
    
    % [coupgrid,~] = spline_spike_coupling(spiketrain,spikeCouplingVec,s);
    % coupgrid_all{n} = coupgrid;
end

ctl_pts_all{3} = spikeHistoryVec;
ctl_pts_all{4} = spikeCouplingVec;

%% Compute test and train indices
numFolds = 10;
T = size(spiketrain_all,2); 
numPts = 3*round(1/params.TimeBin); % 3 seconds. i've tried #'s from 1-10 seconds.. not sure what is best
[train_ind,test_ind] = compute_test_train_ind(numFolds,numPts,T);

%% Run the glm for all cells in the session
output = struct;
for n = 1%:numcells
    fprintf('Starting forward search for cell %d/%d\n', n, numcells);
    
    % spike history terms
    A{3} = shgrid_all{n};
    
    % cell-cell coupling terms
    A(4:end) = [];
    otherCells = find((1:numcells) ~= n);
    for m = 1:numel(otherCells)
        %A{m+3} = coupgrid_all{otherCells(m)};
        A{m+3} = shgrid_all{otherCells(m)}; % just make the coupling filter the same as the spike history filter for now
    end

    % spiketrain for this cell
    spiketrain = spiketrain_all(n,:);
    
    % perform forward search
    tic
    if params.single_models_only
        [testFit_all, trainFit_all, bestModels, bestModel_testfit, parameters, pvals, final_pval] = ...
            single_model_kfold(A, spiketrain, train_ind, test_ind);
    else
        [testFit_all, trainFit_all, bestModels, bestModel_testfit, parameters, pvals, final_pval] = ...
            forward_search_kfold(A, spiketrain, train_ind, test_ind);
    end
    toc
    
    % compute position and speed tuning curves
    pos_tuning = compute_1d_tuning_curve(posx,spiketrain,numposbin,params.TrackStart,params.TrackEnd);
    speed_tuning = compute_1d_tuning_curve(speed,spiketrain,10,0,max_speed);

    % need to re-set the bestModels stuff for the other cells... 
    %param_mean = mean(parameters); % didn't understand the point of this, commented it out
    % param_mean = parameters;
    % glm_tuning_curves = compute_glm_tuning(A,bestModels,param_mean,ctl_pts_all,s,params.TimeBin);
    
    % create an output structure
    output(n).testFit_all = testFit_all;
    output(n).trainFit_all = trainFit_all;
    output(n).bestModels = bestModels;
    output(n).bestModel_testfit = bestModel_testfit;
    output(n).parameters = parameters;
    % output(n).glm_tuning = glm_tuning_curves;
    output(n).pvals = pvals;
    output(n).final_pval = final_pval;
    output(n).pos_tuning = pos_tuning;
    output(n).speed_tuning = speed_tuning;
    
end

return