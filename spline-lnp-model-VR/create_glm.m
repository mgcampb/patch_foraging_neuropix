function [bestModels,allModelTestFits,tuning_curves,final_pval,fig1] = create_glm(posx,speed,spiketrain,params)
% edited by malcolm 2/12/2019
% for compatibility with 1D VR data
% spike_idx is for plotting rasters

%% do some initial processing 

var_name = {'position','speed'};
numVar = numel(var_name);
num_plot_columns = numVar+1;
xbinedges = params.TrackStart:params.SpatialBin:params.TrackEnd;
xbincent = xbinedges+params.SpatialBin/2; xbincent = xbincent(1:end-1);
numposbin = numel(xbincent);
numctrlpoints_pos = 30;

%% compute the input matrices for position and speed

fig1 = figure(1);
s = 0.5; % spline parameter
%%%%%%%%% POSITION %%%%%%%%%
    fprintf('Making position matrix\n');
    
    % plot position coverage and position tuning curve
    [pos_tuning_curve,pos_occupancy] = compute_1d_tuning_curve(posx,spiketrain,numposbin,params.TrackStart,params.TrackEnd);

    fig1 = subplot(4,num_plot_columns,1);
    plot(xbincent,pos_occupancy.*params.TimeBin)
    title('position occupancy')
    axis tight
    
    fig1 = subplot(4,num_plot_columns,1+num_plot_columns);
    plot(xbincent,pos_tuning_curve./params.TimeBin)
    title('position tuning curve')
    ylabel('spikes/s')
    axis tight
    
    % plot spike raster
    
    
    % spline
    x_vec = linspace(params.TrackStart,params.TrackEnd,numctrlpoints_pos);
    x_vec(1) = x_vec(1)-0.01;
    [posgrid,ctl_pts_pos] = spline_1d(posx,x_vec,s);
    A{1} = posgrid;
    ctl_pts_all{1} = ctl_pts_pos;

%%%%%%%%% SPEED %%%%%%%%%
    fprintf('Making speed matrix\n');
    
    % plot coverage and tuning curve
    sorted_speed = sort(speed);
    max_speed = ceil(sorted_speed(round(numel(speed)*0.99))/10)*10;
    [speed_tuning_curve,speed_occupancy] = compute_1d_tuning_curve(speed,spiketrain,10,0,max_speed);

    fig1 = subplot(4,num_plot_columns,3);
    plot(linspace(0,max_speed,10),speed_occupancy.*params.TimeBin,'k','linewidth',2)
    box off
    title('speed occupancy')
    axis tight
    ylabel('seconds')
    
    fig1 = subplot(4,num_plot_columns,3+num_plot_columns);
    plot(linspace(0,max_speed,10),speed_tuning_curve./params.TimeBin,'k','linewidth',2)
    box off
    title('speed tuning curve')
    axis tight
    ylabel('spikes/s')
    
    spdVec = [0:5:max_speed]; spdVec(end) = max_speed; spdVec(1) = -0.1;
    speed(speed > max_speed) = max_speed; %send everything over max to max
    [speedgrid,ctl_pts_speed] = spline_1d(speed,spdVec,s);
    A{2} = speedgrid;
    ctl_pts_all{2} = ctl_pts_speed;

%% fit the model

%%%%%%% COMPUTE TEST AND TRAIN INDICES %%%%%%%%%
numFolds = 10;
T = numel(spiketrain); 
numPts = 3*round(1/params.TimeBin); % 3 seconds. i've tried #'s from 1-10 seconds.. not sure what is best
[train_ind,test_ind] = compute_test_train_ind(numFolds,numPts,T);

%%%%%%%% FORWARD SEARCH PROCEDURE %%%%%%%%%
[allModelTestFits, allModelTrainFits, bestModels, bestModelFits, parameters, pvals, final_pval] = forward_search_kfold(A,spiketrain,train_ind,test_ind);



% plot the model tuning curves (this is an approx)
plotfig = 1;
final_param = parameters{end};
[tuning_curves,fig1] = plot_all_tuning(A,bestModels,final_param,ctl_pts_all,s,plotfig,params.TimeBin);
tuning_curves = nan;

firstModelFit = allModelTestFits{1};
fig1 = subplot(4,num_plot_columns,num_plot_columns*3+1:num_plot_columns*3+num_plot_columns);
errorbar(1:numVar,mean(firstModelFit),std(firstModelFit)/sqrt(10),'.k','linewidth',2)
hold on
plot([1 numVar],[0 0],'--b','linewidth',1);
hold off
box off
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'position','speed'})
ylabel('bits/spike')
axis([0.5  2.5 -inf inf])
%legend('first model fit','baseline')

return