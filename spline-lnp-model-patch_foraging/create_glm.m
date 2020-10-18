function [bestModels,allModelTestFits,tuning_curves,final_pval,fig1] = create_glm(speed,spiketrain,params)
% edited by malcolm 6/16/2020
% for compatibility with patch foraging data

%% do some initial processing 

var_name = {'speed'};
numVar = numel(var_name);
num_plot_columns = numVar;
numctrlpoints_speed = 5;
numbins_tuning_speed = 10;
speed_bin = 0.5;

%% compute the input matrices for position and speed

fig1 = figure(1);
s = 0.5; % spline parameter

%%%%%%%%% SPEED %%%%%%%%%
    fprintf('Making speed matrix\n');
    
    % plot coverage and tuning curve    
    sorted_speed = sort(speed);
    min_speed = floor(sorted_speed(round(numel(speed)*0.01))/speed_bin)*speed_bin;
    max_speed = ceil(sorted_speed(round(numel(speed)*0.99))/speed_bin)*speed_bin;
    spdVec_tuning = linspace(min_speed,max_speed,numbins_tuning_speed);
    [speed_tuning_curve,speed_occupancy] = compute_1d_tuning_curve(speed,spiketrain,numbins_tuning_speed,min_speed,max_speed);

    fig1 = subplot(3,num_plot_columns,1);
    plot(spdVec_tuning,speed_occupancy.*params.TimeBin,'k','linewidth',2)
    xlim([min_speed max_speed]);
    box off
    title('speed occupancy')
    axis tight
    ylabel('seconds')
    
    fig1 = subplot(3,num_plot_columns,2); hold on;
    plot(spdVec_tuning,speed_tuning_curve./params.TimeBin,'k','linewidth',2)
    xlim([min_speed max_speed]);
    box off
    title('speed tuning curve')
    axis tight
    ylabel('spikes/s')
    
    spdVec = linspace(min_speed,max_speed,numctrlpoints_speed);  
    spdVec(1) = spdVec(1)-0.01;
    speed(speed < min_speed) = min_speed; % send everything below min to min
    speed(speed > max_speed) = max_speed; % send everything over max to max
    [speedgrid,ctl_pts_speed] = spline_1d(speed,spdVec,s);
    A{1} = speedgrid;
    ctl_pts_all{1} = ctl_pts_speed;

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
xlim([min_speed max_speed]);
tuning_curves = nan;

firstModelFit = allModelTestFits{1};
fig1 = subplot(3,num_plot_columns,3);
errorbar(1:numVar,mean(firstModelFit),std(firstModelFit)/sqrt(10),'.k','linewidth',2)
hold on
plot([1 numVar],[0 0],'--b','linewidth',1);
hold off
box off
set(gca,'xtick',1)
set(gca,'xticklabel',{'speed'})
ylabel('bits/spike')
axis([0.5  1.5 -inf inf])
%legend('first model fit','baseline')

return