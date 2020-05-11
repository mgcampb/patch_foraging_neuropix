function [tuning_curves, fig1] = plot_all_tuning(A,variables,parameters,ctl_pts_all,s,plotfig,dt)

%% Description
% Given the variables, A, and the parameters,
% this will return the tuning curves for the cell
% if plotfig = 1, this will also plot the tuning curves

% NOTE: I just use A to compute the correct indexes

var_name = {'position','speed'};
numVar = numel(var_name);
num_plot_columns = numVar+1;

variables = sort(variables);
b0 = parameters(1);
param = parameters(2:end);

total_ind = 0;
% position
if ismember(1,variables)
    param_ind = size(A{1},2);
    total_ind = total_ind + param_ind;
    param1 = param(total_ind - param_ind + 1:total_ind);
    scale(1) = mean(exp(A{1}*param1'));
    [pos_y,x] = spline_1d_plot(param1,ctl_pts_all{1},s);
end


% speed
if ismember(2,variables)
    param_ind = size(A{2},2);
    total_ind = total_ind + param_ind;
    param1 = param(total_ind - param_ind + 1:total_ind);
    scale(2) = mean(exp(A{2}*param1'));
    [speed_y,speed_x] = spline_1d_plot(param1,ctl_pts_all{2},s);
end

tuning_curves = {};

if plotfig
    figure(1)
    
    var_k = 1;
    if ismember(var_k,variables)
        fig1 = subplot(4,num_plot_columns,num_plot_columns*2+1);
        scale_factor_ind = setdiff(variables,var_k); scale_factor = scale(scale_factor_ind);
        plot(x,exp(pos_y)*exp(b0)*prod(scale_factor)/dt)
        xlabel('pos')
        ylabel('spikes/s')
        axis tight
        tuning_curves{var_k} = exp(pos_y(:))*exp(b0)*prod(scale_factor)/dt;
        title('LNP pos tuning')
    end
    
    
    var_k = 2;
    if ismember(var_k,variables)
        fig1 = subplot(4,num_plot_columns,num_plot_columns*2+3);
        scale_factor_ind = setdiff(variables,var_k); scale_factor = scale(scale_factor_ind);
        positive_val = speed_x > 0;
        plot(speed_x(positive_val),exp(speed_y(positive_val))*prod(scale_factor)*exp(b0)/dt,'k','linewidth',2);
        box off
        xlabel('speed')
        ylabel('spikes/s')
        axis tight
        tuning_curves{var_k} = exp(speed_y')*exp(b0)*prod(scale_factor)/dt;
        vector_x = speed_x(positive_val); vector_y = tuning_curves{var_k}(positive_val);
        %plot_special_points
        title('LNP speed tuning')
    end
    
    set(gcf, 'Position', [12.2000 12.200 1420 800]);
end
return