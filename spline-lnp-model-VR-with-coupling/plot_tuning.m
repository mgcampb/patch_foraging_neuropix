function [tuning_curves, fig1] = plot_tuning(A,variables,parameters,ctl_pts_all,s,plotfig,dt,fig1,var_name)

%% Description
% Given the variables, A, and the parameters,
% this will return the tuning curves for the cell
% if plotfig = 1, this will also plot the tuning curves

% NOTE: I just use A to compute the correct indexes
numVar = numel(var_name);

variables = sort(variables);
b0 = parameters(1);
param = parameters(2:end);

total_ind = 0;
% position
if ismember(1,variables)
    param_ind = size(A{1},2);
    total_ind = total_ind + param_ind;
    param1 = param(total_ind - param_ind + 1:total_ind);
    [pos_y,~,~] = spline_1d_plot(param1,ctl_pts_all{1},s);
    scale(1) = mean(exp(pos_y(:)));
end

% speed
if ismember(2,variables)
    param_ind = size(A{2},2);
    total_ind = total_ind + param_ind;
    param1 = param(total_ind - param_ind + 1:total_ind);
    [speed_y,speed_x] = spline_1d_plot(param1,ctl_pts_all{2},s);
    scale(2) = mean(exp(speed_y));
end

tuning_curves = {};

if plotfig
    figure(1)
    
    if ismember(1,variables)
        fig1 = subplot(4,numVar,numVar*2+1);
        scale_factor_ind = setdiff(variables,1); scale_factor = scale(scale_factor_ind);
        imagesc(exp(pos_y)*exp(b0)*prod(scale_factor)/dt)
        axis off
        axis tight
        tuning_curves{1} = exp(pos_y(:))*exp(b0)*prod(scale_factor);
        title('LNP position tuning')
    end
    
    if ismember(2,variables)
        fig1 = subplot(4,numVar,numVar*2+2);
        scale_factor_ind = setdiff(variables,2); scale_factor = scale(scale_factor_ind);
        plot(speed_x,exp(speed_y)*exp(b0)*prod(scale_factor)/dt,'k','linewidth',2);
        box off
        xlabel('speed')
        ylabel('spikes/s')
        axis tight
        tuning_curves{2} = exp(speed_y)*exp(b0)*prod(scale_factor);
        title('LNP speed tuning')
    end
    
end

return