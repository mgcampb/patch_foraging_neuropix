function glm_tuning_curves = compute_glm_tuning(A,variables,parameters,ctl_pts_all,s,dt)

%% Description
% Given the variables, A, and the parameters,
% this will return the glm tuning curves for the cell
% edited by malcolm 3/5/2019

variables = sort(variables);
b0 = parameters(1);
param = parameters(2:end);
total_ind = 0;

y = {};
scale = [];
for i = 1:2
    if ismember(i,variables)
        param_ind = size(A{i},2);
        total_ind = total_ind + param_ind;
        param1 = param(total_ind - param_ind + 1:total_ind);
        [y{i},~,~] = spline_1d_plot(param1,ctl_pts_all{i},s);
        scale(i) = mean(exp(y{i}));
    end
end

glm_tuning_curves = {};
for i = 1:2
    if ismember(i,variables)
        scale_factor_ind = setdiff(variables,i);
        scale_factor = scale(scale_factor_ind);
        glm_tuning_curves{i} = exp(y{i}(:))*exp(b0)*prod(scale_factor)/dt;
    end
end

return