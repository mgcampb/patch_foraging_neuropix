
% Plots sum of rew kern coeffs vs either time since rew or time on
% patch coeff, to demonstrate that they tend to have opposite sign
% get beta_all_sig and var_name from the script "fit_glm_all_sessions.m"
% MGC 7/24/2021

hfig = figure('Position',[200 200 700 450]);

plot_col = cool(3);
rew_size = [1 2 4];

% Time since reward
for i = 1:3
    subplot(2,3,i); hold on;
    y1 = sum(beta_all_sig(contains(var_name,'RewKern') & contains(var_name,sprintf('%duL',rew_size(i))),:));
    y2 = beta_all_sig(strcmp(var_name,sprintf('TimeSinceRew_%duL',rew_size(i))),:);
    [r,p] = corr(y1',y2','Type','Spearman');
    scatter(y1,y2,10,plot_col(i,:));
    xlabel('Sum reward kern coeff.');
    ylabel('Time since reward coeff.');
    title(sprintf('Spearman r=%0.3f, p=%0.3f',r,p));
    plot(xlim,[0 0],'k--');
    plot([0 0],ylim,'k--');
end

% Time on patch
for i = 1:3
    subplot(2,3,i+3); hold on;
    y1 = sum(beta_all_sig(contains(var_name,'RewKern') & contains(var_name,sprintf('%duL',rew_size(i))),:));
    y2 = beta_all_sig(strcmp(var_name,sprintf('TimeOnPatch_%duL',rew_size(i))),:);
    [r,p] = corr(y1',y2','Type','Spearman');
    scatter(y1,y2,10,plot_col(i,:));
    xlabel('Sum reward kern coeff.');
    ylabel('Time on patch coeff.');
    title(sprintf('Spearman r=%0.3f, p=%0.3f',r,p));
    plot(xlim,[0 0],'k--');
    plot([0 0],ylim,'k--');
end