paths = struct;

paths.model_fits = 'C:\code\patch_foraging_neuropix\malcolm\glm\GLM_output\allRewSize\no_zscore';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_time_since_reward_kernels_allRewSize';

session_all = dir(fullfile(paths.model_fits,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

opt = struct;
opt.tbin = 0.02;
opt.basis_length = 2;
opt.nbasis = 11;
% opt.num_base_var = 6;
% opt.pval_thresh = 0.05;
opt.rew_size = [1 2 4];

mouse_all = {'75','76','78','79','80'};

%% load all model fits
pval_all = [];
beta_all = [];
mouse = [];
session = [];
for sIdx = 1:numel(session_all)
    dat = load(fullfile(paths.model_fits,session_all{sIdx}));
    pval_all = [pval_all; dat.pval];
    beta_all = [beta_all dat.beta_all];
    var_name = dat.var_name;
    bas = dat.bas;
    t_basis = dat.t_basis;
    mouse_this = find(strcmp(mouse_all,session_all{sIdx}(1:2)));
    mouse = [mouse; repmat(mouse_this,numel(dat.pval),1)];
    session = [session; repmat(sIdx,numel(dat.pval),1)];
end

%% plot: cells per mouse and cells per session
hfig = figure('Position',[200 200 800 400]);
hfig.Name = 'Num neurons per mouse and session';

subplot(1,2,1); hold on;
hc = histcounts(mouse);
for i = 1:numel(hc)
    bar(i,hc(i));
end
xticks(1:numel(hc));
xticklabels(mouse_all);
xlabel('Mouse');
ylabel('Num. cells (after firing rate cutoff)');
title(sprintf('Total = %d neurons',sum(hc)));

subplot(1,2,2); hold on;
hc = histcounts(session);
plot_col = lines(numel(mouse_all));
for i = 1:numel(hc)
    mouse_num = find(strcmp(mouse_all,session_all{i}(1:2)));
    bar(i,hc(i),'FaceColor',plot_col(mouse_num,:));
end
xticks(1:numel(session_all));
%xticklabels(session_all);
%xtickangle(90);
set(gca,'TickLabelInterpreter','none');
xlabel('Session');
ylabel('Num. cells (after firing rate cutoff)');

save_figs(paths.figs,hfig,'png');

%% plot: DVs for 2uL vs 4uL


% Time Since Rew
var1 = 'TimeSinceRew_2uL';
var2 = 'TimeSinceRew_4uL';
hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name);
save_figs(paths.figs,hfig,'png');

% Time On Patch
var1 = 'TimeOnPatch_2uL';
var2 = 'TimeOnPatch_4uL';
hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name);
save_figs(paths.figs,hfig,'png');

% Total Reward
var1 = 'TotalRew_2uL';
var2 = 'TotalRew_4uL';
hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name);
save_figs(paths.figs,hfig,'png');

% Time on Patch vs Total Reward, 4uL
var1 = 'TimeOnPatch_4uL';
var2 = 'TotalRew_4uL';
hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name);
save_figs(paths.figs,hfig,'png');

% Time on Patch vs Total Reward, 2uL
var1 = 'TimeOnPatch_2uL';
var2 = 'TotalRew_2uL';
hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name);
save_figs(paths.figs,hfig,'png');

%% functions

function hfig = plot_var_pair(var1,var2,beta_all,mouse,mouse_all,var_name)

x = beta_all(strcmp(var_name,var1),:);
y = beta_all(strcmp(var_name,var2),:);
keep = abs(x)>0 & abs(y)>0;
x = x(keep);
y = y(keep);
m = mouse(keep);

hfig = figure;
hfig.Name = sprintf('%s vs %s',var1,var2);
hold on;
plot_col = lines(numel(mouse_all));
for i = numel(mouse_all):-1:1
    my_scatter(x(m==i),y(m==i),plot_col(i,:),0.5);
end
grid on;
h = refline(1,0);
plot([0 0],ylim,'k:');
plot(xlim,[0 0],'k:');
xlabel(var1,'Interpreter','none');
ylabel(var2,'Interpreter','none');

lm = fitlm(x,y);
xpred = min(xlim):0.01:max(xlim);
[ypred,yci] = predict(lm,xpred');
err = yci(:,2)-yci(:,1);
shadedErrorBar(xpred,ypred,err);

title(sprintf('%d/%d neurons w non-zero coeff for both',numel(x),size(beta_all,2)));

end