addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_percent_significant_cells';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'STR';
opt.data_set = 'mc';
opt.pval_thresh = 0.05;

%%
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

% get mouse name for each session
mouse = cell(size(session_all));
if strcmp(opt.data_set,'mb')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:2);
    end
elseif strcmp(opt.data_set,'mc')
    for i = 1:numel(session_all)
        mouse{i} = session_all{i}(1:3);
    end
end
uniq_mouse = unique(mouse);

%%
num_sig = nan(numel(session_all),1);
num_tot = nan(numel(session_all),1);
pct_sig = nan(numel(session_all),1);
for i = 1:numel(session_all)
    fprintf('Session %d/%d: %s\n',i,numel(session_all),session_all{i});
    
    fit = load(fullfile(paths.results,session_all{i}));
    
    keep_cell = strcmp(fit.brain_region_rough,opt.brain_region);
    
    sig = fit.pval_full_vs_base(keep_cell)<opt.pval_thresh & sum(abs(fit.beta_all(fit.base_var==0,keep_cell))>0)'>0;
    num_sig(i) = sum(sig);
    num_tot(i) = sum(keep_cell);
    pct_sig(i) = 100*sum(sig)/sum(keep_cell);
end

%%
keep = num_tot>0;
session_all = session_all(keep);
mouse = mouse(keep);
uniq_mouse = unique(mouse);
num_sig = num_sig(keep);
num_tot = num_tot(keep);
pct_sig = pct_sig(keep);

%% make fig

hfig = figure('Position',[200 200 60*numel(session_all) 400]); hold on;
hfig.Name = sprintf('GLM_percent_significant_cells_%s_cohort_%s',opt.data_set,opt.brain_region');
plot_col = lines(numel(uniq_mouse));
for i = 1:numel(pct_sig)
    bar(i,pct_sig(i),'FaceColor',plot_col(strcmp(uniq_mouse,mouse{i}),:));
    text(i,pct_sig(i)+5,sprintf('%d/%d',num_sig(i),num_tot(i)),'HorizontalAlignment','center');
end
xticks(1:numel(pct_sig));
xticklabels(session_all);
xtickangle(90);
set(gca,'TickLabelInterpreter','None');
ylabel('% significant');
ylim([0 100]);
title(sprintf('%s cohort, %s (%d sessions, %d mice)',opt.data_set,opt.brain_region,numel(session_all),numel(uniq_mouse)));
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');