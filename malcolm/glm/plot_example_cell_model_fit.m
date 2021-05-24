addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_example_cell_model_fits';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt.session = '80_20200317';
opt.example_cell_id = 368;
opt.tbin = 0.02;
opt.smooth_sigma_fr = 0.05; % in seconds, for smoothing firing rate trace of example cell
opt.rew_size = [1 2 4];

opt.snippet = [8000 10000]; % which snippet of the session to plot (in bins)


%% load model fit data

fit = load(fullfile(paths.results,opt.session));
if strcmp(opt.example_cell_id,'rand')
    opt.example_cell_id = randsample(fit.good_cells,1);
end

%% load spike data

dat = load(fullfile(paths.data,opt.session));
good_cells_all = dat.sp.cids(dat.sp.cgs==2);

t = dat.velt;
spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
for cIdx = 1:numel(good_cells_all)
    spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
    spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
end
spikecounts_filt = spikecounts_whole_session(:,ismember(good_cells_all,fit.good_cells));

%% get patch times
    
patch_num = nan(size(t));
for i = 1:size(dat.patchCSL,1)
    % include one time bin before patch stop to catch the first reward
    patch_num(t>=(dat.patchCSL(i,2)-opt.tbin) & t<=dat.patchCSL(i,3)) = i;
end
in_patch = ~isnan(patch_num);

%% EXAMPLE NEURON: regressor trace + spike activity + predicted firing rate + smoothed firing rate, model coefficients

cellidx = find(fit.good_cells==opt.example_cell_id);
y = fit.spikecounts(:,cellidx);

ysmooth = gauss_smoothing(spikecounts_filt(:,cellidx),opt.smooth_sigma_fr/opt.tbin);
ysmooth = ysmooth(in_patch);
ypred = exp([ones(size(fit.X_full,1),1) fit.X_full] * fit.beta_all(:,cellidx)); % predict based on GLM

patch_times = [1; find(diff(fit.X_full(:,1))>median(diff(fit.X_full(:,1)))*10)+1];
patch_times = patch_times(patch_times>=opt.snippet(1) & patch_times<=opt.snippet(2));

rew_times = cell(3,1);
for i = 1:numel(opt.rew_size)
    idx = find(contains(fit.var_name,sprintf('RewKern1_%duL',opt.rew_size(i))))-1;
    rew_times{i} = [1; find(diff(fit.X_full(:,idx))>0)+1];
    rew_times{i} = rew_times{i}(rew_times{i}>=opt.snippet(1) & rew_times{i}<=opt.snippet(2));
end

%% plot regressor matrix
hfig = figure('Position',[100 100 1100 600]);
hfig.Name = sprintf('%s c%d regressor matrix snippet',opt.session,opt.example_cell_id);
imagesc(opt.tbin*(opt.snippet(1):opt.snippet(2)),1:67,fit.X_full(opt.snippet(1):opt.snippet(2),:)')
yticks(1:67);
yticklabels(fit.var_name(2:end))
xticks((opt.snippet(1):(10/opt.tbin):opt.snippet(2))*opt.tbin);
xticklabels((opt.snippet(1):(10/opt.tbin):opt.snippet(2))*opt.tbin);
xlabel('time (sec)');
hbar=colorbar;
ylabel(hbar,'z-score');
set(gca,'TickLabelInterpreter','none')
%set(gca,'FontSize',14);
save_figs(paths.figs,hfig,'pdf');

%% plot real and predicted neural activity
hfig = figure('Position',[100 100 2200 400]);
hfig.Name = sprintf('%s c%d firing rate true and predicted snippet',opt.session,opt.example_cell_id);

subplot(2,1,1); hold on;
stairs(opt.tbin*(opt.snippet(1):opt.snippet(2)),y(opt.snippet(1):opt.snippet(2)),'k-');
for j = 1:numel(patch_times)
    plot([patch_times(j) patch_times(j)]*opt.tbin,ylim,'b-');
end
xticks([]);
yticks(0:3);
ylabel('spike counts');
set(gca,'FontSize',14);

subplot(2,1,2); hold on;
plot(opt.tbin*(opt.snippet(1):opt.snippet(2)),ysmooth(opt.snippet(1):opt.snippet(2)),'k-');
plot(opt.tbin*(opt.snippet(1):opt.snippet(2)),ypred(opt.snippet(1):opt.snippet(2)),'r-');
for j = 1:numel(patch_times)
    plot([patch_times(j) patch_times(j)]*opt.tbin,ylim,'b-');
end
plot_col = cool(numel(opt.rew_size));
for i = 1:numel(opt.rew_size)
    for j = 1:numel(rew_times{i})
        plot(rew_times{i}(j)*opt.tbin,max(ylim),'v','Color',plot_col(i,:),'MarkerFaceColor',plot_col(i,:));
    end
end
ylabel('firing rate');
xlim([opt.snippet(1) opt.snippet(2)]*opt.tbin);
xticks([opt.snippet(1) opt.snippet(2)]*opt.tbin);
yticks([]);
xticklabels([opt.snippet(1) opt.snippet(2)]*opt.tbin);
xlh=xlabel('time (sec)');
legend('smoothed firing rate','predicted firing rate');
set(xlh,'Position',[mean(xlim),min(ylim)]);
set(gca,'FontSize',14);
title(sprintf('corr=%0.2f',corr(ypred,ysmooth)));

save_figs(paths.figs,hfig,'png');
save_figs(paths.figs,hfig,'pdf');

%% model fit coefficients
beta_this = fit.beta_all(2:end,cellidx);
hfig = figure('Position',[100 100 2000 600]); hold on;
hfig.Name = sprintf('%s c%d GLM coefficients',opt.session,opt.example_cell_id);
my_scatter(1:numel(beta_this),beta_this,'k',0.5);
plot(xlim,[0 0],'k--');
xticks(1:numel(beta_this));
xticklabels(fit.var_name(2:end));
xtickangle(90);
ylabel('beta');
title(sprintf('%s c%d',opt.session,opt.example_cell_id),'Interpreter','none');
set(gca,'TickLabelInterpreter','none')
set(gca,'FontSize',14);
save_figs(paths.figs,hfig,'png');
