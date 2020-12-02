addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison';
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_correlations_between_pred_and_smoothed_firing_rate';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.tbin = 0.02;
opt.smooth_sigma_fr = 0.05; % in seconds, for smoothing spike counts
opt.brain_region = 'PFC';
opt.data_set = 'mb';

%%
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

%%
hfig = figure('Position',[50 50 2200 1000]);
hfig.Name = sprintf('Correlations between predicted and smoothed firing rate %s cohort %s',opt.data_set,opt.brain_region);
ncol = 7;
nrow = 4;
counter = 1;
for sesh_idx = 1:numel(session_all)
    
    fprintf('Analyzing session %d/%d: %s\n',sesh_idx,numel(session_all),session_all{sesh_idx});
    
    %% load model fit data

    fit = load(fullfile(paths.results,session_all{sesh_idx}));
    if sum(strcmp(fit.brain_region_rough,opt.brain_region))==0
        continue
    end

    %% load spike data

    dat = load(fullfile(paths.data,session_all{sesh_idx}));
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

    %% correlate predicted and smoothed firing rate for all neurons

    r_all = nan(size(fit.good_cells));
    for i = 1:numel(fit.good_cells)
        cellidx = find(fit.good_cells==fit.good_cells(i));
        ysmooth = gauss_smoothing(spikecounts_filt(:,cellidx),opt.smooth_sigma_fr/opt.tbin);
        ysmooth = ysmooth(in_patch);
        ypred = exp([ones(size(fit.X_full,1),1) fit.X_full] * fit.beta_all(:,cellidx)); % predict based on GLM
        r_all(i) = corr(ysmooth,ypred);
    end
    
    %% plot
    brain_region_rough = dat.brain_region_rough(ismember(good_cells_all,fit.good_cells));
    keep = strcmp(brain_region_rough,opt.brain_region);
    if sum(keep)>0
        subplot(nrow,ncol,counter);
        histogram(r_all(keep),20);
        title(session_all{sesh_idx},'Interpreter','none');
        xlabel('Corr');
        ylabel('Num cells');
        counter = counter+1;
    end
    
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');