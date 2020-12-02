% makes pie charts of neurons with non-zero coefficients

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\firing_rate_in_vs_out_of_patch';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mc';
opt.tbin = 0.02;

%%
session_all = dir(fullfile(paths.data,'*.mat'));
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
fr_all = [];
for session_idx = 1:numel(session_all)
    opt.session = session_all{session_idx};
    fprintf('Analyzing session %d/%d: %s\n',session_idx,numel(session_all),opt.session);

    %% load data    
    dat = load(fullfile(paths.data,opt.session));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);
    
    if isfield(dat,'brain_region_rough')
        good_cells_all = good_cells_all(strcmp(dat.brain_region_rough,opt.brain_region));
        if isempty(good_cells_all)
            continue
        end
    else
        continue
    end
        
    
    %% compute binned spikecounts for each cell
    
    t = dat.velt;
    spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
    for cIdx = 1:numel(good_cells_all)
        spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
        spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
    end
    
    %% get patch num for each patch
    
    patch_num = nan(size(t));
    for i = 1:size(dat.patchCSL,1)
        % include one time bin before patch stop to catch the first reward
        patch_num(t>=(dat.patchCSL(i,2)-opt.tbin) & t<=dat.patchCSL(i,3)) = i;
    end
    in_patch = ~isnan(patch_num);
    
    %% compute in patch and out of patch firing rate
    T_in_patch = sum(in_patch) * opt.tbin;
    T_out_patch = sum(~in_patch) * opt.tbin;
    N_in_patch = sum(spikecounts_whole_session(in_patch,:));
    N_out_patch = sum(spikecounts_whole_session(~in_patch,:));
    fr_in_patch = N_in_patch/T_in_patch;
    fr_out_patch = N_out_patch/T_out_patch;
    fr_all = [fr_all [fr_in_patch; fr_out_patch]];
end

%%

hfig = figure('Position',[200 200 1000 400]);
hfig.Name = sprintf('Firing rate in vs out of patch %s %s cohort',opt.brain_region,opt.data_set);

subplot(1,2,1);
plot(fr_all(2,:),fr_all(1,:),'ko');
title(opt.brain_region);
xlabel('Firing rate, out of patch');
ylabel('Firing rate, in patch');
refline(1,0);

subplot(1,2,2);
diff = (fr_all(1,:)-fr_all(2,:))./(fr_all(1,:)+fr_all(2,:));
histogram(diff)
title(opt.brain_region);
ylabel('Num. cells');
xlabel('Firing rate change index');

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');