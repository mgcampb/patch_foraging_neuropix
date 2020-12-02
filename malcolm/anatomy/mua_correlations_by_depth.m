addpath(genpath('C:\code\spikes'));
addpath(genpath('C:\code\npy-matlab'));

paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data\';
paths.figs = 'C:\figs\patch_foraging_neuropix\anatomy\mua_correlations_by_depth_MB_cohort_by_insertion_depth\good_units';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.tbin = 0.2;
opt.depthbin = 100;
opt.data_set = 'mb';

%%
sessions = dir(sprintf('%s*.mat',paths.data));
sessions = {sessions.name}';
if strcmp(opt.data_set,'mb')
    sessions = sessions(~contains(sessions,'mc'));
elseif strcmp(opt.data_set,'mc')
    sessions = sessions(contains(sessions,'mc'));
end

%%
for sesh_idx = 1:numel(sessions)
    fprintf('session %d/%d: %s\n',sesh_idx,numel(sessions),sessions{sesh_idx});
    
    clear anatomy
    load(fullfile(paths.data,sessions{sesh_idx}));
    
    if ~exist('anatomy','var')
        continue
    end
    
    if ~isfield(anatomy,'insertion_depth')
        continue
    end

    [~,spikeDepths] = templatePositionsAmplitudes(sp.temps,sp.winv,sp.ycoords,sp.spikeTemplates,sp.tempScalingAmps);
    spikeDepths = anatomy.insertion_depth-spikeDepths;
    
    if strcmp(opt.data_set,'mb')
        good_cells = anatomy.cell_labels.CellID;
        depth_cell = nan(size(good_cells));
        for cIdx = 1:numel(good_cells)
            depth_cell(cIdx) = median(spikeDepths(sp.clu==good_cells(cIdx)));
        end
    end
        
    opt.tbinedge = 0:opt.tbin:max(sp.st);
    opt.depthbinedge = 0:opt.depthbin:max(spikeDepths);

    % spike counts in each depth/time bin
    good_cells = sp.cids(sp.cgs==2);
    keep_spike = ismember(sp.clu,good_cells);
    N = histcounts2(spikeDepths(keep_spike),sp.st(keep_spike),opt.depthbinedge,opt.tbinedge);

    rho = corr(N');

    %% make fig
    hfig = figure('Position',[200 200 1600 800]);
    
    subplot(1,3,1:2);
    depthbincent = opt.depthbinedge(1:end-1)+opt.depthbin/2;
    imagesc(depthbincent,depthbincent,rho);
    axis square
    title(sessions{sesh_idx},'Interpreter','none');
    colorbar
    xlabel('depth from surface (um)');
    ylabel('depth from surface (um)');
    if strcmp(opt.data_set,'mb')
        if exist('anatomy','var')
            hold on;
            cortex = anatomy.cell_labels.Cortex;
            plot(ones(sum(cortex),1),depth_cell(cortex),'ro','MarkerSize',15)
            plot(depth_cell(cortex),ones(sum(cortex),1),'ro','MarkerSize',15)
            plot(ones(sum(~cortex),1),depth_cell(~cortex),'ko','MarkerSize',15)
            plot(depth_cell(~cortex),ones(sum(~cortex),1),'ko','MarkerSize',15)
        end
    end
    set(gca,'FontSize',18);

    subplot(1,3,3);
    norm_count = sum(N,2)/max(sum(N,2));
    h=plot(norm_count,depthbincent);
    set(h,'HandleVisibility','off');
    set(gca, 'YDir','reverse')
    xlabel('norm MUA activity');
    ylabel('depth from surface (um)');
    if strcmp(opt.data_set,'mb')
        if exist('anatomy','var')
            hold on;
            plot(zeros(sum(cortex),1),depth_cell(cortex),'ro','MarkerSize',15)
            plot(zeros(sum(~cortex),1),depth_cell(~cortex),'ko','MarkerSize',15)
            legend({'cortex units','non-cortex units'},'Location','southeast');
        end
    end
    set(gca,'FontSize',18);
    
    saveas(hfig,fullfile(paths.figs,sessions{sesh_idx}(1:end-4)),'png');
end