paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all';
session_all = session_all(contains(session_all,'mc'));

session_all = {'mc2_20201021'};

for i = 1:numel(session_all)
    dat = load(fullfile(paths.data,session_all{i}));
    good_cells_all = dat.sp.cids(dat.sp.cgs==2);
    brain_region_rough = cell(numel(good_cells_all),1);
    if isfield(dat,'anatomy')
        if isfield(dat.anatomy,'cell_labels')
            brain_region_rough(dat.anatomy.cell_labels.Cortex) = {'PFC'};
            brain_region_rough(~dat.anatomy.cell_labels.Cortex) = {'Sub-PFC'};
        else
            anatomy = dat.anatomy;

            % get depth
            [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps,dat.sp.winv,dat.sp.ycoords,dat.sp.spikeTemplates,dat.sp.tempScalingAmps);
            spike_depths = nan(numel(good_cells_all),1);
            for cidx = 1:numel(good_cells_all)
                spike_depths(cidx) = median(spike_depths_all(dat.sp.clu==good_cells_all(cidx)));
            end

            if isfield(anatomy,'insertion_depth')
                depth_from_surface = spike_depths-anatomy.insertion_depth;
            else
                depth_from_surface = nan;
            end

            if strcmp(dat.anatomy.target,'OFC')
                brain_region_rough(depth_from_surface>-3000) = {'PFC'};
                brain_region_rough(depth_from_surface<=-3000) = {'Sub-PFC'};
            elseif strcmp(dat.anatomy.target,'DMS')
                brain_region_rough(depth_from_surface<=-2000) = {'STR'};
                brain_region_rough(depth_from_surface>-2000) = {'MOs'};
            end
        end
        save(fullfile(paths.data,session_all{i}),'brain_region_rough','-append');
    end
end