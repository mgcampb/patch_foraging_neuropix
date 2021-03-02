% Script to assign brain region to each unit using the anatomy struct
% output by Display_Probe_Track in SHARP-Track code
% MGC 8/19/2020

data_dir = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data\';

data_set_all = dir(fullfile(data_dir,'*.mat'));
data_set_all = {data_set_all.name}';
for i = 1:numel(data_set_all)
    data_set_all{i} = data_set_all{i}(1:end-4);
end
data_set_all = data_set_all(~contains(data_set_all,'mc'));

opt = struct;
opt.scale_depth = false; % whether or not to scale the probe so that insertion depth matches probe tip detected in histology
% otherwise we measure distance from probe tip directly


%% iterate over sessions
for sIdx = 1:numel(data_set_all)
    data_set = data_set_all{sIdx};
    fprintf('Session %d/%d: %s\n',sIdx,numel(data_set_all),data_set);
    % load data
    dat = load(fullfile(data_dir,data_set));
    if isfield(dat,'anatomy3d')
        
        [~, spikeDepths] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);

        % get spike depths for all good units (distance in um from tip of probe)
        good_cells = dat.sp.cids(dat.sp.cgs==2)';
        depth_from_tip = nan(numel(good_cells),1);
        for cIdx = 1:numel(good_cells)
            depth_from_tip(cIdx) = median(spikeDepths(dat.sp.clu==good_cells(cIdx)));
        end

        % compute distance from the surface of the brain (either scaled so that
        % probe tip matches insertion depth or not)
        probe_tip_final = max(dat.anatomy.probe_tip, dat.anatomy.insertion_depth); % if probe_tip < insertion_depth, take insertion_depth instead
        if opt.scale_depth
            depth_from_surface = probe_tip_final - depth_from_tip * dat.anatomy.scale_factor;
        else
            depth_from_surface = probe_tip_final - depth_from_tip;
        end


        %% Get 3d coords

        anatomy3d = dat.anatomy3d;
        Coords = table;
        Coords.CellID = good_cells;
        Coords_this = anatomy3d.probe_entry_point + (depth_from_surface/10) * anatomy3d.probe_direction ;
        Coords_this = Coords_this * 10; % convert to um
        Coords.AP = Coords_this(:,1);
        Coords.DV = Coords_this(:,2);
        Coords.ML = Coords_this(:,3);
        anatomy3d.Coords = Coords;

        %% save results
        save(fullfile(data_dir,data_set),'anatomy3d','-append');
    end
end