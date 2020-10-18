% Script to assign brain region to each unit using the anatomy struct
% output by Display_Probe_Track in SHARP-Track code
% MGC 8/19/2020

data_dir = 'H:\My Drive\processed_neuropix_data\';
data_set = '79_20200229';

opt = struct;
opt.scale_depth = false; % whether or not to scale the probe so that insertion depth matches probe tip detected in histology
% otherwise we measure distance from probe tip directly


%%

% load data
dat = load(fullfile(data_dir,data_set));
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

% assign brain region to each unit
[~,~,bin] = histcounts(depth_from_surface,dat.anatomy.borders);
brain_region = dat.anatomy.brain_region(bin)';

% save results
cell_labels = table();
cell_labels.CellID = good_cells;
cell_labels.BrainRegion = brain_region;
cell_labels.DepthFromTip = depth_from_tip;
cell_labels.DepthFromSurface = depth_from_surface;
anatomy = dat.anatomy;
anatomy.cell_labels = cell_labels;
save(fullfile(data_dir,data_set),'anatomy','-append');