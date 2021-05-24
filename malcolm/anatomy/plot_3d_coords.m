% Script to assign brain region to each unit using the anatomy struct
% output by Display_Probe_Track in SHARP-Track code
% MGC 8/19/2020

paths = struct;
paths.allenCCF = 'C:\code\allenCCF';
paths.npy_matlab = 'C:\code\npy-matlab';
addpath(genpath(paths.allenCCF));
addpath(genpath(paths.npy_matlab));

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

%% make wireframe brain

black_brain = true;
% create a new figure with wireframe
fwireframe = plotBrainGrid([], [], [], black_brain);
hold on; 
fwireframe.InvertHardcopy = 'off';

%% iterate over sessions
for sIdx = 1:numel(data_set_all)
    data_set = data_set_all{sIdx};
    fprintf('Session %d/%d: %s\n',sIdx,numel(data_set_all),data_set);
    % load data
    dat = load(fullfile(data_dir,data_set));
    if isfield(dat,'anatomy3d')
        grp1 = strcmp(dat.brain_region_rough,'PFC');
        grp2 = strcmp(dat.brain_region_rough,'Sub-PFC');
        Coords = dat.anatomy3d.Coords;
        plot3(Coords.AP(grp1)/10,Coords.ML(grp1)/10,Coords.DV(grp1)/10,'w.');
        plot3(Coords.AP(grp2)/10,Coords.ML(grp2)/10,Coords.DV(grp2)/10,'g.');
    end
end