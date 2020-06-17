% example_cell.m
% script to run GLM on an example cell
% MGC 6/16/2020

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';

opt = struct;
opt.session = '80_20200315.mat';
opt.TimeBin = 0.02;

opt.example_cell_id = 365;

%% load data
dat = load(fullfile(paths.data,opt.session));

%% spiketrain
good_cells = dat.sp.cids(dat.sp.cgs==2);
spike_t = dat.sp.st(dat.sp.clu==opt.example_cell_id);
spiketrain = histc(spike_t,dat.velt);

%% run GLM
[bestModels,allModelTestFits,tuning_curves,final_pval,fig1] = create_glm(dat.vel',spiketrain,opt);