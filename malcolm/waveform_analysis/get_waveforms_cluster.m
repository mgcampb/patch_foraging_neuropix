% Script to get raw waveforms from unprocessed NP data, using RC cluster
% MGC 2/19/2021

tic

%% paths

paths = struct;
paths.data_proc = '/n/holystore01/LABS/uchida_users/Users/mcampbell/processed_neuropix_data_patch_foraging';
paths.data_raw = '/n/holyscratch01/uchida_lab/Users/mcampbell';
paths.neuropix_utils = '/n/holystore01/LABS/uchida_users/Users/mcampbell/neuropix_utils';
addpath(genpath(paths.neuropix_utils));
paths.malcolm_functions = '/n/holystore01/LABS/uchida_users/Users/mcampbell/patchforaging_glm/malcolm_functions';
addpath(genpath(paths.malcolm_functions));
paths.spikes = '/n/holystore01/LABS/uchida_users/Users/mcampbell/patchforaging_glm/spikes';
addpath(genpath(paths.spikes));
paths.npy_matlab = '/n/holystore01/LABS/uchida_users/Users/mcampbell/patchforaging_glm/npy-matlab';
addpath(genpath(paths.npy_matlab));

paths.output = '/n/holyscratch01/uchida_lab/Users/mcampbell/waveforms';
if ~isfolder(paths.output)
    mkdir(paths.output);
end

%% analysis options

opt = struct;
opt.gain = 500;
opt.samp_freq = 30000;
opt.samp_before = 30; % samples before each spike to read
opt.samp_after = 60; % samples after each spike to read
opt.num_spikes = 200; % downsample num spikes for speed
opt.data_dir = paths.data_raw;

t_spk = 1000*(-opt.samp_before:opt.samp_after)/opt.samp_freq;

%% session
session_all = {'75_20200313','75_20200315',...
    '76_20200302','76_20200303','76_20200305','76_20200306','76_20200307','76_20200308',...
    '78_20200310','78_20200311','78_20200312','78_20200313'...
    '79_20200225','79_20200226','79_20200227','79_20200228','79_20200229'...
    '80_20200315','80_20200317'};
sesh_idx = str2double(getenv('SLURM_ARRAY_TASK_ID'));
session = session_all{sesh_idx};
session_raw_data = strcat('m',session([1:3 end-5:end]));
    
%% load processed dat
dat = load(fullfile(paths.data_proc,session));
good_cells = dat.sp.cids(dat.sp.cgs==2);

%%
mean_waveform = nan(numel(good_cells),opt.samp_before+opt.samp_after+1);
for i = 1:numel(good_cells)
    fprintf('Getting waveform for session %d/%d (%s), cell %d/%d: %d\n',sesh_idx,numel(session_all),session,i,numel(good_cells),good_cells(i));
    mean_waveform(i,:) = get_waveform(session_raw_data,good_cells(i),opt);
end

% save data
fprintf('Saving data...');
save(fullfile(paths.output,session),'mean_waveform','t_spk','opt');
fprintf('done\n');
toc