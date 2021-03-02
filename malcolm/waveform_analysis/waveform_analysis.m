paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.neuropix_utils = 'C:\code\neuropix_utils';
addpath(genpath(paths.neuropix_utils));
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));

paths.output = 'C:\data\patch_foraging_neuropix\waveforms';
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

t_spk = 1000*(-opt.samp_before:opt.samp_after)/opt.samp_freq;
    
%%
session = '80_20200317';
dat = load(fullfile(paths.data,session));

good_cells = dat.sp.cids(dat.sp.cgs==2);

%%
pb = ParforProgressbar(numel(good_cells));
mean_waveform = nan(numel(good_cells),91);
parfor i = 1:numel(good_cells)
    fprintf('Getting waveform for cell %d/%d: %d\n',i,numel(good_cells),good_cells(i));
    mean_waveform(i,:) = get_waveform('m80_200317',good_cells(i),opt);
    pb.increment();
end

% save data
save(fullfile(paths.output,session),'mean_waveform');

%%
mean_waveform_blsub = nan(size(mean_waveform));
for i = 1:size(mean_waveform_blsub,1)
    mean_waveform_blsub(i,:) = mean_waveform(i,:) - mean(mean_waveform(i,1:opt.samp_before/2));
end

%% pca
[coeff,score,~,~,expl] = pca(zscore(mean_waveform_blsub,[],2));
figure;
plot3(score(:,1),score(:,2),score(:,3),'ko');
xlabel('PC1');
ylabel('PC2');
zlabel('PC3');
grid on;