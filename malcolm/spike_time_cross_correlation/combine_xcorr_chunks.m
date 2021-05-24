% combines output from RCC cluster
% MGC 4/16/2021

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation\20210414_run1';

xcorr_all = [];
xcorr_jit = [];

rez_files = dir(fullfile(paths.results,'chunk*.mat'));
rez_files = {rez_files.name}';
for i = 1:numel(rez_files)
    fprintf('Loading chunk %d/%d\n',i,numel(rez_files));
    rez = load(fullfile(paths.results,rez_files{i}));
    xcorr_all = [xcorr_all; rez.xcorr_all];
    xcorr_jit = [xcorr_jit; rez.xcorr_jit];
end
xcorr_opt = rez.opt;

save(fullfile(paths.results,'rez_combined'),'xcorr_all','xcorr_jit','xcorr_opt');