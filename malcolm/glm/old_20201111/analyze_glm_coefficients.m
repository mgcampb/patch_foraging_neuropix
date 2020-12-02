addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201109_MB_cohort_1SecKerns';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pie_charts\20201109_2uL_and_4uL_MB_cohort_1SecKerns';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

% session_all = {'mc2_20201020','mc4_20201024','mc4_20201026'};

beta_all = [];

for i = 1:numel(session_all)
    fprintf('Loading session %d/%d\n',i,numel(session_all));
    dat = load(fullfile(paths.results,session_all{i}));
    % renormalize
    beta_this = dat.beta_all(:,dat.anatomy.cell_labels.Cortex(ismember(dat.anatomy.cell_labels.CellID,dat.good_cells)));
    beta_all = [beta_all beta_this];
    % beta_all = [beta_all dat.beta_all];
    figure;
    histogram(dat.run_times);
end

X = zscore(beta_all(9:end,:));
[coeff,score,~,~,expl] = pca(X');
figure;
plot(score(:,1),score(:,2),'ko');

% 
% idx1 = 36;
% idx2 = 51;
% 
% keep = abs(beta_all(idx1,:))>0 & abs(beta_all(idx2,:))>0;
% 
% figure;
% plot(beta_all(idx1,keep),beta_all(idx2,keep),'ko');
% refline(1,0);