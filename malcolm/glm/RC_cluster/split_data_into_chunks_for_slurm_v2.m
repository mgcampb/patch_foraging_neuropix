% split data into chunks of roughly 10-20 cells each

% 5/30/2021
% EDITED to only use cells that passed GLM firing rate cut offs
% Cell that crashes fitting and needs to be removed: 78_20200312_c403


paths = struct;
paths.data = 'C:\data\patch_foraging_neuropix\GLM_output\20210816_50ms_bins';
paths.output = 'C:\data\patch_foraging_neuropix\data_chunks';

opt = struct;
opt.target_num_cells_per_chunk = 10; % added hack to make this 5 for m78 because fitting takes forever (long PRTs)

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'));

%%
session = {};
cellID = {};
counter = 1;
for i = 1:numel(session_all)
    fprintf('Analyzing session %d/%d: %s\n',i,numel(session_all),session_all{i});
    dat = load(fullfile(paths.data,session_all{i}));
    if contains(session_all{i},'78') % manually make m78 have only 5 cells per session because these sessions were long and took forever
        cells_per_chunk = 5;
    else
        cells_per_chunk = opt.target_num_cells_per_chunk;
    end
    if isfield(dat,'anatomy') && isfield(dat,'brain_region_rough')
        good_cells = dat.good_cells;
        if strcmp(session_all{i},'78_20200312') % manually remove the one problematic cell from m78 0312
            good_cells(good_cells==403) = [];
        end
        num_chunks = floor(numel(good_cells)/cells_per_chunk);
        chunks = round(linspace(0,numel(good_cells),num_chunks+1));
        for j = 1:num_chunks
            session{counter} = session_all{i};
            cellID{counter} = good_cells(chunks(j)+1:chunks(j+1));
            counter = counter+1;
        end
    end
end

%% save
save(fullfile(paths.output,sprintf('data_chunks_%d_cells_50ms.mat',opt.target_num_cells_per_chunk)),'session','cellID');