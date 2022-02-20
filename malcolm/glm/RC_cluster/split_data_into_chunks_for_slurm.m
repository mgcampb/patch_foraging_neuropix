% split data into chunks of roughly 10-20 cells each

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.output = 'C:\data\patch_foraging_neuropix';

opt = struct;
opt.target_num_cells_per_chunk = 10;

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
    if isfield(dat,'anatomy') && isfield(dat,'brain_region_rough')
        good_cells = dat.sp.cids(dat.sp.cgs==2);
        num_chunks = floor(numel(good_cells)/opt.target_num_cells_per_chunk);
        chunks = round(linspace(0,numel(good_cells),num_chunks+1));
        for j = 1:num_chunks
            session{counter} = session_all{i};
            cellID{counter} = good_cells(chunks(j)+1:chunks(j+1));
            counter = counter+1;
        end
    end
end

%% save
save(fullfile(paths.output,sprintf('data_chunks_%d_cells.mat',opt.target_num_cells_per_chunk)),'session','cellID');