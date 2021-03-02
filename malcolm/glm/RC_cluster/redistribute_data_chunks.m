paths = struct;
paths.results1 = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210204_DVs_run2';
paths.results2 = 'C:\data\patch_foraging_neuropix\GLM_output\run_20210205_DVs';
paths.data_chunks = 'C:\data\patch_foraging_neuropix\data_chunks';

chunks = dir(fullfile(paths.results1,'*.mat'));
chunks = {chunks.name}';

load(fullfile(paths.data_chunks,'data_chunks_20_cells.mat')); % session and cellID

%%
job_time1 = nan(numel(session),1);
job_time2 = nan(numel(session),1);
cells_per_chunk = nan(numel(session),1);
session_all = {};
good_cells_all = [];
for i = 1:numel(chunks)
    chunk_idx = str2double(chunks{i}(6:8));
    fprintf('Processing chunk %d/%d\n',i,numel(chunks));
    dat1 = load(fullfile(paths.results1,chunks{i}));
    dat2 = load(fullfile(paths.results2,chunks{i}));
    job_time1(chunk_idx) = sum(dat1.run_times);
    job_time2(chunk_idx) = sum(dat2.run_times);
    cells_per_chunk(chunk_idx) = numel(dat1.good_cells);
    session_all = [session_all; repmat({dat1.opt.session},numel(dat1.good_cells),1)];
    good_cells_all = [good_cells_all; dat1.good_cells'];
end

job_time1(155) = nan;
job_time2(155) = nan;
cells_per_chunk(155) = 19;

session_all = [session_all; repmat({'78_20200312'},18,1)];
good_cells_all = [good_cells_all; [351;357;359;360;363;367;372;374;376;377;379;382;385;389;390;391;394;395]];

%% redistribute cells

cells_per_session = [10 10 10 10 10 10 10 10 10 5 5 5 10 10 10 10 10 10 10];
tiny_jobs = job_time1<200 & job_time2<200;
small_jobs = job_time1<500 & job_time2<500 & ~tiny_jobs;
huge_jobs = job_time1>1500 | job_time2>1500;
big_jobs = job_time1>1000 | job_time2>1000 & ~huge_jobs;

%% redistribute cells across jobs for efficiency

session_uniq = unique(session);
% combine cells by session
cells_by_session = cell(numel(session_uniq),1);
for i = 1:numel(session_uniq)
    cells_by_session{i} = good_cells_all(strcmp(session_all,session_uniq{i}));
end

session_new = {};
cellID_new = {};
counter = 1;
for i = 1:numel(session_uniq)
    good_cells_this = cells_by_session{i};
    num_chunks = ceil(numel(good_cells_this)/cells_per_session(i));
    chunks = round(linspace(0,numel(good_cells_this),num_chunks+1));
    for j = 1:num_chunks
        session_new{counter} = session_uniq{i};
        cellID_new{counter} = good_cells_this(chunks(j)+1:chunks(j+1));
        counter = counter+1;
    end
end

%% save new data chunks
session = session_new';
cellID = cellID_new';
save(fullfile(paths.data_chunks,'data_chunks_redistributed_384.mat'),'session','cellID');