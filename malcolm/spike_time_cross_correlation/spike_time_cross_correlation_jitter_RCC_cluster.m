% script to compute spike time cross correlation between pairs of cells
% with jitter for assessing statistical significance
% Written to run on RCC cluster, analyzing a chunk of the data
% MGC 4/13/2021

tic

d = datetime; d.Format = 'uuuuMMdd-HHmmss'; % output saved by date

paths = struct;
paths.data = '/n/holystore01/LABS/uchida_users/Users/mcampbell/spike_time_cross_correlation';
paths.results = fullfile(paths.data,'results',char(d));

opt = struct;
opt.data_file = 'data_organized_for_spike_t_crosscorr_RCC_cluster.mat'; % name of file with organized data
opt.chunk_idx = str2double(getenv('SLURM_ARRAY_TASK_ID')); % which data chunk to process
opt.binsize = 0.001; % in seconds
opt.max_lag = 50; % in bins; for cross correlation
opt.num_jitter = 2;
opt.jitter_sd = 0.003; % SD of normally distributed jitter, in seconds

%% load organized data
load(fullfile(paths.data,opt.data_file));

%% get chunk of data to analyze

chunk_this = chunks{opt.chunk_idx};

% allocate matrix for xcorr results
N_pairs = numel(chunk_this);
xcorr_all = nan(N_pairs,opt.max_lag*2+1);
xcorr_jit = nan(N_pairs,opt.max_lag*2+1,opt.num_jitter); % for xcorr on jittered data


%% compute spike time cross correlations for this data chunk
for pIdx = 1:N_pairs
    
    uid1 = pairs_table.UniqueID1{chunk_this(pIdx)};
    uid2 = pairs_table.UniqueID2{chunk_this(pIdx)};
    
    fprintf('Pair %d/%d: %s %s\n',pIdx,N_pairs,uid1,uid2);
    
    cellidx1 = find(strcmp(sig_cells.UniqueID,uid1));
    cellidx2 = find(strcmp(sig_cells.UniqueID,uid2));
    
    st1 = sig_cells.SpikeTimes{cellidx1};
    st2 = sig_cells.SpikeTimes{cellidx2};
    
    %% compute cross corr
    
    % get xcorr time window for this session
    session_idx = strcmp(xcorr_time_window.Session,pairs_table.Session{pIdx});
    minT = xcorr_time_window.MinT(session_idx);
    maxT = xcorr_time_window.MaxT(session_idx);

    st1_bin = histcounts(st1,minT:opt.binsize:maxT);
    st2_bin = histcounts(st2,minT:opt.binsize:maxT);

    st_xcorr = xcorr(st1_bin,st2_bin,opt.max_lag);
    xcorr_all(pIdx,:) = st_xcorr;

    %% jitter
    xcorr_jit_this = nan(opt.max_lag*2+1,opt.num_jitter);
    for jIdx = 1:opt.num_jitter
        jit1 = randn(size(st1)) * opt.jitter_sd;
        jit2 = randn(size(st2)) * opt.jitter_sd;
        st1_jit = st1+jit1;
        st2_jit = st2+jit2;
        st1_bin = histcounts(st1_jit,minT:opt.binsize:maxT);
        st2_bin = histcounts(st2_jit,minT:opt.binsize:maxT);
        st_xcorr = xcorr(st1_bin,st2_bin,opt.max_lag);
        xcorr_jit_this(:,jIdx) = st_xcorr;
    end
    xcorr_jit(pIdx,:,:) = xcorr_jit_this;

end

%% save results
if ~isfolder(paths.results)
    mkdir(paths.results);
end
save(fullfile(paths.results,sprintf('chunk%03d',opt.chunk_idx)),'xcorr_all','xcorr_jit','opt');

toc