% script to compute spike time cross correlation between pairs of cells
% with jitter for assessing statistical significance
% MGC 4/13/2021

d = datetime; d.Format = 'uuuuMMdd-HHmmss'; % output saved by date

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = fullfile('C:\data\patch_foraging_neuropix\spike_time_cross_correlation\jitter',char(d));
paths.spikes = 'C:\code\spikes'; % spikes repo
addpath(genpath(paths.spikes));
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.sig_cells = 'C:\data\patch_foraging_neuropix\sig_cells';
paths.figs = fullfile(paths.results,'figs');

opt = struct;
opt.binsize = 0.001; % in seconds
opt.max_lag = 50; % in bins; for cross correlation
opt.max_dist = 200; % in microns
opt.sig_cells = 'sig_cells_table_20210413'; % which sig_cells table to use
opt.make_plots = false;
opt.num_jitter = 100;
opt.jitter_sd = 0.003; % SD of normally distributed jitter, in seconds

%% load sig_cells table
load(fullfile(paths.sig_cells,opt.sig_cells));

session_all = unique(sig_cells.Session);

%% iterate over sessions
for sesh_idx = 19 % 1:numel(session_all)
    
    opt.session = session_all{sesh_idx};
    fprintf('Session %d/%d: %s\n',sesh_idx,numel(session_all),opt.session);
    
    % only keep cells from the sig cells table for this session
    good_cells = sig_cells.CellID(strcmp(sig_cells.Session,opt.session));

    %% load data
    dat = load(fullfile(paths.data,opt.session));

    %% get spike depths
    [~,spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps,dat.sp.winv,dat.sp.ycoords,dat.sp.spikeTemplates,dat.sp.tempScalingAmps);

    spike_depths = nan(numel(good_cells),1);
    for cIdx = 1:numel(good_cells)
        spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
    end

    %% get pairs

    % all possible pairs
    pairs_idx = nchoosek(1:numel(good_cells),2);

    % only keep pairs within the specified distance
    spike_dist = abs(diff(spike_depths(pairs_idx),[],2));
    keep_pair = spike_dist<opt.max_dist;
    pairs_idx = pairs_idx(keep_pair,:);

    % allocate matrix for xcorr results
    N_pairs = size(pairs_idx,1);
    xcorr_all = nan(N_pairs,opt.max_lag*2+1);
    xcorr_jit = nan(N_pairs,opt.max_lag*2+1,opt.num_jitter); % for xcorr on jittered data


    %% compute spike time cross correlations

    % window for computing cross correlation
    minT = 0;
    maxT = max(dat.sp.st);
    
    pb = ParforProgressbar(N_pairs);
    parfor pIdx = 1:N_pairs

        cell1 = good_cells(pairs_idx(pIdx,1));
        cell2 = good_cells(pairs_idx(pIdx,2));

        st1 = dat.sp.st(dat.sp.clu==cell1);
        st2 = dat.sp.st(dat.sp.clu==cell2);

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
        pb.increment();

    end

    %% save results
    if ~isfolder(paths.results)
        mkdir(paths.results);
    end
    save(fullfile(paths.results,opt.session),'xcorr_all','xcorr_jit','pairs_idx','opt','good_cells');

    %% make plots
    if opt.make_plots
        if ~isfolder(paths.figs)
            mkdir(paths.figs);
        end
        for i = 1:size(xcorr_all,1)
            fprintf('making fig for pair %d/%d\n',i,size(xcorr_all,1));
            this_pair = sprintf('c%d-c%d',good_cells(pairs_idx(i,1)),good_cells(pairs_idx(i,2)));
            h = figure; 
            plot(1000*opt.binsize*(-opt.max_lag:opt.max_lag),xcorr_all(i,:));
            title(this_pair);
            xlabel('lag (ms)'); ylabel('spike time xcorr');
            saveas(h,fullfile(paths.figs,this_pair),'png');
            close(h);
        end
    end
end