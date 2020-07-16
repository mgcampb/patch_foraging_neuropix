%% Perform seqNMF and use this to calculate sequenciness for data on patch

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_matlab/neuroPixelsData/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_matlab/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
tbin_ms = opt.tbin*1000; % for making index vectors
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

opt.maxtime_alignleave = 3000; %MB
opt.maxtime_alignstop = 3000; %MB

%MB *** need to incorporate this leaveBuffer in other plots aligned to patchStop, patchCue, etc
% can later do this trimming more precisely but looking directly at velocity per trial
opt.leaveBuffer_ms = 500; %MB 'buffer' time prior to 'patchLeave' being triggered to leave out of any plots aligned to patchStop or patchCue, to avoid corrupting on patch PSTHs etc w running just before patchLeave (since running will occur at different times relative to patchStop per patch)

opt.additionalBuffer = 200; %MB - ms to cut off from end of trials aligned to patchStop (temporary solution to reduce impact of slight misalignment of trials after running PCA on firing rate while onPatch), this is in addition to the leaveBuffer_ms


sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and perform seqNMF

for sIdx = 3:3 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % behavioral events to align to
    rew_size = mod(dat.patches(:,2),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    rew_ms = dat.rew_ts * 1000;
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
    
    p.patchstop_ms = patchstop_ms(keep);
    p.patchleave_ms = patchleave_ms(keep);
    
    % compute firing rate matrix
%     tic
%     [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
%     toc
    
    % seems to converge to not much
    K = 5;
    L = 50;
    lambda =.005;
    shg; clf
    display('Running seqNMF on first 1000 indices')
    [W,H] = seqNMF(fr_mat(:,1:1000),'K',K, 'L', L,'lambda', lambda);

end