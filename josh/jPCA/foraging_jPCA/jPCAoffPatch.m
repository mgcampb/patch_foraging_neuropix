%% Perform jPCA as described by Churchland et al 2012
%  jPCA package from the Churchland lab website, also requires
%  BioInformatics Toolbox

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

%% Extract FR matrices and idx vectors to FRandTimes structure

FRandTimes = struct;

for sIdx = 1:2 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    
    % compute firing rate matrix
    tic
    [fr_mat, tbincent] = calcFRVsTime(good_cells, dat, opt);
    toc

    FRandTimes(sIdx).fr_mat = fr_mat;

end

%% Make cue and leave-aligned condition averages in jPCA format
%  4 conditions: avg cue forall, avg leave sep by rewsize

jPCA_data = {};
for sIdx = 1:2
    jPCA_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    dat = load(fullfile(paths.data,session));
    patches = dat.patches;
    patchType = patches(:,2);
    
    rew_size = mod(dat.patches(:,2),10);
    N0 = mod(round(dat.patches(:,2)/10),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    rew_ms = dat.rew_ts*1000;
    
    pre_ms = 1000;
    post_ms = 1000;
    times = -pre_ms:tbin_ms:post_ms;
    
    pre_ix = pre_ms / tbin_ms;
    post_ix = post_ms / tbin_ms;

    leave_align_pre = round((patchleave_ms - pre_ms) / tbin_ms);
    leave_align_post = round((patchleave_ms + post_ms) / tbin_ms);
    
    % start with leave-aligned divided by rewsize
    rew_counter = 1;
    for iRewsize = [1,2,4]
        iRewsize_trials = find(rew_size == iRewsize);
        temp_fr_mat = {};
        for j = 1:numel(iRewsize_trials)
            iTrial = iRewsize_trials(j);
            if min(abs(patchleave_ms(iTrial) - rew_ms)) > pre_ix
                leave_ix = round(patchleave_ms(iTrial) / tbin_ms);
                temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,leave_align_pre(iTrial):leave_align_post(iTrial));
            end
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter).A = mean_condition_fr';
        jPCA_data{sIdx}(rew_counter).times = times';
        
        % now align to first rew
        for j = 1:numel(iRewsize_trials)
            iTrial = iRewsize_trials(j);
            [m,i] = min(abs(patchstop_ms(iTrial) - rew_ms));
            first_rew_ix = round(rew_ms(i) / tbin_ms);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,first_rew_ix - pre_ix:first_rew_ix + post_ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter+3).A = mean_condition_fr';
        jPCA_data{sIdx}(rew_counter+3).times = times';
        
        rew_counter = rew_counter + 1;
    end
end


%% Now perform jPCA for leave conditions
close all;
for sIdx = 1:2
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    mouse = session(1:2); 
    date = session(8:end);
    
    % plotting the first jPCA plane for 2000 ms of data, using 6 PCs (the default)
    times = -pre_ms:tbin_ms:post_ms;
%     times = -1000:tbin_ms:0;
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    jPCA_params.substRawPCs = true;
    [Projection, Summary] = jPCA(jPCA_data{sIdx}, times, jPCA_params);
    
    plot_params.substRawPCs = false;
    plot_params.colors = {[.25 1 .25],[.25 .75 .25],[0 .5 0],[.5 .5 .5],[.5 1 1],[.5 .5 1]};
    phaseSpace(Projection, Summary,plot_params);  % makes the plot
    
%     % make a movie!
    movParams.colors = {[.25 1 .25],[.25 .75 .25],[0 .5 0],[.5 .5 .5],[.5 1 1],[.5 .5 1]};
    movParams.times = times;
    movParams.substRawPCs = true;
%     % change this to allow for path change
%     movParams.fname = sprintf('m%s_%s_offPatchjPCA',mouse,date);
%     phaseMovie(Projection,Summary,movParams);
%     
%     % show rotation
%     fname = sprintf('m%s_%s_rotation_offPatch',mouse,date);
%     pixelsToGet = [25 35 280 280];
%     rotationMovie(Projection, Summary, times, 70, 70,pixelsToGet,movParams.colors,fname);  % 70 steps to full rotation.  Show all 70.

end

%% Check out the PETH
close all;
times = -1000:tbin_ms:1000;
for sIdx = 1:2
    % First, jPCA with all neurons
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    jPCA_params.suppressBWrosettes = true;
    jPCA_params.suppressHistograms = true;
    jPCA_params.suppressText = true;
    [Projection, Summary] = jPCA(jPCA_data{sIdx}, times, jPCA_params);
    all_r2_ratio = Summary.R2_Mskew_2D / Summary.R2_Mbest_2D;
    clc % suppress output
    
    % get indices for sorting PETHs, then sort our jPCA data
    fr_mat = jPCA_data{sIdx}(2).A'; % middle of the road choice
    % sort PETH
    fr_mat_norm = zscore(fr_mat,[],2);
    [~,index] = max(fr_mat_norm');
    [~,index_sort] = sort(index);
    for condIdx = 1:6
        jPCA_data{sIdx}(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,index_sort); % sort our jPCA data
    end
    
    labels = {"1uL Leave","2uL Leave","4uL Leave","1uL First Rew","2uL First Rew","4uL First Rew"};
    
%     conditions = [1,3,4,6]; % all conditions
    conditions = 1:6;
    PETH_PCA_jPCAGrid(conditions,jPCA_data{sIdx},Projection,labels,"All Neurons")
end