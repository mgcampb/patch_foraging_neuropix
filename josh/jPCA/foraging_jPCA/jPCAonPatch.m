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
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % Trial level features for decision variable creation
    patches = dat.patches;
    patchCSL = dat.patchCSL;
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
    
    p.patchstop_ms = patchstop_ms(keep);
    p.patchleave_ms = patchleave_ms(keep);
    
    new_fr_mat = true;
    if new_fr_mat == true
        % compute firing rate matrix
        tic
%         [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
        [fr_mat, tbincent] = calcFRVsTime(good_cells,dat,opt); % calc from full matrix
        toc
    end
    
%     patchstop_ms = p_out.patchstop_ms + 9;
%     patchleave_ms = p_out.patchleave_ms + 9;

    leave_buffer = 500;
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms/ tbin_ms) + 1;
    patchleave_ix = round((patchleave_ms - leave_buffer) / tbin_ms) + 1;
    
    FRandTimes(sIdx).fr_mat = fr_mat;
    FRandTimes(sIdx).stop_leave_ms = [patchstop_ms patchleave_ms];
    FRandTimes(sIdx).stop_leave_ix = [patchstop_ix patchleave_ix];
    
end

%% Now turn FRandTimes into jPCA format for 2 second conditions
jPCA_data = {};
for sIdx = 3:3
    jPCA_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms;
    times = -1000:tbin_ms:1000;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
  
    rew_counter = 1;
    
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        temp_fr_mat = {length(trials10x)};
        for j = 1:numel(trials10x)
            iTrial = trials10x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,(stop_ix):stop_ix + sec2ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter).A = mean_condition_fr';
        jPCA_data{sIdx}(rew_counter).times = times';
        
        temp_fr_mat = {length(trials11x)};
        for j = 1:numel(trials11x)
            iTrial = trials11x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:(stop_ix + sec2ix));
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter+3).A = mean_condition_fr'; % 3 reward sizes
        jPCA_data{sIdx}(rew_counter+3).times = times';
        
        rew_counter = rew_counter + 1;
    end
end

%% quick code to visualize jPCA_data PETHs

for sIdx = 3:3
    avg40 = zscore(jPCA_data{sIdx}(3).A',[],2);
    [~,idx] = max(avg40,[],2);
    [~,idx_sort40] = sort(idx);
    figure();colormap('jet')
    subplot(1,2,1)
    imagesc(flipud(avg40(idx_sort40,:)))
    title("Average 40 PETH")
    xlabel("Time (ms)")
    xticks([1 25 50 75 100])
    xticklabels([0 500 1000 1500 2000])
    avg44 = zscore(jPCA_data{sIdx}(6).A',[],2);
    subplot(1,2,2)
    imagesc(flipud(avg44(idx_sort40,:)))
    title("Average 44 PETH")
    xlabel("Time (ms)")
    xticks([1 25 50 75 100])
    xticklabels([0 500 1000 1500 2000])
    
end

%% Now perform jPCA for 2 second conditions
close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    mouse = session(1:2); 
    date = session(8:end);
    
    % plotting the first jPCA plane for 2000 ms of data, using 6 PCs (the default)
    times = -1000:tbin_ms:1000;
%     times = -1000:tbin_ms:0;
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    jPCA_params.suppressBWrosettes = true;
    jPCA_params.suppressText = false;
    [Projection, Summary] = jPCA(jPCA_data{sIdx}, times, jPCA_params);
    plot_params.substRawPCs = false;
    plot_params.colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};
%     [colorStruct, haxP, vaxP] = phaseSpace(Projection, Summary,plot_params);  % makes the plot
    
    % make a movie!
%     movParams.colors = {[.5 .5 .5],[.5 1 1],[.5 .5 1],[0 0 0],[0 1 1],[0 0 1]};
    movParams.colors = {[.5 1 1],[.75 .75 1],[1 .5 1],[0 1 1],[.5 .5 1],[1 0 1]};
    movParams.times = times;
    movParams.substRawPCs = false;
    movParams.conds2plot = 'all';
    % change this to allow for path change
    movParams.fname = sprintf('m%s_%s_RRjPCA',mouse,date);
    phaseMovie(Projection,Summary,movParams);

%     % show rotation
    fname = sprintf('m%s_%s_rotation_2sec',mouse,date);
    pixelsToGet = [70 -90 600 590];
%     rotationMovie(Projection, Summary, times, 70, 70,pixelsToGet,movParams.colors,fname);  % 70 steps to full rotation.  Show all 70.

end

%% ok, let's see what's actually happening in jPC space 
close all;
times = -1000:tbin_ms:1000;

for sIdx = 1:1
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
    fr_mat = jPCA_data{sIdx}(3).A'; % middle of the road choice
    % sort PETH
    fr_mat_norm = zscore(fr_mat,[],2);
    [~,index] = max(fr_mat_norm');
    [~,index_sort] = sort(index);
    for condIdx = 1:6
        jPCA_data{sIdx}(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,index_sort); % sort our jPCA data
    end
    
    labels = {"10","20","40","11","22","44"};
    
    conditions = [1,3,4,6]; % all conditions
    PETH_PCA_jPCAGrid(conditions,jPCA_data{sIdx},Projection,labels,"Avg")
    
    %%% now perform jPCA for the sequency units %%%
    seq_jPCA_data = struct;
    for condIdx = 1:6
        seq_jPCA_data(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,50:150); % pull off the sequency bois
        seq_jPCA_data(condIdx).times = times;
    end
    [Projection, Summary] = jPCA(seq_jPCA_data, times, jPCA_params);
    seq_r2_ratio = Summary.R2_Mskew_2D / Summary.R2_Mbest_2D;
    clc % suppress output
    PETH_PCA_jPCAGrid(conditions,seq_jPCA_data,Projection,labels,"Mid-Responsive Neurons")
    
    %%% now perform jPCA for the ramping units %%%
    ramp_jPCA_data = struct;
    for condIdx = 1:6
        ramp_jPCA_data(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,end-100:end); % pull off the sequency bois
        ramp_jPCA_data(condIdx).times = times;
    end
    [Projection, Summary] = jPCA(ramp_jPCA_data, times, jPCA_params);
    clc % suppress output
    PETH_PCA_jPCAGrid(conditions,ramp_jPCA_data,Projection,labels,"Ramping Neurons")
    ramp_r2_ratio = Summary.R2_Mskew_2D / Summary.R2_Mbest_2D;
    
    %%% now perform jPCA for the ramping units + some reward responsive units %%%
    extreme_jPCA_data = struct;
    for condIdx = 1:6
%         extreme_jPCA_data(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,[1:50 (end-50:end)]); % pull off the sequency bois
        extreme_jPCA_data(condIdx).A = jPCA_data{sIdx}(condIdx).A(:,[1:50 end-50:end]); % pull off the sequency bois
        extreme_jPCA_data(condIdx).times = times;
    end
    
    [Projection, Summary] = jPCA(extreme_jPCA_data, times,jPCA_params);
    clc % suppress output
    PETH_PCA_jPCAGrid(conditions,extreme_jPCA_data,Projection,labels,"50 Ramp + 50 reward Neurons")
    extreme_r2_ratio = Summary.R2_Mskew_2D / Summary.R2_Mbest_2D;

    fprintf("All Neurons skewR2 ratio: %f \n",all_r2_ratio);
    fprintf("Mid-Responsive Neurons skewR2 ratio: %f \n",seq_r2_ratio);
    fprintf("Ramping Neurons skewR2 ratio: %f \n",ramp_r2_ratio);
    fprintf("50 Most Rampy + 50 Most Reward Responsive Neurons skewR2 ratio: %f \n",extreme_r2_ratio);
end

%% Prep jPCA for 2 second single trial analysis
% single trials are now the conditions

jPCA_data = {};
for sIdx = 3:3
    jPCA_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec2ix = 2000/tbin_ms;
    times = -1000:tbin_ms:1000;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(rewsize);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    rew_counter = 1;
%     colors = [.5 .5 .5;.5 1 1; .5 .5 1; 0 0 0; 0 1 1; 0 0 1]; 
    colors = [.5 1 1 ; .75 .75 1 ; 1 .5 1 ; 0 1 1 ; .5 .5 1 ; 1 0 1];
    singleTrialColors = {nTrials};
    for iRewsize = [1,2,4]
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        singleTrialColors(trials10x) = mat2cell(colors(rew_counter,:),1);
        singleTrialColors(trials11x) = mat2cell(colors(3 + rew_counter,:),1);
        rew_counter = rew_counter + 1;
    end
    
    % define jPCA data
    for iTrial = 1:nTrials
        if prts(iTrial) > 2.55
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            jPCA_data{sIdx}(iTrial).A = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec2ix)';
            jPCA_data{sIdx}(iTrial).times = times';
        end
    end

end

%% Perform jPCA on single trial data
close all
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    mouse = session(1:2); 
    date = session(8:end);
    
    % plotting the first jPCA plane for 2000 ms of data, using 6 PCs (the default)
    times = -1000:tbin_ms:1000;
%     times = -1000:tbin_ms:0;
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    jPCA_params.suppressBWrosettes = true;
    [Projection, Summary] = jPCA(jPCA_data{sIdx}([trials10x(1:8); trials11x(1:8)]), times, jPCA_params);
    plot_params.substRawPCs = false;
    plot_params.colors = singleTrialColors([trials10x(1:8); trials11x(1:8)]);
%     plot_params.colors = {[.5 .5 .5],[.5 1 1],[.5 .5 1],[0 0 0],[0 1 1],[0 0 1]};
    phaseSpace(Projection, Summary,plot_params);  % makes the plot
    
    % make a movie!
%     movParams.colors = {[.5 .5 .5],[.5 1 1],[.5 .5 1],[0 0 0],[0 1 1],[0 0 1]};
    movParams.times = times;
    movParams.substRawPCs = false;
    movParams.colors = singleTrialColors([trials10x(1:15); trials11x(1:10)]);
    % change this to allow for path change
    movParams.fname = sprintf('m%s_%s_2secjPCA',mouse,date);
    phaseMovie(Projection,Summary,movParams);
% %     
%     % show rotation
%     fname = sprintf('m%s_%s_rotation_2sec',mouse,date);
%     pixelsToGet = [25 35 280 280];
%     rotationMovie(Projection, Summary, times, 70, 70,pixelsToGet,movParams.colors,fname);  % 70 steps to full rotation.  Show all 70.

end

%% Now turn FRandTimes into jPCA format for 3 second conditions

jPCA_data = {};
for sIdx = 1:1
    jPCA_data{sIdx} = struct;
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec1ix = 1000/tbin_ms;
    sec2ix = 2000/tbin_ms;
    sec3ix = 3000/tbin_ms;
    times = -1000:tbin_ms:2000;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
  
    rew_counter = 1;
    
    for iRewsize = 4 
        trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.55);
        trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.55);
        trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
        trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
        
        temp_fr_mat = {length(trials100x)};
        for j = 1:numel(trials100x)
            iTrial = trials100x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter).A = mean_condition_fr';
        jPCA_data{sIdx}(rew_counter).times = times';
        
        temp_fr_mat = {length(trials110x)};
        for j = 1:numel(trials110x)
            iTrial = trials110x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter+1).A = mean_condition_fr'; % 3 reward sizes
        jPCA_data{sIdx}(rew_counter+1).times = times';
        
        temp_fr_mat = {length(trials101x)};
        for j = 1:numel(trials101x)
            iTrial = trials101x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter+2).A = mean_condition_fr'; % 3 reward sizes
        jPCA_data{sIdx}(rew_counter+2).times = times';
        
        temp_fr_mat = {length(trials111x)};
        for j = 1:numel(trials111x)
            iTrial = trials111x(j);
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            temp_fr_mat{j} = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix);
        end
        
        mean_condition_fr = mean(cat(3,temp_fr_mat{:}),3); % concatenate in third dimension, average over it
        jPCA_data{sIdx}(rew_counter+3).A = mean_condition_fr'; % 3 reward sizes
        jPCA_data{sIdx}(rew_counter+3).times = times';
        
        rew_counter = rew_counter + 4;
    end
end

%% Now perform jPCA for 3 second conditions
close all;

for sIdx = 1:1
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    mouse = session(1:2); 
    date = session(8:end);
    
    % plotting the first jPCA plane for 2000 ms of data, using 6 PCs (the default)
    times = -1000:tbin_ms:3000;
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    jPCA_params.suppressText = false;
    [Projection, Summary] = jPCA(jPCA_data{sIdx}, times, jPCA_params);
    
    plot_params.substRawPCs = false;
    plot_params.colors = {[1 0 1],[.75 .25 .75],[.25 .5 1],[0 1 1]};
%     plot_params.colors = {[0 0 0],rgb('hotpink'),[1 0 0],rgb('indigo')};
    phaseSpace(Projection, Summary,plot_params);  % makes the plot
 
%     % make a movie!
    movParams.colors = {[1 0 1],[.75 .25 .75],[.25 .5 1],[0 1 1]};
%     movParams.colors = {[0 0 0],rgb('hotpink'),[1 0 0],rgb('indigo')};
    movParams.times = times;
    movParams.substRawPCs = false;
    % change this to allow for path change
%     movParams.fname = sprintf('m%s_%s_3secPCA',mouse,date);
    phaseMovie(Projection,Summary,movParams);

%     % show rotation
%     fname = sprintf('m%s_%s_rotation_3sec',mouse,date);
%     pixelsToGet = [25 35 280 280];
%     rotationMovie(Projection, Summary, times, 70, 70,pixelsToGet,movParams.colors,fname);  % 70 steps to full rotation.  Show all 70.

end

%% Prep jPCA for 3 second single trial analysis
% single trials are now the conditions

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    
    sec3ix = 3000/tbin_ms;
    times = -1000:tbin_ms:2000;
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(rewsize);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    rew_counter = 1;

    singleTrialColors = {nTrials};
    
    trials400x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.55);
    trials440x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4 & rew_barcode(:,3) == 0 & prts > 3.55);
    trials404x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 4 & prts > 3.55);
    trials444x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4 & rew_barcode(:,3) == 4 & prts > 3.55);
    
    % define groups for coloring
    group4Vec = nan(nTrials,1);
    group4Vec(trials400x) = 1;
    group4Vec(trials440x) = 2;
    group4Vec(trials404x) = 3;
    group4Vec(trials444x) = 4;
    colors = [1 0 1; .75 .25 .75; .25 .5 1; 0 1 1];
 
    % define jPCA data
    jPCA_3secSingles{sIdx} = struct;
    jPCA_3secSinglesColors{sIdx} = {numel(find(rewsize(iTrial) == 4 && prts(iTrial) > 3.55))};
    for iTrial = 1:nTrials
        if rewsize(iTrial) == 4 && prts(iTrial) > 3.55
            stop_ix = FRandTimes(sIdx).stop_leave_ix(iTrial,1);
            jPCA_3secSingles{sIdx}(iTrial).A = FRandTimes(sIdx).fr_mat(:,stop_ix:stop_ix + sec3ix)';
            jPCA_3secSingles{sIdx}(iTrial).times = times';
            % apply some color grad over trial number here
            jPCA_3secSinglesColors{sIdx}{iTrial} = colors(group4Vec(iTrial),:) * (.5 + (iTrial / nTrials)/2);
        end
    end
    
    % get rid of the empty cells
    jPCA_3secSingles{sIdx} = jPCA_3secSingles{sIdx}(all(~cellfun('isempty',struct2cell(jPCA_3secSingles{sIdx}))));
    jPCA_3secSinglesColors{sIdx} = jPCA_3secSinglesColors{sIdx}(~cellfun('isempty',jPCA_3secSinglesColors{sIdx}));
    jPCA_3secSinglesColors{sIdx} = jPCA_3secSinglesColors{sIdx}(2:end); % hack because of the isempty call
    group4Vec(isnan(group4Vec)) = [];
    
end

%% Now perform jPCA for 3 second conditions single trials
close all;

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    mouse = session(1:2); 
    date = session(8:end);
    
    % plotting the first jPCA plane for 2000 ms of data, using 6 PCs (the default)
    times = -1000:tbin_ms:3000;
    jPCA_params.numPCs = 6;  % default anyway, but best to be specific
    jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
    [Projection, Summary] = jPCA(jPCA_3secSingles{sIdx}, times, jPCA_params);
    
%     plot_params.substRawPCs = false;
%     plot_params.colors = jPCA_3secSinglesColors{sIdx};
%     plot_params.conds2plot = find(group4Vec == 4);
%     phaseSpace(Projection,Summary,plot_params);  % makes the plot
    fr_mat = jPCA_3secSingles{sIdx}(2).A'; % middle of the road choice
    % sort PETH
    fr_mat_norm = zscore(fr_mat,[],2);
    [~,index] = max(fr_mat_norm');
    [~,index_sort] = sort(index);
    for condIdx = 1:5
        jPCA_3secSingles{sIdx}(condIdx).A = jPCA_3secSingles{sIdx}(condIdx).A(:,index_sort); % sort our jPCA data
    end

    labels = {"Trial 1","Trial 2","Trial 3","Trial 4","Trial 5"};
    PETH_PCA_jPCAGrid(1:5,jPCA_3secSingles{sIdx},Projection,labels,"All Neurons")
    
%     make a movie!
    movParams.colors = jPCA_3secSinglesColors{sIdx};
    movParams.substRawPCs = true;
    movParams.conds2plot = find(group4Vec == 1);
    movParams.times = times;
    % change this to allow for path change
    movParams.fname = sprintf('m%s_%s_400LessSmoothPCA',mouse,date);
    phaseMovie(Projection,Summary,movParams);
    
%     % show rotation
%     fname = sprintf('m%s_%s_rotation_3sec',mouse,date);
%     pixelsToGet = [25 35 280 280];
%     rotationMovie(Projection, Summary, times, 70, 70,pixelsToGet,movParams.colors,fname);  % 70 steps to full rotation.  Show all 70.

end
