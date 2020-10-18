%% 
% Use some combination of PC (from on patch PSTH) to predict 
% leave time in which no more reward was delivered

% ie fit a model to predict the downslope of the integrator curve
% exponential? linear? 

% start by visualizing the PC trajectories after the last reward delivery

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

%% Iterate over sessions
reset_figs
tic
trial_pc_traj = {{}};
pc_reductions = {};


for sIdx = 3:3 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
	fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);    
        % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
    tbinedge = opt.tstart:opt.tbin:opt.tend;
    tbincent = tbinedge(1:end-1)+opt.tbin/2;
   
    % behavioral events to align to
    rew_size = mod(dat.patches(:,2),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    rew_ms = dat.rew_ts*1000;
    
    % exclude patchstop rewards and rewards that preceded patchleave by <1s    
    keep = true(size(rew_ms));    
    for rIdx = 1:numel(rew_ms)
        if sum(patchstop_ms<rew_ms(rIdx) & patchleave_ms>rew_ms(rIdx))==0 % only keep rewards in patches
            keep(rIdx) = false;
        end
        if min(abs(rew_ms(rIdx)-patchstop_ms))<500 || min(abs(rew_ms(rIdx)-patchleave_ms))<1000
            keep(rIdx) = false;
        end
    end
    rew_ms = rew_ms(keep);
    
    % get size of each reward
    rew_size_indiv = nan(size(rew_ms));
    for rIdx = 1:numel(rew_ms)
        patch_id = find(patchstop_ms<rew_ms(rIdx) & patchleave_ms>rew_ms(rIdx));
        rew_size_indiv(rIdx) = rew_size(patch_id);
    end
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
    
    %trials.start = dat.patchStop_ts(keep);
    %trials.end = dat.patchLeave_ts(keep);

    p.patchstop_ms = patchstop_ms(keep);
    p.patchleave_ms = patchleave_ms(keep);
    
     % compute firing rate matrix 
    tic
    [fr_mat, p_out, tbincent] = calc_onPatch_FRVsTimeNew6_9_2020(good_cells, dat, trials, p, opt); %MB includes only activity within patches
    toc
    
    fr_mat_zscore = my_zscore(fr_mat); % z-score our psth matrix
    
    % update timestamp vectors according to p_out, including .55 ms bias
    % according to linear regression results
    patchstop_ms = p_out.patchstop_ms + 9;
    patchleave_ms = p_out.patchleave_ms + 9;
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round(patchleave_ms / tbin_ms) + 1,size(fr_mat_zscore,2)); % might not be good
 
    % now perform PCA on concatenated matrix
    tic
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
%     [u,s,v] = svd(fr_mat_zscore);
    toc
%     s = diag(s);
    plot(expl(1:10).*2 / sum(expl.*2))
    title("Variance Explained Principle Components")
    xlabel("Principle Component")
    ylabel("Normalized Variance Explained")
    
    % transform into PC Space with first 3 components
%     reduced_data = u(:,1:3) * diag(s(1:3)) * v(1:3,:); % not currently using this
%     pc_space_traj = v(1:6,:);
%     reconstr = score(1:3,:) * coeff(
    pc_space_traj = score(:,1:6);
    
    % gather trajectories by trial using our new indices
    for iTrial = 1:numel(patchleave_ix)
%         trial_pc_traj{sIdx}{iTrial} = pc_space_traj(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
        trial_pc_traj{sIdx}{iTrial} = pc_space_traj(patchstop_ix(iTrial):patchleave_ix(iTrial),1:3);
    end
    
    pc_reductions{sIdx} = pc_space_traj;
end

%% Visualize reduced PSTH Matrices, align to task events

global gP
gP.cmap{1} = [0 0 0];
gP.cmap{3} = cool(3);
gP.cmap{4} = [0 0 0; winter(3)];

for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    dat = load(fullfile(paths.data,session));
    patches = dat.patches;
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    psth_label = {'cue','stop','leave','rew'};
    t_align = cell(4,1);
    t_start = cell(4,1);
    t_end = cell(4,1);
    % cue currently not used
    t_align{1} = patchcue_ms;
    t_start{1} = patchcue_ms-1000;
    t_end{1} = patchstop_ms;
    % stop
    t_align{2} = patchstop_ms; %MB
    t_start{2} = patchstop_ms; %MB
    t_endmax = patchleave_ms - opt.additionalBuffer;
    t_endmax(patchleave_ms > opt.maxtime_alignstop + patchstop_ms) = patchstop_ms(patchleave_ms > opt.maxtime_alignstop + patchstop_ms) + opt.maxtime_alignstop; % max 4 seconds before stop
    t_end{2} = t_endmax;
    % leave 
    t_align{3} = patchleave_ms;
    t_startmin = patchstop_ms; % changed 6/9/2020 (from +1)
    t_startmin(patchleave_ms > opt.maxtime_alignleave + patchstop_ms) = patchleave_ms(patchleave_ms > opt.maxtime_alignleave + patchstop_ms) - opt.maxtime_alignleave;
    t_start{3} = t_startmin; % maximum of 5 seconds before patch leave
    t_end{3} = patchleave_ms;
    % rew currently not used
    t_align{4} = rew_ms;
    t_start{4} = rew_ms-1000; % -1 to +1 sec rel. to reward
    t_end{4} = rew_ms+1000;

    t_endmax = patchleave_ms;

    % group by rew size:
    grp = cell(4,1);
    grp{1} = rew_size;
    grp{2} = rew_size;
    grp{3} = rew_size;
    grp{4} = rew_size_indiv;
      
    % visualize PCs
    for aIdx = 2:3 % currently just look at stop and leave alignments
        fig_counter = fig_counter+1;
        hfig(fig_counter) = figure('Position',[100 100 2300 700]);
        hfig(fig_counter).Name = sprintf('%s - pca whole session - task aligned - %s',session,psth_label{aIdx});
        for pIdx = 1:6 % plot for first 3 PCs
            subplot(2,6,pIdx);
            plot_timecourse('stream',pc_reductions{sIdx}(:,pIdx),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}/tbin_ms,[],'resample_bin',1);
            atitle(sprintf('PC%d/%s/ALL TRIALS/',pIdx,psth_label{aIdx}));
            subplot(2,6,pIdx+6);
            plot_timecourse('stream',pc_reductions{sIdx}(:,pIdx),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}/tbin_ms,grp{aIdx},'resample_bin',1);
            atitle(sprintf('PC%d/%s/SPLIT BY REW SIZE/',pIdx,psth_label{aIdx}));
        end
    end
end

%% Plot trajectories after delivery of the final reward 

for sIdx = 3:3 % replace this when doing multiple sessions
    figcounter = 1;
    
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;

    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    rew_burnin_sec = .5;
    
    % make barcode matrices
    nTimesteps = 20;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    last_rew_sec = zeros(length(patchCSL),1);
    pc_last_rew = zeros(length(patchCSL),size(trial_pc_traj{sIdx}{1},2));
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        last_rew_sec(iTrial) = max(rew_indices);
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
        % log the PC values at last_rew time 
        if prts(iTrial) - .55 > last_rew_sec(iTrial) + rew_burnin_sec % adjust for neural data cutoff
            pc_last_rew(iTrial,:) = trial_pc_traj{sIdx}{iTrial}((last_rew_sec(iTrial) + rew_burnin_sec) * 1000/tbin_ms,:);
        end
    end
    
    % get a broad look at the behavior we're analyzing
    figure(figcounter);
    scatter(last_rew_sec,prts - last_rew_sec)
    title("Time of last reward vs Time on patch after last rew (ms)")
    xlabel("Time of last reward delivery (ms)")
    ylabel("Time on patch after last rew (ms)")
    % now split by rewsize
    figure(figcounter +  1);
    subplotcounter = 1;
    for iRewsize = [1 2 4]
        subplot(1,3,subplotcounter)
        scatter(last_rew_sec(rewsize == iRewsize) - 1,prts(rewsize == iRewsize) - last_rew_sec(rewsize == iRewsize) - 1)
        title(sprintf("%i uL",iRewsize))
        xlabel("Time of last reward delivery (ms)")
        ylabel("Time on patch after last rew (ms)")
        subplotcounter = subplotcounter + 1;
    end
    
    % next plot the trajectories 
    rewsizes = [1,2,4];
    overall_max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}));
    
    k = 10; % number of trial samples to take
    
    figure(figcounter + 2);
    colormap(cool)
    
    for j = 1:numel(rewsizes)
        iRewsize = rewsizes(j);
        max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(rewsize == iRewsize)));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),3)],trial_pc_traj{sIdx}(rewsize == iRewsize),'un',0); %,'un',0);
        mean_trajectory = mean(cat(3,padded_trajectories{:}),3,'omitnan');
        t_len = size(mean_trajectory,1);

%         display(t_len)
        t = (1:t_len) * (overall_max_len / t_len) + j * overall_max_len; % scale the colormap according to the length of the trial so we use all of it

        % plot a sample of a few trials from this reward size
        trial_sample = randsample(find(rewsize == iRewsize & prts - .55 > last_rew_sec),k);
        max_len = max(last_rew_sec(trial_sample) .* 1000) / tbin_ms;
        
        for iTrial = trial_sample'
            subplot(1,3,j)
            grid()
            last_rew_ix = last_rew_sec(iTrial) * 1000 / tbin_ms;
            t_len = size(trial_pc_traj{sIdx}{iTrial}(last_rew_ix:end,:),1);
            t = (1:t_len) ./ (t_len / max_len);
            patch([trial_pc_traj{sIdx}{iTrial}(last_rew_ix:end,1)' nan],[trial_pc_traj{sIdx}{iTrial}(last_rew_ix:end,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',1);
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(last_rew_ix,1),trial_pc_traj{sIdx}{iTrial}(last_rew_ix,3),'bo') % start
            scatter(trial_pc_traj{sIdx}{iTrial}(last_rew_ix:1000/tbin_ms:end,1),trial_pc_traj{sIdx}{iTrial}(last_rew_ix:1000/tbin_ms:end,3),'kx') % second marker
            scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,3),'rx') % end
            xlabel('PC 1')
            ylabel('PC 3')
            title(sprintf('%i uL Sample Traj',iRewsize));
        end
    end
    
    % now scatter between value of PC3 at time of last rew and time on patch after last rew
    figure(figcounter + 3)
    subplotcounter = 1;
    for iRewsize = [1 2 4]
        subplot(1,3,subplotcounter)
        iRewsizeLeaves = (prts(rewsize == iRewsize & prts - .55 > last_rew_sec + rew_burnin_sec) - last_rew_sec(rewsize == iRewsize & prts - .55 > last_rew_sec + rew_burnin_sec) - 1);
        iRewsizePC1 = pc_last_rew(rewsize == iRewsize & prts - .55 > last_rew_sec + rew_burnin_sec,1); 
        iRewsizePC1 = zscore(iRewsizePC1);
        iRewsizePC3 = pc_last_rew(rewsize == iRewsize & prts - .55 > last_rew_sec + rew_burnin_sec,3); 
        iRewsizePC3 = zscore(iRewsizePC3);
        scatter3(iRewsizePC1,iRewsizePC3,iRewsizeLeaves)
        title(sprintf("%i uL",iRewsize))
        xlabel("PC1 at time of last rew (zscore)")
        ylabel("PC3 at time of last rew (zscore)")
        zlabel("Time on patch after last rew (sec)")
        subplotcounter = subplotcounter + 1;
    end
    
    
    % compare this vs null models in which this relationship is solely due
    % to random noise

end

%% Visualize gradients to figure out what just the initial condition is missing

for sIdx = 3:3 % replace this when doing multiple sessions
    sample = randsample(length(trial_pc_traj{sIdx}),5);
    for j = 1:numel(sample)
        iTrial = sample(j);
        [dX,dY] = gradient(trial_pc_traj{3}{iTrial}(:,[1,3]));
        figure()
        subplot(1,2,1)
        quiver(trial_pc_traj{3}{iTrial}(:,1),trial_pc_traj{3}{iTrial}(:,3),dY(:,1),dY(:,2))
        title("Trajectory Gradient over Time")
        subplot(1,2,2)
        plot(mean(dY,2))
        title("Mean Gradient over Time")
    end
end

%% Show full trajectories of simple reward histories to capture full dynamics (ie RR999,R0999)

sec2idx = round(2000/tbin_ms);
sec1idx = round(1000/tbin_ms);


figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
    gradsR0999 = {{}};
    gradsRR999 = {{}};
    
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;

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
        last_rew_ix = max(rew_indices); 
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % iterate over reward sizes
    for iRewsize = [1,2,4]
        
        trialsR0999 = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == -1);
        trialsRR999 = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == -1);
        trial2secIdVec = zeros(15,1); 
        trial2secIdVec(trialsR0999) = 1;
        trial2secIdVec(trialsRR999) = 2;
        
        for j = 1:numel(trialsR0999)
            figure(figcounter)
            colormap(cool)
            iTrial = trialsR0999(j);
            t = 1:size(trial_pc_traj{sIdx}{iTrial},1);
            patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,3),'rx') % traj end pt
            scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            xlabel("PC1")
            ylabel("PC3")
            title(sprintf('Session %s %i0999 Trajectories through 2 PC Space',session,iRewsize));
            if j == 1
                [~,dY] = gradient(trial_pc_traj{sIdx}{iTrial}(:,[1,3]));
                figure(figcounter+1)
                subplot(1,2,1)
                quiver(trial_pc_traj{sIdx}{iTrial}(:,1),trial_pc_traj{sIdx}{iTrial}(:,3),dY(:,1),dY(:,2))
                title(sprintf("%Sample %i0999 Trial Trajectory Gradient",iRewsize))
                subplot(1,2,2)
                plot(sqrt(mean(dY.^2,2)))
                title(sprintf("Sample %i0999 Trial Gradient Magnitude",iRewsize))
                hold on
                scatter(sec1idx,0,'k*')
                ylim([0,3])
            end
            
            [~,dY] = gradient(trial_pc_traj{3}{iTrial}(:,[1,3]));
            gradsR0999{iRewsize}{iTrial} = dY;
        end
        
        for j = 1:numel(trialsRR999)
            figure(figcounter+2)
            colormap(cool)
            iTrial = trialsRR999(j);
            t = 1:size(trial_pc_traj{sIdx}{iTrial},1);
            patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,3),'rx') % traj end pt
            scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            scatter(trial_pc_traj{sIdx}{iTrial}(sec1idx,1),trial_pc_traj{sIdx}{iTrial}(sec1idx,3),'k*')
            xlabel("PC1")
            ylabel("PC3")
            title(sprintf('Session %s %i%i999 Trajectories through 2 PC Space',session,iRewsize,iRewsize));
            if j == 1
                [~,dY] = gradient(trial_pc_traj{3}{iTrial}(:,[1,3]));
                figure(figcounter+3)
                subplot(1,2,1)
                quiver(trial_pc_traj{3}{iTrial}(:,1),trial_pc_traj{3}{iTrial}(:,3),dY(:,1),dY(:,2))
                hold on 
                scatter(trial_pc_traj{sIdx}{iTrial}(sec1idx,1),trial_pc_traj{sIdx}{iTrial}(sec1idx,3),'k*')
                title(sprintf("Sample %i%i999 Trial Trajectory Gradient",iRewsize,iRewsize))
                subplot(1,2,2)
                plot(mean(dY,2))
                title(sprintf("Sample %i%i999 Trial Gradient Magnitude",iRewsize,iRewsize))
                hold on
                mean_grad = mean(dY,2);
                scatter(sec1idx,mean_grad(sec1idx),'k*')
            end
            [~,dY] = gradient(trial_pc_traj{3}{iTrial}(:,[1,3]));
            gradsRR999{iRewsize}{iTrial} = dY;
        end
        figcounter = figcounter + 4;
       
    end
    
    % prep data structures, visualize gradients
    figure()
    subplotCounter = 1;
    for iRewsize = [1,2,4]
        subplot(1,3,subplotCounter)
        gradsR0999{iRewsize} = gradsR0999{iRewsize}(~cellfun('isempty',gradsR0999{iRewsize}));
        gradsRR999{iRewsize} = gradsRR999{iRewsize}(~cellfun('isempty',gradsRR999{iRewsize}));
        
        max_len = max(cellfun(@(x)  size(x,1),gradsR0999{iRewsize}));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),2)],gradsR0999{iRewsize},'un',0); %,'un',0);
        mean_trajectoryR0999 = mean(cat(2,padded_trajectories{:}),2,"omitnan"); 
        sem_trajectoryR0999 = 1.96 * std(cat(2,padded_trajectories{:}),0,2,"omitnan");
        plot(mean_trajectoryR0999)
%         errorbar(mean_trajectoryR0999,sem_trajectoryR0999)
        hold on
        
        max_len = max(cellfun(@(x)  size(x,1),gradsRR999{iRewsize}));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),2)],gradsRR999{iRewsize},'un',0); %,'un',0);
        mean_trajectoryRR999 = mean(cat(2,padded_trajectories{:}),2,"omitnan"); 
        sem_trajectoryRR999 = 1.96 * std(cat(2,padded_trajectories{:}),0,2,"omitnan");
        plot(mean_trajectoryRR999)
%         errorbar(mean_trajectoryRR999,sem_trajectoryRR999)

        title(sprintf("%i uL Gradient over trial separated by 1 sec rew delivery",iRewsize))
        
        subplotCounter = subplotCounter + 1;
        ylim([-1,.4])
        xlim([0,100])
        scatter(sec1idx,mean_trajectoryRR999(sec1idx),'k*')
        
        legend("R0999","RR999","Reward Delivery")
    end
    
    
end

%% Now just try PC1 to see if we can get predictability here with just start point and slope

% start w/ just dividing by reward size
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    prts = patchleave_ms - patchstop_ms;
    rew_ms = data.rew_ts;
    
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
    last_rew_ix = zeros(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_sec = max(rew_indices);
        last_rew_ix(iTrial) = round(last_rew_sec * 1000 / tbin_ms);
        rew_barcode(iTrial , (last_rew_sec + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    time_after = prts - last_rew_sec; 
    figure()
    boxplot(time_after,rewsize);
    xlabel("Reward size (uL)")
    ylabel("Time stayed after reward")
    title("Distribution of Time stay after reward divided by rewsize")

    rew_burnin_sec = .5;
    post_rew_PC1 = {{}};
    

    for iRewsize = [2,4]
        rewsize_trials = find(rewsize == iRewsize);
        figure()
        colormap(cool)
        hold on
        
        for j = 1:numel(rewsize_trials)
            iTrial = rewsize_trials(j);
            [~,max_ix] = max(trial_pc_traj{sIdx}{iTrial}(last_rew_ix(iTrial):end,1));
            pc1 = trial_pc_traj{sIdx}{iTrial}(last_rew_ix(iTrial):max_ix,1);
            post_rew_PC1{iRewsize}{j} = [linspace(last_rew_ix(iTrial)/tbin_ms,prts(iTrial),length(pc1))' pc1];
            plot(post_rew_PC1{iRewsize}{j}(:,1),post_rew_PC1{iRewsize}{j}(:,2))
        end
        xlim([0,15])
        title(sprintf("%i uL Trial",iRewsize));
        
    end
    
    for iRewsize = [2,4]
        max_len = max(cellfun(@(x)  size(x,1),post_rew_PC1{iRewsize}));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),2)],post_rew_PC1{iRewsize},'un',0); %,'un',0);
        mean_trajectory = mean(cat(2,padded_trajectories{:}),2,"omitnan");
        figure(4)
        hold on
        plot(mean_trajectory)
        legend("2 uL","4 uL")
    end
    
end

