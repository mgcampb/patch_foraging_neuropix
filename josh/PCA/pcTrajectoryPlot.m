%% Plot neural trajectories in reduced PC space over time between trials
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
    plot(expl(1:10) / sum(expl))
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

%% Visualize PC Trajectories
% first just take a random sample of single trials and color with cool gradient

for sIdx = 3:3 % replace this when doing multiple sessions
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_');
    
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    k = 5;
    sample = randsample(size(trial_pc_traj{sIdx},2),k);
    
    max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(sample)));
    figure(1)
    
    for j = 1:numel(sample)
        iTrial = sample(j);
                
        t_len = size(trial_pc_traj{sIdx}{iTrial},1);
        rew_idx = (1000/tbin_ms):(1000/tbin_ms):t_len;
        %         display(t_len)
        t = (1:t_len) ./ (t_len / max_len); % scale the colormap according to the length of the trial so we use all of the colormap

        figure(1)
        colormap(cool(max_len)); % 541 is the max trial length
        subplot(2,1,1)
        patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,2)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
        hold on
        grid()
        scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),'bo') % start
        scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,2),'rx') % end
%         scatter(trial_pc_traj{sIdx}{iTrial}(rew_idx,1),trial_pc_traj{sIdx}{iTrial}(rew_idx,2),'k*') % reward timing (every second)
        xlabel('PC 1')
        ylabel('PC 2')
        title(sprintf('Session %s Random Sample of 20 Single Trial Trajectories through 2 PC Space',session));

        subplot(2,1,2)
        patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,2)' nan],[trial_pc_traj{sIdx}{iTrial}(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
        view(3)
        grid()
        hold on
        scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo')
        scatter3(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,2),trial_pc_traj{sIdx}{iTrial}(end,3),'rx')
%         scatter3(trial_pc_traj{sIdx}{iTrial}(rew_idx,1),trial_pc_traj{sIdx}{iTrial}(rew_idx,2),trial_pc_traj{sIdx}{iTrial}(rew_idx,3),'k*') % reward timing (every second)

        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s Random Sample of 20 Single Trial Trajectories through 3 PC Space',session));
        
        figure(2)
        colormap(cool(max_len)); % 541 is the max trial length
        subplot(2,1,1)
        patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
        hold on
        grid()
        scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % start
        scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,3),'rx') % end
%         scatter(trial_pc_traj{sIdx}{iTrial}(rew_idx,1),trial_pc_traj{sIdx}{iTrial}(rew_idx,2),'k*') % reward timing (every second)
        xlabel('PC 1')
        ylabel('PC 3')
        title(sprintf('Session %s Random Sample of 20 Single Trial Trajectories through 2 PC Space',session));

        subplot(2,1,2)
        patch([trial_pc_traj{sIdx}{iTrial}(:,1)' nan],[trial_pc_traj{sIdx}{iTrial}(:,2)' nan],[trial_pc_traj{sIdx}{iTrial}(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
        view(3)
        grid()
        hold on
        scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo')
        scatter3(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,2),trial_pc_traj{sIdx}{iTrial}(end,3),'rx')
%         scatter3(trial_pc_traj{sIdx}{iTrial}(rew_idx,1),trial_pc_traj{sIdx}{iTrial}(rew_idx,2),trial_pc_traj{sIdx}{iTrial}(rew_idx,3),'k*') % reward timing (every second)

        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s Random Sample of 20 Single Trial Trajectories through 3 PC Space',session));
    end

end

%% Now looking at how individual reward events might affect these trajectories 

sec2idx = round(2000/tbin_ms);
sec3idx = round(3000/tbin_ms);

% drawArrow = @(x,y,varargin) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1),0, varargin{:}) ;

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
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    for iRewsize = [1,2,4]        
        % define trial groups based on reward events
        % two seconds
        trials10x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & prts > 2.55);
        display(trials10x)
        trials11x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & prts > 2.55);
        
        trial2secIdVec = zeros(15,1); 
        trial2secIdVec(trials10x) = 1;
        trial2secIdVec(trials11x) = 2;
        
        % three seconds
        if iRewsize > 1
            trials100x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0 & prts > 3.55);
            trials110x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0 & prts > 3.55);
            trials101x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & prts > 3.55);
            trials111x = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & prts > 3.55);
            trial3secIdVec = zeros(15,1); 
            trial3secIdVec(trials100x) = 1;
            trial3secIdVec(trials110x) = 2;
            trial3secIdVec(trials101x) = 3;
            trial3secIdVec(trials111x) = 4;
        end
        
        k = 5;
        
        for j = 1:k
            iTrial = trials10x(j);
            figure(figcounter)
            colormap(cool(sec2idx))
            t2_1 = 1:sec2idx;
            t2_2 = (1:sec2idx) + sec2idx;
            subplot(2,1,1)
            patch([trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),'rx') % traj end pt
            grad = gradient(trial_pc_traj{sIdx}{iTrial}(sec2idx - 1:sec2idx,[1,3]));
%             quiver(trial_pc_traj{sIdx}{iTrial}(sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),grad(1,1),grad(2,1)) % traj end pt
%             drawArrow(trial_pc_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,3),'linewidth',.5,'color','r','maxheadsize',.5)
            scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            
            xlabel('PC 1')
            ylabel('PC 3')
            title(sprintf('Session %s %i0X Trial Trajectories through 2 PC Space',session,iRewsize));
            grid()
            
            subplot(2,1,2)
            patch([trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,2)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5)
            hold on
            scatter3(trial_pc_traj{sIdx}{iTrial}(sec2idx),trial_pc_traj{sIdx}{iTrial}(sec2idx,2),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),'rx') % traj end pt
            hold on
            scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s %i0X Trial Trajectories through 3 PC Space',session,iRewsize));
            view(3)
            grid()
        end
        
        % similarly plot the RRX
        for j = 1:k
            iTrial = trials11x(j);
            figure(figcounter+1)
            colormap(autumn(sec2idx + 500))
            t2_1 = 1:sec2idx;
            t2_2 = (1:sec2idx) + sec2idx;
            subplot(2,1,1)
            patch([trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(2000/tbin_ms,1),trial_pc_traj{sIdx}{iTrial}(2000/tbin_ms,3),'rx') % traj end pt
%             quiver(trial_pc_traj{sIdx}{iTrial}(sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),grad(1,1),grad(2,1)) % traj end pt
%             drawArrow(trial_pc_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,3),'linewidth',.5,'color','r','maxheadsize',.5)
            scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            scatter(trial_pc_traj{sIdx}{iTrial}(1000/tbin_ms,1),trial_pc_traj{sIdx}{iTrial}(1000/tbin_ms,3),'k*')
            
            xlabel('PC 1')
            ylabel('PC 3')
            title(sprintf('Session %s %i%iX Trial Trajectories through 2 PC Space',session,iRewsize,iRewsize));
            grid()
            
            subplot(2,1,2)
            patch([trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,2)' nan],[trial_pc_traj{sIdx}{iTrial}(1:sec2idx,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5)
            hold on
            scatter3(trial_pc_traj{sIdx}{iTrial}(sec2idx),trial_pc_traj{sIdx}{iTrial}(sec2idx,2),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),'rx') % traj end pt
            scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s %i%iX Trial Trajectories through 3 PC Space',session,iRewsize,iRewsize));
            view(3)
            grid()
        end

        cat10x = cellfun(@(x) x(1:sec2idx,:),trial_pc_traj{sIdx}(trials10x),'un',0);
        cat11x = cellfun(@(x) x(1:sec2idx,:),trial_pc_traj{sIdx}(trials11x),'un',0);

        mean10x = mean(cat(6,cat10x{:}),6);
        mean11x = mean(cat(6,cat11x{:}),6);
        
        if iRewsize > 1
            cat100x = cellfun(@(x) x(1:sec3idx,:),trial_pc_traj{sIdx}(trials100x),'un',0);
            cat110x = cellfun(@(x) x(1:sec3idx,:),trial_pc_traj{sIdx}(trials110x),'un',0);
            cat101x = cellfun(@(x) x(1:sec3idx,:),trial_pc_traj{sIdx}(trials101x),'un',0);
            cat111x = cellfun(@(x) x(1:sec3idx,:),trial_pc_traj{sIdx}(trials111x),'un',0);

            mean100x = mean(cat(6,cat100x{:}),6);
            mean110x = mean(cat(6,cat110x{:}),6);
            mean101x = mean(cat(6,cat101x{:}),6);
            mean111x = mean(cat(6,cat111x{:}),6); 
        end
        
        % first 2-second pooling
        figure(figcounter+2)
        colormap([cool(sec2idx);hot(sec2idx)]) 
        subplot(2,1,1)
        t2_1 = 1:sec2idx;
        t2_2 = (1:sec2idx) + sec2idx;
        t2_rew = (1000/tbin_ms);
        patch([mean10x(:,1)' nan],[mean10x(:,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        hold on
        patch([mean11x(:,1)' nan],[mean11x(:,3)' nan],[t2_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        view(2)
        grid()
        scatter([mean10x(1,1) mean11x(1,1)],[mean10x(1,3) mean11x(1,3)],'bo')
        scatter([mean10x(end,1) mean11x(end,1)],[mean10x(end,3) mean11x(end,3)],'rx')
        scatter(mean11x(t2_rew,1),mean11x(t2_rew,3),'k*')
        xlabel('PC 1')
        ylabel('PC 3')
        title(sprintf('Session %s %i%iX vs %i0X Trajectories through 2 PC Space',session,iRewsize,iRewsize,iRewsize));
        % now 3 dimensions
        subplot(2,1,2)
        patch([mean10x(:,1)' nan],[mean10x(:,2)' nan],[mean10x(:,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        hold on
%         patch([mean10x(:,1)' nan],[ones(size(mean10x(:,2)')) nan],[mean10x(:,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor',[.5,1,1],'LineWidth',1.5) % shadow
%         plot3(mean11x(:,1),mean11x(:,2),mean11x(:,3),'Color',barcode_colorgrad2(2,:),'LineWidth',1.5)
        patch([mean11x(:,1)' nan],[mean11x(:,2)' nan],[mean11x(:,3)' nan],[t2_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
%         patch([mean11x(:,1)' nan],[ones(size(mean11x(:,2)')) nan],[mean11x(:,3)' nan],[t2_1 nan],'FaceColor','none','EdgeColor',[1,.5,.5],'LineWidth',1.5) % shadow
        view(3)
        grid()
        scatter3([mean10x(1,1) mean11x(1,1)],[mean10x(1,2) mean11x(1,2)],[mean10x(1,3) mean11x(1,3)],'bo')
        scatter3([mean10x(end,1) mean11x(end,1)],[mean10x(end,2) mean11x(end,2)],[mean10x(end,3) mean11x(end,3)],'rx')
        scatter3(mean11x(t2_rew,1),mean11x(t2_rew,2),mean11x(t2_rew,3),'k*')
        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s %i%iX vs %i0X Trajectories through 3 PC Space',session,iRewsize,iRewsize,iRewsize));
        
        
        if iRewsize > 1 % rewsize 1 dont stay long enough
            % last, 3-second pooling
            figure(figcounter + 3)
            subplot(2,1,1)
            colormap([cool(sec3idx*2);winter(sec3idx*2);summer(sec3idx*2);hot(sec3idx*2)]) 
            t3_1 = 1:sec3idx;
            t3_2 = (1:sec3idx) + sec3idx;
            t3_3 = (1:sec3idx) + 2 * sec3idx;
            t3_4 = (1:sec3idx) + 3 * sec3idx;
            t3_rew10 = (1000/tbin_ms); % reward timing for star visualization
            t3_rew01 = (2000/tbin_ms);
            t3_rew11 = [(1000/tbin_ms) (2000/tbin_ms)];

            patch([mean100x(:,1)' nan],[mean100x(:,3)' nan],[t3_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) 
            hold on
            patch([mean110x(:,1)' nan],[mean110x(:,3)' nan],[t3_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            patch([mean101x(:,1)' nan],[mean101x(:,3)' nan],[t3_3 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            patch([mean111x(:,1)' nan],[mean111x(:,3)' nan],[t3_4 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) 

            scatter([mean100x(1,1) mean110x(1,1) mean101x(1,1) mean111x(1,1)],[mean100x(1,3) mean110x(1,3) mean101x(1,3) mean111x(1,3)],'bo')
            scatter([mean100x(end,1) mean110x(end,1) mean101x(end,1) mean111x(end,1)],[mean100x(end,3) mean110x(end,3) mean101x(end,3) mean111x(end,3)],'rx')
            scatter([mean110x(t3_rew10,1) mean101x(t3_rew01,1) mean111x(t3_rew11,1)'],[mean110x(t3_rew10,3) mean101x(t3_rew01,3) mean111x(t3_rew11,3)'],'k*')
            xlabel('PC 1')
            ylabel('PC 3')
            
            grid()
            view(2)
            title(sprintf('Session %s 3 Second Reward Timing Averaged %i uL Rew Trial Trajectories through 2 PC Space',session));
            subplot(2,1,2)
            patch([mean100x(:,1)' nan],[mean100x(:,2)' nan],[mean100x(:,3)' nan],[t3_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) 
            hold on
            patch([mean110x(:,1)' nan],[mean110x(:,2)' nan],[mean110x(:,3)' nan],[t3_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            patch([mean101x(:,1)' nan],[mean101x(:,2)' nan],[mean101x(:,3)' nan],[t3_3 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            patch([mean111x(:,1)' nan],[mean111x(:,2)' nan],[mean111x(:,3)' nan],[t3_4 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) 
            grid()
            view(3)
            scatter3([mean100x(1,1) mean110x(1,1) mean101x(1,1) mean111x(1,1)],[mean100x(1,2) mean110x(1,2) mean101x(1,2) mean111x(1,2)],[mean100x(1,3) mean110x(1,3) mean101x(1,3) mean111x(1,3)],'bo')
            scatter3([mean100x(end,1) mean110x(end,1) mean101x(end,1) mean111x(end,1)],[mean100x(end,2) mean110x(end,2) mean101x(end,2) mean111x(end,2)],[mean100x(end,3) mean110x(end,3) mean101x(end,3) mean111x(end,3)],'rx')
            scatter3([mean110x(t3_rew10,1) mean101x(t3_rew01,1) mean111x(t3_rew11,1)'],[mean110x(t3_rew10,2) mean101x(t3_rew01,2) mean111x(t3_rew11,2)'],[mean110x(t3_rew10,3) mean101x(t3_rew01,3) mean111x(t3_rew11,3)'],'k*')
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s 3 Second Reward Timing Averaged %i uL Rew Trial Trajectories through 3 PC Space',session,iRewsize));

            % now just look at 101X vs 110X
            figure(figcounter + 4)
            subplot(2,1,1)
            colormap([hot(sec3idx * 2);cool(sec3idx * 2)]) 
            t3_1 = 1:sec3idx;
            t3_2 = (1:sec3idx) + sec3idx;

            t3_rew10 = (1000/tbin_ms); % reward timing for star visualization
            t3_rew01 = (2000/tbin_ms);

            patch([mean110x(:,1)' nan],[mean110x(:,3)' nan],[t3_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            hold on
            patch([mean101x(:,1)' nan],[mean101x(:,3)' nan],[t3_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)

            scatter([mean110x(1,1) mean101x(1,1)],[mean110x(1,3) mean101x(1,3)],'bo')
            scatter([mean110x(end,1) mean101x(end,1)],[mean110x(end,3) mean101x(end,3)],'rx')
            scatter([mean110x(t3_rew10,1) mean101x(t3_rew01,1)],[mean110x(t3_rew10,3) mean101x(t3_rew01,3)],'k*')
            xlabel('PC 1')
            ylabel('PC 3')
            grid()
            view(2)
            title(sprintf('Session %s %i0%iX vs %i%i0X Trajectories through 2 PC Space',session,iRewsize,iRewsize,iRewsize,iRewsize));
            
            subplot(2,1,2)
            patch([mean110x(:,1)' nan],[mean110x(:,2)' nan],[mean110x(:,3)' nan],[t3_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            hold on
            patch([mean101x(:,1)' nan],[mean101x(:,2)' nan],[mean101x(:,3)' nan],[t3_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5)
            grid()
            view(3)
            scatter3([mean110x(1,1) mean101x(1,1)],[mean110x(1,2) mean101x(1,2)],[mean110x(1,3) mean101x(1,3)],'bo')
            scatter3([mean110x(end,1) mean101x(end,1)],[mean110x(end,2) mean101x(end,2)],[mean110x(end,3) mean101x(end,3)],'rx')
            scatter3([mean110x(t3_rew10,1) mean101x(t3_rew01,1)],[mean110x(t3_rew10,2) mean101x(t3_rew01,2)],[mean110x(t3_rew10,3) mean101x(t3_rew01,3)],'k*')
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s %i0%iX vs %i%i0X Trajectories through 3 PC Space',session,iRewsize,iRewsize,iRewsize,iRewsize));

            figcounter = figcounter + 5;
        else
            figcounter = figcounter + 3;
        end
    end
end

%% Show full trajectories of simple reward histories to capture full dynamics (ie RR999,R0999)

sec2idx = round(2000/tbin_ms);
sec1idx = round(1000/tbin_ms);

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
        trialsR0999 = find(rew_barcode(:,1) == rew_barcode(:,2) == -1);
        trialsRR999 = find(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == -1);
        trial2secIdVec = zeros(15,1); 
        trial2secIdVec(trialsR0999) = 1;
        trial2secIdVec(trialsRR999) = 2;
        
        display(length(trialsR0999));
        display(length(trialsRR999));
        
        figure()
        colormap(cool)
        
        for j = 1:numel(trialsRR999)
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
        end
        
%         % just a demo to compare to jPCA
%         figure()
%         colormap(cool)
%         iTrial = trialsRR999(1);
%         plot(trial_pc_traj{sIdx}{iTrial}(:,1));
%         hold on
%         plot(trial_pc_traj{sIdx}{iTrial}(:,3));
%         legend("PC1","PC3")
%         title(sprintf('Session %s %i%i999 PC1,PC3 over time',session,iRewsize,iRewsize));
%         ylabel("PC Magnitude")
%         xlabel("Time")
    end
end



%% Experimenting with color gradients over time 
% Early mid late w/ gradient 
% Again, not that beautiful

figcounter = 1;
colormap([hot(116);summer(171);cool(541)]) % 541 is the max trial length

for sIdx = 3:3 % replace this when doing multiple sessions
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_');
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    tri_iles = [0,quantile(prts,[.33,.66,1.])];
    
    cumulative_line = 0;
    for q = 1:3
        thisQuartileTrials = find(prts > tri_iles(q) & prts < tri_iles(q + 1));
        
        max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(thisQuartileTrials)));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),3)],trial_pc_traj{sIdx}(thisQuartileTrials),'un',0); %,'un',0);
        mean_trajectory = mean(cat(3,padded_trajectories{:}),3,"omitnan"); 
        
        t_len = size(mean_trajectory,1);
        
        t = (1:t_len) + cumulative_line; % scale the colormap according to the length of the trial so we use all of it
        cumulative_line = cumulative_line + t_len;

        rewtimes = (1000/tbin_ms):(1000/tbin_ms):t_len;
        subplot(2,1,1)
        grid()
        p1 = patch([mean_trajectory(:,1)' nan],[mean_trajectory(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5); % color by PRT
        hold on
        scatter(mean_trajectory(1,1),mean_trajectory(1,3),'bo') % start
        scatter(mean_trajectory(end,1),mean_trajectory(end,3),'rx') % end
        scatter(mean_trajectory(rewtimes,1),mean_trajectory(rewtimes,3),'*','MarkerFaceColor',[.4 .4 .4],'MarkerEdgeColor',[.4 .4 .4]) % reward timing (every second)
        xlabel('PC 1')
        ylabel('PC 3')
        title(sprintf('Session %s early/mid/late PRT Trajectories through 2 PC Space',session));
        
        subplot(2,1,2)
        grid()
        p2 = patch([mean_trajectory(:,1)' nan],[mean_trajectory(:,2)' nan],[mean_trajectory(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5); % color by PRT
        view(3)
        hold on
        scatter3(mean_trajectory(1,1),mean_trajectory(1,2),mean_trajectory(1,3),'bo')
        scatter3(mean_trajectory(end,1),mean_trajectory(end,2),mean_trajectory(end,3),'rx')
        scatter3(mean_trajectory(rewtimes,1),mean_trajectory(rewtimes,2),mean_trajectory(rewtimes,3),'*','MarkerFaceColor',[.4 .4 .4],'MarkerEdgeColor',[.4 .4 .4]) % reward timing (every second)
        
        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s early/mid/late PRT Trial Trajectories through 3 PC Space',session));
        
    end
end

%% Trajectories separated by reward size
% not that beautiful

figcounter = 1;
colormap([hot(726);summer(726);cool(726)]) % 541 is the max trial length
for sIdx = 3:3 % replace this when doing multiple sessions
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_');
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    rewsizes = [1,2,4];
    
    for j = 1:numel(rewsizes)
        
        iRewsize = rewsizes(j);
        max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(rewsize == iRewsize)));
        padded_trajectories = cellfun(@(x) [x ; nan(max_len-size(x,1),3)],trial_pc_traj{sIdx}(rewsize == iRewsize),'un',0); %,'un',0);
        mean_trajectory = mean(cat(3,padded_trajectories{:}),3,'omitnan');
        t_len = size(mean_trajectory,1);

        display(t_len)
        t = (1:t_len) * (726 / t_len) + j * 726; % scale the colormap according to the length of the trial so we use all of it

        rewtimes = (1000/tbin_ms):(1000/tbin_ms):t_len;
        subplot(2,1,1)
        grid()
        p1 = patch([mean_trajectory(:,1)' nan],[mean_trajectory(:,2)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5); % color by PRT
        hold on
        scatter(mean_trajectory(1,1),mean_trajectory(1,2),'bo') % start
        scatter(mean_trajectory(end,1),mean_trajectory(end,2),'rx') % end
        scatter(mean_trajectory(rewtimes,1),mean_trajectory(rewtimes,2),'k*') % reward timing (every second)
        xlabel('PC 1')
        ylabel('PC 2')
        title(sprintf('Session %s small/medium/large rew Trajectories through 2 PC Space',session));
        
        subplot(2,1,2)
        grid()
        p2 = patch([mean_trajectory(:,1)' nan],[mean_trajectory(:,2)' nan],[mean_trajectory(:,3)' nan],[t nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5); % color by PRT
        view(3)
        hold on
        scatter3(mean_trajectory(1,1),mean_trajectory(1,2),mean_trajectory(1,3),'bo')
        scatter3(mean_trajectory(end,1),mean_trajectory(end,2),mean_trajectory(end,3),'rx')
        scatter3(mean_trajectory(rewtimes,1),mean_trajectory(rewtimes,2),mean_trajectory(rewtimes,3),'k*') % reward timing (every second)
        
        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s small/medium/large rew Trial Trajectories through 3 PC Space',session));
        
    end
end



%% Old code: %%


%% Plot averaged trajectories colored by time of leaving, separated by
% rewsize
figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_');
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    
    for iRewsize = [1,2,4]
        iSizeTrials = find(rewsize == iRewsize);
        round_prts = round(prts);
        max_prt = max(round_prts(iSizeTrials));
        % create a linear color map ranging from dark light blue, coloring
        % trajectories by PRT rounded to the nearest second
        if iRewsize == 4
            len = max_prt;
            blue = [0, 0, 1];
            teal = [0 1 1];
            prt_color_grad = [linspace(teal(1),blue(1),len)', linspace(teal(2),blue(2),len)', linspace(teal(3),blue(3),len)'];
        end
        
        if iRewsize == 2
            len = max_prt;
            red = [1, 0, 0];
            yellow = [1 1 0];
            prt_color_grad = [linspace(yellow(1),red(1),len)', linspace(yellow(2),red(2),len)', linspace(yellow(3),red(3),len)'];
        end
        
        if iRewsize == 1
            len = max_prt;
            black = [32/255, 32/255, 32/255];
            grey = [224/255 224/255 224/255];
            prt_color_grad = [linspace(grey(1),black(1),len)', linspace(grey(2),black(2),len)', linspace(grey(3),black(3),len)'];
        end
        figure(figcounter);
        % every trial
        for j = 1:numel(iSizeTrials) % only look at large reward where mice seem to discern frequency
            iTrial = iSizeTrials(j);
            subplot(2,1,1)
            scatter(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,2),'rx') % traj end pt
            hold on
            scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),'bo') % traj start pt
            plot(trial_pc_traj{sIdx}{iTrial}(:,1),trial_pc_traj{sIdx}{iTrial}(:,2),'Color',prt_color_grad(round_prts(iTrial),:)) % color by PRT
            xlabel('PC 1')
            ylabel('PC 2')
            title(sprintf('Session %s %i uL Rew Trial Trajectories through 2 PC Space',session,iRewsize));

            subplot(2,1,2)
            scatter3(trial_pc_traj{sIdx}{iTrial}(end,1),trial_pc_traj{sIdx}{iTrial}(end,2),trial_pc_traj{sIdx}{iTrial}(end,3),'rx') % traj end pt
            hold on
            scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            plot3(trial_pc_traj{sIdx}{iTrial}(:,1),trial_pc_traj{sIdx}{iTrial}(:,2),trial_pc_traj{sIdx}{iTrial}(:,3),'Color',prt_color_grad(round_prts(iTrial),:)) % color by PRT
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s %i uL Rew Trial Trajectories through 3 PC Space',session,iRewsize));
        end

        figure(figcounter+1);
        % prt group averages
        unique_round_prts = unique(round_prts(iSizeTrials));
        for j = 1:numel(unique_round_prts)
            iPRT = unique_round_prts(j);
            max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(round_prts == iPRT)));
            padded_trajectories = cellfun(@(x) [x ; zeros(max_len-size(x,1),3)],trial_pc_traj{sIdx}(round_prts == iPRT),'un',0); %,'un',0);
            mean_trajectory = mean(cat(3,padded_trajectories{:}),3);

            iTrial = iSizeTrials(j);
            subplot(2,1,1)
            scatter(mean_trajectory(end,1),mean_trajectory(end,2),'rx')
            hold on
            scatter(mean_trajectory(1,1),mean_trajectory(1,2),'bo')
            plot(mean_trajectory(:,1),mean_trajectory(:,2),'Color',prt_color_grad(j,:)) % color by PRT
            xlabel('PC 1')
            ylabel('PC 2')
            title(sprintf('Session %s PRT-Averaged %i uL Rew Trial Trajectories through 2 PC Space',session,iRewsize));

            subplot(2,1,2)
            scatter3(mean_trajectory(end,1),mean_trajectory(end,2),mean_trajectory(end,3),'rx')
            hold on
            scatter3(mean_trajectory(1,1),mean_trajectory(1,2),mean_trajectory(1,3),'bo')
            plot3(mean_trajectory(:,1),mean_trajectory(:,2),mean_trajectory(:,3),'Color',prt_color_grad(j,:)) % color by PRT
            xlabel('PC 1')
            ylabel('PC 2')
            zlabel('PC 3')
            title(sprintf('Session %s PRT-Averaged %i uL Rew Trial Trajectories through 3 PC Space',session,iRewsize));
        end
        figcounter = figcounter + 2;
    end
end

%% Now just separate by PRT quartile
% this plot is actually not as nice as the one above
% could be alignment issues

figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_');
    
    % Trial level features
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    quartiles = [0,quantile(prts,[.25,.5,.75,1.])];
    
    len = 4;
    blue = [0, 0, 1];
    teal = [0 1 1];
    prt_color_grad = [linspace(teal(1),blue(1),len)', linspace(teal(2),blue(2),len)', linspace(teal(3),blue(3),len)'];
    
    for q = 1:4
        thisQuartileTrials = find(prts > quartiles(q) & prts < quartiles(q + 1));
        
        max_len = max(cellfun(@(x)  size(x,1),trial_pc_traj{sIdx}(thisQuartileTrials)));
        padded_trajectories = cellfun(@(x) [x ; zeros(max_len-size(x,1),3)],trial_pc_traj{sIdx}(thisQuartileTrials),'un',0); %,'un',0);
        mean_trajectory = mean(cat(3,padded_trajectories{:}),3);
        
        subplot(2,1,1)
        plot(mean_trajectory(:,1),mean_trajectory(:,3),'Color',prt_color_grad(q,:),'LineWidth',1.5) % color by PRT
        hold on
        scatter(mean_trajectory(1,1),mean_trajectory(1,3),'bo')
        scatter(mean_trajectory(end,1),mean_trajectory(end,3),'rx')
        xlabel('PC 1')
        ylabel('PC 3')
        title(sprintf('Session %s PRT-Quartile Averaged Trial Trajectories through 2 PC Space',session));
        
        subplot(2,1,2)
        plot3(mean_trajectory(:,1),mean_trajectory(:,2),mean_trajectory(:,3),'Color',prt_color_grad(q,:),'LineWidth',1.5) % color by PRT
        hold on
        scatter3(mean_trajectory(1,1),mean_trajectory(1,2),mean_trajectory(1,3),'bo')
        scatter3(mean_trajectory(end,1),mean_trajectory(end,2),mean_trajectory(end,3),'rx')
        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('Session %s PRT-Quartile Averaged Trial Trajectories through 3 PC Space',session));
        
    end
end

%% finding trajectories using plot_timecourse as a check

% PCs for large rew size patchStops: 40 v 44
psth_label = {'cue','stop','leave','rew'};
sec2idx = round(2000/tbin_ms);
sec3idx = round(3000/tbin_ms);

% stop
t_align{2} = patchstop_ms;
t_start{2} = patchstop_ms;
t_endmax = patchleave_ms;
t_endmax(patchleave_ms > opt.maxtime_alignstop + patchstop_ms) = patchstop_ms(patchleave_ms > opt.maxtime_alignstop + patchstop_ms) + opt.maxtime_alignstop; % max 4 seconds before stop
t_end{2} = t_endmax;

% leave
t_align{3} = patchleave_ms;
t_start{3} = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
t_end{3} = patchleave_ms+2000;

% stop
x = 9; % MB I have no idea why this number is 9, need to figure it out and align properly
t_align_onpatch{2} = p_out.patchstop_ms + x; %MB
t_start_onpatch{2} = p_out.patchstop_ms + x; %MB
t_endmax = p_out.patchleave_ms;
t_endmax(p_out.patchleave_ms > opt.maxtime_alignstop + p_out.patchstop_ms) = p_out.patchstop_ms(p_out.patchleave_ms > opt.maxtime_alignstop + p_out.patchstop_ms) + opt.maxtime_alignstop; % max 4 seconds before stop
t_end_onpatch{2} = t_endmax;
t_end_onpatch2s{2} = t_endmax - 1000;

figcounter = 1;
for sIdx = 3:3 % replace this when doing multiple sessions
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
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    figcounter = 1;

    % define trial groups based on reward events
    % two seconds
    trials_40 = find(rew_barcode(:,2) == 0);
    trials_44 = find(rew_barcode(:,2) == 4);
    
    barcodeID_Lg = nan(length(patchstop_ms),1);
    barcodeID_Lg(trials_40) = 1;
    barcodeID_Lg(trials_44) = 2;
    
    % this is where we're doing something new
    fig_counter = fig_counter + 1;
    hfig(fig_counter) = figure('Position',[100 100 2300 700]);
    hfig(fig_counter).Name = sprintf('%s - pca onPatch - %s',session,psth_label{2});
    
    for pIdx = 1:6
        tbin_ms = opt.tbin*1000;
        
        subplot(2,3,pIdx);
        [~,~,psth4X(pIdx)] = plot_timecourse('stream',score(:,pIdx)',t_align_onpatch{2}/tbin_ms,t_start_onpatch{2}/tbin_ms,t_end_onpatch2s{2}/tbin_ms,barcodeID_Lg,'resample_bin',1);
        atitle(sprintf('PC%d/%s/',pIdx,psth_label{2}));
    end    
    figcounter = figcounter + 1;
end

subplot(1,2,1)
plot(psth4X(1).mean(1, :), psth4X(3).mean(1, :)); hold on;
plot(psth4X(1).mean(2, :), psth4X(3).mean(2, :));

subplot(1,2,2)
plot3(psth4X(1).mean(1, :), psth4X(2).mean(1, :), psth4X(3).mean(1, :)); hold on;
plot3(psth4X(1).mean(2, :), psth4X(2).mean(2, :), psth4X(3).mean(2, :));

%% troubleshooting alignment issues

% linear fit w/o bias
fitlm(prts, patchleave_ix - patchstop_ix)
% linear fit with bias
fitlm(prts, patchleave_ix - patchstop_ix + 559.5/tbin_ms)

patchstop_ms = p_out.patchstop_ms;
patchleave_ms = p_out.patchleave_ms;
% create index vectors from our update timestamp vectors
patchstop_ix = round((patchstop_ms) / tbin_ms) + 1;
patchleave_ix = min(round((patchleave_ms) / tbin_ms) + 1,size(fr_mat_zscore,2)); % might not be good

% gather trajectories by trial using our new indices
for iTrial = 1:numel(patchleave_ix)
%         trial_pc_traj{sIdx}{iTrial} = pc_space_traj(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
    trial_pc_traj{sIdx}{iTrial} = pc_space_traj(patchstop_ix(iTrial):patchleave_ix(iTrial),:);
end

cell_ix_len = cell2mat(cellfun(@length,trial_pc_traj{3},'uni',false));
ratio = cell_ix_len ./ prts';
figure(1)
subplot(1,3,1)
scatter(prts,patchleave_ix - patchstop_ix)
title("PRT vs patchleaveidx - patchstopidx")
xlabel("PRT")
ylabel("patchleaveidx - patchstopidx")
subplot(1,3,2)
hist(ratio)
title("Histogram of ratio between cell index length and prt")
xlabel("Ratio between cell ix len and prt")
ylabel("Frequency")
subplot(1,3,3)
% so the problems are not being caused by 
scatter(prts,cell_ix_len)
xlabel("PRT")
ylabel("cell index length")
title("PRT vs cell index length")

% so the problems are not being caused by 
scatter(prts,ratio)
xlabel("PRT")
ylabel("ratio")
title("PRT vs cell index length / PRT")

%% old code
%  Manually cut out PSTH time after calculating big PSTH matrix
%     tic
%     fr_mat = calcFRVsTime(good_cells,dat,opt);
%     fr_mat_zscore = my_zscore(fr_mat);
%     toc
%     % convert to ms to indices for FR mat
%     tbin_ms = opt.tbin*1000;
%     patchstop_ix = round(patchstop_ms / tbin_ms);
%     patchleave_ix = round(patchleave_ms / tbin_ms);
%     % slice FR matrix s.t. we only analyze data on patch
%     off_patch_vector = zeros(size(fr_mat,2),1);
%     off_patch_vector(1:patchstop_ix(1)) = -1; % time before first patch
%     off_patch_vector(patchstop_ix(end):end) = -1; % time after last patch
%     
%     for iTrial = 1:numel(patchleave_ix)-1
%         off_patch_vector(patchleave_ix(iTrial):patchstop_ix(iTrial+1)) = -1; % remove time between patches
%     end
%     fr_mat_zscore(:,off_patch_vector < 0) = []; % remove time off patch 
%     % adjust index vectors
%     off_ix = patchstop_ix(2:end) - patchleave_ix(1:end-1);
%     off_ix = [off_ix ; length(off_patch_vector) - patchstop_ix(end)];
%     off_ix(1) = patchstop_ix(1);
%     off_ix = cumsum(off_ix);
%     patchstop_ix = patchstop_ix - off_ix + 1; 
%     patchleave_ix = patchleave_ix - off_ix + 1;
%     % adjust time vectors
%     off_ms = patchstop_ms(2:end) - patchleave_ms(1:end-1);
%     off_ms = [off_ms ; patchleave_ms(end)];
%     off_ms(1) = patchstop_ms(1);
%     off_ms = cumsum(off_ms);
%     patchstop_ms = patchstop_ms - off_ms; 
%     patchleave_ms = patchleave_ms - off_ms;

        % now plot pc trajectories for 2 seconds of performance, colored by
        % whether reward was received at the 2nd second every trial
% individual traces
%         for j = 1:numel(iSizeTrials) 
%             iTrial = iSizeTrials(j);
%             if prts(iTrial) > 2.5 % not sure why needs to be > 2.5- this is a real problem
%                 figure(figcounter)
%                 subplot(2,1,1)
%                 scatter(trial_pc_traj{sIdx}{iTrial}(sec2idx,1),trial_pc_traj{sIdx}{iTrial}(sec2idx,2),'rx') % traj end pt
%                 hold on
%                 scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),'bo') % traj start pt
%                 plot(trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1),trial_pc_traj{sIdx}{iTrial}(1:sec2idx,2),'Color',barcode_colorgrad2(trial2secIdVec(j),:)) % color by PRT
%                 xlabel('PC 1')
%                 ylabel('PC 2')
%                 title(sprintf('Session %s %i uL Rew Trial Trajectories through 2 PC Space',session,iRewsize));
%                 
%                 subplot(2,1,2)
%                 scatter3(trial_pc_traj{sIdx}{iTrial}(sec2idx),trial_pc_traj{sIdx}{iTrial}(sec2idx,2),trial_pc_traj{sIdx}{iTrial}(sec2idx,3),'rx') % traj end pt
%                 hold on
%                 scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
%                 plot3(trial_pc_traj{sIdx}{iTrial}(1:sec2idx,1),trial_pc_traj{sIdx}{iTrial}(1:sec2idx,2),trial_pc_traj{sIdx}{iTrial}(1:sec2idx,3),'Color',barcode_colorgrad2(trial2secIdVec(j),:)) % color by PRT
%                 xlabel('PC 1')
%                 ylabel('PC 2')
%                 zlabel('PC 3')
%                 title(sprintf('Session %s %i uL Rew Trial Trajectories through 3 PC Space',session,iRewsize));
%             end
%             
%             if prts(iTrial) > 3.5
%                 figure(figcounter+1)
%                 subplot(2,1,1)
%                 scatter(trial_pc_traj{sIdx}{iTrial}(sec3idx,1),trial_pc_traj{sIdx}{iTrial}(sec3idx,2),'rx') % traj end pt
%                 hold on
%                 scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),'bo') % traj start pt
%                 plot(trial_pc_traj{sIdx}{iTrial}(1:sec3idx,1),trial_pc_traj{sIdx}{iTrial}(1:sec3idx,2),'Color',barcode_colorgrad4(trial3secIdVec(j),:)) % color by PRT
%                 xlabel('PC 1')
%                 ylabel('PC 2')
%                 title(sprintf('Session %s %i uL Rew Trial Trajectories through 2 PC Space',session,iRewsize));
%                 
%                 subplot(2,1,2)
%                 scatter3(trial_pc_traj{sIdx}{iTrial}(sec3idx),trial_pc_traj{sIdx}{iTrial}(sec3idx,2),trial_pc_traj{sIdx}{iTrial}(sec3idx,3),'rx') % traj end pt
%                 hold on
%                 scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(1,2),trial_pc_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
%                 plot3(trial_pc_traj{sIdx}{iTrial}(1:sec3idx,1),trial_pc_traj{sIdx}{iTrial}(1:sec3idx,2),trial_pc_traj{sIdx}{iTrial}(1:sec3idx,3),'Color',barcode_colorgrad4(trial3secIdVec(j),:)) % color by PRT
%                 xlabel('PC 1')
%                 ylabel('PC 2')
%                 zlabel('PC 3')
%                 title(sprintf('Session %s %i uL Rew Trial Trajectories through 3 PC Space',session,iRewsize));
%             end
%         end
        
        % last, plot barcode-pooled mean traces
%         cat2sec = cellfun(@(x) x(1:sec2idx,:),trial_pc_traj{sIdx}(prts > 2.55),'un',0);
%         cat3sec = cellfun(@(x) x(1:sec3idx,:),trial_pc_traj{sIdx}(prts > 3.55),'un',0);
