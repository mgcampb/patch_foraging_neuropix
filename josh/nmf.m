%% perform NMF to see if we're missing anything 

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

%% Iterate over sessions and perform NMF
reset_figs
trial_factor_traj = {{}};

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
    
    % update timestamp vectors according to p_out, including .55 ms bias
    % according to linear regression results
    patchstop_ms = p_out.patchstop_ms + 9;
    patchleave_ms = p_out.patchleave_ms + 9;
    % create index vectors from our update timestamp vectors
    patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
    patchleave_ix = min(round(patchleave_ms / tbin_ms) + 1,size(fr_mat,2)); % might not be good
 
    % now perform NMF on concatenated matrix

%     max_factors = 20;
%     n_inits = 1; 
%     D = zeros(max_factors,n_inits);
%     for k = 1:max_factors
%         for i = 1:n_inits
%             [W,H,d] = nnmf(fr_mat,k);
%             D(k,i) = d;
%         end
%     end

    % let's just take the 3-factor version
    k = 3;
    [W,H,D] = nnmf(fr_mat,k);

%     % plot D across initializations/number of factors
%     plot(expl(1:10).*2 / sum(expl.*2))
%     title("Variance Explained Principle Components")
%     xlabel("Principle Component")
%     ylabel("Normalized Variance Explained")

    % gather factor trajectories by trial using our new indices
    for iTrial = 1:numel(patchleave_ix)
        trial_factor_traj{sIdx}{iTrial} = H(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
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
        
        figure()
        for j = 1:k
            iTrial = trials10x(j);
            colormap(cool(sec2idx))
            t2_1 = 1:sec2idx;
            t2_2 = (1:sec2idx) + sec2idx;
            subplot(2,1,1)
            patch([trial_factor_traj{sIdx}{iTrial}(1,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(3,1:sec2idx) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_factor_traj{sIdx}{iTrial}(1,sec2idx),trial_factor_traj{sIdx}{iTrial}(3,sec2idx),'rx') % traj end pt
%             grad = gradient(trial_factor_traj{sIdx}{iTrial}(sec2idx - 1:sec2idx,[1,3]));
%             quiver(trial_factor_traj{sIdx}{iTrial}(sec2idx,1),trial_factor_traj{sIdx}{iTrial}(sec2idx,3),grad(1,1),grad(2,1)) % traj end pt
%             drawArrow(trial_factor_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,1),trial_factor_traj{sIdx}{iTrial}(sec2idx-1:sec2idx,3),'linewidth',.5,'color','r','maxheadsize',.5)
            scatter(trial_factor_traj{sIdx}{iTrial}(1,1),trial_factor_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            
            xlabel('Factor 1')
            ylabel('Factor 3')
            title(sprintf('Session %s %i0X Trial Trajectories through 2 Factor Space',session,iRewsize));
            grid()
            
            subplot(2,1,2)
            patch([trial_factor_traj{sIdx}{iTrial}(1,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(2,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(3,1:sec2idx) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5)
            hold on
            scatter3(trial_factor_traj{sIdx}{iTrial}(1,sec2idx),trial_factor_traj{sIdx}{iTrial}(2,sec2idx),trial_factor_traj{sIdx}{iTrial}(3,sec2idx),'rx') % traj end pt
            hold on
            scatter3(trial_factor_traj{sIdx}{iTrial}(1,1),trial_factor_traj{sIdx}{iTrial}(2,1),trial_factor_traj{sIdx}{iTrial}(3,1),'bo') % traj start pt
            xlabel('Factor 1')
            ylabel('Factor 2')
            zlabel('Factor 3')
            title(sprintf('Session %s %i0X Trial Trajectories through 3 Factor Space',session,iRewsize));
            view(3)
            grid()
        end
        
        % similarly plot the RRX
        figure()
        for j = 1:k
            iTrial = trials11x(j);
            colormap(autumn(sec2idx))
            t2_1 = 1:sec2idx;
            t2_2 = (1:sec2idx) + sec2idx;
            subplot(2,1,1)
            patch([trial_factor_traj{sIdx}{iTrial}(1,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(3,1:sec2idx) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5) % color by PRT
            hold on
            scatter(trial_factor_traj{sIdx}{iTrial}(1,sec2idx),trial_factor_traj{sIdx}{iTrial}(3,sec2idx),'rx') % traj end pt
            scatter(trial_factor_traj{sIdx}{iTrial}(1,1),trial_factor_traj{sIdx}{iTrial}(1,3),'bo') % traj start pt
            
            xlabel('Factor 1')
            ylabel('Factor 3')
            title(sprintf('Session %s %i%iX Trial Trajectories through 2 Factor Space',session,iRewsize,iRewsize));
            grid()
            
            subplot(2,1,2)
            patch([trial_factor_traj{sIdx}{iTrial}(1,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(2,1:sec2idx) nan],[trial_factor_traj{sIdx}{iTrial}(3,1:sec2idx) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',.5)
            hold on
            scatter3(trial_factor_traj{sIdx}{iTrial}(1,sec2idx),trial_factor_traj{sIdx}{iTrial}(2,sec2idx),trial_factor_traj{sIdx}{iTrial}(3,sec2idx),'rx') % traj end pt
            scatter3(trial_factor_traj{sIdx}{iTrial}(1,1),trial_factor_traj{sIdx}{iTrial}(2,1),trial_factor_traj{sIdx}{iTrial}(3,1),'bo') % traj start pt
            xlabel('Factor 1')
            ylabel('Factor 2')
            zlabel('Factor 3')
            title(sprintf('Session %s %i%iX Trial Trajectories through 3 Factor Space',session,iRewsize,iRewsize));
            view(3)
            grid()
        end

        cat10x = cellfun(@(x) x(:,1:sec2idx),trial_factor_traj{sIdx}(trials10x),'un',0);
        cat11x = cellfun(@(x) x(:,1:sec2idx),trial_factor_traj{sIdx}(trials11x),'un',0);

        mean10x = mean(cat(3,cat10x{:}),3);
        mean11x = mean(cat(3,cat11x{:}),3);
        
%         if iRewsize > 1
%             cat100x = cellfun(@(x) x(1:sec3idx,:),trial_factor_traj{sIdx}(trials100x),'un',0);
%             cat110x = cellfun(@(x) x(1:sec3idx,:),trial_factor_traj{sIdx}(trials110x),'un',0);
%             cat101x = cellfun(@(x) x(1:sec3idx,:),trial_factor_traj{sIdx}(trials101x),'un',0);
%             cat111x = cellfun(@(x) x(1:sec3idx,:),trial_factor_traj{sIdx}(trials111x),'un',0);
% 
%             mean100x = mean(cat(3,cat100x{:}),3);
%             mean110x = mean(cat(3,cat110x{:}),3);
%             mean101x = mean(cat(3,cat101x{:}),3);
%             mean111x = mean(cat(3,cat111x{:}),3); 
%         end
%         
        % mean 2-second activity
        figure()
        colormap([cool(sec2idx);hot(sec2idx)]) 
        subplot(2,1,1)
        t2_1 = 1:sec2idx;
        t2_2 = (1:sec2idx) + sec2idx;
        t2_rew = (1000/tbin_ms);
        patch([mean10x(1,:) nan],[mean10x(3,:) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        hold on
        patch([mean11x(1,:) nan],[mean11x(3,:) nan],[t2_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        view(2)
        grid()
        scatter([mean10x(1,1) mean11x(1,1)],[mean10x(3,1) mean11x(3,1)],'bo')
        scatter([mean10x(1,end) mean11x(1,end)],[mean10x(3,end) mean11x(3,end)],'rx')
%         scatter(mean11x(1,end),mean11x(2,end),'rx')
        scatter(mean11x(1,t2_rew),mean11x(3,t2_rew),'k*')
        xlabel('Factor 1')
        ylabel('Factor 3')
        title(sprintf('Session %s %i%iX vs %i0X Trajectories through 2 Factor Space',session,iRewsize,iRewsize,iRewsize));
        % now 3 dimensions
        subplot(2,1,2)
        patch([mean10x(1,:) nan],[mean10x(2,:) nan],[mean10x(3,:) nan],[t2_1 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        hold on
        patch([mean11x(1,:) nan],[mean11x(2,:) nan],[mean11x(3,:) nan],[t2_2 nan],'FaceColor','none','EdgeColor','interp','LineWidth',1.5) % color by PRT
        view(3)
        grid()
        scatter3([mean10x(1,1) mean11x(1,1)],[mean10x(2,1) mean11x(2,1)],[mean10x(3,1) mean11x(3,1)],'bo')
        scatter3([mean10x(1,end) mean11x(1,end)],[mean10x(2,end) mean11x(2,end)],[mean10x(3,end) mean11x(3,end)],'rx')
        scatter3(mean11x(1,t2_rew),mean11x(2,t2_rew),mean11x(3,t2_rew),'k*')
        xlabel('Factor 1')
        ylabel('Factor 2')
        zlabel('Factor 3')
        title(sprintf('Session %s %i%iX vs %i0X Trajectories through 3 Factor Space',session,iRewsize,iRewsize,iRewsize));
        
    end
end
