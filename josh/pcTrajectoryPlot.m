%% Plot neural trajectories in reduced PC space over time between trials
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_matlab/neuroPixelsData/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_matlab/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Iterate over sessions
reset_figs
tic
trial_pc_traj = {{}};

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
    N0 = mod(round(dat.patches(:,2)/10),10);
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
    
     % compute firing rate matrix 
    tic
    fr_mat = calcFRVsTime(good_cells,dat,opt);
    fr_mat_zscore = my_zscore(fr_mat);
    toc
    
    % convert to ms to indices for FR mat
    tbin_ms = opt.tbin*1000;
    patchstop_ix = round(patchstop_ms / tbin_ms);
    patchleave_ix = round(patchleave_ms / tbin_ms);
    % slice FR matrix s.t. we only analyze data on patch
    off_patch_vector = zeros(size(fr_mat,2),1);
    off_patch_vector(1:patchstop_ix(1)) = -1; % time before first patch
    off_patch_vector(patchstop_ix(end):end) = -1; % time after last patch

    for iTrial = 1:numel(patchleave_ix)-1
        off_patch_vector(patchleave_ix(iTrial):patchstop_ix(iTrial+1)) = -1; % remove time between patches
    end
    fr_mat_zscore(:,off_patch_vector < 0) = []; % remove time off patch 
    % now adjust index vectors
    off_ix = patchstop_ix(2:end) - patchleave_ix(1:end-1);
    off_ix = [off_ix ; length(off_patch_vector) - patchstop_ix(end)];
    off_ix(1) = patchstop_ix(1);
    off_ix = cumsum(off_ix);

    patchstop_ix = patchstop_ix - off_ix + 1; 
    patchleave_ix = patchleave_ix - off_ix + 1;
    
    % now perform PCA on concatenated matrix
    tic
    [u,s,v] = svd(fr_mat_zscore);
    toc
    s = diag(s);
    plot(s(1:10).*2 / sum(s.*2))
    title("Variance Explained Principle Components")
    xlabel("Principle Component")
    ylabel("Normalized Variance Explained")
    
    % transform into PC Space with first 3 components
    reduced_data = u(:,1:3) * diag(s(1:3)) * v(1:3,:);
    pc_space_traj = v(1:3,:);
    
    % gather trajectories by trial using our new indices
    for iTrial = 1:numel(patchleave_ix)
        trial_pc_traj{sIdx}{iTrial} = pc_space_traj(:,patchstop_ix(iTrial):patchleave_ix(iTrial));
    end
end

%% Visualize PC Trajectories

for sIdx = 3:3 % replace this when doing multiple sessions
    session = erase(sessions{sIdx}(1:end-4),'_');
    for iTrial = 10:20
        figure(sIdx * 2 + 1)
        scatter(trial_pc_traj{sIdx}{iTrial}(1,end),trial_pc_traj{sIdx}{iTrial}(2,end),'bo')
        scatter(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(2,1),'rx')
        hold on
        plot(trial_pc_traj{sIdx}{iTrial}(1,:),trial_pc_traj{sIdx}{iTrial}(2,:))
        xlabel('PC 1')
        ylabel('PC 2')
        title(sprintf('%s Trajectories through 2 PC Space',session));

        figure(sIdx * 2 + 2)
        scatter3(trial_pc_traj{sIdx}{iTrial}(1,end),trial_pc_traj{sIdx}{iTrial}(2,end),trial_pc_traj{sIdx}{iTrial}(3,end),'bo')
        scatter3(trial_pc_traj{sIdx}{iTrial}(1,1),trial_pc_traj{sIdx}{iTrial}(2,1),trial_pc_traj{sIdx}{iTrial}(3,1),'rx')
        hold on
        plot3(trial_pc_traj{sIdx}{iTrial}(1,:),trial_pc_traj{sIdx}{iTrial}(2,:),trial_pc_traj{sIdx}{iTrial}(3,:))
        xlabel('PC 1')
        ylabel('PC 2')
        zlabel('PC 3')
        title(sprintf('%s Trajectories through 3 PC Space',session));
    end
end

