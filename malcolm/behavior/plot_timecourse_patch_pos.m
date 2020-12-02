paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\behavior\patch_pos'; % where to save figs

addpath(genpath('C:\code\HGK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};
sessions = sessions(~contains(sessions,'mc'));

%% BEHAVIOR
reset_figs;
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    fig_counter = fig_counter+1;
    hfig(fig_counter) = figure('Position',[100 100 1500 600]);
    hfig(fig_counter).Name = sprintf('%s_patch_pos',session);
    
    % load data
    dat = load(fullfile(paths.data,session));
    
    
    % position data to plot
    t = dat.velt;
    pos = dat.patch_pos;
    pos(isnan(pos)) = -4;
    pos_ms = interp1(t,pos,0:0.001:max(t));
    
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
    
    % patch cue
    t_align = patchcue_ms;
    t_start = patchcue_ms-1000;
    t_end = patchstop_ms;
    subplot(2,4,1);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end);
    title('Aligned to cue');
    subplot(2,4,5);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end,rew_size);
    title('Aligned to cue');

    % patch stop
    t_align = patchstop_ms;
    t_start = patchcue_ms; 
    t_end = min(patchleave_ms,patchstop_ms+5000); % maximum of 5 seconds after patch stop
    subplot(2,4,2);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end);
    title('Aligned to stop');
    subplot(2,4,6);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end,rew_size);
    title('Aligned to stop');

    % patch leave
    t_align = patchleave_ms;
    t_start = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
    t_end = patchleave_ms+2000;
    subplot(2,4,3);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end);
    title('Aligned to leave');
    subplot(2,4,7);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end,rew_size);
    title('Aligned to leave');

    % reward
    t_align = rew_ms;
    t_start = rew_ms-1000; % -1 to +1 sec rel. to reward
    t_end = rew_ms+1000;
    subplot(2,4,4);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end);   
    title('Aligned to reward');
    subplot(2,4,8);
    plot_timecourse('stream',pos_ms,t_align,t_start,t_end,rew_size_indiv);
    title('Aligned to reward');
    
end
save_figs(paths.figs,hfig,'png');