paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = {'mc2_20201017.mat'};

%% PSTHs: Individual cells - WARNING - takes forever!
psth = {};
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
	fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);    
    
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    % get depth
    [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps,dat.sp.winv,dat.sp.ycoords,dat.sp.spikeTemplates,dat.sp.tempScalingAmps);
    spike_depths = nan(numel(good_cells),1);
    for cidx = 1:numel(good_cells)
        spike_depths(cidx) = median(spike_depths_all(dat.sp.clu==good_cells(cidx)));
    end

    if isfield(dat.anatomy,'insertion_depth')
        depth_from_surface = spike_depths-dat.anatomy.insertion_depth;
    else
        depth_from_surface = nan(size(spike_depths));
    end
   
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
    
    % what to compute
    psth_this = struct;
    psth_this.session = session;
    psth_this.cids = good_cells;
    psth_this.rew_size = rew_size;
    psth_this.N0 = N0;
    psth_this.psth_cue = [];
    psth_this.psth_stop = [];
    psth_this.psth_leave = [];
    psth_this.psth_rew = [];
    psth_this.t_window_cue = [];
    psth_this.t_window_stop = [];
    psth_this.t_window_leave = [];
    psth_this.t_window_rew = [];
    
    % create psth's using Hyunggoo's plot_timecourse
    for cIdx = 1:numel(good_cells)
        fprintf('\tPlot timecourse for cell %d/%d...\n',cIdx,numel(good_cells));
        spiket = dat.sp.st(dat.sp.clu==good_cells(cIdx));
        spiket_ms = spiket*1000;
        
        h = figure('Position',[200 200 2000 800]);
        h.Name = sprintf('%s_c%d_psths %s depthFromSurface=%d',session,good_cells(cIdx),dat.brain_region_rough{cIdx},round(depth_from_surface(cIdx)));
        
        % patch cue
        t_align = patchcue_ms;
        t_start = patchcue_ms-1000;
        t_end = patchstop_ms;
        subplot(2,4,1);
        [~,~,z,t]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
        atitle('CUE/ALL TRIALS');
        psth_this.psth_cue = [psth_this.psth_cue; z.mean];
        psth_this.t_window_cue = t;
        subplot(2,4,5);
        plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end,rew_size);
        atitle('CUE/SPLIT BY REW SIZE');       
        
        % patch stop
        t_align = patchstop_ms;
        t_start = patchcue_ms; 
        t_end = min(patchleave_ms,patchstop_ms+5000); % maximum of 5 seconds after patch stop
        subplot(2,4,2);
        [~,~,z,t]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
        atitle('STOP/ALL TRIALS');
        psth_this.psth_stop = [psth_this.psth_stop; z.mean];
        psth_this.t_window_stop = t;
        subplot(2,4,6);
        plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end,rew_size);
        atitle('STOP/SPLIT BY REW SIZE');
        
        % patch leave
        t_align = patchleave_ms;
        t_start = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
        t_end = patchleave_ms+2000;
        subplot(2,4,3);
        [~,~,z,t]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
        atitle('LEAVE/ALL TRIALS');
        psth_this.psth_leave = [psth_this.psth_leave; z.mean];
        psth_this.t_window_leave = t;
        subplot(2,4,7);
        plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end,rew_size);
        atitle('LEAVE/SPLIT BY REW SIZE');
        
        % reward
        t_align = rew_ms;
        t_start = rew_ms-1000; % -1 to +1 sec rel. to reward
        t_end = rew_ms+1000;
        subplot(2,4,4);
        [~,~,z,t]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
        atitle('REW/ALL TRIALS');
        psth_this.psth_rew = [psth_this.psth_rew; z.mean];
        psth_this.t_window_rew = t;
        subplot(2,4,8);
        plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end,rew_size_indiv);
        atitle('REW/SPLIT BY REW SIZE');
        
        save_figs(fullfile(paths.figs,'psth_indiv_cells',session),h,'png');
        close(h);
    end
    psth{sIdx} = psth_this;
end
% save('psth.mat','psth');