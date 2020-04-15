paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.tbin = 0.001;
opt.smoothSigma_time = 0.1;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% BEHAVIOR
reset_figs;
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);
    fig_counter = fig_counter+1;
    hfig(fig_counter) = figure('Position',[100 100 1500 1200]);
    hfig(fig_counter).Name = sprintf('%s_behavior',session);
    
    % load data
    dat = load(fullfile(paths.data,session));
    
    % behavioral events to align to
    rew_size = mod(dat.patches(:,2),10);
    N0 = mod(round(dat.patches(:,2)/10),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    rew_ms = dat.rew_ts*1000;
    lickt_ms = dat.lick_ts*1000;
    speed_ms = interp1(dat.velt,dat.vel,0:0.001:max(dat.velt));
    
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
    subplot(4,4,1);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
    atitle('CUE/SPEED/ALL TRIALS');
    subplot(4,4,5);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
    atitle('CUE/SPEED/SPLIT BY REW SIZE');
    subplot(4,4,9);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
    atitle('CUE/LICK/ALL TRIALS');
    subplot(4,4,13);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
    atitle('CUE/LICK/SPLIT BY REW SIZE');

    % patch stop
    t_align = patchstop_ms;
    t_start = patchcue_ms; 
    t_end = min(patchleave_ms,patchstop_ms+5000); % maximum of 5 seconds after patch stop
    subplot(4,4,2);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
    atitle('STOP/SPEED/ALL TRIALS');
    subplot(4,4,6);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
    atitle('STOP/SPEED/SPLIT BY REW SIZE');
    subplot(4,4,10);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
    atitle('STOP/LICK/ALL TRIALS');
    subplot(4,4,14);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
    atitle('STOP/LICK/SPLIT BY REW SIZE');
    

    % patch leave
    t_align = patchleave_ms;
    t_start = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
    t_end = patchleave_ms+2000;
    subplot(4,4,3);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end);
    atitle('LEAVE/SPEED/ALL TRIALS');
    subplot(4,4,7);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size);
    atitle('LEAVE/SPEED/SPLIT BY REW SIZE');
    subplot(4,4,11);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
    atitle('LEAVE/LICK/ALL TRIALS');
    subplot(4,4,15);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size);
    atitle('LEAVE/LICK/SPLIT BY REW SIZE');

    % reward
    t_align = rew_ms;
    t_start = rew_ms-1000; % -1 to +1 sec rel. to reward
    t_end = rew_ms+1000;
    subplot(4,4,4);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end);   
    atitle('REW/SPEED/ALL TRIALS');
    subplot(4,4,8);
    plot_timecourse('stream',speed_ms,t_align,t_start,t_end,rew_size_indiv);
    atitle('REW/SPEED/SPLIT BY REW SIZE');
    subplot(4,4,12);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end);
    atitle('REW/LICK/ALL TRIALS');
    subplot(4,4,16);
    plot_timecourse('timestamp',lickt_ms,t_align,t_start,t_end,rew_size_indiv);
    atitle('REW/LICK/SPLIT BY REW SIZE');
end
save_figs(fullfile(paths.figs,'behavior'),hfig,'png');

%% PSTHs: Individual cells - WARNING - takes forever!
psth = {};
parfor sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
	fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);    
    
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
   
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
        h.Name = sprintf('%s_c%d_psths',session,good_cells(cIdx));
        
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
save('psth.mat','psth');

%% PCA on PSTHs
reset_figs;
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('PCA on PSTHs for session %d/%d: %s...\n',sIdx,numel(sessions),session);
    
    psth_by_alignment = {psth{sIdx}.psth_cue,...
        psth{sIdx}.psth_stop,...
        psth{sIdx}.psth_leave,...
        psth{sIdx}.psth_rew};
    t_window_by_alignment = {psth{sIdx}.t_window_cue,...
        psth{sIdx}.t_window_stop,...
        psth{sIdx}.t_window_leave,...
        psth{sIdx}.t_window_rew};
    psth_label = {'cue','stop','leave','rew'};
    
    for aIdx = 1:numel(psth_label) % different alignments (cue, stop, leave, rew)
        fig_counter = fig_counter+1;
        hfig(fig_counter) = figure('Position',[200 200 800 700]);
        hfig(fig_counter).Name = sprintf('%s - pca on psths - %s',session,psth_label{aIdx});

        % get psth for this alignment, and exclude nans
        psth_this = psth_by_alignment{aIdx};
        t_this = t_window_by_alignment{aIdx};
        t_this = t_this(1:10:end);
        t_this = t_this(~all(isnan(psth_this)));
        psth_this = psth_this(:,~all(isnan(psth_this)));
        
        % z score      
        psth_zscore = my_zscore(psth_this);

        % pca
        [coeffs,score,~,~,expl] = pca(psth_zscore);

        % eigenvalue plot
        subplot(2,2,1); hold on; set_fig_prefs;
        my_scatter(1:numel(expl),expl,'r',0.3);
        my_scatter(1:numel(expl),cumsum(expl),'b',0.3);
        ylim([0 100]);
        legend({'Indiv.','Cum.'});
        xlabel('Num. PCs');
        ylabel('% explained variance');
        xlim([-10 size(coeffs,2)+10]);
        title(sprintf('SESSION: %s',session),'Interpreter','none');

        % top two principal components
        subplot(2,2,2); hold on; set_fig_prefs;
        plot(t_this,coeffs(:,1));
        plot(t_this,coeffs(:,2));    
        xlabel(sprintf('Time relative to %s (sec)',psth_label{aIdx}));
        ylabel('PC coefficient');
        legend({'PC1','PC2'},'Location','northwest');
        plot([0 0],ylim,'k:','HandleVisibility','off');

        % cells projected into top two principal components
        subplot(2,2,3); hold on; set_fig_prefs;
        my_scatter(score(:,1),score(:,2),'k',0.3);
        xlabel('PC1 loading');
        ylabel('PC2 loading');
        title(sprintf('%d single units',size(psth_zscore,1)));
        axis equal;
        plot([0 0],ylim,'k:');
        plot(xlim,[0 0],'k:');

        % histogram of component scores
        subplot(2,2,4); hold on; set_fig_prefs;
        [ks1,x1] = ksdensity(score(:,1),'Bandwidth',3);
        [ks2,x2] = ksdensity(score(:,2),'Bandwidth',3);
        plot(x1,ks1);
        plot(x2,ks2);
        plot([0 0],ylim,'k:','HandleVisibility','off');
        legend({'PC1','PC2'},'location','northwest');
        xlabel('PC loading');
        ylabel('Density');
    end
end
save_figs(fullfile(paths.figs,'pca_on_psths'),hfig,'png');

%% PCA on fat firing rate matrix: neurons (N) by time (T) where T is the length of the session
opt.tstart = 0;
opt.tend = max(dat.sp.st);
tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

reset_figs
tic
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
	fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);    
        
    % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
   
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
    
    % compute firing rate mat
    fr_mat = calcFRVsTime(good_cells,dat,opt);
    fr_mat_zscore = my_zscore(fr_mat);

    % pca on firing rate matrix
    [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
    
    % make figure: eigenvalue plot
    fig_counter = fig_counter+1;
    hfig(fig_counter) = figure('Position',[100 100 400 350]);
    hfig(fig_counter).Name = sprintf('%s - pca whole session - eigenvalue plot',session);
    hold on; set_fig_prefs;
    my_scatter(1:numel(expl),expl,'r',0.3);
    my_scatter(1:numel(expl),cumsum(expl),'b',0.3);
    ylim([0 100]);
    legend({'Indiv.','Cum.'});
    xlabel('Num. PCs');
    ylabel('% explained variance');
    xlim([-10 size(coeffs,2)+10]);
    title(sprintf('SESSION: %s',session),'Interpreter','none');
    
    % PC traces aligned to task events
    
    % alignments:
    psth_label = {'cue','stop','leave','rew'};
    t_align = cell(4,1);
    t_start = cell(4,1);
    t_end = cell(4,1);
    % cue
    t_align{1} = patchcue_ms;
    t_start{1} = patchcue_ms-1000;
    t_end{1} = patchstop_ms;
    % stop
    t_align{2} = patchstop_ms;
    t_start{2} = patchcue_ms; 
    t_end{2} = min(patchleave_ms,patchstop_ms+5000); % maximum of 5 seconds after patch stop
    % leave
    t_align{3} = patchleave_ms;
    t_start{3} = max(patchstop_ms,patchleave_ms-5000); % maximum of 5 seconds before patch leave
    t_end{3} = patchleave_ms+2000;
    % rew
    t_align{4} = rew_ms;
    t_start{4} = rew_ms-1000; % -1 to +1 sec rel. to reward
    t_end{4} = rew_ms+1000;
    
    % groupings by rew size:
    groupings = cell(4,1);
    groupings{1} = rew_size;
    groupings{2} = rew_size;
    groupings{3} = rew_size;
    groupings{4} = rew_size_indiv;    
    
    for aIdx = 1:4       
        fig_counter = fig_counter+1;
        hfig(fig_counter) = figure('Position',[100 100 2300 700]);
        hfig(fig_counter).Name = sprintf('%s - pca whole session - task aligned - %s',session,psth_label{aIdx});
        
        for pIdx = 1:6
            subplot(2,6,pIdx);
            plot_timecourse('stream',score(:,pIdx)',t_align{aIdx},t_start{aIdx},t_end{aIdx});
            atitle(sprintf('PC%d/%s/ALL TRIALS/',pIdx,psth_label{aIdx}));
            subplot(2,6,pIdx+6);
            plot_timecourse('stream',score(:,pIdx)',t_align{aIdx},t_start{aIdx},t_end{aIdx},groupings{aIdx});
            atitle(sprintf('PC%d/%s/SPLIT BY REW SIZE/',pIdx,psth_label{aIdx}));
        end
    end
    
end
toc
save_figs(fullfile(paths.figs,'pca_whole_session'),hfig,'png');