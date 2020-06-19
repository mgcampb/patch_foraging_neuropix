%% Perform PCA on firing rates, separated by the depth of recording 

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

%% First visualize the distribution of depths and firing rates within each layer 
for sIdx = 3:3
    session = sessions{sIdx}(1:end-4);
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2)';
    
    % get spike depths for all spikes individually
    % this function comes from the spikes repository
    % depth indicates distance from tip of probe in microns
    [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);

    % take median spike depth for each cell
    spike_depths = nan(size(good_cells));

    parfor cIdx = 1:numel(good_cells)
        spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
    end
 
    hist(spike_depths);
    title("Distribution of spike depths")
end
%% Iterate over sessions, perform PCA within depth quantiles
reset_figs
tic
trial_pc_traj = {{}};
pc_reductions = {{}};

for sIdx = 3:3 % numel(sessions)
    session = sessions{sIdx}(1:end-4);
	fprintf('Loading session %d/%d: %s...\n',sIdx,numel(sessions),session);    
        % load data
    dat = load(fullfile(paths.data,session));
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    
    [~, spike_depths_all] = templatePositionsAmplitudes(dat.sp.temps, dat.sp.winv, dat.sp.ycoords, dat.sp.spikeTemplates, dat.sp.tempScalingAmps);

    % take median spike depth for each cell
    spike_depths = nan(size(good_cells));
    parfor cIdx = 1:numel(good_cells)
        spike_depths(cIdx) = median(spike_depths_all(dat.sp.clu==good_cells(cIdx)));
    end
    depth_quartiles = [0 quantile(spike_depths,[.25,.5,.75,1])];
    
    % time bins
    opt.tstart = 0;
    opt.tend = max(dat.sp.st);
   
    % behavioral events to align to
    rew_size = mod(dat.patches(:,2),10);
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    %MB trial start/stop times to feed into onPatch firing rate matrix
    keep = patchleave_ms > patchstop_ms + opt.leaveBuffer_ms; % only including trials w PRT at least as long as 'leaveBuffer'
    trials.start = patchstop_ms(keep) /1000;
    trials.end = (patchleave_ms(keep) - opt.leaveBuffer_ms) /1000; % including time up to X ms prior to patchleave to reduce influence of running
    trials.length = trials.end - trials.start; % new 6/9/2020
    trials.length = (floor(trials.length .* 10))/10; % new 6/9/2020
    trials.end = trials.start + trials.length; % new 6/9/2020
%     
%     p.patchstop_ms = patchstop_ms(keep);
%     p.patchleave_ms = patchleave_ms(keep);

    
    for q = 2:5
         % compute firing rate matrix 
        quartile_cells = good_cells(spike_depths > depth_quartiles(q-1) & spike_depths < depth_quartiles(q));
        
        tic
        [fr_mat, tbincent] = calcFRVsTime(quartile_cells, dat, opt); %MB includes only activity within patches
        toc

        fr_mat_zscore = my_zscore(fr_mat); % z-score our psth matrix

%         % update timestamp vectors according to p_out, including .55 ms bias
%         % according to linear regression results
%         patchstop_ms = p_out.patchstop_ms + 9;
%         patchleave_ms = p_out.patchleave_ms + 9;
%         % create index vectors from our update timestamp vectors
%         patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
%         patchleave_ix = min(round(patchleave_ms / tbin_ms) + 1,size(fr_mat_zscore,2)); % might not be good

        % now perform PCA on concatenated matrix
        tic
        [coeffs,score,~,~,expl] = pca(fr_mat_zscore');
    %     [u,s,v] = svd(fr_mat_zscore);
        toc
    %     s = diag(s);
        figure(sIdx)
        subplot(2,2,q-1);
        plot(expl(1:10).*2 / sum(expl.*2))
        title(sprintf("Variance Explained by Principle Component for Depth Quartile %i",q-1))
        xlabel("Principle Component")
        ylabel("Proportion Variance Explained")
        
        figure(sIdx * 2 + 1)
        subplot(2,2,q-1);
        hist(mean(fr_mat,2))
        title(sprintf("Dist of avg FR for Depth Quartile %i",q-1))

        pc_reductions{sIdx}{q-1} = score(:,1:6);
    end
end

%% Visualize reduced PSTH Matrices, align to stop and leave, sep by quartile

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
    
    for q = 1:4
        % visualize PCs
        for aIdx = 1:4 % currently just look at stop and leave alignments
            fig_counter = fig_counter+1;
            hfig(fig_counter) = figure('Position',[100 100 2300 700]);
            hfig(fig_counter).Name = sprintf('%s - pca whole session - task aligned - %s',session,psth_label{aIdx});
            for pIdx = 1:6 % plot for first 3 PCs
                subplot(2,6,pIdx);
                plot_timecourse('stream',pc_reductions{sIdx}{q}(:,pIdx),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}/tbin_ms,[],'resample_bin',1);
                atitle(sprintf('PC%d %s-aligned Quartile %i',pIdx,psth_label{aIdx},q));
                subplot(2,6,pIdx+6);
                plot_timecourse('stream',pc_reductions{sIdx}{q}(:,pIdx),t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}/tbin_ms,grp{aIdx},'resample_bin',1);
                atitle(sprintf('PC%d %s-aligned Quartile %i',pIdx,psth_label{aIdx},q));
            end
        end
    end
end


