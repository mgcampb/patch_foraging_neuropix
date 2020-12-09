%% Developing running selectivity metric 
%  To what degree are mice advancing on the end of the patch throughout the
%  trial vs sharply running before making decision? 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = opt.tbin * 1000;
opt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
opt.preLeave_buffer = 500; 
opt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]}; 
mPFC_sessions = [1:8 10:13 15:18 23 25]; 

%% Generate "reward barcodes" to plot by trial groups later
rew_barcodes = cell(numel(sessions),1);
for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));
    
    % Trial data
    patchstop_ms = data.patchCSL(:,2);
    patchleave_ms = data.patchCSL(:,3);
    rew_ms = data.rew_ts;
    patches = data.patches;
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(patches(:,2),10);
    
    % make barcode matrices also want to know where we have no more rewards
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix = max(rew_indices);
        rew_barcode(iTrial , (last_rew_ix + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    rew_barcodes{sIdx} = rew_barcode;
end

%% Visualize behavior: velocity, licking, and position
for sIdx = 10:15
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));  
    
    rew_barcode = rew_barcodes{sIdx};
    session = sessions{sIdx}(1:end-4); 
    session_title = ['m' session(1:2) ' ' session(end-2) '/' session([end-1:end])];  

    % reinitialize ms vectors
    patchstop_ms = data.patchCSL(:,2) * 1000;
    patchleave_ms = data.patchCSL(:,3) * 1000;
    rew_size = mod(data.patches(:,2),10);
    
    % Make 4X group 
    RX_group = nan(length(patchstop_ms),1);
    trials40 = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 0);
    trials44 = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4); 
    trials20 = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) == 0);
    trials22 = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) == 2); 
    RX_group(trials20) = 1; 
    RX_group(trials22) = 2;  
    RX_group(trials40) = 3; 
    RX_group(trials44) = 4; 
    
    % alignment values:
    % stop
    t_align{1} = patchstop_ms;
    t_start{1} = patchstop_ms;
    
    % leave
    t_align{2} = patchleave_ms;
    t_end{2} = patchleave_ms+1000;
    
    % for plotting up to X # of seconds max, w attrition for trials w lower PRTs
    for i = 1:20
        t_endmax = patchleave_ms - 500;
        t_endmax(patchleave_ms > i*1000 + patchstop_ms) = patchstop_ms(patchleave_ms > i*1000 + patchstop_ms) + i*1000;
        t_end{1}{i} = t_endmax;
        
        t_startmax = patchstop_ms;
        t_startmax(patchstop_ms < patchleave_ms - i*1000) = patchleave_ms(patchstop_ms < patchleave_ms - i*1000) - i*1000;
        t_start{2}{i} = t_startmax;
    end
    
    % grouping variables
    gr.uL = rew_size; 
    gr.RX_group = RX_group;
    
    % global variable for use w plot_timecourse
    global gP
    gP.cmap{3} = cool(3);
    
    maxTime = 3;
    
    hfig = figure('Position',[100 100 2300 700]);  
    
    beh_vars = {data.vel data.patch_pos}; 
    beh_varnames = ["Velocity","Patch Position"];
    for vIdx = 1:2
        % Stop-aligned
        aIdx = 1; 
        subplot(2,3,aIdx + 3 * (vIdx - 1));
        vel_stop = plot_timecourse('stream',beh_vars{vIdx},t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}{maxTime}/tbin_ms,gr.uL,'resample_bin',1);
        vel_stop(2).XTick = [0 .05 .1];
        vel_stop(2).XTickLabel = {[0 1 2]};
        vel_stop(2).XLabel.String = "Time Since Patch Stop (s)";
        vel_stop(2).Legend.String = {('1uL') ('2uL') ('4uL')}; 
        title(sprintf("Stop-Aligned %s",beh_varnames(vIdx)))

        % Leave-aligned
        aIdx = 2; maxTime = 3;
        subplot(2,3,aIdx + 3 * (vIdx - 1));
        vel_leave = plot_timecourse('stream',beh_vars{vIdx},t_align{aIdx}/tbin_ms,t_start{aIdx}{maxTime}/tbin_ms,t_end{aIdx}/tbin_ms,gr.uL,'resample_bin',1);
        vel_leave(2).XTick = [-.1 -.05 0];
        vel_leave(2).XTickLabel = {[-2 -1 0]};
        vel_leave(2).XLabel.String = 'Time Before Leave (s)';
        vel_leave(2).Legend.String = {('1uL') ('2uL') ('4uL')};  
        title(sprintf("Leave-Aligned %s",beh_varnames(vIdx)))

        % Stop-aligned separated by t1 reward
        gP.cmap{4} = [.75 .75 1 ; .5 .5 1 ; 1 .5 1 ; 1 0 1];
        aIdx = 1; maxTime = 2;
        subplot(2,3,3 + 3 * (vIdx - 1));
        vel_RX = plot_timecourse('stream',beh_vars{vIdx},t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}{maxTime}/tbin_ms,gr.RX_group,'resample_bin',1);
        vel_RX(2).XTick = [0 .05 .1];
        vel_RX(2).XTickLabel = {[0 1 2]};
        vel_RX(2).XLabel.String = "Time Since Patch Stop (s)";
        vel_RX(2).Legend.String = {('20') ('22') ('40') ('44')};   
        title(sprintf("Stop-Aligned RX %s",beh_varnames(vIdx)))
    end
    suptitle(session_title)
end 

%% Now try to summarize this in monoparametric "running selectivity" metric 
%  try out a variety of pre-leave windows  
%  visualize trials w/ different running selectivity... maybe avg over
%  quartiles or something? 
%  Only sum positive velocity... backwards running is hmm

behavior_struct = struct; 

for sIdx = 1:numel(sessions) 
    session = sessions{sIdx}(1:end-4);
    data = load(fullfile(paths.data,session));    
    patchstop_ms = data.patchCSL(:,2) * 1000; 
    patchleave_ms = data.patchCSL(:,3) * 1000;  
    prts = patchleave_ms - patchstop_ms; 
    patchstop_ix = round(patchstop_ms / tbin_ms);
    patchleave_ix = round(patchleave_ms/ tbin_ms); 
    lick_ts_ms = data.lick_ts * 1000; 
    nTrials = length(patchleave_ms); 
    
    pre_leave_periods = [500 1000 1500] / tbin_ms;
    
    vel_trialed = cell(nTrials,1); 
    pos_trialed = cell(nTrials,1);   
    licks_trialed = cell(nTrials,1);  
    vel_selectivity_metric_sum = nan(nTrials,numel(pre_leave_periods)); 
    vel_selectivity_metric_sum_eqWindow = nan(nTrials,numel(pre_leave_periods)); 
    vel_selectivity_metric_mean = nan(nTrials,numel(pre_leave_periods)); 
    positive_velocity = data.vel; 
    positive_velocity(positive_velocity <= 0) = nan; % only care about forward movement
    for iTrial = 1:nTrials
        vel_trialed{iTrial} = data.vel(patchstop_ix(iTrial):patchleave_ix(iTrial)); 
        pos_trialed{iTrial} = data.patch_pos(patchstop_ix(iTrial):patchleave_ix(iTrial)); 
        licks_trialed{iTrial} = lick_ts_ms(lick_ts_ms > patchstop_ms(iTrial) & lick_ts_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial); 
        
        for k_period = 1:numel(pre_leave_periods)  
            if prts(iTrial) > pre_leave_periods(k_period) * tbin_ms
                background_vel_sum = nansum(positive_velocity(patchstop_ix(iTrial):patchleave_ix(iTrial)-pre_leave_periods(k_period))); 
                pre_leave_vel_sum = nansum(positive_velocity(patchleave_ix(iTrial)-pre_leave_periods(k_period):patchleave_ix(iTrial))); 
                background_vel_mean = nanmean(positive_velocity(patchstop_ix(iTrial):patchleave_ix(iTrial)-pre_leave_periods(k_period))); 
                pre_leave_vel_mean = nanmean(positive_velocity(patchleave_ix(iTrial)-pre_leave_periods(k_period):patchleave_ix(iTrial))); 
                vel_selectivity_metric_sum(iTrial,k_period) = (pre_leave_vel_sum - background_vel_sum) / pre_leave_vel_sum;   
                vel_selectivity_metric_mean(iTrial,k_period) = (pre_leave_vel_mean - background_vel_mean) / pre_leave_vel_mean;     
            end   
            % equal window now to try to get rid of correlation with PRT
            if prts(iTrial) > 2 * pre_leave_periods(k_period) * tbin_ms
                background_vel = nansum(positive_velocity(patchleave_ix(iTrial)-2*pre_leave_periods(k_period):patchleave_ix(iTrial)-pre_leave_periods(k_period)));
                pre_leave_vel = nansum(positive_velocity(patchleave_ix(iTrial)-pre_leave_periods(k_period):patchleave_ix(iTrial)));  
                vel_selectivity_metric_sum_eqWindow(iTrial,k_period) = (pre_leave_vel - background_vel) / pre_leave_vel; 
            end
        end
    end
    behavior_struct(sIdx).vel_trialed = vel_trialed;  
    behavior_struct(sIdx).pos_trialed = pos_trialed; 
    behavior_struct(sIdx).licks_trialed = licks_trialed; 
    behavior_struct(sIdx).vel_selectivity_metric_sum = vel_selectivity_metric_sum;
    behavior_struct(sIdx).vel_selectivity_metric_mean = vel_selectivity_metric_mean; 
    behavior_struct(sIdx).vel_selectivity_metric_sum_eqWindow = vel_selectivity_metric_sum_eqWindow;
end

%% Visualize what velocity selectivity metric looks like acr days for mice
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]}; 
mouse_names = ["m75","m76","m78","m79","m80"]; 
% mouse_grps = {1:3,4:9}; 
% mouse_names = ["mc2","mc4"]; 

close all 
figcounter = 1; 
for mIdx = 2:4
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        visualization_metric = behavior_struct(sIdx).vel_selectivity_metric_sum_eqWindow;
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.data,session));   
        prts = data.patchCSL(:,3) - data.patchCSL(:,2);
        session_title = ['m' session(1:2) ' ' session(end-2) '/' session(end-1:end)]; 
%         session_title = [session(1:3) ' ' session(end-3:end-2) '/' session(end-1:end)]; % for mgc
        
        figure(figcounter)  
        subplot(2,numel(mouse_grps{mIdx}),i);hold on
        nTrials = length(behavior_struct(sIdx).vel_trialed);  
        p2 = scatter(1:nTrials,visualization_metric(:,1),'.');
        p3 = scatter(1:nTrials,visualization_metric(:,2),'.');
        xlabel("Trials") 
        ylabel("Velocity Selectivity Metric (Sum Equal Window)")
        ylim([-3,2])
        if i == 1 
            legend([p2 p3],["500 msec","1000 msec"])
        end
        yline(1,'-','linewidth',1,'HandleVisibility','off') 
        yline(0,'--','linewidth',1,'HandleVisibility','off') 
        title(session_title)
        
        subplot(2,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}))   
        % remove outliers to better visualize distribution  
        no_outliers_vel_selectivity = visualization_metric;
        for k_period = 1:numel(pre_leave_periods) 
            k_mean = mean(visualization_metric(:,k_period)); 
            k_std = std(visualization_metric(:,k_period));   
            no_outliers_vel_selectivity(abs(no_outliers_vel_selectivity(:,k_period) - k_mean)/k_std > 3,k_period) = nan;
        end
        violinplot(no_outliers_vel_selectivity);hold on; 
        yline(1,'-','linewidth',2) 
        yline(0,'--','linewidth',2)  
        ylim([-2.5,2])
        xticklabels(["500 msec","1000 msec","1500 msec"]) 
        ylabel("Velocity Selectivity Metric (Sum Equal Window)") 
        
%         figure(figcounter + 1) 
%         subplot(2,numel(mouse_grps{mIdx}),i); 
%         scatter(prts,visualization_metric(:,1),'.');   
%         not_nan_ix = ~isnan(visualization_metric(:,1));
%         [r,p] = corrcoef(prts(not_nan_ix),visualization_metric(not_nan_ix,1));
%         title(sprintf("PRT vs 500 msec metric \n (r = %.2f,p = %.2f)",r(2),p(2)))
%         subplot(2,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}))
%         scatter(prts,visualization_metric(:,2),'.');  
%         not_nan_ix = ~isnan(visualization_metric(:,2));
%         [r,p] = corrcoef(prts(not_nan_ix),visualization_metric(not_nan_ix,2));
%         title(sprintf("PRT vs 1000 msec metric \n (r = %.2f,p = %.2f)",r(2),p(2)))
    end
    suptitle(mouse_names(mIdx));   
    figcounter = figcounter + 2; 
end 

%% Visualize what single trial velocity traces look like across selectivity metric quartiles 
vel_selectivity_struct = struct; 
figcounter = 1; 
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.data,session));
        
        rew_barcode = rew_barcodes{sIdx};
        session = sessions{sIdx}(1:end-4);
        session_title = ['m' session(1:2) ' ' session(end-2) '/' session([end-1:end])];
        
        % reinitialize ms vectors
        patchstop_ms = data.patchCSL(:,2) * 1000;
        patchleave_ms = data.patchCSL(:,3) * 1000;
        rew_size = mod(data.patches(:,2),10);
        
        % alignment values:
        % stop
        t_align{1} = patchstop_ms;
        t_start{1} = patchstop_ms;
        
        % leave
        t_align{2} = patchleave_ms;
        t_end{2} = patchleave_ms+1000;
        
        % for plotting up to X # of seconds max, w attrition for trials w lower PRTs
        for j = 1:20
            t_endmax = patchleave_ms - 500;
            t_endmax(patchleave_ms > j*1000 + patchstop_ms) = patchstop_ms(patchleave_ms > j*1000 + patchstop_ms) + j*1000;
            t_end{1}{j} = t_endmax;
            
            t_startmax = patchstop_ms;
            t_startmax(patchstop_ms < patchleave_ms - j*1000) = patchleave_ms(patchstop_ms < patchleave_ms - j*1000) - j*1000;
            t_start{2}{j} = t_startmax;
        end
        
        k_period = 2;
        % grouping variables
        gr.uL = rew_size;
        metric = behavior_struct(sIdx).vel_selectivity_metric_sum_eqWindow; 
        gr.vel_selectivity = metric(:,k_period); 
        creeping_quantiles = quantile(metric(:,k_period),3);
        creeping_quantiles = [min(metric(:,k_period)) ...
            creeping_quantiles ...
            max(metric(:,k_period))];
        [~,~,creeping_bin] = histcounts(metric(:,k_period),creeping_quantiles);
        creeping_bin(creeping_bin == 0) = nan;
        gr.creeping_binned = creeping_bin; 
        
        % save to struct 
        vel_selectivity_struct(sIdx).vel_sel_eqWindow1000ms = metric(:,k_period); 
        vel_selectivity_struct(sIdx).vel_sel_eqWindow1000ms_bin = creeping_bin;
        
        % global variable for use w plot_timecourse
        global gP
        gP.cmap{4} = cool(4);
        
        maxTime = 3;
        
        beh_vars = {data.vel data.patch_pos};
        beh_varnames = ["Velocity","Patch Position"];
        for vIdx = 1
            figure(figcounter + vIdx - 1); 
            subplot(2,numel(mouse_grps{mIdx}),i)
            % Stop-aligned
            aIdx = 1;
            vel_stop = plot_timecourse('stream',beh_vars{vIdx},t_align{aIdx}/tbin_ms,t_start{aIdx}/tbin_ms,t_end{aIdx}{maxTime}/tbin_ms,gr.creeping_binned,'resample_bin',1);
            vel_stop(2).XTick = [0 .05 .1];
            vel_stop(2).XTickLabel = {[0 1 2]};
            vel_stop(2).XLabel.String = "Time Since Patch Stop (s)";
            %         vel_stop(2).Legend.String = {('1uL') ('2uL') ('4uL')};
            title(sprintf("%s \n Stop-Aligned %s",session_title,beh_varnames(vIdx)))
            
            % Leave-aligned
            aIdx = 2; maxTime = 3;
            subplot(2,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}))
            vel_leave = plot_timecourse('stream',beh_vars{vIdx},t_align{aIdx}/tbin_ms,t_start{aIdx}{maxTime}/tbin_ms,t_end{aIdx}/tbin_ms,gr.creeping_binned,'resample_bin',1);
            vel_leave(2).XTick = [-.1 -.05 0];
            vel_leave(2).XTickLabel = {[-2 -1 0]};
            vel_leave(2).XLabel.String = 'Time Before Leave (s)';
            %         vel_leave(2).Legend.String = {('1uL') ('2uL') ('4uL')};
            title(sprintf("%s \n Leave-Aligned %s",session_title,beh_varnames(vIdx)))
        end 
    end   
    figcounter = figcounter + 2;
end 

%% Now investigate differences in neural dynamics 



