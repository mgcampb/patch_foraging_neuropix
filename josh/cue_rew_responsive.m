%% Identify the number of cue and reward responsive neurons! 
%  compare mPFC and sub cortical number of responsive neurons?  
%   - can take p-value w/ bootstrap

% take msec window to calculate baseline spike-rate pre reward / cue
%   - significance based on assumption of Poisson noise w/ mean of baseline
%     spike rate (Inagaki and Chen et al.)
% alternatively, could use driscoll transient selection? 
%   - this would be nice in creating some continuity! and allowing for a variety of
%     latencies
%   - just need to repeat for cue!
% - shuffle stimulus rather than neurons 
%   - fewer operations, same effect
% Data structures: 
% 1. fr_mat_trials: cell by mice, sessions... cue:trial stop 
% 2. timesinceCue: 1:time2stop 
%    timesinceRew0: 1:nextRew for first rew (also has second cue presentation)
%    timesinceRew1plus: 1:nextRew for later rews 
% End result: table of cells w/ cellIDs, significant peak ix 

% to get persistent higher activity: 
% 1. t-test for incr FR post prox cue: patch off 
% 2. t-test for incr FR on patch 
% 3. t-test for incr FR just during prox cue 

%% set path
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 
sig_cell_sessions = sig_cells.Session;  

% analysis options
calcFR_opt = struct;
calcFR_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calcFR_opt.smoothSigma_time = .05; % 50 msec smoothing
calcFR_opt.patch_leave_buffer = 0.50; % in seconds; only takes within patch times up to this amount before patch leave
calcFR_opt.min_fr = 0; % minimum firing rate (on patch, excluding buffer) to keep neurons 
calcFR_opt.cortex_only = true; 
calcFR_opt.tstart = 0;
tbin_ms = calcFR_opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 15:18 23 25];   
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]};  
mouse_names = ["m75","m76","m78","m79","m80"]; 
session_titles = cell(numel(mPFC_sessions),1); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end 

%% Load neural data from all good cells into cell array
fr_mat_trials = cell(numel(mouse_grps),1); % cell array of nMice   
spikecounts_full = cell(numel(mouse_grps),1); % for discovery of higher persistent activites  .. no smoothing
vel_full = cell(numel(mouse_grps),1);  
accel_full = cell(numel(mouse_grps),1); 
cue_label = cell(numel(mouse_grps),1); 
cellIDs = cell(numel(mouse_grps),1);  
sigCellCluster = cell(numel(mouse_grps),1); 
brain_region = cell(numel(mouse_grps),1); 

for mIdx = 1:numel(mouse_grps)
    fr_mat_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);     
    spikecounts_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    vel_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    accel_full{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    cue_label{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    cellIDs{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    sigCellCluster{mIdx} = cell(numel(mouse_grps{mIdx}),1); % cluster or NaN
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));    
        good_cells = data.sp.cids(data.sp.cgs==2);  
        nNeurons = length(good_cells);  
        
        % get GLM cluster vector per session
        sig_cellIDs_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).CellID;   
        sig_clusters_session = sig_cells(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4)),:).KMeansCluster;
        
        calcFR_opt.tend = max(data.sp.st);
        fr_mat = calcFRVsTime(good_cells,data,calcFR_opt);  
         
        t = data.velt;
        spikecounts_full{mIdx}{i} = nan(numel(nNeurons),numel(t)-1);
        for cIdx = 1:nNeurons
            spike_t = data.sp.st(data.sp.clu==good_cells(cIdx));
            spikecounts_full{mIdx}{i}(cIdx,:) = histcounts(spike_t,t);
        end
        
        % Get trial event timings
        patchcue_ms = data.patchCSL(:,1)*1000;
        patchstop_ms = data.patchCSL(:,2)*1000;
        patchleave_ms = data.patchCSL(:,3)*1000; 
        patchcue_ix = round(patchcue_ms / tbin_ms) + 1; 
        patchstop_ix = round(patchstop_ms / tbin_ms) + 1;
        patchleave_ix = round((patchleave_ms - 1000 * calcFR_opt.patch_leave_buffer) / tbin_ms) + 1;
        
        % Gather firing rate matrices in trial form
        fr_mat_trials{mIdx}{i} = cell(length(data.patchCSL),1);  
        cue_label{mIdx}{i} = zeros(size(fr_mat,2),1); 
        for iTrial = 1:length(data.patchCSL)
            fr_mat_trials{mIdx}{i}{iTrial} = fr_mat(:,patchcue_ix(iTrial):patchleave_ix(iTrial));   
            cue_label{mIdx}{i}(patchcue_ix(iTrial):patchstop_ix(iTrial)) = 1; % prox cue on
            cue_label{mIdx}{i}(patchstop_ix(iTrial):patchleave_ix(iTrial)) = 2; % patch cue on
        end

        % log information about cells 
        cellIDs{mIdx}{i} = good_cells;    
        brain_region{mIdx}{i} = data.brain_region_rough;    
        sigCellCluster{mIdx}{i} = nan(nNeurons,1); 
        sigCellCluster{mIdx}{i}(ismember(good_cells,sig_cellIDs_session)) = sig_clusters_session; 
        
        % Add speed information for motor confounds check later  
        vel_full{mIdx}{i} = data.vel;
        accel_full{mIdx}{i} = gradient(data.vel);
    end 
    fprintf("%s loading complete \n",mouse_names(mIdx))
end 

%% Add task events: trialed 3 x trial length cell array 

task_vars_trialed = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    task_vars_trialed{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        
        % Load session information
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
        patchcue_sec = data.patchCSL(:,1);
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);  
        rew_sec = data.rew_ts;   
        cue_lens_ix = round((patchstop_sec - patchcue_sec) * 1000 / tbin_ms); % important for labeling task events  
        trial_lens_ix = cellfun(@(x) size(x,2),fr_mat_trials{mIdx}{i}); % note that this includes cue and preleave cutoff

        % Collect trial reward timings
        rew_ix_cell = cell(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < (patchleave_sec(iTrial) - calcFR_opt.patch_leave_buffer)) - patchstop_sec(iTrial));
            rew_ix_cell{iTrial} = rew_indices * 1000 / tbin_ms;  
            rew_ix_cell{iTrial}(rew_ix_cell{iTrial} == 0) = 1; 
            rew_ix_cell{iTrial} = rew_ix_cell{iTrial} + cue_lens_ix(iTrial); % add cue length
        end
        
        % Collect task events in nTrials x 1 cell array
        task_vars_trialed{mIdx}{i} = cell(nTrials,1); 
        for iTrial = 1:nTrials 
            iTrial_trial_len = trial_lens_ix(iTrial); 
            iTrial_cue_len = cue_lens_ix(iTrial);  
            task_vars_trialed{mIdx}{i}{iTrial} = nan(3,iTrial_trial_len); 
            
            % first add time since cue 
            task_vars_trialed{mIdx}{i}{iTrial}(1,1:iTrial_cue_len) = (1:iTrial_cue_len) * tbin_ms / 1000; 
            
            % Add time since reward information
            if length(rew_ix_cell{iTrial}) > 1 % more than one reward 
                task_vars_trialed{mIdx}{i}{iTrial}(2,iTrial_cue_len:rew_ix_cell{iTrial}(2)) = (0:(rew_ix_cell{iTrial}(2)-iTrial_cue_len)) * tbin_ms / 1000; 
                for r = 2:numel(rew_ix_cell{iTrial})
                    rew_ix = rew_ix_cell{iTrial}(r); 
                    task_vars_trialed{mIdx}{i}{iTrial}(3,rew_ix:end) = (0:(iTrial_trial_len-rew_ix)) * tbin_ms / 1000; 
                end 
            else % only one reward
                % only log time since first reward
                task_vars_trialed{mIdx}{i}{iTrial}(2,iTrial_cue_len:end) = (0:(iTrial_trial_len-iTrial_cue_len)) * tbin_ms / 1000;
            end
        end
    end
end

%% Collect number of neurons per session to allow for table pre-allocation  
s_nNeurons = [];  
counter = 1; 
for mIdx = 1:5  
    for i = 1:numel(mouse_grps{mIdx})
        s_nNeurons = [s_nNeurons; length(brain_region{mIdx}{i})];
        counter = counter + 1; 
    end
end 

session_neuron_starts = [0 ; cumsum(s_nNeurons)];  
session_neuron_ranges = cell(numel(mouse_grps),1);
counter = 1; 
for mIdx = 1:5  
    session_neuron_ranges{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        session_neuron_ranges{mIdx}{i} = [session_neuron_starts(counter)+1 session_neuron_starts(counter+1)];  
        counter = counter + 1; 
    end 
end

%% First analyze for presence of persistent changes in firing rate in different cue zones 
% use fr_mat_full and cue_label 
% start table of cells! w/ cellID, persistent activity changes in diff cue zones
% - use wilcoxon ranksum to make it more distribution agnostic 
% - New plan: Poisson mean diff test w/ no smoothing

colors = lines(2);
col_names = ["Mouse","Session","CellID","Region","GLM_Cluster","pValue_cuePatch","pValue_cue","pValue_patch", ... 
             "Cue_peak_pos","Cue_peak_neg","Rew0_peak_pos","Rew0_peak_neg","Rew1plus_peak_pos","Rew1plus_peak_neg"]; 
% col_types = arrayfun(@(x) 'double',1:14,'UniformOutput',false); 
col_types = ["string","string","double","string","double","double","double","double","double","double","double","double","double","double"];
transients_table = table('Size',[sum(s_nNeurons),14],'VariableTypes',col_types,'VariableNames',col_names);

mean_fr_mat_offPatch_pooled = []; 
mean_fr_mat_cuePatch_pooled = []; 
mean_fr_mat_cue_pooled = []; 
mean_fr_mat_patch_pooled = []; 
for mIdx = 1:5
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i); 
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        % get region information
        nNeurons = length(brain_region{mIdx}{i}); 
    
        % get mean fr w.r.t. different cues for visualization
        mean_fr_mat_offPatch = mean(spikecounts_full{mIdx}{i}(:,cue_label{mIdx}{i} == 0),2) / diff(t(1:2));
        mean_fr_mat_cuePatch = mean(spikecounts_full{mIdx}{i}(:,cue_label{mIdx}{i} > 0),2) / diff(t(1:2));
        mean_fr_mat_cue = mean(spikecounts_full{mIdx}{i}(:,cue_label{mIdx}{i} == 1),2) / diff(t(1:2));
        mean_fr_mat_patch =  mean(spikecounts_full{mIdx}{i}(:,cue_label{mIdx}{i} == 2),2) / diff(t(1:2));    
        % log to pooled vectors
        mean_fr_mat_offPatch_pooled = [mean_fr_mat_offPatch_pooled ; mean_fr_mat_offPatch]; 
        mean_fr_mat_cuePatch_pooled = [mean_fr_mat_cuePatch_pooled ; mean_fr_mat_cuePatch]; 
        mean_fr_mat_cue_pooled = [mean_fr_mat_cue_pooled ; mean_fr_mat_cue]; 
        mean_fr_mat_patch_pooled = [mean_fr_mat_patch_pooled ; mean_fr_mat_patch]; 

        % Add to table  
        i_start = session_neuron_ranges{mIdx}{i}(1); 
        i_end = session_neuron_ranges{mIdx}{i}(2);
        transients_table(i_start:i_end,:).Mouse = repmat(mouse_names(mIdx),[1 + i_end - i_start 1]); 
        transients_table(i_start:i_end,:).Session = repmat(string(session_title),[1 + i_end - i_start 1]); 
        transients_table(i_start:i_end,:).CellID = cellIDs{mIdx}{i}'; 
        transients_table(i_start:i_end,:).Region = string(cellfun(@(x) string(x),brain_region{mIdx}{i},'un',0));  
        transients_table(i_start:i_end,:).GLM_Cluster = sigCellCluster{mIdx}{i};
    end
end

%% Visualize persistent activity results

colors = lines(2);
colors = [colors(1,:) ; max(0,colors(1,:) - .25) ; min(1,colors(1,:) + .25) ; colors(2,:); max(0,colors(2,:) - .25) ; min(1,colors(2,:) + .25)];

cortex_binary = strcmp(transients_table.Region,'PFC'); 

pValue_cuePatch = poisscdf(mean_fr_mat_cuePatch_pooled,mean_fr_mat_offPatch_pooled);
pValue_cue = poisscdf(mean_fr_mat_cue_pooled,mean_fr_mat_offPatch_pooled);
pValue_patch = poisscdf(mean_fr_mat_patch_pooled,mean_fr_mat_offPatch_pooled); 
transients_table.pValue_cuePatch = pValue_cuePatch;
transients_table.pValue_cue = pValue_cue;
transients_table.pValue_patch = pValue_patch;

alpha = .01; 
sig_cuePatch = nan(sum(s_nNeurons),1);  
sig_cuePatch(pValue_cuePatch < alpha) = 1; 
sig_cuePatch(pValue_cuePatch > 1-alpha) = 2;  
sig_cuePatch(isnan(sig_cuePatch)) = 0; 
sig_cue = nan(sum(s_nNeurons),1);   
sig_cue(pValue_cue < alpha) = 1; 
sig_cue(pValue_cue > 1-alpha) = 2;  
sig_cue(isnan(sig_cue)) = 0; 
sig_patch = nan(sum(s_nNeurons),1);  
sig_patch(pValue_patch < alpha) = 1; 
sig_patch(pValue_patch > 1-alpha) = 2;  
sig_patch(isnan(sig_patch)) = 0; 

figure()
subplot(2,3,1); hold on
gscatter(mean_fr_mat_offPatch_pooled(cortex_binary),mean_fr_mat_cuePatch_pooled(cortex_binary),sig_cuePatch(cortex_binary),colors(1:3,:),'.oo',[2 2 2]);
% binscatter(mean_fr_mat_offPatch(cortex_binary),mean_fr_mat_cuePatch(cortex_binary),nBins);
% gscatter(mean_fr_mat_offPatch(cortex_binary),mean_fr_mat_cuePatch(cortex_binary),p_cuePatch(cortex_binary) < .01/nNeurons);
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;h.HandleVisibility = 'off'; 
legend("Non-Significant","p < 0.1 Lower FR","p < 0.1 Higher FR")
title("Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Prox Cue + Patch");
subplot(2,3,2) ; hold on
gscatter(mean_fr_mat_offPatch_pooled(cortex_binary),mean_fr_mat_cue_pooled(cortex_binary),sig_cue(cortex_binary),colors(1:3,:),'.oo',[2 2 2],'HandleVisibility','off');
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;
title("Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Prox Cue");
subplot(2,3,3) ; hold on
gscatter(mean_fr_mat_offPatch_pooled(cortex_binary),mean_fr_mat_patch_pooled(cortex_binary),sig_patch(cortex_binary),colors(1:3,:),'.oo',[2 2 2],'HandleVisibility','off');
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;
title("Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Patch");
subplot(2,3,4) ; hold on
gscatter(mean_fr_mat_offPatch_pooled(~cortex_binary),mean_fr_mat_cuePatch_pooled(~cortex_binary),sig_cuePatch(~cortex_binary),colors(4:6,:),'.oo',[2 2 2]);
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;h.HandleVisibility = 'off'; 
legend("Non-Significant","p < 0.1 Lower FR","p < 0.1 Higher FR")
title("Sub-Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Prox Cue + Patch");
subplot(2,3,5) ; hold on
gscatter(mean_fr_mat_offPatch_pooled(~cortex_binary),mean_fr_mat_cue_pooled(~cortex_binary),sig_cue(~cortex_binary),colors(4:6,:),'.oo',[2 2 2],'HandleVisibility','off');
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;
title("Sub-Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Prox Cue");
subplot(2,3,6) ; hold on
gscatter(mean_fr_mat_offPatch_pooled(~cortex_binary),mean_fr_mat_patch_pooled(~cortex_binary),sig_patch(~cortex_binary),colors(4:6,:),'.oo',[2 2 2],'HandleVisibility','off');
h = refline(1,0); xlim([0 60]);ylim([0 60]);
h.LineStyle = '--';h.Color = 'k'; h.LineWidth = 1.5;
title("Sub-Cortex")
xlabel("Mean FR Off Patch");ylabel("Mean FR Patch");

%% Analyze significance of proportions w/ permutation test 

region = transients_table.Region; 
alpha = .01;  
pValues_full = [pValue_cuePatch pValue_cue pValue_patch]; 
prop_sig = nan(6,2); 
for iVar = 1:3
    prop_sig(iVar,1) = mean(pValues_full(region == "PFC",iVar) < alpha); 
    prop_sig(iVar,2) = mean(pValues_full(region == "Sub-PFC",iVar) < alpha); 
    prop_sig(iVar+3,1) = mean(pValues_full(region == "PFC",iVar) > 1 - alpha);  
    prop_sig(iVar+3,2) = mean(pValues_full(region == "Sub-PFC",iVar) > 1- alpha); 
end

new_testing = false; 
if new_testing == true
    true_diff = prop_sig(:,1) - prop_sig(:,2);
    pvalue_perm = nan(length(true_diff),1);
    nShuffles = 10000;
    iVars = [1:3 1:3];
    for iRespType = 1:length(true_diff)
        iVar = iVars(iRespType);
        
        sig_count = 0;
        % Now perform permutation test to determins significance
        if iRespType <= 3 % test sig lower
            for iShuffle = 1:nShuffles
                shuffle_region = region(randperm(length(region)));
                iShuffle_pfc = mean(pValues_full(shuffle_region == "PFC",iVar) < alpha);
                iShuffle_subPfc = mean(pValues_full(shuffle_region == "Sub-PFC",iVar) < alpha);
                if iShuffle_pfc - iShuffle_subPfc > true_diff(iRespType)
                    sig_count = sig_count + 1;
                end
            end
        else % test sig higher
            for iShuffle = 1:nShuffles
                shuffle_region = region(randperm(length(region)));
                iShuffle_pfc = mean(pValues_full(shuffle_region == "PFC",iVar) > 1 - alpha);
                iShuffle_subPfc = mean(pValues_full(shuffle_region == "Sub-PFC",iVar) < 1 - alpha);
                if iShuffle_pfc - iShuffle_subPfc > true_diff(iRespType)
                    sig_count = sig_count + 1;
                end
            end
        end
        pvalue_perm(iRespType) = sig_count / nShuffles;
    end
end

labels = ["Sig Lower FR Patch+Cue","Sig Lower FR Cue","Sig Lower FR Patch" ...
          "Sig Higher FR Patch+Cue","Sig Higher FR Cue","Sig Higher FR Patch"];
        
figure();hold on
b = bar(prop_sig,'FaceColor','flat'); 
b(1).CData = colors(1,:); 
b(2).CData = [0 0 0];
xticks(1:6);xticklabels(labels) 
legend("PFC","Sub-PFC") 
text(b(1).XData(pvalue_perm < .0001),1.05 * b(1).YData(pvalue_perm < .0001), "***",'FontSize',14,'HorizontalAlignment','center' );   
for i = 1:numel(pvalue_perm)  
    if pvalue_perm(i) < .0001
        plot([b(1).XEndPoints(i) b(2).XEndPoints(i)],1.025 * [b(1).YData(i) b(1).YData(i)],'k','linewidth',1.5,'HandleVisibility','off')
    end
end 
text(b(1).XData(pvalue_perm < .01 & pvalue_perm > .0001),1.05 * b(1).YData(pvalue_perm < .01 & pvalue_perm > .0001), "*",'FontSize',14,'HorizontalAlignment','center' );   
for i = 1:numel(pvalue_perm)  
    if pvalue_perm(i) < .01 && pvalue_perm(i) > .0001
        plot([b(1).XEndPoints(i) b(2).XEndPoints(i)],1.025 * [b(1).YData(i) b(1).YData(i)],'k','linewidth',1.5,'HandleVisibility','off')
    end
end
xtickangle(45);
ylabel("Proportion of Significant Transient Cells") 
title(sprintf("Proportion of Cells With Significant Task-Related Transients \n Separated by Region"))

%% Visualize transient discovery PETH
% separate training and visualization data 

var_bins{1} = 0:.025:.75; 
var_bins{2} = 0:.050:3; 
var_bins{3} = 0:.050:3;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

taskvar_peth = cell(3,1); 
pvalue_peth = cell(3,1);  
pos_peak_ix = cell(3,1); 
for iVar = 1:3 
    taskvar_peth{iVar} = nan(sum(s_nNeurons),length(var_bins{iVar})-1);  
    pvalue_peth{iVar} = nan(sum(s_nNeurons),length(var_bins{iVar})-1);   
    pos_peak_ix{iVar} = nan(sum(s_nNeurons),2); % [train test] 
end

counter = 0;  
bar = waitbar(0,"Finding Transients"); % progress tracking 
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})   
        % Load session information
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
        
        i_start = session_neuron_ranges{mIdx}{i}(1); 
        i_end = session_neuron_ranges{mIdx}{i}(2); 

        % Use all trials for cue
        transient_opt.vars = 1;  
        sort_trials = 2:2:nTrials; 
        vis_trials = 1:2:nTrials; 
        [transient_struct_tmp,taskvar_peth_cell,pvalue_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},vis_trials,tbin_ms,var_bins,transient_opt);
        pos_peak_ix{1}(i_start:i_end,1) = transient_struct_tmp.peak_ix_pos;
        taskvar_peth{1}(i_start:i_end,:) = taskvar_peth_cell{1}; 
        pvalue_peth{1}(i_start:i_end,:) = pvalue_peth_cell{1}; 
        transient_struct_tmp = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},sort_trials,tbin_ms,var_bins,transient_opt);
        pos_peak_ix{1}(i_start:i_end,2) = transient_struct_tmp.peak_ix_pos;
        % Use rewsize 2, 4 uL trials for time since reward 
        rewsize24_trials = rewsize > 1; 
        sort_trials = rewsize24_trials(2:2:length(rewsize24_trials)); 
        vis_trials = rewsize24_trials(1:2:length(rewsize24_trials)); 
        for iVar = 2:3
            transient_opt.vars = iVar; 
            [transient_struct_tmp,taskvar_peth_cell,pvalue_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},vis_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix{iVar}(i_start:i_end,1) = transient_struct_tmp.peak_ix_pos;
            taskvar_peth{iVar}(i_start:i_end,:) = taskvar_peth_cell{iVar}; 
            pvalue_peth{iVar}(i_start:i_end,:) = pvalue_peth_cell{iVar}; 
            transient_struct_tmp = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},sort_trials,tbin_ms,var_bins,transient_opt);
            pos_peak_ix{iVar}(i_start:i_end,2) = transient_struct_tmp.peak_ix_pos;
        end  
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),bar) 
    end
end 
close(bar); 

%% Visualize results 
close all
var_names = ["Cue","Rew t = 0","Rew t = 1+"]; 
ROI = "Sub-PFC"; 
region_bool = strcmp(ROI,transients_table.Region) | ~strcmp(ROI,transients_table.Region);   
pos_peak_ix_roi = cell(3,1);  
taskvar_peth_roi = cell(3,1);  
pvalue_peth_roi = cell(3,1);  
for iVar = 1:3 
    pos_peak_ix_roi{iVar} = pos_peak_ix{iVar}(region_bool,:); 
    taskvar_peth_roi{iVar} = taskvar_peth{iVar}(region_bool,:); 
    pvalue_peth_roi{iVar} = pvalue_peth{iVar}(region_bool,:); 
end

for iVar = 1:3
    [~,train_sort] = sort(pos_peak_ix_roi{iVar}(:,1));  
    train_sort = train_sort(ismember(train_sort,find(~isnan(pos_peak_ix_roi{iVar}(:,1))))); % get rid of non significant cells
    [~,test_sort] = sort(pos_peak_ix_roi{iVar}(:,2));  
    test_sort = test_sort(ismember(test_sort,find(~isnan(pos_peak_ix_roi{iVar}(:,2))))); % get rid of non significant cells
    figure() 
    ax(1) = subplot(2,2,1);
    imagesc(flipud(zscore(taskvar_peth_roi{iVar}(train_sort,:),[],2)))
    caxis([-3 3])   
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar))) 
    title(sprintf("Z-Scored %s Responsivity",var_names(iVar))) 
    ylabel("Train Sort")
    ax(2) = subplot(2,2,2);
    imagesc(flipud(log(pvalue_peth_roi{iVar}(train_sort,:)))) 
    colorbar()  
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar))) 
    title(sprintf("log p-value %s Responsivity",var_names(iVar))) 
    ylabel("Train Sort")
    ax(3) = subplot(2,2,3);
    imagesc(flipud(zscore(taskvar_peth_roi{iVar}(test_sort,:),[],2)))
    caxis([-3 3])  
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar))) 
    title(sprintf("Z-Scored %s Responsivity",var_names(iVar))) 
    ylabel("Test Sort")
    ax(4) = subplot(2,2,4);
    imagesc(flipud(log(pvalue_peth_roi{iVar}(test_sort,:)))) 
    colorbar() 
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar))) 
    title(sprintf("log p-value %s Responsivity",var_names(iVar))) 
    ylabel("Test Sort")
    
    % set colormaps
    colormap(ax(1),parula)
    colormap(ax(2),bone)
    colormap(ax(3),parula)
    colormap(ax(4),bone) 
    suptitle(ROI)
end 

figure()
for iVar = 1:3
    subplot(1,3,iVar)
    scatter(.1 * rand(length(pos_peak_ix_roi{iVar}),1) + pos_peak_ix_roi{iVar}(:,1),.1 * rand(length(pos_peak_ix_roi{iVar}),1) + pos_peak_ix_roi{iVar}(:,2),'.')   
    both_nonNan = all(~isnan(pos_peak_ix_roi{iVar}),2);
    [r,p] = corrcoef(pos_peak_ix_roi{iVar}(both_nonNan,1),pos_peak_ix_roi{iVar}(both_nonNan,2));
    title(sprintf("Time Since %s Train vs Test Peaks (n = %i) \n (r = %.3f,p = %.3f)",var_names(iVar),length(find(both_nonNan)),r(2),p(2))) 
    xlabel(sprintf("Time Since %s Train Peak",var_names(iVar))) 
    ylabel(sprintf("Time Since %s Test Peak",var_names(iVar))) 
end

% cross cue sort reward responsivity?
[~,cue_sort] = sort(pos_peak_ix_roi{1}(:,2));
cue_sort = cue_sort(ismember(cue_sort,find(~isnan(pos_peak_ix_roi{1}(:,2))))); % get rid of non significant cells 
for iVar = 2:3  
    figure() 
    [~,rew_sort] = sort(pos_peak_ix_roi{iVar}(:,1));
    rew_sort = rew_sort(ismember(rew_sort,find(~isnan(pos_peak_ix_roi{iVar}(:,1))))); % get rid of non significant cells
    subplot(2,2,1);
    imagesc(flipud(zscore(taskvar_peth_roi{1}(cue_sort,:),[],2)))
    caxis([-3,3]) 
    title("Z-Scored Cue Responsivity") 
    xticks(1:10:numel(var_bins{1})) 
    xticklabels(var_bins{1}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(1)))  
    ylabel(sprintf("Time Since %s Sort",var_names(1)))
    subplot(2,2,3);
    imagesc(flipud(zscore(taskvar_peth_roi{1}(rew_sort,:),[],2)))  
    caxis([-3,3]) 
    title("Z-Scored Cue Responsivity") 
    xticks(1:10:numel(var_bins{1})) 
    xticklabels(var_bins{1}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(1)))  
    ylabel(sprintf("Time Since %s Sort",var_names(iVar)))
    subplot(2,2,2)
    imagesc(flipud(zscore(taskvar_peth_roi{iVar}(cue_sort,:),[],2)))
    caxis([-3,3])
    title(sprintf("Z-Scored Time Since %s Responsivity",var_names(iVar))) 
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar))) 
    ylabel(sprintf("Time Since %s Sort",var_names(1)))
    subplot(2,2,4) 
    imagesc(flipud(zscore(taskvar_peth_roi{iVar}(rew_sort,:),[],2))) 
    caxis([-3,3]) 
    title(sprintf("Z-Scored Time Since %s Responsivity",var_names(iVar))) 
    xticks(1:10:numel(var_bins{iVar})) 
    xticklabels(var_bins{iVar}(1:10:end)) 
    xlabel(sprintf("Time Since %s",var_names(iVar)))  
    ylabel(sprintf("Time Since %s Sort",var_names(iVar)))
end  

mouse_num = cell2mat(arrayfun(@(x) str2double(transients_table.Mouse{x}(2:3)), 1:length(transients_table.Mouse),'un',0))'; 

for glm_cluster = 1:3 
    figure() 
    glm_bool = transients_table.GLM_Cluster == glm_cluster; 
    mouse_num_clust = mouse_num(glm_bool); 
    for iVar = 1:3 
        sig_bool = all(~isnan(pos_peak_ix{iVar}),2); 
        sig_bool = sig_bool(glm_bool);
        subplot(1,3,iVar);hold on
        imagesc(log(pvalue_peth{iVar}(glm_bool,:))) 
        colormap('bone') 
        if iVar > 1
            gscatter(zeros(length(find(glm_bool)),1),1:length(find(glm_bool)),mouse_num_clust,[],[],[],'HandleVisibility','off')   
            gscatter(zeros(length(find(glm_bool)),1) + length(var_bins{iVar}),1:length(find(glm_bool)),sig_bool,[0 0 0;1 0 0],'o',10,'HandleVisibility','off')
        else  
            gscatter(zeros(length(find(glm_bool)),1),1:length(find(glm_bool)),mouse_num_clust) 
            gscatter(zeros(length(find(glm_bool)),1) + length(var_bins{iVar}),1:length(find(glm_bool)),sig_bool,[0 0 0;1 0 0],'o',10,'HandleVisibility','off')
            legend([mouse_names "No Sig Transient" "Sig Transient"])
        end
        ylim([0 length(find(glm_bool))])
        xticks(1:10:numel(var_bins{iVar}))
        xticklabels(var_bins{iVar}(1:10:end))
        xlabel(sprintf("Time Since %s",var_names(iVar)))
    end
    suptitle(sprintf("GLM KMeans Cluster %i",glm_cluster))
end

%% Now look for transient responsivity w/ shuffle testing

var_bins{1} = 0:.025:.75; 
var_bins{2} = 0:.050:3; 
var_bins{3} = 0:.050:3;  
transient_opt = struct; 
transient_opt.visualization = false;  
transient_opt.preRew_buffer = round(3 * calcFR_opt.smoothSigma_time * 1000 / tbin_ms);

taskvar_peth = cell(3,1); 
pvalue_peth = cell(3,1);  
for iVar = 1:3 
    taskvar_peth{iVar} = nan(sum(s_nNeurons),length(var_bins{iVar})-1);  
    pvalue_peth{iVar} = nan(sum(s_nNeurons),length(var_bins{iVar})-1);   
end

counter = 0;  
b = waitbar(0,"Finding Transients"); % progress tracking 
transient_structs = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps) 
    transient_structs{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        transient_structs{mIdx}{i} = cell(3,1); 
        % Load session information
        sIdx = mouse_grps{mIdx}(i);   
        data = load(fullfile(paths.data,sessions{sIdx}));
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
    
        i_start = session_neuron_ranges{mIdx}{i}(1); 
        i_end = session_neuron_ranges{mIdx}{i}(2); 
        
        % Use all trials for cue
        transient_opt.vars = 1; 
        [transient_structs{mIdx}{i}{1},taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},1:nTrials,tbin_ms,var_bins,transient_opt);
        taskvar_peth{1}(i_start:i_end,:) = taskvar_peth_cell{1}; 
        % Use rewsize 2, 4 uL trials for time since reward 
        for iVar = 2:3
            transient_opt.vars = iVar;
            [transient_structs{mIdx}{i}{iVar},taskvar_peth_cell] = driscoll_transient_discovery2(fr_mat_trials{mIdx}{i},task_vars_trialed{mIdx}{i},rewsize > 1,tbin_ms,var_bins,transient_opt);
            taskvar_peth{iVar}(i_start:i_end,:) = taskvar_peth_cell{iVar}; 
        end
        
        % Add to table  
        transients_table(i_start:i_end,:).Cue_peak_pos = transient_structs{mIdx}{i}{1}.peak_ix_pos;
        transients_table(i_start:i_end,:).Cue_peak_neg = transient_structs{mIdx}{i}{1}.peak_ix_neg;
        transients_table(i_start:i_end,:).Rew0_peak_pos = transient_structs{mIdx}{i}{2}.peak_ix_pos;
        transients_table(i_start:i_end,:).Rew0_peak_neg = transient_structs{mIdx}{i}{2}.peak_ix_neg;
        transients_table(i_start:i_end,:).Rew1plus_peak_pos = transient_structs{mIdx}{i}{3}.peak_ix_pos;
        transients_table(i_start:i_end,:).Rew1plus_peak_neg = transient_structs{mIdx}{i}{3}.peak_ix_neg;
        
        counter = counter + 1; 
        waitbar(counter/numel(mPFC_sessions),b) 
    end
end 
close(b);

%% Now analyze transients_table 

%% 1) Permutation testing to assess difference in significant task-responsive neurons 

region = transients_table.Region;
colors = lines(3); 
PFC_nNeurons = length(find(region == "PFC"));
subPFC_nNeurons = length(find(region == "Sub-PFC")); 

cue_peak_pos = transients_table.Cue_peak_pos; 
rew0_peak_pos = transients_table.Rew0_peak_pos; 
rew1plus_peak_pos = transients_table.Rew1plus_peak_pos;  
cue_peak_neg = transients_table.Cue_peak_neg; 
rew0_peak_neg = transients_table.Rew0_peak_neg; 
rew1plus_peak_neg = transients_table.Rew1plus_peak_neg;   
peak_ix = [cue_peak_pos rew0_peak_pos rew1plus_peak_pos cue_peak_neg rew0_peak_neg rew1plus_peak_neg];
 
nShuffles = 1000; 
new_testing = true; 
if new_testing == true
    frac_sig = nan(size(peak_ix,2),2);
    pvalue_perm = nan(size(peak_ix,2),1);
    
    for iPeakType = 1:size(peak_ix,2)
        this_peak_ix = peak_ix(:,iPeakType);
        
        true_pfc = mean(~isnan(this_peak_ix(region == "PFC")));
        true_subPfc = mean(~isnan(this_peak_ix(region == "Sub-PFC")));
        frac_sig(iPeakType,:) = [true_pfc true_subPfc];
        true_diff = true_pfc - true_subPfc;
        
        sig_count = 0;
        % Now perform permutation test to determins significance
        for iShuffle = 1:nShuffles
            shuffle_region = region(randperm(length(region)));
            iShuffle_pfc = mean(~isnan(this_peak_ix(shuffle_region == "PFC")));
            iShuffle_subPfc = mean(~isnan(this_peak_ix(shuffle_region == "Sub-PFC")));
            if iShuffle_pfc - iShuffle_subPfc > true_diff
                sig_count = sig_count + 1;
            end
        end
        pvalue_perm(iPeakType) = sig_count / nShuffles;
    end
end

labels = ["Positive Cue Transient","Positive Rew0 Transient","Positive Rew1Plus Transient",...
          "Negative Cue Transient","Negative Rew0 Transient","Negative Rew1Plus Transient",];
figure();hold on
b = bar(frac_sig,'FaceColor','flat'); 
b(1).CData = colors(1,:); 
b(2).CData = [0 0 0];
xticks(1:6);xticklabels(labels) 
legend("PFC","Sub-PFC") 
text(b(1).XData(pvalue_perm < .0001),1.05 * b(1).YData(pvalue_perm < .0001), "***",'FontSize',14,'HorizontalAlignment','center' );   
for i = 1:numel(pvalue_perm)  
    if pvalue_perm(i) < .0001
        plot([b(1).XEndPoints(i) b(2).XEndPoints(i)],1.025 * [b(1).YData(i) b(1).YData(i)],'k','linewidth',1.5,'HandleVisibility','off')
    end
end
xtickangle(45);
ylabel("Proportion of Significant Transient Cells") 
title(sprintf("Proportion of Cells With Significant Task-Related Transients \n Separated by Region") )

%% 2) check out correlation between cue and reward responsivity 
% Now select for brain region of interest
close all
ROI = "PFC";
cue_peak_pos = transients_table(region == ROI,:).Cue_peak_pos;
rew0_peak_pos = transients_table(region == ROI,:).Rew0_peak_pos;
rew1plus_peak_pos = transients_table(region == ROI,:).Rew1plus_peak_pos;
cue_peak_neg = transients_table(region == ROI,:).Cue_peak_neg;
rew0_peak_neg = transients_table(region == ROI,:).Rew0_peak_neg;
rew1plus_peak_neg = transients_table(region == ROI,:).Rew1plus_peak_neg;

PFC_nNeurons = length(find(region == ROI));

figure() 
subplot(1,3,1)
scatter(.1* rand(PFC_nNeurons,1) + cue_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew0_peak_pos,'.');hold on  
xlim([0 max(var_bins{1})])
ylim([0 max(var_bins{2})]) 
yticks([0:.2:1 1.5 2 2.5 3])
xlabel("Positive Proximity Cue Peak Location")
ylabel("Patch Onset Peak Location") 
subplot(1,3,2)
scatter(.1* rand(PFC_nNeurons,1) + cue_peak_neg, ... 
        .1* rand(PFC_nNeurons,1) + rew0_peak_pos,'.');hold on  
xlim([0 max(var_bins{1})])
ylim([0 max(var_bins{2})]) 
yticks([0:.2:1 1.5 2 2.5 3])
xlabel("Negative Proximity Cue Peak Location")
ylabel("Patch Onset Peak Location") 
subplot(1,3,3)
scatter(.1* rand(PFC_nNeurons,1) + cue_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew1plus_peak_pos,'.')
% binscatter(.1* rand(PFC_nNeurons,1) + cue_peak_pos, ... 
%         .1* rand(PFC_nNeurons,1) + rew1plus_peak_pos,10)
xlim([0 max(var_bins{1})])
ylim([0 max(var_bins{2})]) 
yticks([0:.2:1 1.5 2 2.5 3])  
xlabel("Positive Proximity Cue Peak Location")
ylabel("t = 1+ Reward Peak Location")
    
figure()
scatter(.1* rand(PFC_nNeurons,1) + rew0_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew1plus_peak_pos,'.')  
unity = refline(1,0); 
unity.LineWidth = 1.5;
unity.LineStyle = '--';
unity.Color = [0 0 0]; 
xlim([0 4])
ylim([0 4]) 

%% 3) Relationship between GLM cluster and transient peak 
close all
ROI = "PFC";
glm_cluster = transients_table(region == ROI,:).GLM_Cluster;

figure() 
subplot(1,2,1)
gscatter(.1* rand(PFC_nNeurons,1) + cue_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew0_peak_pos,glm_cluster);hold on  
xlim([0 max(var_bins{1})])
ylim([0 max(var_bins{2})]) 
yticks([0:.2:1 1.5 2 2.5 3])
xlabel("t = 0 Reward Peak Location")
ylabel("Patch Onset Peak Location") 
legend(["Cluster 1","Cluster 2","Cluster 3"])
subplot(1,2,2)
gscatter(.1* rand(PFC_nNeurons,1) + cue_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew1plus_peak_pos,glm_cluster,[],[],[],'HandleVisibility','off')
xlim([0 max(var_bins{1})])
ylim([0 max(var_bins{2})]) 
yticks([0:.2:1 1.5 2 2.5 3])  
xlabel("Proximity Cue Peak Location")
ylabel("t = 1+ Reward Peak Location") 

figure()
gscatter(.1* rand(PFC_nNeurons,1) + rew0_peak_pos, ... 
        .1* rand(PFC_nNeurons,1) + rew1plus_peak_pos,glm_cluster)  
unity = refline(1,0); 
unity.LineWidth = 1.5;
unity.LineStyle = '--';
unity.Color = [0 0 0];  
unity.HandleVisibility = 'off';
xlim([0 max(var_bins{2})])
ylim([0 max(var_bins{2})])
xlabel("t = 0 Reward Peak Location")
ylabel("t = 1+ Reward Peak Location") 
legend(["Cluster 1","Cluster 2","Cluster 3"])

figure() 
subplot(1,2,1)
percentClu1 = mean(~isnan(rew0_peak_pos(glm_cluster == 1)));
percentClu2 = mean(~isnan(rew0_peak_pos(glm_cluster == 2)));
percentClu3 = mean(~isnan(rew0_peak_pos(glm_cluster == 3)));    
bar([percentClu1 percentClu2 percentClu3]) 
title(sprintf('Proportion of Significant Post-Rew0 Peaks Discovered \n Divided By GLM Cluster')) 
xticklabels(["Cluster 1","Cluster 2","Cluster 3"])
ylabel("Proportion of Cells With Significant Post-Rew0 Peak")
subplot(1,2,2) 
histogram(glm_cluster(~isnan(glm_cluster) & isnan(rew0_peak_pos)),'Normalization','Probability') 
title(sprintf("Distribution of Cluster \n Among Undiscovered GLM SigCells"))
xticklabels(["Cluster 1","Cluster 2","Cluster 3"]) 
ylabel("Density")

%% 4) Does significant encoding of one variable (ie cue) affect likelihood to encode others? 
% mainly interesting for cue-reward 
% test if this is true for non transient-reward responsive cells 
close all

lines3 = lines(3); 
colors = [lines3(3,:); lines3(3,:)+.25 ; lines3(1,:) ; lines3(1,:)+.25 ; lines3(2,:) ; lines3(2,:)+.25];
colors(colors > 1) = 1;

% close all
ROI = "PFC";
cue_peak_pos = transients_table(region == ROI,:).Cue_peak_pos;
rew0_peak_pos = transients_table(region == ROI,:).Rew0_peak_pos;
rew1plus_peak_pos = transients_table(region == ROI,:).Rew1plus_peak_pos; 
peak_ix = [cue_peak_pos rew0_peak_pos rew1plus_peak_pos];

var_names = ["Cue","Rew t = 0","Rew t = 1+"];

nShuffles = 1000; 
frac_sig = nan(size(peak_ix,2),6); 
pvalue_perm = nan(size(peak_ix,2),2); 
for iPeakType = 1:size(peak_ix,2) 
    this_peak_ix = peak_ix(:,iPeakType);   
%     frac_sig(iPeakType,iPeakType) = mean(~isnan(this_peak_ix)); 
    other_peakTypes = setdiff(1:3,iPeakType);  
    counter = 0; 
    for j_PeakType = 1:numel(other_peakTypes)
        jPeakType = other_peakTypes(j_PeakType);  
        this_peak_ix2 = peak_ix(:,jPeakType); 
        frac_sig(iPeakType,1+2*(jPeakType-1)) = mean(~isnan(this_peak_ix2(~isnan(this_peak_ix))));  
        frac_sig(iPeakType,1+2*(jPeakType-1)+1) = mean(~isnan(this_peak_ix2(isnan(this_peak_ix))));   
        
        true_diff = frac_sig(iPeakType,1+2*(jPeakType-1)) - frac_sig(iPeakType,1+2*(jPeakType-1)+1);
        sig_count = 0;
        % Now perform permutation test to determines significance
        for iShuffle = 1:nShuffles
            shuffle_this_peak_ix = this_peak_ix(randperm(length(this_peak_ix)));
            iShuffle_sig = mean(~isnan(this_peak_ix(~isnan(shuffle_this_peak_ix))));
            iShuffle_nan = mean(~isnan(this_peak_ix(isnan(shuffle_this_peak_ix))));
            if iShuffle_sig - iShuffle_nan > true_diff
                sig_count = sig_count + 1;
            end
        end
        pvalue_perm(iPeakType,j_PeakType) = sig_count / nShuffles;
        counter = counter + 1; 
    end
end

figure()
b = bar(frac_sig,'FaceColor','flat'); 
for i = 1:numel(b) 
    b(i).CData = colors(i,:);   
    if mod(i,2) == 0
        b(i).Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end 
legend(var_names)
title("Relationships between Proportions of Significant Encoding")

xticklabels(arrayfun(@(x) sprintf("Divided by %s Significant",var_names(x)),1:length(var_names)))
ylabel("Proportion Significant Cells")
