%% RX_stretch2: A late stage visualization function 
%  Visualize RXNil dynamics within selected neurons across sessions and mice 
%  - Include x sec pre cue onset and y sec post patch offset 
%       - Relationship between evidence accumulation and ITI dynamics
%  - Use jPCA style of visualization 
%  - Stretch to median QRT and PRT values across sessions 
%  - Make it easy to choose different neural populations
%       - (PFC,Sub-PFC,GLM significant,Transient Selection)

%% Set paths
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat';
load(paths.transients_table);  
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData')); 

% analysis options
calcFR_opt = struct;
calcFR_opt.tstart = 0;
calcFR_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calcFR_opt.smoothSigma_time = .100; % 100 msec smoothing
tbin_sec = calcFR_opt.tbin;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 14:18 23 25];   
mouse_grps = {1:2,3:8,10:13,14:18,[23 25]};  % note this should be 14:18
mouse_names = ["m75","m76","m78","m79","m80"];
session_titles = cell(numel(mouse_grps),1);
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];
        session_titles{mIdx}{i} = session_title;
    end
end   

%% Load data into {mouse}{session} cell arrays

% How much time to keep before cue and after patch leave
pre_cue_sec = 0.5; 
post_leave_sec = 0.5; 
pre_cue_ix = round(pre_cue_sec / tbin_sec); 
post_leave_ix = round(post_leave_sec / tbin_sec); 
sec1ix = round(1 / tbin_sec);

fr_mat_trialed = cell(numel(mouse_grps),1); 
prts = cell(numel(mouse_grps),1); 
qrts = cell(numel(mouse_grps),1); 
RXNil = cell(numel(mouse_grps),1);  
GLM_cluster = cell(numel(mouse_grps),1); 
brain_region = cell(numel(mouse_grps),1); 
transient_peak = cell(numel(mouse_grps),1); 

for mIdx = 1:numel(mouse_grps)
    fr_mat_trialed{mIdx} = cell(numel(mouse_grps{mIdx}),5); % 5 segments per trial  
    prts{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    RXNil{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    GLM_cluster{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    transient_peak{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        % grab ALL NEURONS... later pare down using cell labels from transients table
        good_cells = data.sp.cids(data.sp.cgs==2);   
        calcFR_opt.tend = max(data.sp.st); 
        fr_mat = calcFRVsTime(good_cells,data,calcFR_opt);  
        
        % gather patch information 
        rewsize = mod(data.patches(:,2),10);  
        rew_sec = data.rew_ts;
        patchcue_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);
        i_prts = patchleave_sec - patchstop_sec; 
        i_qrts = patchstop_sec - patchcue_sec; 
        nTrials = length(i_qrts); 
        
        % Make RXNil vector
        i_RXNil = nan(nTrials,1); 
        for iTrial = 1:nTrials
            if i_prts(iTrial) >= 1 % only care if we have at least 1 second on patch
                rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
                if isequal(rew_indices,1)
                    i_RXNil(iTrial) = 10*rewsize(iTrial);
                elseif isequal(rew_indices,[1 ; 2]) 
                    i_RXNil(iTrial) = 10*rewsize(iTrial) + rewsize(iTrial);
                end
            end
        end  
        
        % get indexing vectors to produce trialed fr_mat
        patchcue_ix = round(patchcue_sec / tbin_sec);
        patchstop_ix = round(patchstop_sec / tbin_sec); 
        patchleave_ix = round(patchleave_sec / tbin_sec);

        % collect firing rate matrix
        i_precue_fr_mat = cell(nTrials,1); 
        i_cue_fr_mat = cell(nTrials,1); 
        i_sec0_fr_mat = cell(nTrials,1); 
        i_sec1plus_fr_mat = cell(nTrials,1); 
        i_postleave_fr_mat = cell(nTrials,1); 
        for iTrial = 1:length(data.patchCSL) 
            if ~isnan(i_RXNil(iTrial)) % don't need to save these guys; should never access
                i_precue_fr_mat{iTrial} = fr_mat(:,patchcue_ix(iTrial) - pre_cue_ix:patchcue_ix(iTrial)-1);
                i_cue_fr_mat{iTrial} = fr_mat(:,patchcue_ix(iTrial):patchstop_ix(iTrial)-1);
                i_sec0_fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial):patchstop_ix(iTrial)+sec1ix-1);
                i_sec1plus_fr_mat{iTrial} = fr_mat(:,patchstop_ix(iTrial)+sec1ix:patchleave_ix(iTrial)-1);
                i_postleave_fr_mat{iTrial} = fr_mat(:,patchleave_ix(iTrial):patchleave_ix(iTrial) + post_leave_ix-1);
            end
        end  
        
        % log data to cell arrays 
        % neural data
        fr_mat_trialed{mIdx}{i,1} = i_precue_fr_mat;
        fr_mat_trialed{mIdx}{i,2} = i_cue_fr_mat;
        fr_mat_trialed{mIdx}{i,3} = i_sec0_fr_mat;
        fr_mat_trialed{mIdx}{i,4} = i_sec1plus_fr_mat;
        fr_mat_trialed{mIdx}{i,5} = i_postleave_fr_mat; 
        % trial information
        prts{mIdx}{i} = i_prts;
        qrts{mIdx}{i} = i_prts;
        RXNil{mIdx}{i} = i_RXNil;  
        % neuron information 
        session_table = transients_table(strcmp(transients_table.Session,session_title),:); 
        GLM_cluster{mIdx}{i} = session_table.gmm_cluster;
        brain_region{mIdx}{i} = session_table.Region;
        transient_peak{mIdx}{i} = [session_table.Rew0_peak_pos session_table.Rew1plus_peak_pos];
    end
end

%% Z-scored PRT (add this above later)  
trialtypes = [10 11 20 22 40 44]; 
zscored_prts = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)  
    zscored_prts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})     
        nTrials = length(RXNil{mIdx}{i}); 
        zscored_prts{mIdx}{i} = nan(nTrials,1); 
        for i_tt = 1:numel(trialtypes) 
            tt = trialtypes(i_tt); 
            zscored_prts{mIdx}{i}(RXNil{mIdx}{i} == tt) = zscore(prts{mIdx}{i}(RXNil{mIdx}{i} == tt)); 
        end 
%         zscored_prts{mIdx}{i} = zscored_prts{mIdx}{i}';
    end
end

%% Stretch, avg, and concatenate trial types within mice
trialtypes = [10 11 20 22 40 44]; 
mouse_RXNil_mean = cell(numel(mouse_grps),numel(trialtypes)); 
% structures for pooling across mice
xMice_fr_mat = cell(numel(mouse_grps),5); % pool sessions within mice
xMice_RXNil = cell(numel(mouse_grps),1); 
xMice_s_ix = cell(numel(mouse_grps),1); 
xMice_s_ix_neurons = cell(numel(mouse_grps),1); 
xMice_GLM_cluster = cell(numel(mouse_grps),1); 
xMice_brain_region = cell(numel(mouse_grps),1); 
xMice_transient_peak = cell(numel(mouse_grps),1); 
xMice_all_tt_sessions = cell(numel(mouse_grps),1);  
xMice_prts = cell(numel(mouse_grps),1);  
xMice_zscored_prts = cell(numel(mouse_grps),1);  

for mIdx = 1:numel(mouse_grps) 
    session_unique_tts = cellfun(@(x) x(~isnan(x)),cellfun(@unique, RXNil{mIdx}, 'un', 0),'un',0);
    all_tt_sessions = cellfun(@(x) isequal(x',trialtypes),session_unique_tts,'un',1);
    xMice_all_tt_sessions{mIdx} = find(all_tt_sessions); 
    
    % Optionally do some session selection here
    % Gather neural data from all sessions for this mouse
    m_precue_fr_mat = fr_mat_trialed{mIdx}(all_tt_sessions,1);
    m_precue_fr_mat = cat(1,m_precue_fr_mat{:}); 
    m_cue_fr_mat = fr_mat_trialed{mIdx}(all_tt_sessions,2); 
    m_cue_fr_mat = cat(1,m_cue_fr_mat{:}); 
    m_sec0_fr_mat = fr_mat_trialed{mIdx}(all_tt_sessions,3); 
    m_sec0_fr_mat = cat(1,m_sec0_fr_mat{:}); 
    m_sec1plus_fr_mat = fr_mat_trialed{mIdx}(all_tt_sessions,4); 
    m_sec1plus_fr_mat = cat(1,m_sec1plus_fr_mat{:});
    m_postleave_fr_mat = fr_mat_trialed{mIdx}(all_tt_sessions,5); 
    m_postleave_fr_mat = cat(1,m_postleave_fr_mat{:}); 
    m_prts = prts{mIdx}(all_tt_sessions); 
    m_prts = cat(1,m_prts{:}); 
    m_zscored_prts = zscored_prts{mIdx}(all_tt_sessions); 
    m_zscored_prts = cat(1,m_zscored_prts{:}); 
    
    % Then trial types and session information 
    m_RXNil = RXNil{mIdx}(all_tt_sessions);
    m_RXNil = cat(1,m_RXNil{:});    
    s_nTrials = cellfun(@length,RXNil{mIdx}(all_tt_sessions));
    s_ix = arrayfun(@(i) i + zeros(s_nTrials(i),1),(1:numel(s_nTrials))','un',0);
    s_ix = cat(1,s_ix{:}); 
    these_sessions = unique(s_ix); 
    
    % Neuron information to subselect populations
    m_GLM_cluster = GLM_cluster{mIdx}(all_tt_sessions);
    m_GLM_cluster = cat(1,m_GLM_cluster{:});  
    m_brain_region = brain_region{mIdx}(all_tt_sessions);
    m_brain_region = cat(1,m_brain_region{:});   
    m_transient_peak = transient_peak{mIdx}(all_tt_sessions); 
    m_transient_peak = cat(1,m_transient_peak{:});  
    s_nNeurons = cellfun(@length,GLM_cluster{mIdx}(all_tt_sessions)); 
    s_ix_neurons = arrayfun(@(i) i + zeros(s_nNeurons(i),1),(1:numel(s_nNeurons))','un',0);
    s_ix_neurons = cat(1,s_ix_neurons{:}); 
    
    % add to pooled datastructures 
    xMice_fr_mat{mIdx,1} = m_precue_fr_mat;
    xMice_fr_mat{mIdx,2} = m_cue_fr_mat;
    xMice_fr_mat{mIdx,3} = m_sec0_fr_mat;
    xMice_fr_mat{mIdx,4} = m_sec1plus_fr_mat;
    xMice_fr_mat{mIdx,5} = m_postleave_fr_mat;
    xMice_RXNil{mIdx} = m_RXNil;
    xMice_s_ix{mIdx} = 10 * mIdx + s_ix; % add 10 * mIdx so we can differentiate betw mice 
    xMice_s_ix_neurons{mIdx} = 10 * mIdx + s_ix_neurons;
    xMice_GLM_cluster{mIdx} = m_GLM_cluster;
    xMice_brain_region{mIdx} = m_brain_region;
    xMice_transient_peak{mIdx} = m_transient_peak; 
    xMice_prts{mIdx} = m_prts;
    xMice_zscored_prts{mIdx} = m_zscored_prts;
    
    % Iterate over trial types 
    for i_trialtype = 1:numel(trialtypes) 
        iTrialtype = trialtypes(i_trialtype); 
        these_trials = m_RXNil == iTrialtype; 
        tt_s_ix = s_ix(these_trials); 
        
        % get median trial section lengths for stretching
        median_cue_len = median(cellfun(@(x) size(x,2),m_cue_fr_mat(these_trials)));
        sec1plus_median_len = median(cellfun(@(x) size(x,2),m_sec1plus_fr_mat(these_trials)));
        
        % collect fr_mats from 5 trial periods  
        tt_precue_fr_mat = m_precue_fr_mat(these_trials);
        tt_cue_fr_mat = cellfun(@(x) imresize(x,[size(x,1),median_cue_len]),m_cue_fr_mat(these_trials),'un',0);
        tt_sec0_fr_mat = m_sec0_fr_mat(these_trials);
        tt_sec1plus_fr_mat = cellfun(@(x) imresize(x,[size(x,1),sec1plus_median_len]),m_sec1plus_fr_mat(these_trials),'un',0);
        tt_postleave_fr_mat = m_postleave_fr_mat(these_trials); 
        
        % concatenate to make full trials 
        tt_fr_mat_full = cat(2,tt_precue_fr_mat,tt_cue_fr_mat,tt_sec0_fr_mat,tt_sec1plus_fr_mat,tt_postleave_fr_mat); 
        tt_fr_mat_full = arrayfun(@(row) cat(2,tt_fr_mat_full{row,:}),(1:length(find(these_trials)))','un',0);   
        % and average neural responses within sessions, then concatenate vertically 
        tt_fr_mat_full_mean = []; 
        for i_s_ix = 1:numel(these_sessions)
            this_session_ix = these_sessions(i_s_ix); 
            session_tt_fr_mat = tt_fr_mat_full(tt_s_ix == this_session_ix);  
            session_tt_fr_mat = cat(3,session_tt_fr_mat{:});
            tt_fr_mat_full_mean = [tt_fr_mat_full_mean ; mean(session_tt_fr_mat,3)];
        end 
        mouse_RXNil_mean{mIdx,i_trialtype} = tt_fr_mat_full_mean;
    end
    
end

%% Perform PCA and NMF. Analyze elbow plot

RXNil_pca = cell(numel(mouse_grps),numel(trialtypes)); 
figure();hold on
for mIdx = 1:numel(mouse_grps)
    session_unique_tts = cellfun(@(x) x(~isnan(x)),cellfun(@unique, RXNil{mIdx}, 'un', 0),'un',0);
    all_tt_sessions = cellfun(@(x) isequal(x',trialtypes),session_unique_tts,'un',1);
    m_RXNil_mean = mouse_RXNil_mean(mIdx,:); 
    
    % Neuron information to subselect populations
    m_GLM_cluster = GLM_cluster{mIdx}(all_tt_sessions);
    m_GLM_cluster = cat(1,m_GLM_cluster{:});  
    m_brain_region = brain_region{mIdx}(all_tt_sessions);
    m_brain_region = cat(1,m_brain_region{:});   
    m_transient_peak = transient_peak{mIdx}(all_tt_sessions); 
    m_transient_peak = cat(1,m_transient_peak{:}); 
    
    % select population
    these_neurons = strcmp(m_brain_region,"PFC"); 
%     these_neurons = ~isnan(m_GLM_cluster); 
    roi_m_RXNil_mean = cellfun(@(x) x(these_neurons,:),m_RXNil_mean,'un',0); 
    
    % concatenate to perform dimensionality reduction 
    tt_starts = [0 cumsum(cellfun(@(x) size(x,2), roi_m_RXNil_mean))];
    roi_m_RXNil_mean_full = cat(2,roi_m_RXNil_mean{:}); 
    [coeff,score,~,~,explained] = pca(zscore(roi_m_RXNil_mean_full,[],2)');  
    for tt = 1:numel(trialtypes)
        RXNil_pca{mIdx,tt} = score(tt_starts(tt)+1:tt_starts(tt+1),:)';
    end
    
%     % for nmf
%     d = nan(10,1); 
%     for k = 1:10 
%         [~,~,d(k)] = nnmf(roi_m_RXNil_mean_full',k); 
%     end
    
    % elbow plot
    subplot(1,2,1) ; hold on
    plot(explained(1:10) / sum(explained))
    subplot(1,2,2) ; hold on
    plot(cumsum(explained(1:10)) / sum(explained))
%     % relation of coeff to glm cluster 
%     subplot(1,3,3) ; hold on
%     gscatter(coeff(:,1),coeff(:,3),m_GLM_cluster(~isnan(m_GLM_cluster)))
end

%% Now visualize
vis_mice = [1 2 4 5]; 
figure()
for m_ix = 1:numel(vis_mice)  
    mIdx = vis_mice(m_ix); 
    ax = subplot(1,numel(vis_mice),m_ix);hold on;
    for tt = 1:numel(trialtypes) 
        plot3(ax,RXNil_pca{mIdx,tt}(1,:),RXNil_pca{mIdx,tt}(2,:),RXNil_pca{mIdx,tt}(3,:))
    end
    grid()
end 

%% Now perform same process, but concatenate across mice 

pool_mice = [1 2 4 5]; % similar PRTs 

pooled_fr_mat = arrayfun(@(row) cat(1,xMice_fr_mat{pool_mice,row}),1:5,'un',0);
[pooled_precue_fr_mat, pooled_cue_fr_mat, pooled_sec0_fr_mat, pooled_sec1plus_fr_mat, pooled_postleave_fr_mat] = pooled_fr_mat{:};

pooled_RXNil = xMice_RXNil(pool_mice); 
pooled_RXNil = cat(1,pooled_RXNil{:}); 
pooled_s_ix = xMice_s_ix(pool_mice); % add 10 * mIdx so we can differentiate betw mice 
pooled_s_ix = cat(1,pooled_s_ix{:});  
pooled_s_ix_neurons = xMice_s_ix_neurons(pool_mice); 
pooled_s_ix_neurons = cat(1,pooled_s_ix_neurons{:}); 
pooled_GLM_cluster = GLM_cluster(pool_mice); 
pooled_GLM_cluster = cat(1,pooled_GLM_cluster{:}); 
pooled_GLM_cluster = cat(1,pooled_GLM_cluster{:}); 
these_sessions = unique(pooled_s_ix); 

% this data structure is within mice pooled, but with xMice stretching so
% we can go back and project these guys onto our PC space
mouse_poolStretch_fr_mat_full = cell(numel(mouse_grps),numel(trialtypes));  
% Data structure w/ pooled acr all mice neuron avgs per trialtype
pooled_RXNil_mean = cell(1,numel(trialtypes)); 
sec1plus_median_lens = nan(numel(trialtypes),1); 

% first get median cue length across trial types.. will make visualization easier 
median_cue_len = floor(median(cellfun(@(x) size(x,2),pooled_cue_fr_mat(~isnan(pooled_RXNil)))));

% Iterate over trial types
for i_trialtype = 1:numel(trialtypes)
    iTrialtype = trialtypes(i_trialtype);
    these_trials = pooled_RXNil == iTrialtype;
    tt_s_ix = pooled_s_ix(these_trials);
    
    % get median trial section lengths for stretching
    sec1plus_median_len = median(cellfun(@(x) size(x,2),pooled_sec1plus_fr_mat(these_trials))); 
    sec1plus_median_lens(i_trialtype) = sec1plus_median_len;
    
    % collect fr_mats from 5 trial periods
    tt_precue_fr_mat = pooled_precue_fr_mat(these_trials);
    tt_cue_fr_mat = cellfun(@(x) imresize(x,[size(x,1),median_cue_len]),pooled_cue_fr_mat(these_trials),'un',0);
    tt_sec0_fr_mat = pooled_sec0_fr_mat(these_trials);
    tt_sec1plus_fr_mat = cellfun(@(x) imresize(x,[size(x,1),sec1plus_median_len]),pooled_sec1plus_fr_mat(these_trials),'un',0);
    tt_postleave_fr_mat = pooled_postleave_fr_mat(these_trials);
    
    % concatenate to make full trials
    tt_fr_mat_full = cat(2,tt_precue_fr_mat,tt_cue_fr_mat,tt_sec0_fr_mat,tt_sec1plus_fr_mat,tt_postleave_fr_mat);
    tt_fr_mat_full = arrayfun(@(row) cat(2,tt_fr_mat_full{row,:}),(1:length(find(these_trials)))','un',0);
    % and average neural responses within sessions, then concatenate vertically
    tt_fr_mat_full_mean = [];
    for i_s_ix = 1:numel(these_sessions)
        this_session_ix = these_sessions(i_s_ix);
        session_tt_fr_mat = tt_fr_mat_full(tt_s_ix == this_session_ix);
        session_tt_fr_mat = cat(3,session_tt_fr_mat{:});
        mouse_poolStretch_fr_mat_full{floor(this_session_ix / 10),i_trialtype} = [mouse_poolStretch_fr_mat_full{floor(this_session_ix / 10),i_trialtype} ; mean(session_tt_fr_mat,3)];
        tt_fr_mat_full_mean = [tt_fr_mat_full_mean ; mean(session_tt_fr_mat,3)];
    end
    pooled_RXNil_mean{i_trialtype} = tt_fr_mat_full_mean;
end

%% Perform PCA across mice 
pooled_RXNil_pca = cell(1,numel(trialtypes)); 
neuron_selection = "GLM"; 
% figure();hold on

% Neuron information to subselect populations
pooled_GLM_cluster = xMice_GLM_cluster(pool_mice); 
pooled_GLM_cluster = cat(1,pooled_GLM_cluster{:});
pooled_brain_region = xMice_brain_region(pool_mice); 
pooled_brain_region = cat(1,pooled_brain_region{:}); 
pooled_transient_peak = xMice_transient_peak(pool_mice); 
pooled_transient_peak = cat(1,pooled_transient_peak{:}); 

% select population 
if neuron_selection == "PFC"
    these_neurons = strcmp(pooled_brain_region,"PFC"); 
elseif neuron_selection == "GLM"
    these_neurons = ~isnan(pooled_GLM_cluster);
elseif neuron_selection == "Not GLM" 
    these_neurons = strcmp(pooled_brain_region,"PFC") & isnan(pooled_GLM_cluster);
else 
    throw(MException("MyComponent:noSuchVariable","neuron_selection must be PFC or GLM"))
end 

pooled_GLM_cluster_selection = pooled_GLM_cluster(these_neurons);

roi_pooled_RXNil_mean = cellfun(@(x) x(these_neurons,:),pooled_RXNil_mean,'un',0);
% now go into our per mouse datastructure and use same subselection
roi_mouse_poolStretch_RXNil_mean = cell(numel(pool_mice),numel(trialtypes)); 
neuron_starts = [0 cumsum(cellfun(@length,xMice_brain_region(pool_mice)))'];
for m = 1:numel(pool_mice) 
    mIdx = pool_mice(m); 
    roi_mouse_poolStretch_tmp = cellfun(@(x) x(these_neurons((neuron_starts(m)+1):neuron_starts(m+1)),:),mouse_poolStretch_fr_mat_full(mIdx,:),'un',0);
    roi_mouse_poolStretch_RXNil_mean(m,:) = roi_mouse_poolStretch_tmp;
end

% concatenate to perform dimensionality reduction
tt_starts = [0 cumsum(cellfun(@(x) size(x,2), roi_pooled_RXNil_mean))];
roi_pooled_RXNil_mean_full = cat(2,roi_pooled_RXNil_mean{:});
[coeff,score,latent,~,explained,mu] = pca(zscore(roi_pooled_RXNil_mean_full,[],2)');
for tt = 1:numel(trialtypes)
    pooled_RXNil_pca{tt} = score(tt_starts(tt)+1:tt_starts(tt+1),:)';
end 

% % reduce within mice to overlay traces 
% for m = 1:numel(pool_mice) 
%     
% end
lines2 = lines(2);
% elbow plot
figure()
plot(explained(1:10),'k','linewidth',1.5);hold on
scatter(1:10,explained(1:10),[],lines2(1,:),'k','linewidth',1.5)
% plot(cumsum(explained(1:10)),'linewidth',1.5);hold on
% scatter(1:10,cumsum(explained(1:10)),[],lines2(1,:),'linewidth',1.5)
title("Variance explained")
ylabel("Variance explained per component (%)") 
xlabel("Principal Component")
set(gca,'fontsize',14)
% subplot(1,2,2)
% plot(cumsum(explained(1:10)) / sum(explained),'linewidth',1.5)
% title("Cumulative variance explained")
% subplot(1,3,3) 
% plot(-diff(explained(1:10) / sum(explained)),'linewidth',1.5)
% title("\Delta variance explained") 
suptitle(neuron_selection)

%% Visualize pooled traces 

pre_cue_ix = round(pre_cue_sec / tbin_sec); 
post_leave_ix = round(post_leave_sec / tbin_sec); 

colors = {[.5 1 1],[0 1 1],[.75 .75 1],[.5 .5 1],[1 .5 1],[1 0 1]}; 
figure(); hold on
for tt = 1:6
    % pre-cue
    plot3(pooled_RXNil_pca{tt}(1,1:pre_cue_ix), ...
          pooled_RXNil_pca{tt}(2,1:pre_cue_ix), ...
          pooled_RXNil_pca{tt}(3,1:pre_cue_ix),'linewidth',2,'color',[.5 .5 .5]) 
    % cue
    plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix:pre_cue_ix + median_cue_len), ...
          pooled_RXNil_pca{tt}(2,pre_cue_ix:pre_cue_ix + median_cue_len), ...
          pooled_RXNil_pca{tt}(3,pre_cue_ix:pre_cue_ix + median_cue_len),'linewidth',2,'color',[.2 .7 .2]) 
    % trial 
    plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix), ...
          pooled_RXNil_pca{tt}(2,pre_cue_ix + median_cue_len:end-post_leave_ix), ... 
          pooled_RXNil_pca{tt}(3,pre_cue_ix + median_cue_len:end-post_leave_ix),'linewidth',2,'color',colors{tt}) 

%     % post leave
%     plot3(pooled_RXNil_pca{tt}(1,end-post_leave_ix:end), ...
%           pooled_RXNil_pca{tt}(2,end-post_leave_ix:end), ...
%           pooled_RXNil_pca{tt}(2,end-post_leave_ix:end),'linewidth',1.5,'color',[.3 .3 .3])
    
    tick_interval = 10;  
    tt_len = length(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix));

    % add time ticks for cue
    time_ticks = (pre_cue_ix+tick_interval):tick_interval:(pre_cue_ix + median_cue_len - tick_interval);
    plot3(pooled_RXNil_pca{tt}(1,time_ticks),pooled_RXNil_pca{tt}(2,time_ticks),pooled_RXNil_pca{tt}(3,time_ticks), ... 
          'ko', 'markerSize', 8, 'markerFaceColor',[.2 .7 .2]); 
    % add time ticks for on patch
    time_ticks = ((pre_cue_ix + median_cue_len)+tick_interval):tick_interval:(tt_len - tick_interval);
    plot3(pooled_RXNil_pca{tt}(1,time_ticks),pooled_RXNil_pca{tt}(2,time_ticks),pooled_RXNil_pca{tt}(3,time_ticks), ... 
          'ko', 'markerSize', 8, 'markerFaceColor',colors{tt}); 
      
    % add mark for reward
    if mod(tt,2) == 0
        plot3(pooled_RXNil_pca{tt}(1,(pre_cue_ix + median_cue_len)+50),pooled_RXNil_pca{tt}(2,(pre_cue_ix + median_cue_len)+50),pooled_RXNil_pca{tt}(3,(pre_cue_ix + median_cue_len)+50), ... 
              'kd', 'markerSize', 15, 'markerFaceColor',colors{tt}); 
    end 
    
        % add some marks to make the trajectories more interpretable 
    plot3(pooled_RXNil_pca{tt}(1,1),pooled_RXNil_pca{tt}(2,1),pooled_RXNil_pca{tt}(3,1), ...
          'ko', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5]);
    plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix),pooled_RXNil_pca{tt}(2,pre_cue_ix),pooled_RXNil_pca{tt}(3,pre_cue_ix), ...
          'ko', 'markerSize', 15, 'markerFaceColor',[.2 .7 .2]);  
    
    % add O at start of trial
    plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len),pooled_RXNil_pca{tt}(2,pre_cue_ix + median_cue_len),pooled_RXNil_pca{tt}(3,pre_cue_ix + median_cue_len), ... 
          'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
    % add X at end of trial
    plot3(pooled_RXNil_pca{tt}(1,end-post_leave_ix),pooled_RXNil_pca{tt}(2,end-post_leave_ix),pooled_RXNil_pca{tt}(3,end-post_leave_ix), ... 
          'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
    % add X at end of trial
    plot3(pooled_RXNil_pca{tt}(1,end-post_leave_ix),pooled_RXNil_pca{tt}(2,end-post_leave_ix),pooled_RXNil_pca{tt}(3,end-post_leave_ix), ... 
          'kx', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5],'linewidth',2);
%     
%     % add arrow to end of line
%     penultimate_pt = [pooled_RXNil_pca{tt}(1,end-post_leave_ix-1),pooled_RXNil_pca{tt}(2,end-post_leave_ix-1),pooled_RXNil_pca{tt}(3,end-post_leave_ix-1)];
%     ultimate_pt = [pooled_RXNil_pca{tt}(1,end-post_leave_ix),pooled_RXNil_pca{tt}(2,end-post_leave_ix),pooled_RXNil_pca{tt}(3,end-post_leave_ix)];
%     projection_scale = 10;
% %     arrow3(penultimate_pt,ultimate_pt,[],.4,.8,.4,.4)
%     quiver3(penultimate_pt(1),penultimate_pt(2),penultimate_pt(3),projection_scale * (ultimate_pt(1) - penultimate_pt(1)),projection_scale * (ultimate_pt(2) - penultimate_pt(2)),projection_scale * (ultimate_pt(3) - penultimate_pt(3)),...
%             'linewidth',1,'maxHeadSize',10,'color',colors{tt})
        
%     ARROW3(penultimate_pt,ultimate_pt,S,W,H,IP,ALPHA,BETA)
%     zl = zlim();
    
    % precue
%     plot3(pooled_RXNil_pca{tt}(1,1:pre_cue_ix), ...
%           pooled_RXNil_pca{tt}(2,1:pre_cue_ix), ...
%           zeros(size(pooled_RXNil_pca{tt}(2,1:pre_cue_ix))) + zl(1),'linewidth',1,'color',min(1,.5 * [.5 .5 .5]) )
%     % cue 
%     plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix:pre_cue_ix + median_cue_len), ...
%           pooled_RXNil_pca{tt}(2,pre_cue_ix:pre_cue_ix + median_cue_len), ...
%           zeros(size(pooled_RXNil_pca{tt}(2,pre_cue_ix:pre_cue_ix + median_cue_len))) + zl(1),'linewidth',1,'color',min(1,.5 * [.5 .5 .5])) 
%     % trial 
%     plot3(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix), ...
%           pooled_RXNil_pca{tt}(2,pre_cue_ix + median_cue_len:end-post_leave_ix), ... 
%           zeros(size(pooled_RXNil_pca{tt}(2,pre_cue_ix + median_cue_len:end-post_leave_ix))) + zl(1),'linewidth',1,'color',min(1,.5 * [.5 .5 .5]))   
% %     zlim(zl)
end 
xlabel("PC1");ylabel("PC2");zlabel("PC3")
xticklabels([]);yticklabels([]);zticklabels([]) 
grid()  
view(-135,30)
title("R0Nil Trial Population Dynamics")
set(gca,'fontsize',15)

%% Visualize pooled traces but now separate by reward at t = 0

pre_cue_ix = round(pre_cue_sec / tbin_sec); 
post_leave_ix = round(post_leave_sec / tbin_sec); 

colors = {[.5 1 1],[0 1 1],[.75 .75 1],[.5 .5 1],[1 .5 1],[1 0 1]}; 
trial_grp = {[1 3 5],[2 4 6]};
figure(); hold on 
for i_trial_grp = 1:2
    for second_pc = 2:3
        for i_tt = 1:numel(trial_grp{i_trial_grp}) %  [1 3 5] 
            tt = trial_grp{i_trial_grp}(i_tt);
            subplot(2,2,2 * (second_pc-2) + i_trial_grp);hold on
            % pre-cue
            plot(pooled_RXNil_pca{tt}(1,1:pre_cue_ix), ...
                pooled_RXNil_pca{tt}(second_pc,1:pre_cue_ix),'linewidth',1,'color',[.5 .5 .5])
            % cue
            plot(pooled_RXNil_pca{tt}(1,pre_cue_ix:pre_cue_ix + median_cue_len), ...
                pooled_RXNil_pca{tt}(second_pc,pre_cue_ix:pre_cue_ix + median_cue_len),'linewidth',1,'color',[.2 .7 .2])
            % trial
            plot(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix), ...
                pooled_RXNil_pca{tt}(second_pc,pre_cue_ix + median_cue_len:end-post_leave_ix),'linewidth',1,'color',colors{tt})
            
            % add some marks to make the trajectories more interpretable
            plot(pooled_RXNil_pca{tt}(1,1),pooled_RXNil_pca{tt}(second_pc,1), ...
                'ko', 'markerSize', 6, 'markerFaceColor',[.5 .5 .5]);
            plot(pooled_RXNil_pca{tt}(1,pre_cue_ix),pooled_RXNil_pca{tt}(second_pc,pre_cue_ix), ...
                'ko', 'markerSize', 6, 'markerFaceColor',[.2 .7 .2]);
            tt_len = length(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix));
            
            % add O at start of trial
            plot(pooled_RXNil_pca{tt}(1,pre_cue_ix + median_cue_len),pooled_RXNil_pca{tt}(second_pc,pre_cue_ix + median_cue_len), ...
                'ko', 'markerSize', 6, 'markerFaceColor',colors{tt});
            tick_interval = 10;
            % add time ticks for cue
            time_ticks = (pre_cue_ix+tick_interval):tick_interval:(pre_cue_ix + median_cue_len);
            plot(pooled_RXNil_pca{tt}(1,time_ticks),pooled_RXNil_pca{tt}(second_pc,time_ticks), ...
                'ko', 'markerSize', 3, 'markerFaceColor',[.2 .7 .2]);
            % add time ticks for on patch
            time_ticks = ((pre_cue_ix + median_cue_len)+tick_interval):tick_interval:tt_len;
            plot(pooled_RXNil_pca{tt}(1,time_ticks),pooled_RXNil_pca{tt}(second_pc,time_ticks), ...
                'ko', 'markerSize', 3, 'markerFaceColor',colors{tt});
            
            % add mark for reward
            if mod(tt,2) == 0
                plot(pooled_RXNil_pca{tt}(1,(pre_cue_ix + median_cue_len)+50),pooled_RXNil_pca{tt}(second_pc,(pre_cue_ix + median_cue_len)+50), ...
                    'kd', 'markerSize', 8, 'markerFaceColor',colors{tt});
            end
            
            % add arrow to end of line
            penultimate_pt = [pooled_RXNil_pca{tt}(1,end-post_leave_ix-1),pooled_RXNil_pca{tt}(second_pc,end-post_leave_ix-1)];
            ultimate_pt = [pooled_RXNil_pca{tt}(1,end-post_leave_ix),pooled_RXNil_pca{tt}(second_pc,end-post_leave_ix)];
        end
        xlabel("PC1");ylabel(sprintf("PC%i",second_pc))
        xticklabels([]);yticklabels([]);zticklabels([])
%         grid() 
        if second_pc == 2 
            if i_trial_grp == 1
                title("R0Nil Population Dynamics")
            else 
                title("RRNil Population Dynamics")
            end
        end
        set(gca,'fontsize',14)
    end 
end
suptitle("RXNil Trial Population Dynamics")
suptitle(neuron_selection)

%% pcs vs time
vis_trial_types = 1:6;
for pc_vis = 1
    figure()
    for tt = vis_trial_types
        plot((1:pre_cue_ix) * tbin_sec - pre_cue_sec - median_cue_len * tbin_sec,pooled_RXNil_pca{tt}(pc_vis,1:pre_cue_ix),'linewidth',1,'color',[.5 .5 .5])
        hold on
        plot((pre_cue_ix:(pre_cue_ix + median_cue_len))* tbin_sec - pre_cue_sec - median_cue_len * tbin_sec,pooled_RXNil_pca{tt}(pc_vis,pre_cue_ix:(pre_cue_ix + median_cue_len))  ,'linewidth',2,'color',[.2 .7 .2])
        t_len = length(pooled_RXNil_pca{tt}(pc_vis,(pre_cue_ix + median_cue_len):end-post_leave_ix))-1;
        plot(((pre_cue_ix + median_cue_len):(pre_cue_ix + median_cue_len + t_len))* tbin_sec - pre_cue_sec - median_cue_len * tbin_sec,pooled_RXNil_pca{tt}(pc_vis,(pre_cue_ix + median_cue_len):end-post_leave_ix),'linewidth',2,'color',colors{tt})
        
        % add mark for reward
        if mod(tt,2) == 0
            plot(tbin_sec * (pre_cue_ix + median_cue_len+50)- pre_cue_sec - median_cue_len*tbin_sec,pooled_RXNil_pca{tt}(pc_vis,(pre_cue_ix + median_cue_len)+50),'kd', 'markerSize', 15, 'markerFaceColor',colors{tt}); 
        end 

        % add some marks to make the trajectories more interpretable
        plot(-pre_cue_sec - median_cue_len * tbin_sec,pooled_RXNil_pca{tt}(pc_vis,1),'ko', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5]);
        plot(- median_cue_len * tbin_sec,pooled_RXNil_pca{tt}(pc_vis,pre_cue_ix),'ko', 'markerSize', 15, 'markerFaceColor',[.2 .7 .2]);
        
        % add O at start of trial
        plot(tbin_sec * (pre_cue_ix + median_cue_len)- pre_cue_sec - median_cue_len*tbin_sec,pooled_RXNil_pca{tt}(pc_vis,pre_cue_ix + median_cue_len),'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
        % add X at end of trial
        plot(tbin_sec * (pre_cue_ix + median_cue_len + t_len)- pre_cue_sec - median_cue_len*tbin_sec,pooled_RXNil_pca{tt}(pc_vis,end-post_leave_ix),'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
        % add X at end of trial
        plot(tbin_sec * (pre_cue_ix + median_cue_len + t_len)- pre_cue_sec - median_cue_len*tbin_sec,pooled_RXNil_pca{tt}(pc_vis,end-post_leave_ix),'kx', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5],'linewidth',2);
    end
    title(sprintf("RXNil PC %i Dynamics",pc_vis))
    ylabel(sprintf("Neural Projection onto PC %i",pc_vis))
    set(gca,'fontsize',16)
end
%% Visualize magnitude of gradient over time, w/ same coloring scheme 
close all
tt_lens = cellfun(@(x) size(x,2),pooled_RXNil_pca);
tt_justTrial_lens = tt_lens - median_cue_len - pre_cue_ix - post_leave_ix; % just stretch this part
median_justTrial_len = floor(median(tt_justTrial_lens));
figure() ;hold on
pcs = 1:3; 
tt_rewsizes = [1 1 2 2 4 4];
gradient_trialtype = [];
for tt = 1:6
    pt0 = pooled_RXNil_pca{tt}(pcs,1);
    dist_from_init = vecnorm(pooled_RXNil_pca{tt}(pcs,1:end-post_leave_ix) - pt0,2);
    grad_tt = vecnorm(gradient(pooled_RXNil_pca{tt}(pcs,1:end-post_leave_ix)));  
    grad_tt_justTrial = grad_tt(median_cue_len+1:end);
    gradient_trialtype = [gradient_trialtype ; grad_tt_justTrial' tt_rewsizes(tt) + zeros(length(grad_tt_justTrial),1)];
%     grad_tt_justTrial = interp1(1:length(grad_tt_justTrial),grad_tt_justTrial,linspace(1,length(grad_tt_justTrial),median_justTrial_len));
    grad_tt = [grad_tt(1:median_cue_len) grad_tt_justTrial]; 
    grad_tt = smoothdata(grad_tt,'gaussian',10); % apply some smoothing
    % last, check out evolution of angle to original state
    angle = acos(pt0' * pooled_RXNil_pca{tt}(pcs,1:end-post_leave_ix) ./ (norm(pt0)*vecnorm(pooled_RXNil_pca{tt}(pcs,1:end-post_leave_ix))));
    angle = angle * 180 / pi; % convert to degrees

    % plot dist from ITI evolution 
    subplot(3,1,1);hold on;grid()
    plot(1:pre_cue_ix,dist_from_init(1:pre_cue_ix),'color',[.5 .5 .5],'linewidth',1.5)
    plot(pre_cue_ix:(pre_cue_ix+median_cue_len),dist_from_init(pre_cue_ix:pre_cue_ix + median_cue_len),'color',[.2 .7 .2],'linewidth',1.5) 
    tt_len = length(dist_from_init((median_cue_len+pre_cue_ix):end));
    plot((pre_cue_ix+median_cue_len) + (0:tt_len-1),dist_from_init((pre_cue_ix+median_cue_len):end),'color',colors{tt},'linewidth',1.5)
    xticks(0:25:length(dist_from_init)) 
    xticklabels((0:25:length(dist_from_init))*tbin_sec - (pre_cue_sec))
    xlabel("Time since Cue Onset (sec)") 
    ylabel("Distance From ITI State (A.U.)") 
    title("Quantified RNil Trial Dynamics") 
%     xline(pre_cue_ix + median(median_cue_len_lens) + 50,'linewidth',1.5,'color',[.6 .6 .6])
    % plot gradient evolution
    subplot(3,1,2);hold on;grid()
    plot(1:pre_cue_ix,grad_tt(1:pre_cue_ix),'color',[.5 .5 .5],'linewidth',1.5)
    plot(pre_cue_ix:(pre_cue_ix+median_cue_len),grad_tt(pre_cue_ix:pre_cue_ix + median_cue_len),'color',[.2 .7 .2],'linewidth',1.5) 
    tt_len = length(grad_tt((median_cue_len+pre_cue_ix):end));
    plot((pre_cue_ix+median_cue_len) + (0:tt_len-1),grad_tt((pre_cue_ix+median_cue_len):end),'color',colors{tt},'linewidth',1.5) 
%     xticks(0:length(grad_tt)/5:length(grad_tt)) 
%     xticklabels((0:length(grad_tt)/5:length(grad_tt)) / length(grad_tt)) 
    xticks(0:25:length(angle)) 
    xticklabels((0:25:length(dist_from_init))*tbin_sec - (pre_cue_sec))
    xlabel("Time since Cue Onset (sec)") 
    ylabel("Magnitude of Gradient (A.U.)") 
%     xline(pre_cue_ix + median(median_cue_len_lens) + 50,'linewidth',1.5,'color',[.6 .6 .6])
    % plot angle evolution
    subplot(3,1,3);hold on;grid()
    plot(1:pre_cue_ix,angle(1:pre_cue_ix),'color',[.5 .5 .5],'linewidth',1.5)
    plot(pre_cue_ix:(pre_cue_ix+median_cue_len),angle(pre_cue_ix:pre_cue_ix + median_cue_len),'color',[.2 .7 .2],'linewidth',1.5) 
    tt_len = length(angle((median_cue_len+pre_cue_ix):end));
    plot((pre_cue_ix+median_cue_len) + (0:tt_len-1),angle((pre_cue_ix+median_cue_len):end),'color',colors{tt},'linewidth',1.5)
    xticks(0:25:length(angle)) 
    xticklabels((0:25:length(dist_from_init))*tbin_sec - (pre_cue_sec))
    xlabel("Time since Cue Onset (sec)") 
    ylabel("Angle Traversed From ITI State (degrees)")  
%     xline(pre_cue_ix + median(median_cue_len_lens) + 50,'linewidth',1.5,'color',[.6 .6 .6])    
end 

%% Check out position on pooled 2Nil trial
pcs = 1:3; 
[closest12,dist12] = dsearchn(pooled_RXNil_pca{3}(pcs,:)',pooled_RXNil_pca{1}(pcs,:)');
[closest24,dist24] = dsearchn(pooled_RXNil_pca{3}(pcs,:)',pooled_RXNil_pca{5}(pcs,:)');
[closest22_rew,dist22_rew] = dsearchn(pooled_RXNil_pca{3}(pcs,:)',pooled_RXNil_pca{4}(pcs,:)');
[closest12_rew,dist12_rew] = dsearchn(pooled_RXNil_pca{3}(pcs,:)',pooled_RXNil_pca{2}(pcs,:)');
[closest24_rew,dist24_rew] = dsearchn(pooled_RXNil_pca{3}(pcs,:)',pooled_RXNil_pca{6}(pcs,:)');

figure();hold on
plot(closest12,'color',colors{1},'linewidth',2)
plot(closest24,'color',colors{5},'linewidth',2)
plot(closest12_rew,'color',colors{2},'linewidth',2)
plot(closest24_rew,'color',colors{6},'linewidth',2)
plot(closest22_rew,'color',colors{4},'linewidth',2)
h = refline(1,0); 
h.Color = [.4 .4 .4]; 
h.LineStyle = '--'; 
h.LineWidth = 1.5;

%% Pad trials per session with zeros so that we can bootstrap and see trial-by-trial variability in initial point
trialtypes = [10 11 20 22 40 44]; 
pooled_sessions = unique(pooled_s_ix);
pooled_mIdx = floor(pooled_sessions/10); 
pooled_i = mod(pooled_sessions,10); 
n_pooled_sessions = length(pooled_sessions); 
s_nNeurons = nan(n_pooled_sessions,1); 
s_nNeurons2 = nan(n_pooled_sessions,1); 

% iterate over sessions used in pooled PCA
for i_this_session = 1:n_pooled_sessions
    this_session = pooled_sessions(i_this_session); 
    s_nNeurons(i_this_session) = length(find(these_neurons & pooled_s_ix_neurons == this_session));  
    % these match up now
    mIdx = pooled_mIdx(i_this_session); 
    i = xMice_all_tt_sessions{mIdx}(pooled_i(i_this_session));
    if neuron_selection == "PFC"
        s_nNeurons2(i_this_session) = length(find(strcmp(brain_region{mIdx}{i},"PFC")));
    elseif neuron_selection == "GLM"
        s_nNeurons2(i_this_session) = length(find(~isnan(GLM_cluster{mIdx}{i})));
    end
end 

total_nNeurons = sum(s_nNeurons); 

% mean and std of response per neuron used to calculate the coeffs matrix 
means = mean(roi_pooled_RXNil_mean_full,2); 
stds = std(roi_pooled_RXNil_mean_full,[],2);
neuron_starts = cumsum([0 ; s_nNeurons]); 
pca_trials = cell(numel(pool_mice),1); % for visualization of single trials, initial pointa analysis
nanPadded_trials = cell(numel(pool_mice),1); % for regression vs PRT within RXNil tt, within timepoint
zeroPadded_trials = cell(numel(pool_mice),1); % for averaging, projecting, then bootstrapping stats 
time_trials = cell(numel(pool_mice),1); % time since patch onset
session_counter = 1; 
for m_ix = 1:numel(pool_mice) 
    mIdx = pool_mice(m_ix); 
    pca_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1); 
    nanPadded_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1); 
    zeroPadded_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    time_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    for i_i = 1:numel(xMice_all_tt_sessions{mIdx})
        i = xMice_all_tt_sessions{mIdx}(i_i);  
        nTrials = length(RXNil{mIdx}{i}); 
        nNeurons = s_nNeurons(session_counter);  
        pca_trials{m_ix}{i_i} = cell(nTrials,1); 
        nanPadded_trials{m_ix}{i_i} = cell(nTrials,1); 
        zeroPadded_trials{m_ix}{i_i} = cell(nTrials,1); 
        time_trials{m_ix}{i_i} = cell(nTrials,1); 
        
        if neuron_selection == "PFC"
            i_these_neurons = strcmp(brain_region{mIdx}{i},"PFC");
        elseif neuron_selection == "GLM"
            i_these_neurons = ~isnan(GLM_cluster{mIdx}{i});
        elseif neuron_selection == "Not GLM"
            i_these_neurons = isnan(GLM_cluster{mIdx}{i}) & strcmp(brain_region{mIdx}{i},"PFC");
        end
        
        s_neuron_ix = (neuron_starts(session_counter)+1):neuron_starts(session_counter+1);
        s_means = means(s_neuron_ix);
        s_stds = stds(s_neuron_ix);
        for iTrial = 1:nTrials 
            tt = find(trialtypes == RXNil{mIdx}{i}(iTrial),1); % RXNil trialtype in {1:6}
            if ~isempty(tt) % this was an RX trial, let's do some stuff  
                disp(tt)
                tt_sec1plus_median_len = sec1plus_median_lens(tt);  
                
                 % collect fr_mats from 5 trial periods
                iTrial_precue_fr_mat = fr_mat_trialed{mIdx}{i,1}{iTrial}(i_these_neurons,:);
                iTrial_cue_fr_mat = imresize(fr_mat_trialed{mIdx}{i,2}{iTrial}(i_these_neurons,:),[nNeurons,median_cue_len]);
                iTrial_sec0_fr_mat = fr_mat_trialed{mIdx}{i,3}{iTrial}(i_these_neurons,:); 
                iTrial_sec1plus_fr_mat = imresize(fr_mat_trialed{mIdx}{i,4}{iTrial}(i_these_neurons,:),[nNeurons,tt_sec1plus_median_len]);
                iTrial_postleave_fr_mat = fr_mat_trialed{mIdx}{i,5}{iTrial}(i_these_neurons,:);

                % concatenate to make full trials
                iTrial_fr_mat_full = cat(2,iTrial_precue_fr_mat,iTrial_cue_fr_mat,iTrial_sec0_fr_mat,iTrial_sec1plus_fr_mat,iTrial_postleave_fr_mat);
                iTrial_fr_mat_norm = (iTrial_fr_mat_full - s_means); %  ./ s_stds;
                t_len = size(iTrial_fr_mat_norm,2); 
                time_trials{m_ix}{i_i}{iTrial} = (1:t_len)' * tbin_sec - pre_cue_sec - median_cue_len * tbin_sec;
                
                % pad to do stuff w/ FR later (note no standardization)
                nanPadded_trials{m_ix}{i_i}{iTrial} = nan(total_nNeurons,t_len);  
                zeroPadded_trials{m_ix}{i_i}{iTrial} = zeros(total_nNeurons,t_len);  
                zeroPadded_fr_mat_norm = zeros(total_nNeurons,t_len);  
                nanPadded_trials{m_ix}{i_i}{iTrial}(s_neuron_ix,:) = iTrial_fr_mat_full;
                zeroPadded_trials{m_ix}{i_i}{iTrial}(s_neuron_ix,:) = iTrial_fr_mat_full;
                zeroPadded_fr_mat_norm(s_neuron_ix,:) = iTrial_fr_mat_norm;
                pca_trials{m_ix}{i_i}{iTrial} = coeff(:,1:20)' * zeroPadded_fr_mat_norm; % only really care about 10 PCs
            end
        end 
        session_counter = session_counter + 1; 
    end
end


%% 1a) Does initial state affect PRT within RXNil trialtype? 
% - this cell takes a second b/c the matrix concatenation is pretty big
% Now concatenate across mice and sessions
pooled_pca_trials = cat(1,pca_trials{:}); 
pooled_pca_trials = cat(1,pooled_pca_trials{:}); 
pooled_zeroPadded_trials = cat(1,zeroPadded_trials{:}); 
pooled_zeroPadded_trials = cat(1,pooled_zeroPadded_trials{:}); 
pooled_nanPadded_trials = cat(1,nanPadded_trials{:}); 
pooled_nanPadded_trials = cat(1,pooled_nanPadded_trials{:}); 
pooled_prts = cat(1,xMice_prts(pool_mice)); 
pooled_prts = cat(1,pooled_prts{:}); 
pooled_zscored_prts = cat(1,xMice_zscored_prts(pool_mice)); 
pooled_zscored_prts = cat(1,pooled_zscored_prts{:});  
pooled_time = cat(1,time_trials{:}); 
pooled_time = cat(1,pooled_time{:}); 

pca_full = cat(2,pooled_pca_trials{:});
zeroPadded_fr_mat_full = cat(2,pooled_zeroPadded_trials{:});
nanPadded_fr_mat_full = cat(2,pooled_nanPadded_trials{:});
time_full = cat(1,pooled_time{:})'; 

patch_onset_ix = median_cue_len + pre_cue_ix; 

% 1c) Corrcoef to find timecourse of significant correlation between z-scored PRT and neural activity 
% multiple timepoints, one PC 
t_centers = (-pre_cue_sec - median_cue_len * tbin_sec + tbin_sec):tbin_sec:1.5;
t_edges = [(-pre_cue_sec - median_cue_len * tbin_sec + tbin_sec):tbin_sec:1.5 1.5+tbin_sec] - tbin_sec / 2;
[~,~,binned_t] = histcounts(time_full,t_edges); 

tts_of_interest = [10 11 20 22 40 44]; 
these_trials = ismember(pooled_RXNil(~isnan(pooled_RXNil)),tts_of_interest);
pcs = 1:3;
pt0 = mean(pca_full(pcs,binned_t == 1),2);
dist_from_init = vecnorm(pca_full(pcs,:) - pt0,2);

% estimate "manifold"
% manifold = pooled_RXNil_pca{3}(pcs,:)'; 
% manifold_tts = [40]; 
% manifold_timepoints = 1:200; 
% manifold_pca_trials = pooled_pca_trials(ismember(pooled_RXNil,manifold_tts));
% manifold_pca_trials = cellfun(@(x) x(pcs,manifold_timepoints),manifold_pca_trials,'un',0); 
% manifold = mean(cat(3,manifold_pca_trials{:}),3)'; 
% 
% figure();scatter3(pca_full(1,1:20:end),pca_full(2,1:20:end),pca_full(3,1:20:end),5,'.');hold on
% plot3(manifold_2(:,1),manifold_2(:,2),manifold_2(:,3),'linewidth',2)
% hold on;plot3(manifold(:,1),manifold(:,2),manifold(:,3),'linewidth',2)

%%
close all
% now align poin w/ closest manifold point
manifold_closest = dsearchn(manifold,pca_full(pcs,:)'); % NEED TO MAKE THIS IN OUR NEW PADDED PC SPACE.. some weird SCALING stuff here?

r_zscored_prt = nan(numel(pcs),max(binned_t)); 
p_zscored_prt = nan(numel(pcs),max(binned_t));  
p_lm = nan(max(binned_t),1); 
r_dist = nan(max(binned_t),1); 
p_dist = nan(max(binned_t),1); 
r_manifold = nan(max(binned_t),1); 
p_manifold = nan(max(binned_t),1); 
for iPC = 1:numel(pcs)
    RXNil_zscored_prts = pooled_zscored_prts(~isnan(pooled_RXNil));
    for i_tBin = 1:max(binned_t)
        these_time_ix = binned_t == i_tBin;
        iTime_neural_data = pca_full(iPC,these_time_ix)';
        
        [r,p] = corrcoef(iTime_neural_data(these_trials),RXNil_zscored_prts(these_trials));
        r_zscored_prt(iPC,i_tBin) = r(2);
        p_zscored_prt(iPC,i_tBin) = p(2);
        
        if iPC == numel(pcs) 
            all_pcs = pca_full(pcs,these_time_ix)';
            mdl = fitlm(all_pcs(these_trials,:),RXNil_zscored_prts(these_trials));
            p_lm(i_tBin) = coefTest(mdl);
%             iTime_dist_data = dist_from_init(these_time_ix)';
%             [r,p] = corrcoef(iTime_dist_data(these_trials),RXNil_zscored_prts(these_trials));
%             r_dist(i_tBin) = r(2);
%             p_dist(i_tBin) = p(2);
%             
%             iTime_manifold_data = manifold_closest(these_time_ix)';
%             [r,p] = corrcoef(iTime_manifold_data(these_trials),RXNil_zscored_prts(these_trials));
%             if mod(i_tBin,10) == 0 
%                 figure();
%                 scatter(iTime_manifold_data(these_trials),RXNil_zscored_prts(these_trials))
%             end
%             r_manifold(i_tBin) = r(2);
%             p_manifold(i_tBin) = p(2);
        end
    end
end 

figure()
subplot(2,1,1)
plot(t_centers,r_zscored_prt','linewidth',3);hold on
% plot(t_centers,r_dist,'k','linewidth',2) 
% plot(t_centers,r_manifold,'color',[.5,.5,.5],'linewidth',2) 
ylabel("Pearson Correlation Coefficient") 
xlim([min(t_centers) max(t_centers)]) 
ylim([-.3 .3]) 
yline(0,'--','linewidth',1.5) 
xline(t_centers(1) + pre_cue_sec,'--','linewidth',1.5) 
xline(t_centers(1) + pre_cue_sec + ((median_cue_len-1) * tbin_sec),'--','linewidth',1.5) 
set(gca,'fontsize',16)
subplot(2,1,2)
plot(t_centers,log10(p_zscored_prt)','linewidth',3);hold on 
% plot(t_centers,log10(p_dist),'k','linewidth',2) 
% plot(t_centers,log10(p_manifold),'color',[.5,.5,.5],'linewidth',2) 
yline(log10(.05),'--','linewidth',1.5)  
xline(t_centers(1) + pre_cue_sec,'--','linewidth',1.5) 
xline(t_centers(1) + pre_cue_sec + ((median_cue_len-1) * tbin_sec),'--','linewidth',1.5) 
ylabel(sprintf("Log10 p-value \n Against Constant Model") )
xlim([min(t_centers) max(t_centers)]) 
suptitle(neuron_selection)
set(gca,'fontsize',16)
xlabel("Time on Patch (sec)")
legend(arrayfun(@(x) sprintf("PC %i",x),1:numel(pcs)))

%% Ok let's make some visualizations to show what's going on here 

%% 1c) how do pcs vary w/ different zscored prt 
% ok we need to do some binning here, single traces are not nice enough
trialtypes = [40];
trialtype_names = [10 11 20 22 40 44];
trialtype_rewsizes = [1 1 2 2 3 3];
nDiv = 3; 
cool3 = cool(3);
cool_prtShades = nan(nDiv*3,3);
for i_rewsize = 1:3
    cool_prtShades(1:nDiv,i_rewsize) = linspace(.9, cool3(1,i_rewsize), nDiv);
    cool_prtShades((nDiv+1):2*nDiv,i_rewsize) = linspace(.9, cool3(2,i_rewsize), nDiv);
    cool_prtShades((2*nDiv+1):nDiv*3,i_rewsize) = linspace(.9, cool3(3,i_rewsize), nDiv);
end
tt_quantile_colors{1} = cool_prtShades(1:nDiv,:); 
tt_quantile_colors{2} = cool_prtShades((nDiv+1):2*nDiv,:); 
tt_quantile_colors{3} = cool_prtShades((2*nDiv+1):nDiv*3,:); 

trial_precue_offset = pre_cue_ix * tbin_sec + median_cue_len * tbin_sec; 
trial_cue_offset = median_cue_len * tbin_sec; 
for pc_vis = 1
    for i_tt = 1:numel(trialtypes)
        tt = trialtypes(i_tt);
        tt_ix = find(trialtype_names == tt);
        rewsize_ix = trialtype_rewsizes(tt_ix);
        these_trials = find(pooled_RXNil == tt);
        these_zscored_prts = pooled_zscored_prts(these_trials);
        [~,~,prt_bin] = histcounts(these_zscored_prts, [min(these_zscored_prts) quantile(these_zscored_prts,nDiv-1) max(these_zscored_prts)]);
        
        hold on
        for i_bin = unique(prt_bin)'
            bin_trials = these_trials(prt_bin == i_bin);
            this_bin_pca = pooled_pca_trials(bin_trials);
            this_bin_pca = cat(3,this_bin_pca{:}); 
            this_bin_pca_mean = median(this_bin_pca,3);
            this_bin_pca_sem = 1.96 * std(this_bin_pca,[],3) / sqrt(numel(bin_trials));
            t_len = size(this_bin_pca_mean,2);
            % do shaded errorbar
%             shadedErrorBar(1:t_len,this_bin_pca_mean(pc_vis,:),this_bin_pca_sem(pc_vis,:),'')
            
            shadedErrorBar(tbin_sec * (1:pre_cue_ix) - trial_precue_offset,this_bin_pca_mean(pc_vis,1:pre_cue_ix),this_bin_pca_sem(pc_vis,1:pre_cue_ix),'lineProps',{'color',[.5 .5 .5],'linewidth',1.5})
            shadedErrorBar(tbin_sec * (pre_cue_ix:(pre_cue_ix+median_cue_len)) - trial_precue_offset,this_bin_pca_mean(pc_vis,pre_cue_ix:pre_cue_ix + median_cue_len),this_bin_pca_sem(pc_vis,pre_cue_ix:pre_cue_ix + median_cue_len),'lineProps',{'color',[.2 .7 .2],'linewidth',1.5}) 
            tt_len = length(this_bin_pca_mean(pc_vis,(median_cue_len+pre_cue_ix):end));
            shadedErrorBar(tbin_sec * ((pre_cue_ix+median_cue_len) + (1:tt_len-1)-1) - trial_time_offset,this_bin_pca_mean(pc_vis,(pre_cue_ix+median_cue_len+1):end),this_bin_pca_sem(pc_vis,(pre_cue_ix+median_cue_len+1):end),'lineProps',{'color',tt_quantile_colors{rewsize_ix}(i_bin,:),'linewidth',1.5}) 
        end
    end
end


%% 1d) Visualize relation between starting pt and PRT w/ colored scatter 
% one timepoint, multiple PCs 
figure();hold on
for t_of_interest = 0:.1:1.5
% t_of_interest = .8;  
    tbin_of_interest = find(abs(t_centers - t_of_interest) < tbin_sec / 2,1); 
    neural_data = pca_full(1:3,binned_t == tbin_of_interest)';
%     disp(tbin_of_interest)
    scatter3(neural_data(:,1),neural_data(:,2),neural_data(:,3),[],zscore(RXNil_zscored_prts),'.') ;hold on 
end

%% 1c.i) A more compelling visualization of how trajectories change dependent on z-scored PRT 
% average within PRT quartiles for given trialtype, plot 3d

trialtypes = [10 11 20 22 40 44]; 
nDiv = 4; % divide into nDiv PRT groups per trialtype
prt_sep_trajectories = cell(numel(trialtypes),nDiv); 
for i_tt = 1:numel(trialtypes) 
    tt = trialtypes(i_tt);
    these_trials = pooled_RXNil == tt;  
    these_zscored_prts = pooled_zscored_prts(these_trials);  
    these_nanPadded_trials = pooled_nanPadded_trials(these_trials); 
    [~,~,prt_bin] = histcounts(these_zscored_prts, [min(these_zscored_prts) quantile(these_zscored_prts,nDiv-1) max(these_zscored_prts)]);

    for i_quantile = 1:nDiv  
        quantile_trials = these_nanPadded_trials(prt_bin == i_quantile); 
        mean_quantile_trial = nanmean(cat(3,quantile_trials{:}),3);  
        mean_quantile_trial(all(isnan(mean_quantile_trial),2),:) = 0; 
        mean_quantile_trial_norm = (mean_quantile_trial - means) ./ stds;
        prt_sep_trajectories{i_tt,i_quantile} = coeff(:,1:10)' * mean_quantile_trial_norm;
    end
end

%% 1c.ii) Visualize trajectories separated by PRT within trialtype

cool3 = cool(3);
cool_prtShades = nan(nDiv*3,3);
for i_rewsize = 1:3
    cool_prtShades(1:nDiv,i_rewsize) = linspace(.9, cool3(1,i_rewsize), nDiv);
    cool_prtShades((nDiv+1):2*nDiv,i_rewsize) = linspace(.9, cool3(2,i_rewsize), nDiv);
    cool_prtShades((2*nDiv+1):nDiv*3,i_rewsize) = linspace(.9, cool3(3,i_rewsize), nDiv);
end

tt_quantile_colors{1} = cool_prtShades(1:nDiv,:); 
tt_quantile_colors{2} = cool_prtShades((nDiv+1):2*nDiv,:); 
tt_quantile_colors{3} = cool_prtShades((2*nDiv+1):nDiv*3,:); 

figure();hold on
for i_tt = [5]  
    tt = trialtypes(i_tt); 
    i_rewsize = min(3,floor(tt/10)); % min(3,floor(mod(tt,10)));
    for i_quantile = [1 2 nDiv]
        % pre-cue
        plot(prt_sep_trajectories{i_tt,i_quantile}(1,1:pre_cue_ix), ...
              prt_sep_trajectories{i_tt,i_quantile}(2,1:pre_cue_ix),'linewidth',2,'color',[.5 .5 .5]) 
        % cue
        plot(prt_sep_trajectories{i_tt,i_quantile}(1,pre_cue_ix:pre_cue_ix + median_cue_len), ...
              prt_sep_trajectories{i_tt,i_quantile}(2,pre_cue_ix:pre_cue_ix + median_cue_len),'linewidth',2,'color',[.2 .7 .2]) 
        % trial 
        plot(prt_sep_trajectories{i_tt,i_quantile}(1,pre_cue_ix + median_cue_len:end-post_leave_ix), ...
              prt_sep_trajectories{i_tt,i_quantile}(2,pre_cue_ix + median_cue_len:end-post_leave_ix),'linewidth',2,'color',tt_quantile_colors{i_rewsize}(i_quantile,:)) 
    end
end

%% 2a) Bootstrap to find SEM of quantified dynamics 
%  - Subsample trials to get nBootstraps x nTrialtypes cell array of means
%  - How many trials to choose?
trialtypes = [10 11 20 22 40 44]; 
nSamples = 500; 
sample_size = 50; 
tt_lens = round(pre_cue_ix + median_cue_len + sec1plus_median_lens + post_leave_ix + 50); 
dist_from_init_bootstrap = cell(numel(trialtypes),1);
manifold = pooled_RXNil_pca{3}(1:3,:)'; 
manifold_closest = cell(numel(trialtypes),1);
grad_bootstrap = cell(numel(trialtypes),1);
angle_bootstrap = cell(numel(trialtypes),1);
pc1_bootstrap = cell(numel(trialtypes),1);
pcs = 1:3; 
f = waitbar(0,'Drawing bootstrapped trajectory quantifications');
for i_tt = 1:numel(trialtypes) 
    tt = trialtypes(i_tt);
    this_t_len = tt_lens(i_tt); 
    these_trials = find(pooled_RXNil == tt);  
    these_nanPadded_trials = pooled_nanPadded_trials(these_trials); 
    
    % just going to end up w/ these metrics.. throw out actual sampled data 
    dist_from_init_bootstrap{i_tt} = nan(nSamples,this_t_len); 
    manifold_closest{i_tt} = nan(nSamples,this_t_len); 
    grad_bootstrap{i_tt} = nan(nSamples,this_t_len); 
    angle_bootstrap{i_tt} = nan(nSamples,this_t_len); 
    pc1_bootstrap{i_tt} = nan(nSamples,this_t_len); 
    
    trialrange = 1:length(these_trials); 
    
    for b = 1:nSamples 
        sample_trial_ix = randi(length(these_trials),sample_size,1);
        sample_trials = these_nanPadded_trials(sample_trial_ix); 
        mean_sample_trial = nanmean(cat(3,sample_trials{:}),3);  
        mean_sample_trial(all(isnan(mean_sample_trial),2),:) = 0; 
        mean_sample_trial_norm = (mean_sample_trial - means) ./ stds;
        sample_pca = coeff(:,1:10)' * mean_sample_trial_norm;
        
        % now calculate metrics
        pt0 = sample_pca(pcs,1);
        dist_from_init_bootstrap{i_tt}(b,:) = vecnorm(sample_pca(pcs,1:end) - pt0,2);
        manifold_closest{i_tt}(b,:) = dsearchn(manifold,sample_pca(pcs,:)');
        grad_tt = vecnorm(gradient(sample_pca(pcs,1:end)),2,1); % vecnorm(gradient(sample_pca(pcs,1:end)));
        grad_tt_justTrial = grad_tt(median_cue_len+1:end);
        grad_tt = [grad_tt(1:median_cue_len) grad_tt_justTrial]; 
        grad_bootstrap{i_tt}(b,:) = smoothdata(grad_tt,'gaussian',1); % apply some smoothing
        % last, check out evolution of angle to original state
        angle = acos(pt0' * sample_pca(pcs,1:end) ./ (norm(pt0)*vecnorm(sample_pca(pcs,1:end))));
        angle_bootstrap{i_tt}(b,:) = real(angle) * 180 / pi; % convert to degrees 
        pc1_bootstrap{i_tt}(b,:) = sample_pca(1,:);
        % 
        waitbar((b + (i_tt-1)*nSamples) / (numel(trialtypes)*nSamples),f) 
    end 
end 

close(f);

%% A better metric of trajectory speed

these_trials1nil = find(pooled_RXNil == 10);
these_nanPadded_trials1nil = pooled_nanPadded_trials(these_trials1nil);
these_trials2nil = find(pooled_RXNil == 20);
these_nanPadded_trials2nil = pooled_nanPadded_trials(these_trials2nil);
these_trials4nil = find(pooled_RXNil == 40);
these_nanPadded_trials4nil = pooled_nanPadded_trials(these_trials4nil);

bootstrap_12ix = nan(nSamples,size(pooled_RXNil_pca{1},2));
bootstrap_24ix = nan(nSamples,size(pooled_RXNil_pca{5},2));
bootstrap_12dist = nan(nSamples,size(pooled_RXNil_pca{1},2));
bootstrap_24dist = nan(nSamples,size(pooled_RXNil_pca{5},2));

nSamples = 500; 
sample_size = 50; 
f = waitbar(0,'Drawing bootstrapped trajectory quantifications');
for b = 1:nSamples
    % 1nil
    sample_trial_ix1nil = randi(length(these_trials1nil),sample_size,1);
    sample_trials1nil = these_nanPadded_trials1nil(sample_trial_ix1nil);
    mean_sample_trial1nil = nanmean(cat(3,sample_trials1nil{:}),3);
    mean_sample_trial1nil(all(isnan(mean_sample_trial1nil),2),:) = 0;
    mean_sample_trial_norm1nil = (mean_sample_trial1nil - means) ./ stds;
    sample_pca1nil = coeff(:,1:10)' * mean_sample_trial_norm1nil;
    % 2nil
    sample_trial_ix2nil = randi(length(these_trials2nil),sample_size,1);
    sample_trials2nil = these_nanPadded_trials2nil(sample_trial_ix2nil);
    mean_sample_trial2nil = nanmean(cat(3,sample_trials2nil{:}),3);
    mean_sample_trial2nil(all(isnan(mean_sample_trial2nil),2),:) = 0;
    mean_sample_trial_norm2nil = (mean_sample_trial2nil - means) ./ stds;
    sample_pca2nil = coeff(:,1:10)' * mean_sample_trial_norm2nil;
    % 4nil
    sample_trial_ix4nil = randi(length(these_trials4nil),sample_size,1);
    sample_trials4nil = these_nanPadded_trials4nil(sample_trial_ix4nil);
    mean_sample_trial4nil = nanmean(cat(3,sample_trials4nil{:}),3);
    mean_sample_trial4nil(all(isnan(mean_sample_trial4nil),2),:) = 0;
    mean_sample_trial_norm4nil = (mean_sample_trial4nil - means) ./ stds;
    sample_pca4nil = coeff(:,1:10)' * mean_sample_trial_norm4nil;
    
    % bootstrap closest points
    [bootstrap_12ix(b,:),bootstrap_12dist(b,:)] = dsearchn(sample_pca2nil(1:3,:)',sample_pca1nil(1:3,:)');
    [bootstrap_24ix(b,:),bootstrap_24dist(b,:)]  = dsearchn(sample_pca2nil(1:3,:)',sample_pca4nil(1:3,:)');
    waitbar(b/nSamples,f) 
end
close(f); 

%% 2c) Plot errorbar Now visualize dynamics with errorbars

% close all
tt_lens = cellfun(@(x) size(x,2),pooled_RXNil_pca);
tt_justTrial_lens = tt_lens - median_cue_len - pre_cue_ix - post_leave_ix; % just stretch this part
median_justTrial_len = floor(median(tt_justTrial_lens)); 
vis_ix = 150; % just look at first second
% vis_trial_grps = {1:2 3:4 5:6};
% vis_trial_grps = {1 2 3 4 5 6};
vis_trial_grps = {1 3 5};
% colors_tt = {[0 1 1],[.5 .5 1],[1 0 1]}; % {[.5 1 1],[0 1 1],[.75 .75 1],[.5 .5 1],[1 .5 1],[1 0 1]}; 
colors_tt = {[.5 1 1],[0 1 1],[.75 .75 1],[.5 .5 1],[1 .5 1],[1 0 1]};  

var_to_plot = pc1_bootstrap; 
var_to_plot_name = "PC1"; 

figure()
subplot(1,2,1);hold on
for i_tt = 1:6
    % distance from initial point
%     dist_from_init_tt = dist_from_init_bootstrap{i_tt}; 
%     mean_dist_from_init = mean(dist_from_init_tt,1);
%     ci_dist_from_init = 1.96 * std(dist_from_init_tt,[],1);
    var_to_plot_tt = var_to_plot{i_tt}; 
    mean_var_to_plot = mean(var_to_plot_tt,1);
    ci_var_to_plot = 1.96 * std(var_to_plot_tt,[],1);
%     ci_dist_from_init(1:end-2) = 0;
    % gradient
    grad_tt = grad_bootstrap{i_tt}; 
    mean_grad = mean(grad_tt,1);
    ci_grad = 1.95 * std(grad_tt,[],1); 
    % angle from init point 
    mean_angle_tt = angle_bootstrap{i_tt}; 
    mean_angle = mean(mean_angle_tt,1); 
    ci_angle = 1.95 * std(mean_angle_tt,[],1);
    
    tt_len = length(mean_var_to_plot((median_cue_len+pre_cue_ix):end));
    
%     shadedErrorBar(1:pre_cue_ix,mean_dist_from_init(1:pre_cue_ix),ci_dist_from_init(1:pre_cue_ix),'lineProps',{'color',[.5 .5 .5],'linewidth',2.5})
%     shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_dist_from_init(pre_cue_ix:pre_cue_ix + median_cue_len),ci_dist_from_init(pre_cue_ix:pre_cue_ix + median_cue_len),'lineProps',{'color',[.2 .7 .2],'linewidth',2.5}) 
%     shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_dist_from_init((pre_cue_ix+median_cue_len):end),ci_dist_from_init((pre_cue_ix+median_cue_len):end),'lineProps',{'color',colors_tt{i_tt},'linewidth',2.5}) 
    
    h = plot(1:pre_cue_ix,mean_var_to_plot(1:pre_cue_ix),'color',[.5 .5 .5],'linewidth',2.5);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    h = plot(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_var_to_plot(pre_cue_ix:pre_cue_ix + median_cue_len),'color',[.2 .7 .2],'linewidth',2.5); 
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    plot((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_var_to_plot((pre_cue_ix+median_cue_len):end),'color',colors_tt{i_tt},'linewidth',2.5) 
    h = errorbar(pre_cue_ix+median_cue_len + tt_len-1,mean_var_to_plot(end),ci_var_to_plot(end),'linewidth',1.5,'MarkerFaceColor','k'); 
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    h.Color = colors_tt{i_tt};
end  

max_t_len = max(cellfun(@(x) size(x,2),dist_from_init_bootstrap));
patch_onset = max_t_len - median_cue_len - pre_cue_sec / tbin_ms * 1000; 

xticks([(pre_cue_sec/tbin_sec) patch_onset_ix:50:max_t_len])
xticklabels(([-(median_cue_len) (patch_onset_ix:50:max_t_len) - patch_onset_ix]) * tbin_sec)
xlabel("Time since patch onset (sec)")
ylabel(sprintf("%s (A.U.)",var_to_plot_name))
title("Quantified RXNil Trial Dynamics")
legend("10Nil","11Nil","20Nil","22Nil","40Nil","44Nil")
set(gca,'FontSize',14)

max_t_len = max(cellfun(@(x) size(x,2),dist_from_init_bootstrap));
patch_onset = max_t_len - median_cue_len - pre_cue_sec / tbin_ms * 1000; 

xticks([(pre_cue_sec/tbin_sec) patch_onset_ix:50:max_t_len])
xticklabels(([-(median_cue_len) (patch_onset_ix:50:max_t_len) - patch_onset_ix]) * tbin_sec)
xlabel("Time since patch onset (sec)")
ylabel("Distance From ITI State (A.U.)")
set(gca,'FontSize',13)

% then separate traces by reward size to show reward effect
subplot(1,2,2)
offsets = 20 * [-1 -1 0 0 1 1];
patch_onset_ix = median_cue_len + pre_cue_sec / tbin_ms * 1000 + 1;
for i_tt = 1:6
    % distance from initial point
    var_to_plot_tt = var_to_plot{i_tt}; 
    mean_var_to_plot = mean(var_to_plot_tt,1);
    mean_var_to_plot(patch_onset_ix:end) = mean_var_to_plot(patch_onset_ix:end) + offsets(i_tt);
    ci_var_to_plot = 1.96 * std(var_to_plot_tt,[],1);
    
    shadedErrorBar(1:pre_cue_ix,mean_var_to_plot(1:pre_cue_ix),ci_var_to_plot(1:pre_cue_ix),'lineProps',{'color',[.5 .5 .5],'linewidth',1.5})
    shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_var_to_plot(pre_cue_ix:pre_cue_ix + median_cue_len),ci_var_to_plot(pre_cue_ix:pre_cue_ix + median_cue_len),'lineProps',{'color',[.2 .7 .2],'linewidth',1.5}) 
    tt_len = length(mean_var_to_plot((median_cue_len+pre_cue_ix):end));
    shadedErrorBar((pre_cue_ix+median_cue_len) + (1:tt_len-1),mean_var_to_plot((pre_cue_ix+median_cue_len+1):end),ci_var_to_plot((pre_cue_ix+median_cue_len+1):end),'lineProps',{'color',colors_tt{i_tt},'linewidth',1.5}) 
    set(gca,'FontSize',14)
end  
xticks([(pre_cue_sec/tbin_sec) patch_onset_ix:50:max_t_len])
xticklabels(([-(median_cue_len) (patch_onset_ix:50:max_t_len) - patch_onset_ix]) * tbin_sec)
xlabel("Time since patch onset (sec)")
ylabel(sprintf("%s (A.U.)",var_to_plot_name))
title("Quantified RXNil Trial Dynamics")
set(gca,'FontSize',14)

% mean_bootstrap_12ix = mean(bootstrap_12ix); 
% ci_bootstrap_12ix = 1.96 * std(bootstrap_12ix); 
% mean_bootstrap_24ix = mean(bootstrap_24ix); 
% ci_bootstrap_24ix = 1.96 * std(bootstrap_24ix); 
% 
% mean_bootstrap_12dist = mean(bootstrap_12dist); 
% ci_bootstrap_12dist = 1.96 * std(bootstrap_12dist); 
% mean_bootstrap_24dist = mean(bootstrap_24dist); 
% ci_bootstrap_24dist = 1.96 * std(bootstrap_24dist); 
% 
% subplot(3,1,2);hold on; grid()
% shadedErrorBar(1:pre_cue_ix,mean_bootstrap_12ix(1:pre_cue_ix),ci_bootstrap_12ix(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
% shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_12ix(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_12ix(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
% tt_len = length(mean_bootstrap_12ix((median_cue_len+pre_cue_ix):end));
% shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_12ix((pre_cue_ix+median_cue_len):end),ci_bootstrap_12ix((pre_cue_ix+median_cue_len):end),'lineprops',{'color',colors{1},'linewidth',1.5})
% 
% shadedErrorBar(1:pre_cue_ix,mean_bootstrap_24ix(1:pre_cue_ix),ci_bootstrap_24ix(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
% shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_24ix(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_24ix(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
% tt_len = length(mean_bootstrap_24ix((median_cue_len+pre_cue_ix):end));
% shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_24ix((pre_cue_ix+median_cue_len):end),ci_bootstrap_24ix((pre_cue_ix+median_cue_len):end),'lineprops',{'color',colors{5},'linewidth',1.5})
% 
% xticks(0:25:length(mean_angle))
% xticklabels((0:25:length(mean_dist_from_init))*tbin_sec - (pre_cue_sec))
% yticks(0:50:length(mean_angle))
% yticklabels((0:50:length(mean_dist_from_init))*tbin_sec - (pre_cue_sec))
% xlabel("Time since Cue Onset (sec)")
% ylabel("Closest timepoint")
% h = refline(1,0); 
% h.Color = colors{3}; 
% h.LineStyle = '-'; 
% h.LineWidth = 2;
% set(gca,'FontSize',13)

%% Calculate p-values for the distance after 500ms post rew and dist at patch leave
p_end = nan(6,6); 
p_rew = nan(6,6); 
patch_onset_ix = median_cue_len + pre_cue_sec / tbin_ms * 1000 + 1;
t_post_rew = 500; 
ix_post_rew = 500 / tbin_ms; 
for i_tt1 = 1:6
    for i_tt2 = 1:6
        % distance from initial point
        dist_from_init_tt1 = dist_from_init_bootstrap{i_tt1}; % [n_bootstrap x n_timepoints]
        dist_from_init_tt2 = dist_from_init_bootstrap{i_tt2}; 
        
        % first pull end of trial
        dist_end_tt1 = dist_from_init_tt1(:,end);
        dist_end_tt2 = dist_from_init_tt2(:,end);
        p_end(i_tt1,i_tt2) = length(find(dist_end_tt1 < dist_end_tt2)) / nSamples; 
        
        % then pull t_post_rew ms post reward
        dist_rew_tt1 = dist_from_init_tt1(:,patch_onset_ix + 1000 / tbin_ms + ix_post_rew);
        dist_rew_tt2 = dist_from_init_tt2(:,patch_onset_ix + 1000 / tbin_ms + ix_post_rew);
        p_rew(i_tt1,i_tt2) = length(find(dist_rew_tt1 < dist_rew_tt2)) / nSamples; 
    end
end

%% take mean and std error of mean for "KiNeT" quantifications
mean_bootstrap_12ix = mean(bootstrap_12ix); 
ci_bootstrap_12ix = 1.96 * std(bootstrap_12ix); 
mean_bootstrap_24ix = mean(bootstrap_24ix); 
ci_bootstrap_24ix = 1.96 * std(bootstrap_24ix); 

mean_bootstrap_12dist = mean(bootstrap_12dist); 
ci_bootstrap_12dist = 1.96 * std(bootstrap_12dist); 
mean_bootstrap_24dist = mean(bootstrap_24dist); 
ci_bootstrap_24dist = 1.96 * std(bootstrap_24dist); 

figure() 
subplot(2,1,1);hold on
shadedErrorBar(1:pre_cue_ix,mean_bootstrap_12ix(1:pre_cue_ix),ci_bootstrap_12ix(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_12ix(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_12ix(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
tt_len = length(mean_bootstrap_12ix((median_cue_len+pre_cue_ix):end));
shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_12ix((pre_cue_ix+median_cue_len):end),ci_bootstrap_12ix((pre_cue_ix+median_cue_len):end),'lineprops',{'color',[0 1 1],'linewidth',1.5})

shadedErrorBar(1:pre_cue_ix,mean_bootstrap_24ix(1:pre_cue_ix),ci_bootstrap_24ix(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_24ix(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_24ix(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
tt_len = length(mean_bootstrap_24ix((median_cue_len+pre_cue_ix):end));
shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_24ix((pre_cue_ix+median_cue_len):end),ci_bootstrap_24ix((pre_cue_ix+median_cue_len):end),'lineprops',{'color',[1 0 1],'linewidth',1.5})

xticks(0:25:length(mean_angle))
xticklabels((0:25:length(mean_dist_from_init))*tbin_sec - (pre_cue_sec))
yticks(0:25:length(mean_angle))
yticklabels((0:25:length(mean_dist_from_init))*tbin_sec - (pre_cue_sec))
xlabel("Time since Cue Onset (sec)")
ylabel(sprintf("Time of Closest Point \n on 2 uL Trajectory"))
h = refline(1,0); 
h.Color = [.4 .4 .4]; 
h.LineStyle = '--'; 
h.LineWidth = 1.5;
set(gca,'fontsize',14)
% 
subplot(2,1,2);hold on

shadedErrorBar(1:pre_cue_ix,mean_bootstrap_12dist(1:pre_cue_ix),ci_bootstrap_12dist(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_12dist(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_12dist(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
tt_len = length(mean_bootstrap_12dist((median_cue_len+pre_cue_ix):end));
shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_12dist((pre_cue_ix+median_cue_len):end),ci_bootstrap_12dist((pre_cue_ix+median_cue_len):end),'lineprops',{'color',[0 1 1],'linewidth',1.5})
shadedErrorBar(1:pre_cue_ix,mean_bootstrap_24dist(1:pre_cue_ix),ci_bootstrap_24dist(1:pre_cue_ix),'lineprops',{'color',[.5 .5 .5],'linewidth',1.5})
shadedErrorBar(pre_cue_ix:(pre_cue_ix+median_cue_len),mean_bootstrap_24dist(pre_cue_ix:pre_cue_ix + median_cue_len),ci_bootstrap_24dist(pre_cue_ix:pre_cue_ix + median_cue_len),'lineprops',{'color',[.2 .7 .2],'linewidth',1.5})
tt_len = length(mean_bootstrap_24dist((median_cue_len+pre_cue_ix):end));
shadedErrorBar((pre_cue_ix+median_cue_len) + (0:tt_len-1),mean_bootstrap_24dist((pre_cue_ix+median_cue_len):end),ci_bootstrap_24dist((pre_cue_ix+median_cue_len):end),'lineprops',{'color',[1 0 1],'linewidth',1.5})
ylabel(sprintf("Distance to Closest Point \n on 2 uL Trajectory (A.U.)"))
xticks(0:25:length(mean_angle))
xticklabels((0:25:length(mean_dist_from_init))*tbin_sec - (pre_cue_sec))
xlabel("Time since Cue Onset (sec)")
set(gca,'fontsize',14)

% Now add significance bars using pvalue from bootstrap
% grad_all_tts = grad_bootstrap(cell2mat(vis_trial_grps));
% tt_lens = cellfun(@(x) size(x,2), grad_all_tts);
% grad_all_tts = cellfun(@(x) x(:,1:min(tt_lens)),grad_all_tts,'un',0); 
% % grad_all_tts = cat(1,grad_all_tts{:}); 
% tt_labels = arrayfun(@(x) x+zeros(nSamples,1),cell2mat(vis_trial_grps),'un',0);
% tt_labels = cat(1,tt_labels{:});
% p_value = nan(min(tt_lens),2); 
% for i_tt = 2:3 
%     diff = grad_all_tts{1} - grad_all_tts{i_tt};
%     for t = 1:min(tt_lens) 
%         p_value(t,i_tt-1) = length(find(diff(:,t) > 0)) / nSamples;
%     end
% end

%% Visualize contribution of different clusters to coeffs matrix 
gmm_colors = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255;
cluster_means = nan(5,3); 
cluster_sems = nan(5,3); 
for i_cluster = 1:5 
    cluster_means(i_cluster,:) = median(coeff(pooled_GLM_cluster_selection == i_cluster,1:3));
    cluster_sems(i_cluster,:) = 1.96 * std(coeff(pooled_GLM_cluster_selection == i_cluster,1:3)) / sqrt(length(find(pooled_GLM_cluster_selection == i_cluster) ));
end
figure()
% set(gca,'colororder',gmm_colors)
b = bar(cluster_means','facecolor','flat');hold on

colororder(gmm_colors)
b = bar(cluster_means');
x = arrayfun(@(i_feature) b(i_feature).XEndPoints,1:5,'un',0);
x = cat(2,x{:});
cluster_means = cluster_means';
cluster_sems = cluster_sems';
errorbar(x,cluster_means(:),cluster_sems(:),'k.','linewidth',1.5)
ylabel("PC Weight")
xticklabels(["PC 1","PC 2","PC 3"])
set(gca,'fontsize',14)




