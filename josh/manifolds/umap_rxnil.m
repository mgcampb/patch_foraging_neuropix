%% Moving RX-Nil into UMAP to make the point that we have low-D embedding dimensionality
%% First load up PCA to show why we need this; this is just copied from RX_stretch2.m 

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
zscored_prts = cell(numel(mouse_grps),1); 
trialtypes = [10 11 20 22 40 44]; % for zscored PRT

for mIdx = 1:numel(mouse_grps)
    fr_mat_trialed{mIdx} = cell(numel(mouse_grps{mIdx}),5); % 5 segments per trial  
    prts{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    RXNil{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    GLM_cluster{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    transient_peak{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    zscored_prts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
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
        
        % zscore PRT within trial type
        zscored_prts{mIdx}{i} = nan(nTrials,1); 
        for i_tt = 1:numel(trialtypes) 
            tt = trialtypes(i_tt); 
            zscored_prts{mIdx}{i}(RXNil{mIdx}{i} == tt) = zscore(prts{mIdx}{i}(RXNil{mIdx}{i} == tt)); 
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
end

%% Perform PCA across mice

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

lines2 = lines(2);
% elbow plot
n_to_plot = 20; 
figure()
% plot(cumsum(explained(1:10)),'linewidth',1.5);hold on
% scatter(1:10,cumsum(explained(1:10)),[],lines2(1,:),'linewidth',1.5)
plot((explained(1:n_to_plot)),'linewidth',1.5);hold on
scatter(1:n_to_plot,(explained(1:n_to_plot)),[],lines2(1,:),'linewidth',1.5)
title("Variance explained")
ylabel("Variance explained per component (%)") 
xlabel("Principal Component")
% ylim([0 100])
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
tts = [1:6];
figure()
rxnil_dynamics_plot3(pooled_RXNil_pca,tts,colors,pre_cue_ix,median_cue_len,post_leave_ix)
title("R0Nil Trial Population Dynamics")
xlabel("PC1")
ylabel("PC2")
% zlabel("PC3")
set(gca,'fontsize',15)

%% Pad trials per session with zeros so that we can bootstrap and see trial-by-trial variability in initial point

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
trial_types_trials = cell(numel(pool_mice),1); % rxnil trial type
zscored_prt_trials = cell(numel(pool_mice),1); % zscored prt
mouseID_trials = cell(numel(pool_mice),1); % mouse ID
session_counter = 1; 
for m_ix = 1:numel(pool_mice) 
    mIdx = pool_mice(m_ix); 
    pca_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1); 
    nanPadded_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1); 
    zeroPadded_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    time_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    trial_types_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    zscored_prt_trials{m_ix} = cell(numel(pool_mice),1); % zscored prt
    mouseID_trials{m_ix} = cell(numel(xMice_all_tt_sessions{mIdx}),1);  
    for i_i = 1:numel(xMice_all_tt_sessions{mIdx})
        i = xMice_all_tt_sessions{mIdx}(i_i);  
        nTrials = length(RXNil{mIdx}{i}); 
        nNeurons = s_nNeurons(session_counter);  
        pca_trials{m_ix}{i_i} = cell(nTrials,1); 
        nanPadded_trials{m_ix}{i_i} = cell(nTrials,1); 
        zeroPadded_trials{m_ix}{i_i} = cell(nTrials,1); 
        time_trials{m_ix}{i_i} = cell(nTrials,1); 
        trial_types_trials{m_ix}{i_i} = cell(nTrials,1); 
        zscored_prt_trials{m_ix}{i_i} = cell(nTrials,1); 
        mouseID_trials{m_ix}{i_i} = cell(nTrials,1); 
        
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
                tt_sec1plus_median_len = sec1plus_median_lens(tt);  
                
                 % collect fr_mats from 5 trial periods
                iTrial_precue_fr_mat = fr_mat_trialed{mIdx}{i,1}{iTrial}(i_these_neurons,:);
                iTrial_cue_fr_mat = imresize(fr_mat_trialed{mIdx}{i,2}{iTrial}(i_these_neurons,:),[nNeurons,median_cue_len]);
                iTrial_sec0_fr_mat = fr_mat_trialed{mIdx}{i,3}{iTrial}(i_these_neurons,:); 
                iTrial_sec1plus_fr_mat = imresize(fr_mat_trialed{mIdx}{i,4}{iTrial}(i_these_neurons,:),[nNeurons,tt_sec1plus_median_len]);
                iTrial_postleave_fr_mat = fr_mat_trialed{mIdx}{i,5}{iTrial}(i_these_neurons,:);

                % concatenate to make full trials
                iTrial_fr_mat_full = cat(2,iTrial_precue_fr_mat,iTrial_cue_fr_mat,iTrial_sec0_fr_mat,iTrial_sec1plus_fr_mat,iTrial_postleave_fr_mat);
                iTrial_fr_mat_norm = (iTrial_fr_mat_full - s_means) ./ s_stds;
                t_len = size(iTrial_fr_mat_norm,2); 
                time_trials{m_ix}{i_i}{iTrial} = (1:t_len)' * tbin_sec - pre_cue_sec - median_cue_len * tbin_sec; % aligned to patch onset
                
                % pad to do stuff w/ FR later (note no standardization)
                nanPadded_trials{m_ix}{i_i}{iTrial} = nan(total_nNeurons,t_len);  
                zeroPadded_trials{m_ix}{i_i}{iTrial} = zeros(total_nNeurons,t_len);  
                zeroPadded_fr_mat_norm = zeros(total_nNeurons,t_len);  
                nanPadded_trials{m_ix}{i_i}{iTrial}(s_neuron_ix,:) = iTrial_fr_mat_full;
                zeroPadded_trials{m_ix}{i_i}{iTrial}(s_neuron_ix,:) = iTrial_fr_mat_full;
                zeroPadded_fr_mat_norm(s_neuron_ix,:) = iTrial_fr_mat_norm;
                pca_trials{m_ix}{i_i}{iTrial} = coeff(:,1:20)' * zeroPadded_fr_mat_norm; % only really care about 10 PCs
                trial_types_trials{m_ix}{i_i}{iTrial} = tt + zeros(t_len,1); 
                zscored_prt_trials{m_ix}{i_i}{iTrial} = zscored_prts{mIdx}{i}(iTrial) + zeros(t_len,1);
                mouseID_trials{m_ix}{i_i}{iTrial} = mIdx + zeros(t_len,1); 
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
pooled_trial_types = cat(1,trial_types_trials{:}); 
pooled_trial_types = cat(1,pooled_trial_types{:}); 
pooled_mouseID = cat(1,mouseID_trials{:}); 
pooled_mouseID = cat(1,pooled_mouseID{:}); 

pooled_zscored_prts_timesteps = cat(1,zscored_prt_trials{:}); 
pooled_zscored_prts_timesteps = cat(1,pooled_zscored_prts_timesteps{:});  

pca_full = cat(2,pooled_pca_trials{:});
zeroPadded_fr_mat_full = cat(2,pooled_zeroPadded_trials{:});
nanPadded_fr_mat_full = cat(2,pooled_nanPadded_trials{:});
time_full = cat(1,pooled_time{:})'; 
trial_types_full = cat(1,pooled_trial_types{:})'; 
mouseID_full = cat(1,pooled_mouseID{:})'; 
pooled_zscored_prt_full = cat(1,pooled_zscored_prts_timesteps{:})';

%% Demo to show why this is interesting .. maybe
%  Maybe we don't even need to get out of avg space, this is not beautiful
manifold_pcs = 1:3;
manifold_tt = 44; 
tt_ix = find(ismember(trialtypes,manifold_tt));
manifold_pca_trials = pooled_pca_trials(ismember(pooled_RXNil,manifold_tt));
manifold_pca_trials = cellfun(@(x) x(manifold_pcs,:),manifold_pca_trials,'un',0); 
manifold = mean(cat(3,manifold_pca_trials{:}),3)'; 

figure();
for this_tt_ix = tt_ix
    plot_these_indices = find(trial_types_full == this_tt_ix); 
    plot_these_indices = plot_these_indices(1:1:end);
    pre_cue_indices = plot_these_indices(time_full(plot_these_indices) < -(pre_cue_ix * tbin_sec)); 
    cue_indices = plot_these_indices(time_full(plot_these_indices) > -(pre_cue_ix * tbin_sec) & time_full(plot_these_indices) < 0); 
    trial_indices = plot_these_indices(time_full(plot_these_indices) > 0 & time_full(plot_these_indices) < post_leave_ix * tbin_sec); 
    colormap([fliplr(.5:0.01:1)' zeros(51,1) fliplr(.5:0.01:1)'])
    %plot traces around
    % scatter3(pca_full(1,plot_these_indices),pca_full(2,plot_these_indices),pca_full(3,plot_these_indices),1,mouseID_full(plot_these_indices),'o');hold on
    scatter3(pca_full(1,pre_cue_indices),pca_full(2,pre_cue_indices),pca_full(3,pre_cue_indices),1,[.75 .75 .75],'o');hold on
    scatter3(pca_full(1,cue_indices),pca_full(2,cue_indices),pca_full(3,cue_indices),1,[.4 .9 .4],'o');hold on
    scatter3(pca_full(1,trial_indices),pca_full(2,trial_indices),pca_full(3,trial_indices),1,time_full(trial_indices),'o');hold on
    % plot mean
    plot3(manifold(1:pre_cue_ix,1),manifold(1:pre_cue_ix,2),manifold(1:pre_cue_ix,3),'linewidth',3,'color',[.5 .5 .5])
    plot3(manifold(pre_cue_ix:pre_cue_ix + median_cue_len,1),manifold(pre_cue_ix:pre_cue_ix + median_cue_len,2),...
           manifold(pre_cue_ix:pre_cue_ix + median_cue_len,3),'linewidth',3,'color',[.2 .7 .2])
    plot3(manifold(pre_cue_ix + median_cue_len:end-post_leave_ix,1),manifold(pre_cue_ix + median_cue_len:end-post_leave_ix,2),...
           manifold(pre_cue_ix + median_cue_len:end-post_leave_ix,3),'linewidth',3,'color',colors{tt_ix})
end
% xlim([-50,50])

%% Now spin up UMAP

% run umap
% [X_umap,~,clusterID] = run_umap(zscore(roi_pooled_RXNil_mean_full,[],2)','n_components',2,'n_neighbors',30);
n_runs = 1; 
n_neighbors_test = [5 15 25 35 50 60 80 100]; 
n_components = 2;
for i_n_neighbors = 1:numel(n_neighbors_test)
    n_neighbors = n_neighbors_test(i_n_neighbors);
    X_umap = run_umap(zscore(roi_pooled_RXNil_mean_full,[],2)','n_components',n_components,'n_neighbors',n_neighbors,'verbose','none');
%     [X_umap,~,clusterID] = run_umap(zscore(score(:,1:3)),'n_components',2,'n_neighbors',5);
    X_umap_tts = cell(numel(trialtypes),1);

    % divide by trial type
    tt_starts = [0 cumsum(cellfun(@(x) size(x,2), roi_pooled_RXNil_mean))];
    tt_starts(end) = tt_starts(end) - 1;
    for tt = 1:numel(trialtypes)
        X_umap_tts{tt} = X_umap(tt_starts(tt)+1:tt_starts(tt+1),:)';
    end 

    % Now visualize...
    pre_cue_ix = round(pre_cue_sec / tbin_sec); 
    post_leave_ix = round(post_leave_sec / tbin_sec); 

    colors = {[.5 1 1],[0 1 1],[.75 .75 1],[.5 .5 1],[1 .5 1],[1 0 1]}; 

    tts = 1:6;
    subplot(2,4,i_n_neighbors)
    rxnil_dynamics_plot2(X_umap_tts,tts,colors,pre_cue_ix,median_cue_len,post_leave_ix)
    title(sprintf("%i neighbors",n_neighbors))
    set(gca,'fontsize',15)
end

%% Make some synthetic data for UMAP to check out 
%% first a ring w/ linear projection up
noise_mean = 5;
noise_std = 1.5;
n_points = 1000; 
r = 5; R = 20;
[x,y] = gen_ring_data(1000,R,r,noise_mean,noise_std); 
scatter(x,y);

higher_dim = 100;
synth_data = [x y] * 3 * randn(2,100); % project into high-d space

%% UMAP on synthetic data
[X_umap,~,clusterID] = run_umap(synth_data,'n_components',2,'n_neighbors',15); % this isn't doing exactly what we would think... 

%% Visualize UMAP results
figure();
subplot(1,2,1)
scatter(x,y,[],X_umap(:,1))
subplot(1,2,2)
scatter(X_umap(:,1),X_umap(:,2))

%% Next synthetic data w/ low intrinsic dimensionality and high or low-D embedding dimensionality 
n_neurons = 50; 
phi_highD = linspace(0,10,n_neurons); 
phi_lowD = 10 + zeros(n_neurons,1); 
G = 5;
sigma_low = 20;
sigma_high = 1;
n_trials = 10;
S = repmat((1:.1:10)',[1 n_trials])';
X_lowEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_lowD,G,sigma_low,S); 
X_highEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S); 

figure()
subplot(1,2,1)
imagesc(flipud(mean(cat(3,X_lowEmbedding),3)))
subplot(1,2,2)
imagesc(flipud(mean(cat(3,X_highEmbedding),3)))

%% Now perform PCA and UMAP to see what they both discover
X_highD_full = reshape(X_highEmbedding,[n_neurons,size(S,2) * n_trials]);
X_lowD_full = reshape(X_lowEmbedding,[n_neurons,size(S,2) * n_trials]);
X_highD_avg = mean(cat(3,X_highEmbedding),3);
X_lowD_avg = mean(cat(3,X_lowEmbedding),3);

%% First perform PCA 
[~,X_lowD_score] = pca(X_lowD_full');
[~,X_highD_score] = pca(X_highD_full');

%% Then UMAP
X_lowD_umap = run_umap(X_lowD_full','n_components',2,'n_neighbors',15); 
X_highD_umap = run_umap(X_highD_full','n_components',2,'n_neighbors',15); 

% X_highD_umap_avg = run_umap(X_highD_avg','n_components',2,'n_neighbors',15); 
% X_lowD_umap_avg = run_umap(X_lowD_avg','n_components',2,'n_neighbors',15); 
%% Visualize
t = S';
figure()
subplot(2,2,1)
scatter(X_lowD_score(:,1),X_lowD_score(:,2),[],t(:))
xlabel("PC1")
ylabel("PC2")
set(gca,'fontsize',15)
subplot(2,2,2)
scatter(X_highD_score(:,1),X_highD_score(:,2),[],t(:))
xlabel("PC1")
ylabel("PC2")
set(gca,'fontsize',15)
subplot(2,2,3)
scatter(X_lowD_umap(:,1),X_lowD_umap(:,2),[],t(:))
xlabel("UMAP1")
ylabel("UMAP2")
set(gca,'fontsize',15)
subplot(2,2,4)
scatter(X_highD_umap(:,1),X_highD_umap(:,2),[],t(:))
xlabel("UMAP1")
ylabel("UMAP2")
set(gca,'fontsize',15)

% Successful check that running on averaged data works
% subplot(2,2,3)
% scatter(X_lowD_umap_avg(:,1),X_lowD_umap_avg(:,2),[],t(:,1))
% subplot(2,2,4)
% scatter(X_highD_umap_avg(:,1),X_highD_umap_avg(:,2),[],t(:,1))


%% helper functions
function [x,y] = gen_ring_data(n_points,R,r,noise_mean,noise_std)
% Just generate points on ring
Th = rand(n_points,1) * 2 * pi; 

% now we generate n x n matrices for x,y,z according to eqn of torus
x = (R+r.*cos(Th)) + (randn(n_points,1) + noise_mean) * noise_std;
y = (R+r.*sin(Th)) + (randn(n_points,1) + noise_mean) * noise_std; 
end

function R = gen_manifold_embedding_dataset(n_neurons,phi,G,sigma,S) 
    % Generate neural population response from neurons w/ gaussian
    % responsivity
    % ----
    % Arguments
    % phi: stimulus preference per neuron [n_neurons x 1]
    % G: gain
    % S: 1D stimulus [n_trials x t_len]
    % ----
    % Returns
    % R: [n_neurons x t_len x n_trials]
    
    [n_trials,t_len] = size(S); 
    
    R = nan(n_neurons,t_len,n_trials); 
    
    % iterate over trials
    for i_trial = 1:n_trials
        for i_timestep = 1:t_len
            R(:,i_timestep,i_trial) = G * exp(-((S(i_trial,i_timestep) - phi) / (sqrt(2.0 * sigma))).^2);
            R(:,i_timestep,i_trial) = poissrnd(R(:,i_timestep,i_trial)); % add poisson noise
        end
    end
end


