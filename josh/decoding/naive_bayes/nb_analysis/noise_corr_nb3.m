%% Analyze correlation in decoded time between cluster 1 and cluster 2
%  Only analyze sessions w/ good cluster 1 and cluster 2 decoding (20+ cells) 

paths = struct;
paths.nb_results = './structs/nb_results14_Mar_2021.mat';
load(paths.nb_results); 
paths.neuro_data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% close all
paths.beh_data = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));

paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC.mat';
load(paths.sig_cells);  

sessions = dir(fullfile(paths.beh_data,'*.mat'));
sessions = {sessions.name};  

% to pare down to just recording sessions
recording_sessions = dir(fullfile(paths.neuro_data,'*.mat'));
recording_sessions = {recording_sessions.name};
% to just use recording sessions
recording_session_bool = cellfun(@(x) ismember(x,recording_sessions),sessions);
sessions = sessions(recording_session_bool);
tbin_ms = .02 * 1000;

mouse_grps = {1:2,3:8,10:13,14:18,[23 25]}; 
pool_sessions = nb_results.clu12_pool_sessions; % only use good clu1 and clu2 sessions
analysis_mice = find(~cellfun(@isempty ,pool_sessions));
analysis_sessions = arrayfun(@(i_mouse) mouse_grps{i_mouse}(nb_results.clu12_pool_sessions{i_mouse}),1:length(mouse_grps),'un',0);
mouse_names = ["m75","m76","m78","m79","m80"];
session_titles = cell(numel(analysis_sessions),1);
for mIdx = 1:numel(analysis_sessions)
    for i = 1:numel(analysis_sessions{mIdx})
        sIdx = analysis_sessions{mIdx}(i);
        session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];
        session_titles{mIdx}{i} = session_title;
    end
end  

feature_names = cellfun(@(x) x.name, nb_results.dataset_opt.features{5}{1});
var_dt = diff(nb_results.var_bins{1}{1}{1}(1:2));
y_true = nb_results.y_true;

%% Load decoded time since and time on patch from clusters 1 and 2

y_hat = nb_results.y_hat;
nMice = numel(analysis_sessions); 

timeSince_hat = cell(nMice,1); 
timePatch_hat = cell(nMice,1); 

% to load from decoding directly
timeSince_ix = 1; 
timePatch_ix = 2; 

% these are just from the pop_decoding_session_sep
rewsizes = [1 2 4];
rewsize = cell(nMice,1); 
RXNil = cell(nMice,1); 

% choose which features to reformat for analysis
trial_decoding_features = 1:2;

for mIdx = 1:numel(analysis_sessions)
    for i_i = 1:numel(analysis_sessions{mIdx})
        i = analysis_sessions{mIdx}(i_i); 
        within_mouse_ix = nb_results.pool_sessions{mIdx}(i_i); % session within mouse mPFC sessions
        sIdx = analysis_sessions{mIdx}(i_i);  
        session = sessions{sIdx}(1:end-4); 
        
        data = load(fullfile(paths.beh_data,sessions{sIdx}));  
        session_title = session([1:2 end-2:end]);
        session_rewsize = mod(data.patches(:,2),10); 
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);   
        rew_sec = data.rew_ts;
        session_prts = patchleave_sec - patchstop_sec;  
        nTrials = length(session_rewsize);
        
        % Make RXNil vector
        session_RXNil = nan(nTrials,1); 
        last_rew_sec = nan(nTrials,1); 
        i_last_rew_ix = nan(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
            if session_prts(iTrial) >= 1 % only care if we have at least 1 second on patch
                if isequal(rew_indices,1)
                    session_RXNil(iTrial) = 10*session_rewsize(iTrial);
                elseif isequal(rew_indices,[1 ; 2])
                    session_RXNil(iTrial) = 10*session_rewsize(iTrial) + session_rewsize(iTrial);
                end
            end
            last_rew_sec(iTrial) = rew_indices(end);
            i_last_rew_ix(iTrial) = round(((rew_indices(end)-1) * 1000) / tbin_ms);
        end
        session_postRew_prts = session_prts - last_rew_sec; 
        i_last_rew_ix(i_last_rew_ix == 0) = 1; 
    
        % reformat decoded time
        for i_feature = 1:numel(trial_decoding_features)
            iFeature = trial_decoding_features(i_feature);
            timeSince_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeSince_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            timePatch_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timePatch_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
        end

        % log trial/behavior information
        rewsize{mIdx}{i_i} = session_rewsize;
        RXNil{mIdx}{i_i} = session_RXNil; 
    end
end 
clear y_hat % we now have this in an easier form to work with

%% 1) Visualize noise correlation basic w/ heatmap

% First, just visualize predictions across trials
decoded_time_hat = timePatch_hat; % which time decoding are we analyzing
time_true_ix = timePatch_ix; 
analyze_ix = round(1000 / tbin_ms); % what time window
close all
for m_ix = 1:numel(analysis_mice) 
    mIdx = analysis_mice(m_ix); 
    % gather decoded time variable from clusters 1 and 2
    mouse_cluster1time = cellfun(@(x) x{1}, decoded_time_hat{mIdx},'un',0);
    mouse_cluster1time = cat(1,mouse_cluster1time{:});
    mouse_cluster2time = cellfun(@(x) x{2}, decoded_time_hat{mIdx},'un',0);
    mouse_cluster2time = cat(1,mouse_cluster2time{:});
    % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
    pad_cluster1time = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_cluster1time,'un',0);
    mouse_cluster1time = var_dt * cat(1,pad_cluster1time{:}); % now this is a matrix
    pad_cluster2time = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_cluster2time,'un',0);
    mouse_cluster2time = var_dt * cat(1,pad_cluster2time{:}); % now this is a matrix
    % also get true time
    y_true_tmp = y_true{mIdx}(pool_sessions{mIdx},time_true_ix);
    true_time = cat(1,y_true_tmp{:});
    pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
    true_time = var_dt * cat(1,pad_trueTime{:});

    subplot(numel(analysis_mice),3,1 + 3 * (m_ix - 1));
    imagesc(true_time); caxis([0 1]) 
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Time on Patch",mouse_names(mIdx)))
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
    subplot(numel(analysis_mice),3,2 + 3 * (m_ix - 1));
    imagesc(mouse_cluster1time); % - mean(mouse_cluster1time,1)); caxis([0 1]) 
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Cluster1-Decoded Time on Patch",mouse_names(mIdx)))
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
    subplot(numel(analysis_mice),3,3 + 3 * (m_ix - 1));
    imagesc(mouse_cluster2time); %  - mean(mouse_cluster2time,1));
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Cluster2-Decoded Time on Patch",mouse_names(mIdx))) 
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
end

%% 2) Compare noise correlation to shuffled data 
decoded_time_hat = timePatch_hat_sameRewsize; % which time decoding are we analyzing
true_time_trials = timePatch_true; 
analyze_ix = round(2000 / tbin_ms); % what time window
n_shuffles = 1000; 

trialtypes = [10 20 40];  

for mIdx = 1:numel(mouse_grps)
    % load RX for pooled sessions
    mouse_RX = RX{mIdx}(pool_sessions{mIdx});
    mouse_RX = cat(1,mouse_RX{:});
    % gather decoded time variable from clusters 1 and 2
    mouse_cluster2time = cellfun(@(x) x{2}, decoded_time_hat{mIdx}(pool_sessions{mIdx}),'un',0);
    mouse_cluster2time = cat(1,mouse_cluster2time{:});
    mouse_cluster3time = cellfun(@(x) x{3}, decoded_time_hat{mIdx}(pool_sessions{mIdx}),'un',0);
    mouse_cluster3time = cat(1,mouse_cluster3time{:});
    % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
    pad_cluster2time = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_cluster2time,'un',0);
    mouse_cluster2time = bin_dt * cat(1,pad_cluster2time{:}); % now this is a matrix
    pad_cluster3time = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_cluster3time,'un',0);
    mouse_cluster3time = bin_dt * cat(1,pad_cluster3time{:}); % now this is a matrix
    % also get true time
    true_time = true_time_trials{mIdx}(pool_sessions{mIdx});
    true_time = cat(1,true_time{:});
    pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
    true_time = cat(1,pad_trueTime{:});
    
    for i_tt = 1:numel(trialtypes)
        these_trials = find(mouse_RX == trialtypes(i_tt));
        nTrials_tt = length(these_trials); 
        % gather decoded time from two populations
        cluster2time_tt = mouse_cluster2time(these_trials,:)'; 
        cluster3time_tt = mouse_cluster3time(these_trials,:)'; 
        time_true_tt = true_time(these_trials,:)'; 
        
        % concatenate trials 
        cat_cluster2_time_tt = cluster2time_tt(:); 
        cat_cluster3_time_tt = cluster3time_tt(:);
        non_nan_ix = ~isnan(cat_cluster2_time_tt); 
        r = corrcoef(cat_cluster2_time_tt(non_nan_ix),cat_cluster3_time_tt(non_nan_ix));  
        r_unshuffled_tt = r(2); 
        
        r_shuffled_tt = nan(n_shuffles,1); 
        % now get distribution of correlation coefficients shuffling trials (within trial type)
        for shuffle = 1:n_shuffles 
            cluster2time_tt = mouse_cluster2time(these_trials(randperm(nTrials_tt)),:)';  
            cluster3time_tt = mouse_cluster3time(these_trials(randperm(nTrials_tt)),:)';  
            cat_cluster2time_tt = cluster2time_tt(:); 
            cat_cluster3time_tt = cluster3time_tt(:);  
            non_nan_ix = ~isnan(cat_cluster2time_tt) & ~isnan(cat_cluster3time_tt); 
            r = corrcoef(cat_cluster2time_tt(non_nan_ix),cat_cluster3time_tt(non_nan_ix));  
            r_shuffled_tt(shuffle) = r(2); 
        end 
        subplot(numel(trialtypes),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on
        histogram(r_shuffled_tt) 
        xline(r_unshuffled_tt,'k--','linewidth',1.5) 
        if mIdx == 1 
            ylabel(sprintf("%i Trials",trialtypes(i_tt)))
        end 
        if i_tt == numel(trialtypes) 
            xlabel("Cluster 2-3 Decoding PearsonR")
        end 
        if i_tt == 1
            title(mouse_names(mIdx))
        end
    end
end


