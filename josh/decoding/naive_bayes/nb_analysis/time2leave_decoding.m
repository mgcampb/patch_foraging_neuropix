%% Looking at some details in timecourse of time until leave decoding 
% 1) Visualize time2leave w/ error bars, separated by PRT post last rew 

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

% mPFC_sessions = [1:8 10:13 14:18 23 25];   
% mouse_grps = {1:2,3:8,10:13,14:18,[23 25]};  % note this should be 14:18
mouse_grps = {1:2,3:8,10:13,14:18,[23 25]}; 
pool_sessions = nb_results.pool_sessions;
analysis_mice = find(~cellfun(@isempty ,pool_sessions));
analysis_sessions = arrayfun(@(i_mouse) mouse_grps{i_mouse}(nb_results.pool_sessions{i_mouse}),1:length(mouse_grps),'un',0);
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
rdbu3 = cbrewer('div',"RdBu",10);
rdbu3 = rdbu3([3 7 end],:);
var_bins = nb_results.var_bins;
var_dt = diff(nb_results.var_bins{1}{1}{1}(1:2));

%% 0) Reformat decoded time, load rewsize, RXNil, prts, and postRew_prts
%   Cell format: timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature}{iTrial}
%   Cell format: timeSince_sameRewsize_hat{mIdx}{i}{i_feature}{iTrial}

y_hat = nb_results.y_hat;
nMice = numel(analysis_sessions); 

timeUntil_hat = cell(nMice,1); 

% to load from decoding directly
timeUntil_ix = 3; 

% these are just from the pop_decoding_session_sep
rewsizes = [1 2 4];
rewsize = cell(nMice,1); 
postRew_prts = cell(nMice,1); 
last_rew_ix = cell(nMice,1); 

% choose which features to reformat for analysis
trial_decoding_features = 1:6;

n_cells = cell(nMice,1); 

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
        last_rew_sec = nan(nTrials,1); 
        i_last_rew_ix = nan(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
            last_rew_sec(iTrial) = rew_indices(end)-1;
            i_last_rew_ix(iTrial) = round(((rew_indices(end)-1) * 1000) / tbin_ms);
        end
        session_postRew_prts = session_prts - last_rew_sec; 
        i_last_rew_ix(i_last_rew_ix == 0) = 1; 
    
        % reformat decoded time
        for i_feature = 1:numel(trial_decoding_features)
            iFeature = trial_decoding_features(i_feature);
            timeUntil_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeUntil_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
        end

        % log trial/behavior information
        rewsize{mIdx}{i_i} = session_rewsize;
        postRew_prts{mIdx}{i_i} = session_postRew_prts; 
        last_rew_ix{mIdx}{i_i} = i_last_rew_ix;
    end
end 
clear y_hat % we now have this in an easier form to work with

%% 1) Time until leave visualization, separated by PRT after last reward
% start by visualizing leave-aligned decoded time until leave 
% color by time since last rew 

analyze_ix = [3000/tbin_ms 3000/tbin_ms 5000 / tbin_ms 3000/tbin_ms 3000/tbin_ms];
smoothing_sigma = 1; 
vis_rewsizes = [1 2 4];

vis_mice = 1:5;
close all
for i_feature = [1 2 5 6]
    figure();hold on
    for m_ix = 1:numel(analysis_mice)
        mIdx = analysis_mice(m_ix);  % 1:numel(mouse_grps)
        session_var_bins = var_bins{mIdx}{iRewsize}{timeUntil_ix};
        % load last reward ix for pooled sessions
        mouse_last_rew_ix = cat(1,last_rew_ix{mIdx}{:}); 
        % load post rew residence times
        mouse_postRew_prts = cat(1,postRew_prts{mIdx}{:}); 
        nTrials = length(mouse_postRew_prts); 
        % load reward size .. would probably be good to divide by this
        mouse_rewsize = cat(1,rewsize{mIdx}{:}); 
        
        % gather decoded time variable from i_feature
        mouse_timeUntil = cellfun(@(x) x{i_feature}, timeUntil_hat{mIdx},'un',0);
        mouse_timeUntil = cat(1,mouse_timeUntil{:});
        % now just pull off decoded timesince after last reward
        mouse_timeUntil = arrayfun(@(iTrial) mouse_timeUntil{iTrial}(mouse_last_rew_ix(iTrial):end-25),(1:nTrials)','un',0); 
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeUntil = cellfun(@(x) [nan(max(0,analyze_ix(mIdx) - (length(x)-1)),1) ; x(end-min((length(x)-1),analyze_ix(mIdx)):end)]',mouse_timeUntil,'un',0);
        mouse_timeUntil = var_dt * cat(1,pad_timeUntil{:}); % now this is a matrix
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);
            cmap = cbrewer('div',"RdBu",nTrials_tt);
            
            % grab decoded variable
            timeuntil_tt = mouse_timeUntil(these_trials,:); 
            prts_these_trials = mouse_postRew_prts(these_trials);
            
            terciles = [0 quantile(prts_these_trials,2) max(prts_these_trials)];
            [~,~,tercile_bin] = histcounts(prts_these_trials,terciles);
            
            subplot(numel(vis_rewsizes),numel(analysis_mice),numel(analysis_mice) * (i_rewsize - 1) + m_ix);hold on
            for i_tercile = 1:max(tercile_bin)
                tt_tercile_mean = nanmean(timeuntil_tt(tercile_bin == i_tercile,:));
                tt_tercile_sem = nanstd(timeuntil_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile))); 
                shadedErrorBar((0:analyze_ix(mIdx))*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
            end
            
            xlim([0 analyze_ix(mIdx)*tbin_ms/1000]) 
            ylim([0 max(session_var_bins)]) 
            xticks(0:1:analyze_ix(mIdx)*tbin_ms/1000)
            xticklabels(fliplr(0:1:analyze_ix(mIdx)*tbin_ms/1000))
            
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end
            if m_ix == 1 && i_rewsize == 1
                legend(["Earliest Leaves","Middle Leaves","Late Leaves"])
            end
            if i_rewsize == numel(vis_rewsizes)
                xlabel("Time Until Leave (sec)");
            end
            if m_ix == 1
                ylabel(sprintf("%i uL \n Decoded Time Until Leave (sec)",iRewsize));
            end 
            
        end
    end 
    suptitle(feature_names(i_feature))
end
