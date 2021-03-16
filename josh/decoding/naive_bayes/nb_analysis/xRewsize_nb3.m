%% Analysis of cross reward size naive bayes decoding

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

y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;
var_dt = diff(var_bins{1}{1}{1}(1:2));

%% 0) Reformat yhat into xRewsize, get RX
%   Cell format: timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature}{iTrial}
%   Cell format: timeSince_sameRewsize_hat{mIdx}{i}{i_feature}{iTrial}

y_hat = nb_results.y_hat;
nMice = numel(analysis_sessions); 

timeSince_hat_xRewsize = cell(nMice,1); 
timePatch_hat_xRewsize = cell(nMice,1); 
timeUntil_hat_xRewsize = cell(nMice,1); 

% to load from decoding directly
timeSince_ix = 1; 
timePatch_ix = 2; 
timeUntil_ix = 3; 

% these are just from the pop_decoding_session_sep
rewsizes = [1 2 4];
rewsize = cell(nMice,1); 
RX = cell(nMice,1); 
prts = cell(nMice,1); 
postRew_prts = cell(nMice,1); 
last_rew_ix = cell(nMice,1); 

% choose which features to reformat for analysis
trial_decoding_features = 1:5;

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
        floor_prts = floor(patchleave_sec - patchstop_sec); 
        rew_sec = data.rew_ts;
        nTrials = length(session_rewsize);
        
        rew_barcode = zeros(length(data.patchCSL) , 15);
        i_last_rew_ix = nan(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            i_last_rew_ix(iTrial) = round(((rew_indices(end)) * 1000) / tbin_ms);
            % make rew_barcode for time on patch evaluation separation
            rew_barcode(iTrial , (max(rew_indices+1)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices+1) = session_rewsize(iTrial);
        end 
        
        % collect RX and RXX reward schedule labels
        session_RXX = nan(nTrials,1);
        for iRewsize = [1 2 4] 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0 & rew_barcode(:,3) <= 0) = double(sprintf("%i00",iRewsize)); 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize) = double(sprintf("%i0%i",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize));
        end   

        % reformat decoded time into xRewssize
        for trained_rewsize = 1:numel(rewsizes)
            for i_feature = 1:numel(trial_decoding_features)
                timeSince_hat_xRewsize{mIdx}{i_i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeSince_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
                timePatch_hat_xRewsize{mIdx}{i_i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timePatch_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
                timeUntil_hat_xRewsize{mIdx}{i_i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeUntil_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
            end
        end 
        
        % get number of cells per cluster
        session_sigcells = sig_cells(strcmp(sig_cells.Session,session),:); 
        glm_clusters_session = session_sigcells.GMM_cluster; 
        n_cells{mIdx}{i_i} = [arrayfun(@(x) length(find(glm_clusters_session == x)),(1:4)) length(glm_clusters_session)];
        
        % log trial/behavior information
        rewsize{mIdx}{i_i} = session_rewsize;
        RX{mIdx}{i_i} = floor(session_RXX/10);
        last_rew_ix{mIdx}{i_i} = i_last_rew_ix;
    end
end 
clear y_hat % we now have this in an easier form to work with


%% 1) Mean + sem visualization of gain modulation
smoothing_sigma = 1;
analyze_ix = round(2000 / tbin_ms);
cool3 = cool(3);  
for i_feature = [1 2 5]
    figure();hold on
    for m_ix = 1:numel(analysis_mice)
        mIdx = analysis_mice(m_ix); 
        % load RX for pooled sessions (look at R0 here)
        mouse_RX = cat(1,RX{mIdx}{:});
        
        % make sure these are the same!
        decoded_time_hat = timePatch_hat_xRewsize;
        y_true_tmp = y_true{mIdx}(pool_sessions{mIdx},timePatch_ix);
        true_time = cat(1,y_true_tmp{:}); 
        pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        for i_trained_rewsize = 1:numel(rewsizes) 
            % gather decoded time variable from decoder trained on i_rewsize
            decodedTime = cellfun(@(x) x{i_trained_rewsize}, decoded_time_hat{mIdx},'un',0);
            decodedTime = cat(1,decodedTime{:}); 
            decodedTime_trainedRewsize = decodedTime(:,i_feature); % with i_feature
            decodedTime_trainedRewsize = cat(1,decodedTime_trainedRewsize{:}); 
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_decodedTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',decodedTime_trainedRewsize,'UniformOutput',false);
            decodedTime_trainedRewsize = var_dt * cat(1,pad_decodedTime{:}); % now this is a matrix
            
            for i_true_rewsize = 1:numel(rewsizes)
                iRewsize_true = rewsizes(i_true_rewsize); 
                these_trials = round(mouse_RX/10) == iRewsize_true;
                nTrials_true_rewsize = length(find(these_trials)); 
                
                decodedTime_hat = decodedTime_trainedRewsize(these_trials,:); 
                
                mean_decodedTime = nanmean(decodedTime_hat);
                sem_decodedTime = 3 * nanstd(decodedTime_hat) / sqrt(nTrials_true_rewsize);
                
                subplot(numel(rewsizes),numel(analysis_mice),numel(analysis_mice) * (i_true_rewsize - 1) + m_ix);hold on
%                 shadedErrorBar((1:analyze_ix)*tbin_ms/1000,mean_decodedTime,sem_decodedTime,'lineProps',{'color',cool3(i_trained_rewsize,:)})

%                 % for single trial visualization
%                 decodedTime_hat = decodedTime_trainedRewsize(these_trials,:);
%                 trueTime = true_time(these_trials,:);
%                 for iTrial = 1:15 % nTrials_true_rewsize
%                     plot(trueTime(iTrial,:),gauss_smoothing(decodedTime_hat(iTrial,:),smoothing_sigma),'color',cool3(i_trained_rewsize,:),'linewidth',.5)
%                 end
                
                ylim([0 2]) 
                plot([0 2],[0,2],'k--','linewidth',1.5)  
                if m_ix == 1
                    ylabel(sprintf("%i uL Trials \n Decoded time",iRewsize_true))
                end 
                if i_true_rewsize == 1 
                    title(sprintf("%s \n %s",feature_names(i_feature),mouse_names(mIdx)))
                end
            end
            if i_trained_rewsize == numel(rewsizes)
                xlabel("True time")
            end
        end
    end
end

%% 2) Formalize by fitting slopes to estimate gain differences
analyze_ix = round(1000 / tbin_ms);
cool3repeat = repmat(cool(3),[3,1]);  
x = [1:3 5:7 9:11]; 
vis_mice = [2 3 5];
for i_feature = 1:5
    figure();hold on
    for m_ix = 1:numel(analysis_mice) 
        mIdx = analysis_mice(m_ix); 
        % load RX for pooled sessions (look at R0 here)
        mouse_RX = cat(1,RX{mIdx}{:});
        
        slope = nan(numel(rewsizes)^2,1);
        slope_sem = nan(numel(rewsizes)^2,1);
        
        % make sure these are the same!
        decoded_time_hat = timePatch_hat_xRewsize;
        y_true_tmp = y_true{mIdx}(pool_sessions{mIdx},timePatch_ix);
        true_time = cat(1,y_true_tmp{:}); 
        pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
        true_time = var_dt * cat(1,pad_trueTime{:});
        
        for i_trained_rewsize = 1:numel(rewsizes) 
            % gather decoded time variable from decoder trained on i_rewsize
            decodedTime = cellfun(@(x) x{i_trained_rewsize}, decoded_time_hat{mIdx},'un',0);
            decodedTime = cat(1,decodedTime{:}); 
            decodedTime_trainedRewsize = decodedTime(:,i_feature); % with i_feature
            decodedTime_trainedRewsize = cat(1,decodedTime_trainedRewsize{:}); 
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_decodedTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',decodedTime_trainedRewsize,'UniformOutput',false);
            decodedTime_trainedRewsize = var_dt * cat(1,pad_decodedTime{:}); % now this is a matrix
            
            for i_true_rewsize = 1:numel(rewsizes)
                iRewsize_true = rewsizes(i_true_rewsize); 
                these_trials = round(mouse_RX/10) == iRewsize_true; % don't care about sec1 onwards
                nTrials_true_rewsize = length(find(these_trials)); 
                
                decodedTime_trueRewsize = decodedTime_trainedRewsize(these_trials,:)'; 
                trueTime_trueRewsize = true_time(these_trials,:)';
                
                mdl = fitlm(trueTime_trueRewsize(:),decodedTime_trueRewsize(:)); %,'intercept',false);
                slope(3 * (i_true_rewsize - 1) + i_trained_rewsize) = mdl.Coefficients.Estimate(2);
                slope_sem(3 * (i_true_rewsize - 1) + i_trained_rewsize) = mdl.Coefficients.SE(2);
            end  
        end
        subplot(1,numel(analysis_mice),m_ix);hold on
        for i = 1:numel(x)
            bar(x(i),slope(i),'FaceColor',cool3repeat(i,:),'FaceAlpha',.5) 
            errorbar(x(i),slope(i),slope_sem(i),'k') 
        end 
        yline(1,'k--','linewidth',1.5) 
        xticks([2 6 10]) 
        xticklabels(["1 uL","2 uL","4 uL"])
        xlabel("Decoded Reward Size") 
        if m_ix == 1
            ylabel("Fit slope between true and decoded time")
        end
        title(sprintf("%s \n %s Model Fits",feature_names(i_feature),mouse_names(mIdx))) 
        ylim([0 4])
    end
end