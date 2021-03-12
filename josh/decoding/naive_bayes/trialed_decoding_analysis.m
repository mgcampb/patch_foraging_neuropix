%% Script to do analysis on y_hat_trials created from popDecoding_sessionSep 
%   Cell format: y_hat_trials{mIdx}{i}{iTrial}{iVar}{trained_rewsize}{i_feature}
load('./structs/yhat_trials.mat');
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
addpath('/Users/joshstern/Documents/UchidaLab_NeuralData'); 

% analysis options
tbin_ms = .02 * 1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mPFC_sessions = [1:8 10:13 15:18 23 25];   
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]};  % note this should be 14:18
mouse_names = ["m75","m76","m78","m79","m80"];
session_titles = cell(numel(mouse_grps),1);
for mIdx = 1:numel(mouse_grps)
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];
        session_titles{mIdx}{i} = session_title;
    end
end  

%% 0.i) Collect predictions to reformat s.t. easier to pool trials
%   Cell format: timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature}{iTrial}
%   Cell format: timeSince_sameRewsize_hat{mIdx}{i}{i_feature}{iTrial}
timeSince_hat = cell(numel(y_hat_trials),1); 
timePatch_hat = cell(numel(y_hat_trials),1); 
timeUntil_hat = cell(numel(y_hat_trials),1);  

timeSince_hat_sameRewsize = cell(numel(y_hat_trials),1); 
timePatch_hat_sameRewsize = cell(numel(y_hat_trials),1); 
timeUntil_hat_sameRewsize = cell(numel(y_hat_trials),1);  

% to load from decoding directly
% timeSince_true = cell(numel(y_hat_trials),1); 
% timePatch_true = cell(numel(y_hat_trials),1); 
% timeUntil_true = cell(numel(y_hat_trials),1); % just make this flip of timeUntil
timeSince_ix = 1; 
timePatch_ix = 2; 
timeUntil_ix = 3; 

% these are just from the pop_decoding_session_sep
rewsizes = [1 2 4];
trial_decoding_features = 1:4;

for mIdx = 1:5  
    for i = 1:numel(y_hat_trials{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx}(1:end-4); 
        data = load(fullfile(paths.data,sessions{sIdx}));  
        i_rewsize = min(3,mod(data.patches(:,2),10)); 
        nTrials = length(i_rewsize);
        
        for i_feature = 1:numel(trial_decoding_features)
            timeSince_hat_sameRewsize{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timeSince_ix}{i_rewsize(iTrial)}{i_feature},(1:nTrials)','un',0);
            timePatch_hat_sameRewsize{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timePatch_ix}{i_rewsize(iTrial)}{i_feature},(1:nTrials)','un',0);
            timeUntil_hat_sameRewsize{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timeUntil_ix}{i_rewsize(iTrial)}{i_feature},(1:nTrials)','un',0);
        end
        
        for trained_rewsize = 1:numel(rewsizes)
            for i_feature = 1:numel(trial_decoding_features)
                timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timeSince_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
                timePatch_hat{mIdx}{i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timePatch_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
                timeUntil_hat{mIdx}{i}{trained_rewsize}{i_feature} = arrayfun(@(iTrial) y_hat_trials{mIdx}{i}{iTrial}{timeUntil_ix}{trained_rewsize}{i_feature},(1:nTrials)','un',0);
            end
        end 
%         % just some slight reformatting for true time
%         timeSince_true{mIdx}{i} = y{mIdx}{i,timeSince_ix}; 
%         timePatch_true{mIdx}{i} = y{mIdx}{i,timePatch_ix}; 
%         timeUntil_true{mIdx}{i} = cellfun(@(x) fliplr(x),y{mIdx}{i,timePatch_ix},'un',0); 
    end
end 

%% 0.ii) Re-load task variables, bin, and get trial information

nMice = numel(mouse_grps);

timeSince_true = cell(nMice,1); 
timePatch_true = cell(nMice,1); 
timeUntil_true = cell(nMice,1); 

% time decoding by reward event
timeSince_hat_rews = cell(nMice,1); 
timePatch_hat_rews = cell(nMice,1); 

% trial information
rewsize = cell(nMice,1);
RX = cell(nMice,1);  
RXNil = cell(nMice,1); 
RXX = cell(nMice,1);  
rew_time = cell(nMice,1);  
rew_num = cell(nMice,1);   
last_rew_ix = cell(nMice,1);   
postRew_rts = cell(nMice,1); 
prts = cell(nMice,1); 
zscored_qrts = cell(nMice,1); 

% reward event information 
rewsize_rews = cell(nMice,1); 
rew_time_rews = cell(nMice,1); 
rew_num_rews = cell(nMice,1); 

for mIdx = 1:nMice
    % true time
    timeSince_true{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    timePatch_true{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    timeUntil_true{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    % decoded per reward 
    timeSince_hat_rews{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    timePatch_hat_rews{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    % trial stuff
    rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); % one vector of rewsizes per session   
    RX{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    RXNil{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    RXX{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_time{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_num{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    prts{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    zscored_qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    % for post last reward analysis
    last_rew_ix{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    postRew_rts{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    % reward event information
    rewsize_rews{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_time_rews{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    rew_num_rews{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    
    for i = 1:numel(mouse_grps{mIdx}) 
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx}(1:end-4); 
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        % Load session information
        nTrials = length(data.patchCSL); 
        session_rewsize = mod(data.patches(:,2),10); 
        patchcue_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);   
        qrts = patchstop_sec - patchcue_sec; 
        session_prts = patchleave_sec - patchstop_sec;  
        floor_prts = floor(session_prts); % for rew barcode
        rew_sec = data.rew_ts;  
        % index vectors
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round(data.patchCSL(:,3)*1000 / tbin_ms) + 1; 
        prts_ix = patchleave_ix - patchstop_ix + 1;

        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        session_rew_time_rews = cell(nTrials,1);  
        session_rew_num_rews = cell(nTrials,1); 
        session_rewsize_rews = cell(nTrials,1); 
        nTimesteps = 15; 
        nTrials = length(session_rewsize); 
        rew_barcode = zeros(length(data.patchCSL) , nTimesteps);
        i_last_rew_ix = nan(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            session_rew_time_rews{iTrial} = rew_indices; 
            session_rew_num_rews{iTrial} = (1:numel(rew_indices))'; 
            session_rewsize_rews{iTrial} = session_rewsize(iTrial) + zeros(numel(rew_indices),1); 
            rew_sec_cell{iTrial} = rew_indices(rew_indices >= 1); 
            i_last_rew_ix(iTrial) = round(((rew_indices(end)) * 1000) / tbin_ms);
            % make rew_barcode for time on patch evaluation separation
            % Note we add 1 to rew_indices here because we are now 1 indexing
            rew_barcode(iTrial , (max(rew_indices+1)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices+1) = session_rewsize(iTrial);
        end 
        i_postRew_rts = session_prts - i_last_rew_ix * tbin_ms / 1000;
        i_last_rew_ix(i_last_rew_ix == 0) = 1; 
        session_rew_time_rews = cat(1,session_rew_time_rews{:}); 
        session_rew_num_rews = cat(1,session_rew_num_rews{:}); 
        session_rewsize_rews = cat(1,session_rewsize_rews{:});
        nRews = length(session_rew_time_rews); 
        
        % collect RX and RXX reward schedule labels
        session_RXX = nan(nTrials,1);
        for iRewsize = [1 2 4] 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0) = double(sprintf("%i00",iRewsize)); 
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize) = double(sprintf("%i0%i",iRewsize,iRewsize));
            session_RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize));
        end   
        
        % Make RXNil vector
        session_RXNil = nan(nTrials,1); 
        for iTrial = 1:nTrials
            if session_prts(iTrial) >= 1 % only care if we have at least 1 second on patch
                rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
                if isequal(rew_indices,1)
                    session_RXNil(iTrial) = 10*session_rewsize(iTrial);
                elseif isequal(rew_indices,[1 ; 2]) 
                    session_RXNil(iTrial) = 10*session_rewsize(iTrial) + session_rewsize(iTrial);
                end
            end
        end  
        
        % zscore
        session_zscored_qrts = nan(size(session_RXNil));
        RXNil_tts = unique(session_RXNil(~isnan(session_RXNil)));
        for i_RXNil_tt = 1:numel(RXNil_tts)
            session_zscored_qrts(session_RXNil == RXNil_tts(i_RXNil_tt)) = zscore(qrts(session_RXNil == RXNil_tts(i_RXNil_tt)));
        end
        
        % Create task variables and bin according to some discretization  
        var_bins{1} = 0:.05:10; % bins to classify timeSinceReward (sec) 
        var_bins{2} = 0:.05:10; % bins to classify timeOnPatch (sec) 
        var_bins{3} = 0:.05:10; % bins to classify time2Leave (sec) 
        
        % log which features we don't have for this session!
        empty_features = arrayfun(@(i_feature) isempty(timeSince_hat_sameRewsize{mIdx}{i}{i_feature}{1}),1:numel(trial_decoding_features));
        nonempty_features = find(~empty_features);
        
        timeSinceReward_binned = cell(nTrials,1); 
        timeOnPatch_binned = cell(nTrials,1); 
        time2Leave_binned = cell(nTrials,1);   
        session_rewTime = cell(nTrials,1); 
        session_rewNum = cell(nTrials,1); 
        session_timeSince_rews = cell(1,length(trial_decoding_features)); 
        session_timePatch_rews = cell(1,length(trial_decoding_features)); 
        rew_counter = 0; 
        for iTrial = 1:nTrials 
            rew_counter = rew_counter + 1; 
            trial_len_ix = prts_ix(iTrial);
            timeSinceReward_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;
            timeOnPatch_binned{iTrial} = (1:trial_len_ix) * tbin_ms / 1000;  
            time2Leave_binned{iTrial} = fliplr(timeOnPatch_binned{iTrial}); 
            session_rewTime{iTrial} = zeros(trial_len_ix,1); 
            session_rewNum{iTrial} = zeros(trial_len_ix,1);  
            
            if numel(rew_sec_cell{iTrial}) == 0
                for f = 1:numel(nonempty_features)
                    i_feature = nonempty_features(f); 
                    session_timeSince_rews{i_feature}{iTrial}{1} = timeSince_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(1:end); 
                    session_timePatch_rews{i_feature}{iTrial}{1} = timePatch_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(1:end); 
                end 
            else 
                for f = 1:numel(nonempty_features)
                    i_feature = nonempty_features(f); 
                    rew_ix1 = round(rew_sec_cell{iTrial}(1) * 1000 / tbin_ms);
                    session_timeSince_rews{i_feature}{iTrial}{1} = timeSince_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(1:rew_ix1); 
                    session_timePatch_rews{i_feature}{iTrial}{1} = timePatch_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(1:rew_ix1); 
                end 
            end
            
            for r = 1:numel(rew_sec_cell{iTrial})
                rew_ix = (rew_sec_cell{iTrial}(r)) * 1000 / tbin_ms; 
                timeSinceReward_binned{iTrial}(rew_ix:end) =  (1:length(timeSinceReward_binned{iTrial}(rew_ix:end))) * tbin_ms / 1000;
                if r == numel(rew_sec_cell{iTrial}) % if it is our last reward
%                     time2Leave_binned{iTrial} = fliplr(timeSinceReward_binned{iTrial}(rew_ix:end)); % add to time2leave  
                    session_rewTime{iTrial}(rew_ix:end) = rew_sec_cell{iTrial}(r); 
                    session_rewNum{iTrial}(rew_ix:end) = r;
                end  
                
                if r == numel(rew_sec_cell{iTrial}) % last reward
                    for f = 1:numel(nonempty_features)
                        i_feature = nonempty_features(f); 
                        session_timeSince_rews{i_feature}{iTrial}{r+1} = timeSince_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(rew_ix:end);
                        session_timePatch_rews{i_feature}{iTrial}{r+1} = timePatch_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(rew_ix:end);
                    end
                else 
                    for f = 1:numel(nonempty_features)
                        i_feature = nonempty_features(f); 
                        next_rew_ix = (rew_sec_cell{iTrial}(r+1)) * 1000 / tbin_ms; 
                        session_timeSince_rews{i_feature}{iTrial}{r+1} = timeSince_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                        session_timePatch_rews{i_feature}{iTrial}{r+1} = timePatch_hat_sameRewsize{mIdx}{i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                    end
                end
                rew_counter = rew_counter + 1; 
            end   
        end 
        
        % ok now go back and concatenate everyone
        for f = 1:numel(nonempty_features)
            i_feature = nonempty_features(f);
            session_timeSince_rews{i_feature} = cat(2,session_timeSince_rews{i_feature}{:})';
            session_timePatch_rews{i_feature} = cat(2,session_timePatch_rews{i_feature}{:})';
        end

        % save trial information to data structures
        timeSince_true{mIdx}{i} = timeSinceReward_binned;
        timePatch_true{mIdx}{i} = timeOnPatch_binned;
        timeUntil_true{mIdx}{i} = time2Leave_binned;
        timeSince_hat_rews{mIdx}{i} = session_timeSince_rews;
        timePatch_hat_rews{mIdx}{i} = session_timePatch_rews;
        RXNil{mIdx}{i} = session_RXNil;
        prts{mIdx}{i} = session_prts;
        zscored_qrts{mIdx}{i} = session_zscored_qrts;
        RX{mIdx}{i} = floor(session_RXX/10);
        RXX{mIdx}{i} = session_RXX; 
        rew_time{mIdx}{i} = session_rewTime;
        rew_num{mIdx}{i} = session_rewNum; 
        rewsize{mIdx}{i} = session_rewsize; 
        last_rew_ix{mIdx}{i} = i_last_rew_ix;
        postRew_rts{mIdx}{i} = i_postRew_rts;
        
        rewsize_rews{mIdx}{i} = session_rewsize_rews;
        rew_time_rews{mIdx}{i} = session_rew_time_rews;
        rew_num_rews{mIdx}{i} = session_rew_num_rews;
    end 
%     disp([mIdx i])
end 

%% 0.iii) Visualize predictions with scatterplot 

close all
bin_dt = diff(var_bins{1}(1:2));
n_bins_hat = 40;  
n_bins_true = length(var_bins{1})-1;
i_feature = 4;
for mIdx = 5
    figure() 
    for i = 1:numel(y_hat_trials{mIdx}) 
        i_rewsize = rewsize{mIdx}{i};
        for i_trial_rewsize = 1:numel(rewsizes) 
            timePatch_hat_full = timePatch_hat{mIdx}{i}{i_trial_rewsize}{i_feature}(i_rewsize == iRewsize);
            timePatch_hat_full = bin_dt * cat(1,timePatch_hat_full{:});
            timePatch_true_full = timePatch_true{mIdx}{i}(i_rewsize == iRewsize);
            timePatch_true_full = bin_dt * cat(2,timePatch_true_full{:})';
            
            noise = .1 * randn(size(timePatch_true_full));
            subplot(1,numel(y_hat_trials{mIdx}),i)
            % scatter or binscatter
            scatter(noise + timePatch_true_full,noise + timePatch_hat_full,'.') 
%             
            [r,p] = corrcoef(timePatch_true_full(~isnan(timePatch_true_full)), timePatch_hat_full(~isnan(timePatch_true_full))); 
            a = binscatter(timePatch_true_full,timePatch_hat_full,[n_bins_true,n_bins_hat]);
            values = flipud(a.Values'); 
            imagesc(values)  
            title(sprintf("%s (r = %.3f, p = %.3f)",session_titles{mIdx}{i},r(2),p(2))) 
            xlim([1 40]);ylim([1 40])
        end
    end
end

%% 1b) Formalize results by performing correlation coefficient analysis 

r_prt = cell(numel(mouse_grps),1); 
p_prt = cell(numel(mouse_grps),1); 
for mIdx = 1:5 
    r_prt{mIdx} = cell(numel(trial_decoding_features),1); 
    p_prt{mIdx} = cell(numel(trial_decoding_features),1); 
    % load RXNil for pooled sessions
    mouse_RXNil = RXNil{mIdx}(pool_sessions{mIdx});
    mouse_RXNil = cat(1,mouse_RXNil{:}); 
    % load PRTs for pooled sessions 
    mouse_prts = prts{mIdx}(pool_sessions{mIdx});
    mouse_prts = cat(1,mouse_prts{:}); 
    
    for i_feature = 1:numel(trial_decoding_features)
        r_prt{mIdx}{i_feature} = nan(numel(trialtypes),analyze_ix); 
        p_prt{mIdx}{i_feature} = nan(numel(trialtypes),analyze_ix); 
        
        % gather decoded time variable from i_feature
        mouse_timePatch = cellfun(@(x) x{i_feature}, timeSince_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timePatch = cat(1,mouse_timePatch{:});
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timePatch = cellfun(@(x) [x(1:min(length(x),100)) ; nan(max(0,100 - length(x)),1)]',mouse_timePatch,'UniformOutput',false);
        mouse_timePatch = cat(1,pad_timePatch{:}); % now this is a matrix 
        
        for i_tt = 1:numel(trialtypes)
            these_trials = find(mouse_RXNil == trialtypes(i_tt));
            nTrials_tt = length(these_trials);
            
            % grab decoded variable
            time_on_patch_tt = mouse_timePatch(these_trials,:);
            prts_tt = mouse_prts(these_trials);
            
            % get corrcoef per timepoint 
            for i_time = 1:analyze_ix  
                i_time_decoding = time_on_patch_tt(:,i_time);
                [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_tt(~isnan(i_time_decoding)));
                
                r_prt{mIdx}{i_feature}(i_tt,i_time) = r(2); 
                p_prt{mIdx}{i_feature}(i_tt,i_time) = p(2);
            end
        end
    end
end

%% 1a) RXNil corrcoef for PRT within trialtype 
%  Start w/ just time on patch
feature_names = ["Cluster 1","Cluster 2","Cluster 3","All GLM Cells"];
analyze_ix = 2000 / tbin_ms;
smoothing_sigma = 1; 
trialtypes = [10 20 40]; % [10 20 40]; % RXNil trialtypes to look at 
close all
pool_sessions = {(2),(4:5),[1 2 4],(3),(1:2)}; % based on mutual information kinda

for i_feature = 1:4
    figure();hold on
    for mIdx = 1:5 % 1:numel(mouse_grps)
        % load RXNil for pooled sessions
        mouse_RXNil = RXNil{mIdx}(pool_sessions{mIdx});
        mouse_RXNil = cat(1,mouse_RXNil{:});
        % load PRTs for pooled sessions
        mouse_prts = prts{mIdx}(pool_sessions{mIdx});
        mouse_prts = cat(1,mouse_prts{:});
        % gather decoded time variable from i_feature
        mouse_timePatch = cellfun(@(x) x{i_feature}, timeSince_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timePatch = cat(1,mouse_timePatch{:});
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timePatch = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_timePatch,'UniformOutput',false);
        mouse_timePatch = bin_dt  * cat(1,pad_timePatch{:}); % now this is a matrix
        
        for i_tt = 1:numel(trialtypes)
            these_trials = find(mouse_RXNil == trialtypes(i_tt));
            nTrials_tt = length(these_trials);
            cmap = cbrewer('div',"RdBu",nTrials_tt);
            
            % grab decoded variable
            time_on_patch_tt = mouse_timePatch(these_trials,:);
            [~,prt_sort] = sort(mouse_prts(these_trials));
            prt_sorted_these_trials = these_trials(prt_sort);
            time_on_patch_tt_prt_sort = time_on_patch_tt(prt_sort,:);
            
            subplot(numel(trialtypes),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on
            for iTrial = 1:nTrials_tt
                plot(gauss_smoothing(time_on_patch_tt_prt_sort(iTrial,:),1),'color',cmap(iTrial,:),'linewidth',.25)
            end
            
            for i_time = 1:analyze_ix
                if p_prt{mIdx}{i_feature}(i_tt,i_time) < .05 
                    text(i_time,2.05,'*','HorizontalAlignment','center')
                end
            end
            ylim([0 2.1])
            
            if i_tt == 1
                title(mouse_names(mIdx))
            end
            if i_tt == numel(trialtypes)
                xlabel("True Time");
            end
            if mIdx == 1
                ylabel(sprintf("%i \n Decoded Time",trialtypes(i_tt)));
            end
            yticklabels([])
            xticklabels([])
        end
    end
    suptitle(feature_names(i_feature))
end

%% 1c) Visualize r_prt and p_prt

sig_threshold = .05; 
x = tbin_ms * (1:analyze_ix) / 1000;
corr_vis_tts = 1:6; 
vis_features = 1:4; 
feature_colors = lines(4); 
for mIdx = 1:numel(mouse_grps) 
    for i_tt = 1:numel(corr_vis_tts)
        subplot(numel(corr_vis_tts),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on 
        for i_feature = 1:numel(vis_features)
            for i_time = 1:(analyze_ix-1)
                if p_prt{mIdx}{i_feature}(i_tt,i_time) < sig_threshold
                    plot(x(i_time:i_time+1),r_prt{mIdx}{i_feature}(i_tt,i_time:i_time+1),'-','linewidth',1.5,'color',feature_colors(i_feature,:))
                else 
                    plot(x(i_time:i_time+1),r_prt{mIdx}{i_feature}(i_tt,i_time:i_time+1),':','color',feature_colors(i_feature,:))
                end
            end 
        end 
        if mIdx == 1 
            ylabel(sprintf("%i \n PRT PearsonR",trialtypes(i_tt)))
        end  
        if i_tt == 1 
            title(mouse_names(mIdx))
        end
        if i_tt == numel(corr_vis_tts) 
            xlabel("True time")
        end
    end 
    
    if mIdx == 5
        legend(feature_names(vis_features))
    end
end

%% 2b) Analyze correlation coefficient for timesince after last rew

analyze_ix = [2000/tbin_ms 2000/tbin_ms 3000 / tbin_ms 2000/tbin_ms 2000/tbin_ms];
% smoothing_sigma = 1; 
vis_rewsizes = [1 2 4];
rdbu3 = cbrewer('div',"RdBu",10);
rdbu3 = rdbu3([3 7 end],:);

r_prt = cell(numel(mouse_grps),1); 
p_prt = cell(numel(mouse_grps),1); 
for mIdx = 1:5 
    r_prt{mIdx} = cell(numel(trial_decoding_features),1); 
    p_prt{mIdx} = cell(numel(trial_decoding_features),1); 
    
    % load last reward ix for pooled sessions
    mouse_last_rew_ix = last_rew_ix{mIdx}(pool_sessions{mIdx});
    mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
    % load post rew residence times
    mouse_postRew_rts = postRew_rts{mIdx}(pool_sessions{mIdx});
    mouse_postRew_rts = cat(1,mouse_postRew_rts{:});
    nTrials = length(mouse_postRew_rts);
    % load reward size .. would probably be good to divide by this
    mouse_rewsize = rewsize{mIdx}(pool_sessions{mIdx});
    mouse_rewsize = cat(1,mouse_rewsize{:});
    
    for i_feature = 1:numel(trial_decoding_features)
        r_prt{mIdx}{i_feature} = nan(numel(rewsizes),analyze_ix(mIdx));
        p_prt{mIdx}{i_feature} = nan(numel(rewsizes),analyze_ix(mIdx));
        
        % gather decoded time variable from i_feature
        mouse_timeSince = cellfun(@(x) x{i_feature}, timeSince_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeSince = cat(1,mouse_timeSince{:});
        % now just pull off decoded timesince after last reward
        mouse_timeSince = arrayfun(@(iTrial) mouse_timeSince{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0);
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeSince = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_timeSince,'UniformOutput',false);
        mouse_timeSince = cat(1,pad_timeSince{:}); % now this is a matrix
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);
            
            % grab decoded variable
            time_since_rew_tt = mouse_timeSince(these_trials,:);
            mouse_postRew_rts_tt = mouse_postRew_rts(these_trials);
            
            % get corrcoef per timepoint 
            for i_time = 1:analyze_ix(mIdx)  
                i_time_decoding = time_since_rew_tt(:,i_time);
                [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),mouse_postRew_rts_tt(~isnan(i_time_decoding)));
                
                r_prt{mIdx}{i_feature}(i_rewsize,i_time) = r(2); 
                p_prt{mIdx}{i_feature}(i_rewsize,i_time) = p(2);
            end
        end
    end
end

%% 2a) Correlation between decoded time since reward on last reward vs post last rew residence time 
% analyze_ix = 2000 / tbin_ms;
smoothing_sigma = 1; 
vis_rewsizes = [1 2 4];

for i_feature = 4
    figure();hold on
    for mIdx = 1:5 % 1:numel(mouse_grps)
        % load last reward ix for pooled sessions
        mouse_last_rew_ix = last_rew_ix{mIdx}(pool_sessions{mIdx});
        mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
        % load post rew residence times
        mouse_postRew_rts = postRew_rts{mIdx}(pool_sessions{mIdx});
        mouse_postRew_rts = cat(1,mouse_postRew_rts{:}); 
        nTrials = length(mouse_postRew_rts); 
        % load reward size .. would probably be good to divide by this
        mouse_rewsize = rewsize{mIdx}(pool_sessions{mIdx});
        mouse_rewsize = cat(1,mouse_rewsize{:});
        
        % gather decoded time variable from i_feature
        mouse_timeSince = cellfun(@(x) x{i_feature}, timeSince_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeSince = cat(1,mouse_timeSince{:});
        % now just pull off decoded timesince after last reward
        mouse_timeSince = arrayfun(@(iTrial) mouse_timeSince{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0); 
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeSince = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_timeSince,'UniformOutput',false);
        mouse_timeSince = bin_dt * cat(1,pad_timeSince{:}); % now this is a matrix
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);
            cmap = cbrewer('div',"RdBu",nTrials_tt);
            
            % grab decoded variable
            timesince_tt = mouse_timeSince(these_trials,:); 
            prts_these_trials = mouse_postRew_rts(these_trials);
%             [~,postRew_rt_sort] = sort(mouse_postRew_rts(these_trials));
%             prt_sorted_these_trials = these_trials(postRew_rt_sort);
%             timesince_tt_prt_sort = timesince_tt(postRew_rt_sort,:);

            terciles = [0 quantile(prts_these_trials,2) max(prts_these_trials)];
            [~,~,tercile_bin] = histcounts(prts_these_trials,terciles);
            
             subplot(numel(vis_rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_rewsize - 1) + mIdx);hold on
            
            for i_tercile = 1:max(tercile_bin)
                tt_tercile_mean = nanmean(timesince_tt(tercile_bin == i_tercile,:));
                tt_tercile_sem = nanstd(timesince_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile))); 
                shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
            end
            
%             subplot(numel(vis_rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_rewsize - 1) + mIdx);hold on
%             for iTrial = 1:nTrials_tt
%                 plot((1:analyze_ix(mIdx)) * tbin_ms/1000,gauss_smoothing(timesince_tt_prt_sort(iTrial,:),1),'color',cmap(iTrial,:),'linewidth',.25)
%             end
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end
            if i_rewsize == numel(vis_rewsizes)
                xlabel("True Time");
            end
            if mIdx == 1
                ylabel(sprintf("%i uL \n Decoded Time",iRewsize));
            end
            
            for i_time = 1:analyze_ix(mIdx)
                if p_prt{mIdx}{i_feature}(i_tt,i_time) < .05 
                    text(i_time*tbin_ms/1000,2.05,'*','HorizontalAlignment','center')
                end
            end
            ylim([0 2.1])
            
%             yticklabels([])
%             xticklabels([])
        end
    end 
    suptitle(feature_names(i_feature))
end

%% 2c) Analyze r_prt for residence time after last rew
sig_threshold = .05; 
x = tbin_ms * (1:analyze_ix) / 1000;
corr_vis_tts = 1:6; 
vis_features = 1:4; 
feature_colors = lines(4); 
for mIdx = 1:numel(mouse_grps) 
    for i_rewsize = 1:numel(rewsizes)
        subplot(numel(rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_rewsize - 1) + mIdx);hold on 
        for i_feature = 1:numel(vis_features)
            for i_time = 1:(analyze_ix-1)
                if p_prt{mIdx}{i_feature}(i_rewsize,i_time) < sig_threshold
                    plot(x(i_time:i_time+1),r_prt{mIdx}{i_feature}(i_rewsize,i_time:i_time+1),'-','linewidth',1.5,'color',feature_colors(i_feature,:))
                else 
                    plot(x(i_time:i_time+1),r_prt{mIdx}{i_feature}(i_rewsize,i_time:i_time+1),':','color',feature_colors(i_feature,:))
                end
            end 
        end 
        if mIdx == 1 
            ylabel(sprintf("%i uL \n PRT PearsonR",rewsizes(i_rewsize)))
        end  
        if i_rewsize == 1 
            title(mouse_names(mIdx))
        end
        if i_rewsize == numel(rewsizes) 
            xlabel("True time")
        end
    end 
    
    if mIdx == 5
        legend(feature_names(vis_features))
    end
end


%% 3) Gain modulation in cross-reward size decoding 
analyze_ix = round(2000 / tbin_ms);
cool3 = cool(3);  

for i_feature = 4
    figure();hold on
    for mIdx = 1:numel(mouse_grps)
        % load RX for pooled sessions (look at R0 here)
        mouse_RX = RX{mIdx}(pool_sessions{mIdx});
        mouse_RX = cat(1,mouse_RX{:});
        
        % make sure these are the same!
        decoded_time_hat = timePatch_hat;
        true_time = timePatch_true{mIdx}(pool_sessions{mIdx});
        true_time = cat(1,true_time{:});
        pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        for i_trained_rewsize = 1:numel(rewsizes) 
            % gather decoded time variable from decoder trained on i_rewsize
            decodedTime = cellfun(@(x) x{i_trained_rewsize}, decoded_time_hat{mIdx}(pool_sessions{mIdx}),'un',0);
            decodedTime = cat(1,decodedTime{:}); 
            decodedTime_trainedRewsize = decodedTime(:,i_feature); % with i_feature
            decodedTime_trainedRewsize = cat(1,decodedTime_trainedRewsize{:}); 
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_decodedTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',decodedTime_trainedRewsize,'UniformOutput',false);
            decodedTime_trainedRewsize = bin_dt * cat(1,pad_decodedTime{:}); % now this is a matrix
            
            for i_true_rewsize = 1:numel(rewsizes)
                iRewsize_true = rewsizes(i_true_rewsize); 
                these_trials = mouse_RX == 10 * iRewsize_true; 
                nTrials_true_rewsize = length(find(these_trials)); 
                
                decodedTime_trueRewsize = decodedTime_trainedRewsize(these_trials,:);
                trueTime_trueRewsize = true_time(these_trials,:);
                
                subplot(numel(rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_true_rewsize - 1) + mIdx);hold on
                
                for iTrial = 1:15 % nTrials_true_rewsize
                    plot(trueTime_trueRewsize(iTrial,:),gauss_smoothing(decodedTime_trueRewsize(iTrial,:),smoothing_sigma),'color',cool3(i_trained_rewsize,:),'linewidth',.5)
                    noise = .1*randn(1,analyze_ix);
                    %                     scatter(noise + trueTime_trueRewsize(iTrial,:),noise + decodedTime_trueRewsize(iTrial,:),5,cool3(i_trained_rewsize,:))
                end
                if mIdx == 1
                    ylabel(sprintf("%i uL Trials \n Decoded time",iRewsize_true))
                end 
                if i_true_rewsize == 1 
                    title(sprintf("%s \n Cross-Reward Size Decoding",mouse_names(mIdx)))
                end
            end
        end
        if i_trained_rewsize == numel(rewsizes)
            xlabel("True time")
        end
    end
end

%% 3b) Visualize gain modulation by plotting nanmean / nansem and fitting linear model to slope

analyze_ix = round(1000 / tbin_ms);
cool3 = cool(3);  
for i_feature = 2:4
    figure();hold on
    for mIdx = 1:numel(mouse_grps)
        % load RX for pooled sessions (look at R0 here)
        mouse_RX = RX{mIdx}(pool_sessions{mIdx});
        mouse_RX = cat(1,mouse_RX{:});
        
        % make sure these are the same!
        decoded_time_hat = timePatch_hat;
        true_time = timePatch_true{mIdx}(pool_sessions{mIdx});
        true_time = cat(1,true_time{:});
        pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        for i_trained_rewsize = 1:numel(rewsizes) 
            % gather decoded time variable from decoder trained on i_rewsize
            decodedTime = cellfun(@(x) x{i_trained_rewsize}, decoded_time_hat{mIdx}(pool_sessions{mIdx}),'un',0);
            decodedTime = cat(1,decodedTime{:}); 
            decodedTime_trainedRewsize = decodedTime(:,i_feature); % with i_feature
            decodedTime_trainedRewsize = cat(1,decodedTime_trainedRewsize{:}); 
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_decodedTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',decodedTime_trainedRewsize,'UniformOutput',false);
            decodedTime_trainedRewsize = bin_dt * cat(1,pad_decodedTime{:}); % now this is a matrix
            
            for i_true_rewsize = 1:numel(rewsizes)
                iRewsize_true = rewsizes(i_true_rewsize); 
%                 these_trials = mouse_RX == 10 * iRewsize_true;  
                these_trials = round(mouse_RX/10) == iRewsize_true;
                nTrials_true_rewsize = length(find(these_trials)); 
                
                decodedTime_trueRewsize = decodedTime_trainedRewsize(these_trials,:); 
                
                mean_decodedTime = nanmean(decodedTime_trueRewsize);
                sem_decodedTime = 3 * nanstd(decodedTime_trueRewsize) / sqrt(nTrials_true_rewsize);
                
                subplot(numel(rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_true_rewsize - 1) + mIdx);hold on
                shadedErrorBar((1:analyze_ix)*tbin_ms/1000,mean_decodedTime,sem_decodedTime,'lineProps',{'color',cool3(i_trained_rewsize,:)})
                ylim([0 2]) 
                plot([0 1],[0,1],'k--','linewidth',1.5)  
                if mIdx == 1
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

%% 3c) Comparison of linear model slope coefficients across reward sizes/ mice

analyze_ix = round(1000 / tbin_ms);
cool3repeat = repmat(cool(3),[3,1]);  
x = [1:3 5:7 9:11]; 
vis_mice = [2 3 5];
for i_feature = 2:4
    figure();hold on
    for m_ix = 1:numel(vis_mice) 
        mIdx = vis_mice(m_ix); 
        % load RX for pooled sessions (look at R0 here)
        mouse_RX = RX{mIdx}(pool_sessions{mIdx});
        mouse_RX = cat(1,mouse_RX{:});
        
        % make sure these are the same!
        decoded_time_hat = timePatch_hat;
        true_time = timePatch_true{mIdx}(pool_sessions{mIdx});
        true_time = cat(1,true_time{:});
        pad_trueTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) nan(1,max(0,analyze_ix - length(x)))],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        slope = nan(numel(rewsizes)^2,1); 
        slope_sem = nan(numel(rewsizes)^2,1);
        
        for i_trained_rewsize = 1:numel(rewsizes) 
            % gather decoded time variable from decoder trained on i_rewsize
            decodedTime = cellfun(@(x) x{i_trained_rewsize}, decoded_time_hat{mIdx}(pool_sessions{mIdx}),'un',0);
            decodedTime = cat(1,decodedTime{:}); 
            decodedTime_trainedRewsize = decodedTime(:,i_feature); % with i_feature
            decodedTime_trainedRewsize = cat(1,decodedTime_trainedRewsize{:}); 
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_decodedTime = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',decodedTime_trainedRewsize,'UniformOutput',false);
            decodedTime_trainedRewsize = bin_dt * cat(1,pad_decodedTime{:}); % now this is a matrix
            
            for i_true_rewsize = 1:numel(rewsizes)
                iRewsize_true = rewsizes(i_true_rewsize); 
%                 these_trials = mouse_RX == 10 * iRewsize_true;  
                these_trials = round(mouse_RX/10) == iRewsize_true; % don't care about sec1 onwards
                nTrials_true_rewsize = length(find(these_trials)); 
                
                decodedTime_trueRewsize = decodedTime_trainedRewsize(these_trials,:)'; 
                trueTime_trueRewsize = true_time(these_trials,:)';
                
                mdl = fitlm(trueTime_trueRewsize(:),decodedTime_trueRewsize(:)); %,'intercept',false);
                slope(3 * (i_true_rewsize - 1) + i_trained_rewsize) = mdl.Coefficients.Estimate(2);
                slope_sem(3 * (i_true_rewsize - 1) + i_trained_rewsize) = mdl.Coefficients.SE(2);
            end  
        end
        subplot(1,numel(vis_mice),m_ix);hold on
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
%         ylim([-1 1.25])
        ylim([0 2.25])
    end
end

%% 4a) Time until leave decoding fidelity in terms of auc/PR
% start by visualizing leave-aligned decoded time until leave 
% color by time since last rew 
analyze_ix = round(4000/tbin_ms);
leave_detection_thresholds = .1:.1:2;
for i_feature = 1
    figure();hold on
    for mIdx = 1:5 % 1:numel(mouse_grps)
        % load last reward ix for pooled sessions
        mouse_last_rew_ix = last_rew_ix{mIdx}(pool_sessions{mIdx});
        mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
        % load post rew residence times
        mouse_postRew_rts = postRew_rts{mIdx}(pool_sessions{mIdx});
        mouse_postRew_rts = cat(1,mouse_postRew_rts{:}); 
        nTrials = length(mouse_postRew_rts); 
        % load reward size .. would probably be good to divide by this
        mouse_rewsize = rewsize{mIdx}(pool_sessions{mIdx});
        mouse_rewsize = cat(1,mouse_rewsize{:});
        
        % gather decoded time variable from i_feature
        % the one difference here is that we align to leave rather than reward event
        mouse_timeUntil = cellfun(@(x) x{i_feature}, timeUntil_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeUntil = cat(1,mouse_timeUntil{:});
        % now just pull off decoded timesince after last reward
        mouse_timeUntil = arrayfun(@(iTrial) mouse_timeUntil{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0); 
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeUntil = cellfun(@(x) [nan(max(0,analyze_ix - (length(x)-1)),1) ; x(end-min((length(x)-1),analyze_ix):end)]',mouse_timeUntil,'un',0);
        mouse_timeUntil = cat(1,pad_timeUntil{:}); % now this is a matrix
        
        % make sure this is same as above variable!!
        true_time = timeUntil_true{mIdx}(pool_sessions{mIdx});
        true_time = cat(1,true_time{:});
        pad_trueTime = cellfun(@(x) [nan(1,max(0,analyze_ix - (length(x)-1)),1) x(end-min((length(x)-1),analyze_ix):end)],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);
            cmap = cbrewer('div',"RdBu",nTrials_tt);
            decoded_timeuntil_tt = bin_dt * mouse_timeUntil(these_trials,:); 
            
            [~,postRew_rt_sort] = sort(mouse_postRew_rts(these_trials));
            prt_sorted_these_trials = these_trials(postRew_rt_sort);
            timeuntil_tt_prt_sort = decoded_timeuntil_tt(postRew_rt_sort,:);
            
            subplot(numel(vis_rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_rewsize - 1) + mIdx);hold on
            for iTrial = 1:nTrials_tt
                plot(gauss_smoothing(timeuntil_tt_prt_sort(iTrial,:),1),'color',cmap(iTrial,:),'linewidth',.25)
            end
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end
            if i_rewsize == numel(vis_rewsizes)
                xlabel("True Time");
            end
            if mIdx == 1
                ylabel(sprintf("%i uL \n Decoded Time",iRewsize));
            end
            yticklabels([])
            xticklabels([])
        end
    end 
    suptitle(feature_names(i_feature))
end

%% 4b) Decoded time since reward accuracy metrics

analyze_ix = round(5000/tbin_ms);
leave_detection_thresholds = .1:.1:1; % 2;
for i_feature = 1
    figure();hold on
    for mIdx = 1:numel(mouse_grps)
        % load last reward ix for pooled sessions
        mouse_last_rew_ix = last_rew_ix{mIdx}(pool_sessions{mIdx});
        mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
        % load post rew residence times
        mouse_postRew_rts = postRew_rts{mIdx}(pool_sessions{mIdx});
        mouse_postRew_rts = cat(1,mouse_postRew_rts{:}); 
        nTrials = length(mouse_postRew_rts); 
        % load reward size .. would probably be good to divide by this
        mouse_rewsize = rewsize{mIdx}(pool_sessions{mIdx});
        mouse_rewsize = cat(1,mouse_rewsize{:});
        
        % gather decoded time variable from i_feature
        % the one difference here is that we align to leave rather than reward event
        mouse_timeUntil = cellfun(@(x) x{i_feature}, timeUntil_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeUntil = cat(1,mouse_timeUntil{:});
        % now just pull off decoded timesince after last reward
        mouse_timeUntil = arrayfun(@(iTrial) mouse_timeUntil{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0); 
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeUntil = cellfun(@(x) [nan(max(0,analyze_ix - (length(x)-1)),1) ; x(end-min((length(x)-1),analyze_ix):end)]',mouse_timeUntil,'un',0);
        mouse_timeUntil = cat(1,pad_timeUntil{:}); % now this is a matrix
        
        % make sure this is same as above variable!!
        true_time = timeUntil_true{mIdx}(pool_sessions{mIdx});
        true_time = cat(1,true_time{:});
        pad_trueTime = cellfun(@(x) [nan(1,max(0,analyze_ix - (length(x)-1)),1) x(end-min((length(x)-1),analyze_ix):end)],true_time,'un',0);
        true_time = cat(1,pad_trueTime{:});
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);

            % grab decoded variable
            decoded_timeuntil_tt = (bin_dt * mouse_timeUntil(these_trials,:))'; % transpose so we can cat trials
            cat_decoded_timeuntil_tt = decoded_timeuntil_tt(:);  
            nan_ix = isnan(cat_decoded_timeuntil_tt); 
            true_timeuntil_tt = true_time(these_trials,:)'; % transpose so we can cat trials
            
            % data structures to hold metrics over leave detection thresholds
            prec = nan(numel(leave_detection_thresholds),1); 
            recall = nan(numel(leave_detection_thresholds),1); 
            F = nan(numel(leave_detection_thresholds),1); 
            confusionmats = cell(numel(leave_detection_thresholds),1);  
            prop_pos = nan(numel(leave_detection_thresholds),1); 
            
%             figure()
            for i_threshold = 1:numel(leave_detection_thresholds)  
                t_threshold = leave_detection_thresholds(i_threshold); 
                threshold_true_timeuntil_tt = true_timeuntil_tt < t_threshold; 
                threshold_decoded_timeuntil_tt = decoded_timeuntil_tt < t_threshold; 
                nPoints = length(threshold_decoded_timeuntil_tt(:));
                
                cat_threshold_true_timeuntil_tt = threshold_true_timeuntil_tt(:);
                cat_threshold_decoded_timeuntil_tt = threshold_decoded_timeuntil_tt(:);
                prop_pos(i_threshold) = length(find(cat_threshold_true_timeuntil_tt))/nPoints;
                
                % evaluation metrics
                confusionmats{i_threshold} = confusionmat(cat_threshold_true_timeuntil_tt(~nan_ix),cat_threshold_decoded_timeuntil_tt(~nan_ix));
                prec(i_threshold) = confusionmats{i_threshold}(2,2) / (sum(confusionmats{i_threshold}(:,2)));
                recall(i_threshold) = confusionmats{i_threshold}(2,2) / (sum(confusionmats{i_threshold}(2,:)));
                F(i_threshold) = 2 * (prec(i_threshold) * recall(i_threshold)) / (prec(i_threshold) + recall(i_threshold));
                 % visualization of classification problem per threshold
%                 subplot(1,numel(leave_detection_thresholds),i_threshold)
%                 scatter(.1*randn(nPoints,1)+decoded_timeuntil_tt(:),.1*randn(nPoints,1)+threshold_true_timeuntil_tt(:),5);hold on
%                 xline(t_threshold,'k--','linewidth',1.5) 
%                 title(sprintf("Precision: %.3f \n Recall: %.3f \n FScore: %.3f",prec(i_threshold),recall(i_threshold),F(i_threshold))) 
%                 if i_threshold == 1
%                     ylabel("True Label") 
%                     yticks([0 1]) 
%                     yticklabels(["No Leave" "Leave"]) 
%                 else 
%                     yticks([])
%                 end
%                 xlabel(sprintf("Decoded \n Time Until Leave"))
            end
            subplot(numel(vis_rewsizes),numel(mouse_grps),numel(mouse_grps) * (i_rewsize - 1) + mIdx);hold on
            plot(prec,'linewidth',1.5) 
            plot(recall,'linewidth',1.5)
            plot(prop_pos,'linewidth',1.5)
            ylim([0 1])
            yline(.5,'k--','linewidth',1.5)
            if mIdx == 1
                ylabel("Error Metric")
            end 
            if i_rewsize == 3 
                xlabel("Time threshold")
            end
        end
    end 
    legend(["Precision","Recall","Proportion Positive Labels"])
    suptitle(feature_names(i_feature))
end

%% 4c) Better time until leave visualization
% start by visualizing leave-aligned decoded time until leave 
% color by time since last rew 
% analyze_ix = 15000 / tbin_ms; 

analyze_ix = [4000/tbin_ms 4000/tbin_ms 15000 / tbin_ms 4000/tbin_ms 4000/tbin_ms];
smoothing_sigma = 1; 
vis_rewsizes = [1 2 4];
rdbu3 = cbrewer('div',"RdBu",10);
rdbu3 = rdbu3([3 7 end],:);

vis_mice = 1:5;
close all
for i_feature = 2:4
    figure();hold on
    for m_ix = 1:numel(vis_mice)
        mIdx = vis_mice(m_ix);  % 1:numel(mouse_grps)
        % load last reward ix for pooled sessions
        mouse_last_rew_ix = last_rew_ix{mIdx}(pool_sessions{mIdx});
        mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
        % load post rew residence times
        mouse_postRew_rts = postRew_rts{mIdx}(pool_sessions{mIdx});
        mouse_postRew_rts = cat(1,mouse_postRew_rts{:}); 
        nTrials = length(mouse_postRew_rts); 
        % load reward size .. would probably be good to divide by this
        mouse_rewsize = rewsize{mIdx}(pool_sessions{mIdx});
        mouse_rewsize = cat(1,mouse_rewsize{:});
        
        % gather decoded time variable from i_feature
        mouse_timeUntil = cellfun(@(x) x{i_feature}, timeUntil_hat_sameRewsize{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeUntil = cat(1,mouse_timeUntil{:});
        % now just pull off decoded timesince after last reward
        mouse_timeUntil = arrayfun(@(iTrial) mouse_timeUntil{iTrial}(mouse_last_rew_ix(iTrial):end-25),(1:nTrials)','un',0); 
        % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
        pad_timeUntil = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_timeUntil,'UniformOutput',false);
        mouse_timeUntil = bin_dt * cat(1,pad_timeUntil{:}); % now this is a matrix
        
        for i_rewsize = 1:numel(vis_rewsizes) 
            iRewsize = vis_rewsizes(i_rewsize); 
            these_trials = find(mouse_rewsize == iRewsize);
            nTrials_tt = length(these_trials);
            cmap = cbrewer('div',"RdBu",nTrials_tt);
            
            % grab decoded variable
            timeuntil_tt = mouse_timeUntil(these_trials,:); 
            prts_these_trials = mouse_postRew_rts(these_trials);
%             [~,postRew_rt_sort] = sort(mouse_postRew_rts(these_trials));
%             prt_sorted_these_trials = these_trials(postRew_rt_sort);
%             timeuntil_tt_prt_sort = timeuntil_tt(postRew_rt_sort,:); 
            
            terciles = [0 quantile(prts_these_trials,2) max(prts_these_trials)];
            [~,~,tercile_bin] = histcounts(prts_these_trials,terciles);
            
            subplot(numel(vis_rewsizes),numel(vis_mice),numel(vis_mice) * (i_rewsize - 1) + m_ix);hold on
            
            for i_tercile = 1:max(tercile_bin)
                tt_tercile_mean = nanmean(timeuntil_tt(tercile_bin == i_tercile,:));
                tt_tercile_sem = nanstd(timeuntil_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile))); 
                shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
            end
%             for iTrial = 1:nTrials_tt
%                 plot(gauss_smoothing(timeuntil_tt_prt_sort(iTrial,:),3),'color',cmap(iTrial,:),'linewidth',.25)
%             end
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end
            if m_ix == 1 && i_rewsize == 1
                legend(["Earliest Leaves","Middle Leaves","Late Leaves"])
            end
            if i_rewsize == numel(vis_rewsizes)
                xlabel("Time since last reward");
            end
            if m_ix == 1
                ylabel(sprintf("%i uL \n Decoded Time Until Leave",iRewsize));
            end

            ylim([0 2])
            
%             yticklabels(1:analyze_ix)
%             xticklabels([])
        end
    end 
    suptitle(feature_names(i_feature))
end


%% 5a) Error correlation analysis 
% Are errors in cluster 2-3 time decoding correlated? 
% Assess significance vs trial shuffled correlations

% First, just visualize predictions across trials
decoded_time_hat = timePatch_hat_sameRewsize; % which time decoding are we analyzing
true_time_trials = timePatch_true; 
analyze_ix = round(1000 / tbin_ms); % what time window
close all
for mIdx = 1:numel(mouse_grps)
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
%     figure(); 
    subplot(5,3,1 + 3 * (mIdx - 1));imagesc(true_time)  
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Time on Patch",mouse_names(mIdx)))
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
    subplot(5,3,2 + 3 * (mIdx - 1));imagesc(mouse_cluster2time);  
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Cluster2-Decoded Time on Patch",mouse_names(mIdx)))
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
    subplot(5,3,3 + 3 * (mIdx - 1));imagesc(mouse_cluster3time)  
    caxis([0 max(true_time,[],'all')])
    title(sprintf("%s \n Cluster3-Decoded Time on Patch",mouse_names(mIdx))) 
    xticks([1 25 50])
    xticklabels([1 25 50] * tbin_ms / 1000) 
    xlabel("True Time on Patch") 
    ylabel("Trials")
end

%% 5b) Start w/ bulk correlation between cluster 2 and 3 compared to trial shuffled, within trialtypes
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

%% 5c) correlation coefficient across time since patch onset 

decoded_time_hat = timePatch_hat_sameRewsize; % which time decoding are we analyzing
true_time_trials = timePatch_true; 
analyze_ix = round(1500 / tbin_ms); % what time window
n_shuffles = 100; 

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
        cluster2time_tt = mouse_cluster2time(these_trials,:); 
        cluster3time_tt = mouse_cluster3time(these_trials,:); 
        time_true_tt = true_time(these_trials,:)'; 
        
        r_unshuffled_tt = nan(analyze_ix,1); 
        for i_time = 1:analyze_ix
            % concatenate trials 
            i_time_cluster2_time_tt = cluster2time_tt(:,i_time); 
            i_time_cluster3_time_tt = cluster3time_tt(:,i_time);
            non_nan_ix = ~isnan(i_time_cluster2_time_tt); 
            r = corrcoef(i_time_cluster2_time_tt(non_nan_ix),i_time_cluster3_time_tt(non_nan_ix));  
            r_unshuffled_tt(i_time) = r(2);  
        end

        r_shuffled_tt = nan(n_shuffles,analyze_ix); 
        % now get distribution of correlation coefficients shuffling trials (within trial type)
        for shuffle = 1:n_shuffles 
            cluster2time_tt = mouse_cluster2time(these_trials(randperm(nTrials_tt)),:);  
            cluster3time_tt = mouse_cluster3time(these_trials(randperm(nTrials_tt)),:);
            for i_time = 1:analyze_ix
                % concatenate trials
                i_time_cluster2_time_tt = cluster2time_tt(:,i_time);
                i_time_cluster3_time_tt = cluster3time_tt(:,i_time);
                non_nan_ix = ~isnan(i_time_cluster2_time_tt) & ~isnan(i_time_cluster3_time_tt);
                r = corrcoef(i_time_cluster2_time_tt(non_nan_ix),i_time_cluster3_time_tt(non_nan_ix));
                r_shuffled_tt(shuffle,i_time) = r(2);
            end
        end
        
        subplot(numel(trialtypes),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on
        plot((1:analyze_ix)*tbin_ms/1000,r_unshuffled_tt,'linewidth',1.5); 
        shadedErrorBar((1:analyze_ix)*tbin_ms/1000,mean(r_shuffled_tt,1),std(r_shuffled_tt,1))

        if mIdx == 1 
            ylabel(sprintf("%i Trials",trialtypes(i_tt)))
        end 
        if i_tt == numel(trialtypes) 
            xlabel("Time on patch (sec)")
        end 
        if i_tt == 1
            title(mouse_names(mIdx))
        end
    end
end 

%% 5d) Cross correlation analysis.. directionality? 
decoded_time_hat = timePatch_hat_sameRewsize; % which time decoding are we analyzing
true_time_trials = timePatch_true; 
analyze_ix = round(2000 / tbin_ms); % what time window
n_shuffles = 100; 
max_lag = 25; 

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
        [unshuffled_xcorr,lags] = xcorr(cat_cluster2_time_tt(non_nan_ix),cat_cluster3_time_tt(non_nan_ix),max_lag,'normalize');
        
        shuffled_xcorr = nan(n_shuffles,1+2*max_lag); 
        shuffled_xcorr_asymm = nan(n_shuffles,1+2*max_lag);
        % now get distribution of correlation coefficients shuffling trials (within trial type)
        for shuffle = 1:n_shuffles 
            cluster2time_tt = mouse_cluster2time(these_trials(randperm(nTrials_tt)),:)';  
            cluster3time_tt = mouse_cluster3time(these_trials(randperm(nTrials_tt)),:)'; 
            non_nan_ix = ~isnan(cluster2time_tt) & ~isnan(cluster3time_tt); 
            shuffled_xcorr(shuffle,:) = xcorr(cluster2time_tt(non_nan_ix),cluster3time_tt(non_nan_ix),max_lag,'normalize');
            shuffled_xcorr_asymm(shuffle,:) = shuffled_xcorr(shuffle,:) - fliplr(shuffled_xcorr(shuffle,:)); 
        end 
        
        % plot xcorr
        figure(1)
        subplot(numel(trialtypes),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on
        plot((-max_lag:max_lag)*tbin_ms/1000,unshuffled_xcorr,'linewidth',1.5)
        shadedErrorBar((-max_lag:max_lag)*tbin_ms/1000,mean(shuffled_xcorr),1.96*std(shuffled_xcorr))
        if mIdx == 1
            ylabel(sprintf("Cluster 2-3 xcorr \n %i Trials",trialtypes(i_tt)))
        end
        if i_tt == numel(trialtypes)
            xlabel("Time Lag")
        end
        if i_tt == 1
            title(mouse_names(mIdx))
        end
        
        % plot xcorr asymmetry
        figure(2)
        subplot(numel(trialtypes),numel(mouse_grps),numel(mouse_grps) * (i_tt - 1) + mIdx);hold on
        plot((-max_lag:max_lag)*tbin_ms/1000,unshuffled_xcorr - flipud(unshuffled_xcorr),'linewidth',1.5) 
        shadedErrorBar((-max_lag:max_lag)*tbin_ms/1000,mean(shuffled_xcorr_asymm),1.96*std(shuffled_xcorr_asymm))  
        xlim([0 max_lag*tbin_ms/1000])
%         xline(r_unshuffled_tt,'k--','linewidth',1.5) 
        if mIdx == 1 
            ylabel(sprintf("Cluster 2-3 xcorr Asymmetry \n %i Trials",trialtypes(i_tt)))
        end 
        if i_tt == numel(trialtypes) 
            xlabel("Time Lag")
        end 
        if i_tt == 1
            title(mouse_names(mIdx))
        end
    end
end

%% 5th inning stretch: reorganize variables into reward event cell arrays 
% ultimate goal: 
%   decoded_time_rews [nRews analyze_ix] 
%   rew_size_rews     [nRews 1] 
%   rew_time_rews     [nRews 1] 
%   rew_num_rews      [nRews 1]

%% 6a) Analysis of gain modulation at later reward events 
%  gain of timeSince decoding different at later reward events?


% rew num has greater effect than rew time?


orRd3 = .9 * cbrewer('seq','OrRd',3);
analyze_ix = round(2000 / tbin_ms); %
vis_rewsizes = [1 2 4]; 
vis_rewtimes = 0:2;

% make a darkening cool color scheme
cool3 = cool(3);
cool9_darkening = zeros(9,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool9_darkening(1:3,i) = fliplr(linspace(.3, cool3(1,i), 3));
    cool9_darkening(4:6,i) = fliplr(linspace(.3, cool3(2,i), 3));
    cool9_darkening(7:9,i) = fliplr(linspace(.3, cool3(3,i), 3));
end
cool9_cell{1} = cool9_darkening(1:3,:);
cool9_cell{2} = cool9_darkening(4:6,:);
cool9_cell{3} = cool9_darkening(7:9,:);

vis_mice = 1:5;

for i_feature = 2
    figure()
    for m_ix = 1:numel(vis_mice)  
        mIdx = vis_mice(m_ix); 
        % decoded time since reward per reward event
        mouse_timeSince_hat_rews = cellfun(@(x) x{i_feature}, timeSince_hat_rews{mIdx}(pool_sessions{mIdx}),'un',0);
        mouse_timeSince_hat_rews = cat(1,mouse_timeSince_hat_rews{:}); 
        
        % ok now turn these into matrix of size [nRews analyze_ix]
        pad_timeSince_hat_rews = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_timeSince_hat_rews,'un',0);
        mouse_timeSince_hat_rews = bin_dt * cat(1,pad_timeSince_hat_rews{:}); % now this is a matrix
        
        % reward size, time, number per reward event
        mouse_rewsize_rews = rewsize_rews{mIdx}(pool_sessions{mIdx});  
        mouse_rewsize_rews = cat(1,mouse_rewsize_rews{:}); 
        mouse_rew_time_rews = rew_time_rews{mIdx}(pool_sessions{mIdx});  
        mouse_rew_time_rews = cat(1,mouse_rew_time_rews{:}); 
        mouse_rew_num_rews = rew_num_rews{mIdx}(pool_sessions{mIdx});  
        mouse_rew_num_rews = cat(1,mouse_rew_num_rews{:}); 
        
        for i_rewsize = 1:numel(vis_rewsizes)
            iRewsize = vis_rewsizes(i_rewsize); 
            subplot(numel(vis_rewsizes),numel(vis_mice),numel(vis_mice) * (i_rewsize - 1) + m_ix);hold on
            
            for i_rewtime = 1:numel(vis_rewtimes)
                these_trials = mouse_rewsize_rews == iRewsize & mouse_rew_time_rews == vis_rewtimes(i_rewtime) & ismember(mouse_rew_num_rews,[1 2]); 
                tt_nTrials = length(find(these_trials)); 
                tt_mouse_timeSince_hat_rews = mouse_timeSince_hat_rews(these_trials,:); 
%                 for trial = 1:size(tt_mouse_timeSince_hat_rews,1) 
%                     plot(gauss_smoothing(tt_mouse_timeSince_hat_rews(trial,:),smoothing_sigma),'color',cool6_cell{i_rewsize}(i_rewtime,:),'linewidth',.5)
%                 end
                tt_mean = nanmean(tt_mouse_timeSince_hat_rews); 
                tt_sem = 1.96 * nanstd(tt_mouse_timeSince_hat_rews) / sqrt(tt_nTrials);  
                shadedErrorBar((1:analyze_ix)*tbin_ms/1000,tt_mean,tt_sem,'lineProps',{'color',cool9_cell{i_rewsize}(i_rewtime,:)}) 
                
                if i_rewsize == 1 
                    title(mouse_names(mIdx))
                end
                if i_rewsize == numel(vis_rewsizes) 
                    xlabel("True Time Since Reward")
                end 
                if m_ix == 1
                    ylabel(sprintf("%i uL \n Decoded Time Since Reward",iRewsize))
                    legend(["t = 0","t = 1","t = 2"])
                end
            end  
            
        end
    end 
    suptitle(feature_names(i_feature))
end

%% old code

% B = mnrfit(decoded_timeuntil_tt(:),threshold_true_timeuntil_tt(:)+1);
% pihat = mnrval(B,threshold_true_timeuntil_tt(:));
% 
% [x1,y1,t1,fpr(i_threshold),optimal_pts(i_threshold,:)] = perfcurve(threshold_true_timeuntil_tt(:),decoded_timeuntil_tt(:),'true');
% [x,y,t2,prec(i_threshold)] = perfcurve(threshold_true_timeuntil_tt(:),decoded_timeuntil_tt(:),1,'XCrit','prec');
% prop_pos(i_threshold) = length(find(threshold_true_timeuntil_tt==1))/length(threshold_true_timeuntil_tt(:));
% 
% [x1,y1,t1,fpr(i_threshold),optimal_pts(i_threshold,:)] = perfcurve(threshold_true_timeuntil_tt(:),pihat(:,1),1,'XCrit','tpr');
% [x,y,t2,prec(i_threshold)] = perfcurve(threshold_true_timeuntil_tt(:),pihat(:,1),1,'XCrit','prec');
% prop_pos(i_threshold) = length(find(threshold_true_timeuntil_tt==1))/length(threshold_true_timeuntil_tt(:));
                