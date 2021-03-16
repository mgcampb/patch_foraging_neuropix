%% Script to analyze differences in decoding across reward times 

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
var_dt = diff(nb_results.var_bins{1}{1}{1}(1:2));

%% 0) Reformat decoded time into timesince_hat_rews, make rewsize_rews,rewtime_rews,rewnum_rews

y_hat = nb_results.y_hat;
nMice = numel(analysis_sessions); 

timeSince_hat = cell(nMice,1); 
timePatch_hat = cell(nMice,1); 
timeUntil_hat = cell(nMice,1); 
totalRews_hat = cell(nMice,1); 
timeSince_hat_rews = cell(nMice,1); 
timePatch_hat_rews = cell(nMice,1); 
timeUntil_hat_rews = cell(nMice,1); 
totalRew_hat_rews = cell(nMice,1); 

% to load from decoding
timeSince_ix = 1; 
timePatch_ix = 2; 
timeUntil_ix = 3; 
totalRews_ix = 4; 

rewsizes = [1 2 4];

% reward event information 
rewsize_rews = cell(nMice,1); 
rew_time_rews = cell(nMice,1); 
rew_num_rews = cell(nMice,1); 
RXX = cell(nMice,1); 

% choose which features to reformat for analysis
trial_decoding_features = 1:6;

n_cells = cell(nMice,1); 

for mIdx = 1:numel(analysis_sessions)
    for i_i = 1:numel(analysis_sessions{mIdx})
        i = analysis_sessions{mIdx}(i_i); 
        within_mouse_ix = nb_results.pool_sessions{mIdx}(i_i); % session within mouse mPFC sessions
        sIdx = analysis_sessions{mIdx}(i_i);  
        session = sessions{sIdx}(1:end-4); 
        
        % load behavior + trial information  
        data = load(fullfile(paths.beh_data,sessions{sIdx}));  
        session_title = session([1:2 end-2:end]);
        session_rewsize = mod(data.patches(:,2),10); 
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);   
        rew_sec = data.rew_ts;
        session_prts = patchleave_sec - patchstop_sec;  
        floor_prts = floor(session_prts); 
        patchstop_ix = round(data.patchCSL(:,2)*1000 / tbin_ms) + 1;
        patchleave_ix = round(data.patchCSL(:,3)*1000 / tbin_ms) + 1; 
        prts_ix = patchleave_ix - patchstop_ix + 1;
        nTrials = length(session_rewsize);
        
        % Collect trial reward timings
        rew_sec_cell = cell(nTrials,1);
        session_rew_time_rews = cell(nTrials,1);  
        session_rew_num_rews = cell(nTrials,1); 
        session_rewsize_rews = cell(nTrials,1); 
        nTimesteps = 15; 
        nTrials = length(session_rewsize); 
        
        % make rew_barcode for RXX labels
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
        
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
            session_rew_time_rews{iTrial} = rew_indices; 
            session_rew_num_rews{iTrial} = (1:numel(rew_indices))'; 
            session_rewsize_rews{iTrial} = session_rewsize(iTrial) + zeros(numel(rew_indices),1); 
            rew_sec_cell{iTrial} = rew_indices(rew_indices >= 1); 
        end 
        session_rew_time_rews = cat(1,session_rew_time_rews{:}); 
        session_rew_num_rews = cat(1,session_rew_num_rews{:}); 
        session_rewsize_rews = cat(1,session_rewsize_rews{:});
        nRews = length(session_rew_time_rews); 
        
        % reformat decoded time
        for i_feature = 1:numel(trial_decoding_features)
            iFeature = trial_decoding_features(i_feature);
            timeSince_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeSince_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            timePatch_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timePatch_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            timeUntil_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{timeUntil_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            totalRews_hat{mIdx}{i_i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{within_mouse_ix}{iTrial}{totalRews_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
        end
        
        session_rewTime = cell(nTrials,1); 
        session_rewNum = cell(nTrials,1); 
        session_timeSince_rews = cell(1,length(trial_decoding_features)); 
        session_timePatch_rews = cell(1,length(trial_decoding_features)); 
        session_timeUntil_rews = cell(1,length(trial_decoding_features)); 
        session_totalRew_rews = cell(1,length(trial_decoding_features)); 
        rew_counter = 0; 
        for iTrial = 1:nTrials 
            rew_counter = rew_counter + 1; 
            trial_len_ix = prts_ix(iTrial);
            session_rewTime{iTrial} = zeros(trial_len_ix,1); 
            session_rewNum{iTrial} = zeros(trial_len_ix,1);  
            
            if numel(rew_sec_cell{iTrial}) == 0
                for i_feature = 1:numel(trial_decoding_features)
                    session_timeSince_rews{i_feature}{iTrial}{1} = timeSince_hat{mIdx}{i_i}{i_feature}{iTrial}(1:end); 
                    session_timePatch_rews{i_feature}{iTrial}{1} = timePatch_hat{mIdx}{i_i}{i_feature}{iTrial}(1:end); 
                    session_timeUntil_rews{i_feature}{iTrial}{1} = timeUntil_hat{mIdx}{i_i}{i_feature}{iTrial}(1:end); 
                    session_totalRew_rews{i_feature}{iTrial}{1} = totalRews_hat{mIdx}{i_i}{i_feature}{iTrial}(1:end); 
                end 
            else 
                for i_feature = 1:numel(trial_decoding_features)
                    rew_ix1 = round(rew_sec_cell{iTrial}(1) * 1000 / tbin_ms);
                    session_timeSince_rews{i_feature}{iTrial}{1} = timeSince_hat{mIdx}{i_i}{i_feature}{iTrial}(1:rew_ix1); 
                    session_timePatch_rews{i_feature}{iTrial}{1} = timePatch_hat{mIdx}{i_i}{i_feature}{iTrial}(1:rew_ix1); 
                    session_timeUntil_rews{i_feature}{iTrial}{1} = timeUntil_hat{mIdx}{i_i}{i_feature}{iTrial}(1:rew_ix1); 
                    session_totalRew_rews{i_feature}{iTrial}{1} = totalRews_hat{mIdx}{i_i}{i_feature}{iTrial}(1:rew_ix1); 
                end 
            end
            % now iterate over every probabilistic reward event
            for r = 1:numel(rew_sec_cell{iTrial})
                rew_ix = (rew_sec_cell{iTrial}(r)) * 1000 / tbin_ms; 
                if r == numel(rew_sec_cell{iTrial}) % if it is our last reward
                    session_rewTime{iTrial}(rew_ix:end) = rew_sec_cell{iTrial}(r); 
                    session_rewNum{iTrial}(rew_ix:end) = r;
                end  
                
                if r == numel(rew_sec_cell{iTrial}) % last reward
                    for i_feature = 1:numel(trial_decoding_features)
                        session_timeSince_rews{i_feature}{iTrial}{r+1} = timeSince_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:end);
                        session_timePatch_rews{i_feature}{iTrial}{r+1} = timePatch_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:end);
                        session_timeUntil_rews{i_feature}{iTrial}{r+1} = timeUntil_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:end);
                        session_totalRew_rews{i_feature}{iTrial}{r+1} = totalRews_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:end);
                    end
                else 
                    for i_feature = 1:numel(trial_decoding_features)
                        next_rew_ix = (rew_sec_cell{iTrial}(r+1)) * 1000 / tbin_ms; 
                        session_timeSince_rews{i_feature}{iTrial}{r+1} = timeSince_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                        session_timePatch_rews{i_feature}{iTrial}{r+1} = timePatch_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                        session_timeUntil_rews{i_feature}{iTrial}{r+1} = timeUntil_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                        session_totalRew_rews{i_feature}{iTrial}{r+1} = totalRews_hat{mIdx}{i_i}{i_feature}{iTrial}(rew_ix:next_rew_ix);
                    end
                end
                rew_counter = rew_counter + 1; 
            end   
        end
        
        % ok now go back and concatenate everyone
        for i_feature = 1:numel(trial_decoding_features)
            session_timeSince_rews{i_feature} = cat(2,session_timeSince_rews{i_feature}{:})';
            session_timePatch_rews{i_feature} = cat(2,session_timePatch_rews{i_feature}{:})';
            session_timeUntil_rews{i_feature} = cat(2,session_timeUntil_rews{i_feature}{:})';
            session_totalRew_rews{i_feature} = cat(2,session_totalRew_rews{i_feature}{:})';
        end
        
        % save trial information to data structures
        timeSince_hat_rews{mIdx}{i_i} = session_timeSince_rews;
        timePatch_hat_rews{mIdx}{i_i} = session_timePatch_rews;
        timeUntil_hat_rews{mIdx}{i_i} = session_timeUntil_rews;
        totalRew_hat_rews{mIdx}{i_i} = session_totalRew_rews;
        rewsize_rews{mIdx}{i_i} = session_rewsize_rews;
        rew_time_rews{mIdx}{i_i} = session_rew_time_rews;
        rew_num_rews{mIdx}{i_i} = session_rew_num_rews;
        RXX{mIdx}{i_i} = session_RXX;
    end
end
clear y_hat % we now have this in an easier form to work with

%% 1) Visualize decoding separated by reward timing

analyze_ix = round([3000 3000 5000 3000 3000] / tbin_ms); %
vis_rewsizes = [1 2 4]; 
vis_rewtimes = 0:2;
smoothing_sigma = 1;

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

decoded_hat_rews = timeSince_hat_rews;
this_var_name = "Time Since Rew";

for i_feature = [1 2 3 5]
    figure()
    for m_ix = 1:numel(analysis_mice)  
        mIdx = analysis_mice(m_ix); 
        % decoded time since reward per reward event
        mouse_decoded_hat_rews = cellfun(@(x) x{i_feature}, decoded_hat_rews{mIdx},'un',0);
        mouse_decoded_hat_rews = cat(1,mouse_decoded_hat_rews{:}); 
        
        % ok now turn these into matrix of size [nRews analyze_ix]
        pad_decoded_hat_rews = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_decoded_hat_rews,'un',0);
        if this_var_name ~= "Total Reward"
            mouse_decoded_hat_rews = var_dt * cat(1,pad_decoded_hat_rews{:}); % now this is a matrix
        else 
            mouse_decoded_hat_rews = cat(1,pad_decoded_hat_rews{:}); % now this is a matrix 
        end
        
        % reward size, time, number per reward event
        mouse_rewsize_rews = cat(1,rewsize_rews{mIdx}{:});
        mouse_rew_time_rews = cat(1,rew_time_rews{mIdx}{:}); 
        mouse_rew_num_rews = cat(1,rew_num_rews{mIdx}{:}); 
        
        for i_rewsize = 1:numel(vis_rewsizes)
            iRewsize = vis_rewsizes(i_rewsize); 
            subplot(numel(vis_rewsizes),numel(analysis_mice),numel(analysis_mice) * (i_rewsize - 1) + m_ix);hold on
            
            for i_rewtime = 1:numel(vis_rewtimes)
                these_trials = mouse_rewsize_rews == iRewsize & mouse_rew_time_rews == vis_rewtimes(i_rewtime) & ismember(mouse_rew_num_rews,[1 2]); 
                tt_nTrials = length(find(these_trials));  
                if tt_nTrials > 1
                    tt_mouse_decoded_hat_rews = mouse_decoded_hat_rews(these_trials,:); 

                    tt_mean = nanmean(tt_mouse_decoded_hat_rews); 
                    tt_sem = 1.96 * nanstd(tt_mouse_decoded_hat_rews) / sqrt(tt_nTrials);
                    shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_mean,tt_sem,'lineProps',{'color',cool9_cell{i_rewsize}(i_rewtime,:)})
                end
            end
            
            % subplot formatting
            ylim([0 analyze_ix(mIdx) *tbin_ms/1000])
            xlim([0 analyze_ix(mIdx) *tbin_ms/1000])
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end
            if i_rewsize == numel(vis_rewsizes)
                xlabel(sprintf("True %s (sec)",this_var_name))
            end
            
            if m_ix == 1
                ylabel(sprintf("%i uL \n Decoded %s (sec)",iRewsize,this_var_name))
                legend(["t = 0","t = 1","t = 2"])
            end
            set(gca,'fontsize',12)
        end
    end
    suptitle(feature_names(i_feature))
end

%% 2) Visualize decoding separated by reward number

analyze_ix = round([2000 2000 2000 2000 2000] / tbin_ms); %
vis_rewsizes = [1 2 4]; 
vis_rewnums = 1:3;
smooth_sigma = 1;
smoothing = true; 

decoded_hat_rews = totalRew_hat_rews;
this_var_name = "Total Reward";
close all
for i_feature = 4
    figure()
    for m_ix = 1:numel(analysis_mice)  
        mIdx = analysis_mice(m_ix); 
        % decoded time since reward per reward event
        mouse_decoded_hat_rews = cellfun(@(x) x{i_feature}, decoded_hat_rews{mIdx},'un',0);
        mouse_decoded_hat_rews = cat(1,mouse_decoded_hat_rews{:}); 
        
        % ok now turn these into matrix of size [nRews analyze_ix]
        pad_decoded_hat_rews = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_decoded_hat_rews,'un',0);
        if this_var_name ~= "Total Reward"
            mouse_decoded_hat_rews = var_dt * cat(1,pad_decoded_hat_rews{:}); % now this is a matrix
        else 
            mouse_decoded_hat_rews = 1 + cat(1,pad_decoded_hat_rews{:}); % now this is a matrix 
        end
        
        % smooth?
        if smoothing == true
            smoothed = arrayfun(@(iTrial) gauss_smoothing(mouse_decoded_hat_rews(iTrial,:),smooth_sigma),(1:size(mouse_decoded_hat_rews,1))','un',0);
            mouse_decoded_hat_rews = cat(1,smoothed{:});
        end
        
        % reward size, time, number per reward event
        mouse_rewsize_rews = cat(1,rewsize_rews{mIdx}{:});
        mouse_rew_time_rews = cat(1,rew_time_rews{mIdx}{:}); 
        mouse_rew_num_rews = cat(1,rew_num_rews{mIdx}{:}); 
        
        for i_rewsize = 1:numel(vis_rewsizes)
            iRewsize = vis_rewsizes(i_rewsize); 
            subplot(numel(vis_rewsizes),numel(analysis_mice),numel(analysis_mice) * (i_rewsize - 1) + m_ix);hold on
            
            for i_rewnum = 1:numel(vis_rewnums)
                these_trials = mouse_rewsize_rews == iRewsize & mouse_rew_num_rews == vis_rewnums(i_rewnum); % & ismember(mouse_rew_time_rews,0:1) ; 
                tt_nTrials = length(find(these_trials));  
                if tt_nTrials > 1
                    tt_mouse_decoded_hat_rews = mouse_decoded_hat_rews(these_trials,:); 

                    tt_mean = nanmean(tt_mouse_decoded_hat_rews); 
                    tt_sem = 1.96 * nanstd(tt_mouse_decoded_hat_rews) / sqrt(tt_nTrials);
                    shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_mean,tt_sem,'lineProps',{'color',cool9_cell{i_rewsize}(4 - i_rewnum,:)})
                end
            end
            
            % subplot formatting 
            if this_var_name ~= "Total Reward"
                ylim([0 analyze_ix(mIdx) *tbin_ms/1000])
                xlim([0 analyze_ix(mIdx) *tbin_ms/1000])
            else  
                ylim([0 6])
                xlim([0 analyze_ix(mIdx) *tbin_ms/1000])
            end
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end 
            
            if i_rewsize == numel(vis_rewsizes) 
                if this_var_name ~= "Total Reward"
                    xlabel(sprintf("True %s (sec)",this_var_name))
                else 
                    xlabel("Time Since Reward (sec)")
                end
            end
            
            if m_ix == 1
                ylabel(sprintf("%i uL \n Decoded %s (sec)",iRewsize,this_var_name))
                legend(["Rew 1","Rew 2","Rew 3"])
            end
            set(gca,'fontsize',12)
        end
    end
    suptitle(feature_names(i_feature))
end

%% 3) Visualize Decoding separated by RXX

cool12_lightening = zeros(12,3);
for i = 1:3
    cool12_lightening(1:4,i) = fliplr(linspace(.1, cool3(1,i), 4));
    cool12_lightening(5:8,i) = fliplr(linspace(.1, cool3(2,i), 4));
    cool12_lightening(9:12,i) = fliplr(linspace(.1, cool3(3,i), 4));
end
cool12_cell{1} = flipud(cool12_lightening(1:4,:));
cool12_cell{2} = flipud(cool12_lightening(5:8,:));
cool12_cell{3} = flipud(cool12_lightening(9:12,:));

analyze_ix = round(3000 / tbin_ms); %
vis_rewsizes = [1 2 4]; 
trialtypes = {[100, 110, 101, 111],[200,220,202,222],[400,440,404,444]};
smooth_sigma = 1;
smoothing = true; 

decoded_hat = totalRews_hat;
this_var_name = "Total Reward";
close all
for i_feature = 1:5
    figure()
    for m_ix = 1:numel(analysis_mice)  
        mIdx = analysis_mice(m_ix); 
        % decoded time since reward per reward event
        mouse_decoded_hat = cellfun(@(x) x{i_feature}, decoded_hat{mIdx},'un',0);
        mouse_decoded_hat = cat(1,mouse_decoded_hat{:}); 
        
        % ok now turn these into matrix of size [nRews analyze_ix]
        pad_decoded_hat = cellfun(@(x) [x(1:min(length(x),analyze_ix)) ; nan(max(0,analyze_ix - length(x)),1)]',mouse_decoded_hat,'un',0);
        if this_var_name ~= "Total Reward"
            mouse_decoded_hat = var_dt * cat(1,pad_decoded_hat{:}); % now this is a matrix
        else 
            mouse_decoded_hat = 1 + cat(1,pad_decoded_hat{:}); % now this is a matrix 
        end
        
        % smooth?
        if smoothing == true
            smoothed = arrayfun(@(iTrial) gauss_smoothing(mouse_decoded_hat(iTrial,:),smooth_sigma),(1:size(mouse_decoded_hat,1))','un',0);
            mouse_decoded_hat = cat(1,smoothed{:});
        end
        
        % reward size, time, number per reward event
        mouse_RXX = cat(1,RXX{mIdx}{:});
        
        for i_rewsize = 1:numel(vis_rewsizes)
            iRewsize = vis_rewsizes(i_rewsize); 
            subplot(numel(vis_rewsizes),numel(analysis_mice),numel(analysis_mice) * (i_rewsize - 1) + m_ix);hold on
            
            for i_tt = 1:numel(trialtypes{i_rewsize})
                these_trials = mouse_RXX == trialtypes{i_rewsize}(i_tt); % & ismember(mouse_rew_time_rews,0:1) ; 
                tt_nTrials = length(find(these_trials));  
                if tt_nTrials > 1
                    tt_mouse_decoded_hat = mouse_decoded_hat(these_trials,:); 

                    tt_mean = nanmean(tt_mouse_decoded_hat); 
                    tt_sem = 1.96 * nanstd(tt_mouse_decoded_hat) / sqrt(tt_nTrials);
                    shadedErrorBar((1:analyze_ix)*tbin_ms/1000,tt_mean,tt_sem,'lineProps',{'color',cool12_cell{i_rewsize}(i_tt,:)})
                end
            end
            
            % subplot formatting 
            if this_var_name ~= "Total Reward"
                ylim([0 analyze_ix * tbin_ms/1000])
                xlim([0 analyze_ix *tbin_ms/1000])
            else  
                ylim([0 6])
                xlim([0 analyze_ix *tbin_ms/1000])
            end
            if i_rewsize == 1
                title(mouse_names(mIdx))
            end 
            
            if i_rewsize == numel(vis_rewsizes) 
                if this_var_name ~= "Total Reward"
                    xlabel(sprintf("True %s (sec)",this_var_name))
                else 
                    xlabel("Time Since Reward (sec)")
                end
            end
            
            if m_ix == 1
                ylabel(sprintf("%i uL \n Decoded %s (sec)",iRewsize,this_var_name))
%                 legend(["Rew 1","Rew 2","Rew 3"])
            end
            set(gca,'fontsize',12)
        end
    end
    suptitle(feature_names(i_feature))
end 

