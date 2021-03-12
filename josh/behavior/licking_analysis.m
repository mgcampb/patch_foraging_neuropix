%% Analysis of licking behavior
%  1) What does timecourse of licking look lick in trials?
%  2) Is licking predictive of patch leave? 
%  3) Reward history / trial type? ** would be nice

paths = struct; 
% just use recording days
paths.neuro_data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% use all sessions w/ behavioral data
paths.beh_data = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path (currently do not have lick timestamps)
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

sessions = dir(fullfile(paths.neuro_data,'*.mat'));
sessions = {sessions.name};   

% lick indexing / smoothing options
tbin_sec = 0.02; % time bin for whole session rate matrix (in sec)
smooth_sigma_sec = .05; % 50 msec smoothing  
tstart = 0; 

mouse_grps = {1:2,3:8,10:13,15:18,[23 25]};   
mPFC_sessions = [1:8 10:13 15:18 23 25];   
mouse_names = ["m75","m76","m78","m79","m80"]; 
session_titles = cell(numel(mPFC_sessions),1); 
for i = 1:numel(mPFC_sessions)
    sIdx = mPFC_sessions(i);   
    session_title = ['m' sessions{sIdx}(1:2) ' ' sessions{sIdx}(end-6) '/' sessions{sIdx}(end-5:end-4)];  
    session_titles{i} = session_title;
end 

% % to pare down to just recording sessions
% recording_sessions = dir(fullfile(paths.neuro_data,'*.mat'));
% recording_sessions = {recording_sessions.name};
% % to just use recording sessions
% recording_session_bool = cellfun(@(x) ismember(x,recording_sessions),sessions); 

%% Load licks and some trial information  

mouse_rewsize = cell(numel(mouse_grps),1); 
mouse_N0 = cell(numel(mouse_grps),1); 
mouse_prts = cell(numel(mouse_grps),1);   
mouse_qrts = cell(numel(mouse_grps),1);  
mouse_RX = cell(numel(mouse_grps),1);  
mouse_RXX = cell(numel(mouse_grps),1); 
mouse_RNil = cell(numel(mouse_grps),1); 
rew_barcodes = cell(numel(mouse_grps),1);  
lick_ts_trials = cell(numel(mouse_grps),1); 
lick_counts_trials = cell(numel(mouse_grps),1); 
licks_smoothed_trials = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    mouse_prts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_N0{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    mouse_qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_RX{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    mouse_RNil{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_RXX{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    rew_barcodes{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    lick_ts_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    lick_counts_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    licks_smoothed_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.neuro_data,session));   
        
        % get patch information
        rewsize = mod(data.patches(:,2),10);   
        N0 = round(mod(data.patches(:,2),100)/10);
        N0(N0 == 3) = .125;
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5;   
        
        % get behavior timing information 
        rew_sec = data.rew_ts; 
        patchcue_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2); 
        patchleave_sec = data.patchCSL(:,3);
        patchstop_ix = round(patchstop_sec / tbin_sec); 
        patchleave_ix = round(patchleave_sec / tbin_sec); 
        cue_rts = patchstop_sec - patchcue_sec;
        prts = patchleave_sec - patchstop_sec;   
        floor_prts = floor(prts);  

        % make barcode matrices
        nTimesteps = 30; 
        nTrials = length(rewsize); 
        rew_barcode = zeros(length(data.patchCSL) , nTimesteps);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
            rew_barcode(iTrial , (max(rew_indices)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
        end  
        
        % turn RX and RXX information into strings for easy access 
        % this is not beautiful but dont judgeee
        RX = nan(nTrials,1); 
        RXX = nan(nTrials,1); 
        RNil = nan(nTrials,1); 
        for iRewsize = [1 2 4] 
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0) = double(sprintf("%i0",iRewsize));
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize) = double(sprintf("%i%i",iRewsize,iRewsize));  
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0) = double(sprintf("%i00",iRewsize)); 
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize) = double(sprintf("%i0%i",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize)); 
            RNil(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) < 0) = iRewsize;
        end
        
        lick_ts = data.lick_ts; 
        tstart = 0; 
        tend = max(patchleave_sec); 
        tbinedge = tstart:tbin_sec:(tend+tbin_sec);
        % compute binned lick counts
        lick_counts = histcounts(lick_ts,tbinedge) / tbin_sec;
        % smoothed lick rate
        licks_smoothed = gauss_smoothing(lick_counts,smooth_sigma_sec/tbin_sec); 
        
        % Turn this into trialed form for analysis 
        lick_ts_trials{mIdx}{i} = cell(nTrials,1);  
        lick_counts_trials{mIdx}{i} = cell(nTrials,1);  
        licks_smoothed_trials{mIdx}{i} = cell(nTrials,1);  
        for iTrial = 1:nTrials 
            lick_ts_trials{mIdx}{i}{iTrial} = lick_ts(lick_ts > patchstop_sec(iTrial) & lick_ts < patchleave_sec(iTrial)) - patchstop_sec(iTrial); 
            lick_counts_trials{mIdx}{i}{iTrial} = lick_counts(patchstop_ix(iTrial):patchleave_ix(iTrial)); 
            licks_smoothed_trials{mIdx}{i}{iTrial} = licks_smoothed(patchstop_ix(iTrial):patchleave_ix(iTrial)); 
        end
        
        % log data 
        mouse_rewsize{mIdx}{i} = rewsize;  
        mouse_N0{mIdx}{i} = N0;  
        mouse_prts{mIdx}{i} = prts;   
        mouse_qrts{mIdx}{i} = cue_rts;  
        mouse_RX{mIdx}{i} = RX; 
        mouse_RXX{mIdx}{i} = RXX;  
        mouse_RNil{mIdx}{i} = RNil;
        rew_barcodes{mIdx}{i} = rew_barcode; 
    end
end 

%% 1) What does timecourse of licking look lick different trial types? 
%  - Start with RX / RXX analysis

% RX in single sessions
RX_colors = [.5 1 1;0 1 1; .75 .75 1 ; .5 .5 1 ; 1 .5 1; 1 0 1];
for mIdx = 1:5 
    figure() 
    for i = 1:numel(mouse_grps{mIdx})  
        counter = 1; 
        for iRewsize = [1 2 4]
            subplot(3,numel(mouse_grps{mIdx}),i + numel(mouse_grps{mIdx}) * (min(3,iRewsize)-1)); hold on
            for trial_type = [double(sprintf("%i0",iRewsize)) double(sprintf("%i%i",iRewsize,iRewsize))]
                these_trials = find(mouse_RX{mIdx}{i} == trial_type);
                lick_counts_trialtype = cat(1,licks_smoothed_trials{mIdx}{i}(these_trials));
                t_lens = cellfun(@length,lick_counts_trialtype);
                max_t_len = max(t_lens);   
                lick_counts_trialtype = arrayfun(@(i) [lick_counts_trialtype{i} nan(1,max_t_len-t_lens(i))],1:length(t_lens),'un',0)';  
                lick_counts_trialtype = cat(1,lick_counts_trialtype{:});  
                plot(nanmean(lick_counts_trialtype),'color',RX_colors(counter,:))
                counter = counter + 1; 
                xlim([0 100]) 
                ylim([0 12])
            end
        end
    end 
end

%% 1b) RX and RXX pooling across sessions within mice 

figure() 
for mIdx = 1:5
    mouse_smoothed_licks = cat(1,licks_smoothed_trials{mIdx}{:}); 
    m_RX = cat(1,mouse_RX{mIdx}{:});
    
    counter = 1;
    for iRewsize = [1 2 4]
        subplot(3,numel(mouse_grps),mIdx + numel(mouse_grps) * (min(3,iRewsize)-1)); hold on
        for trial_type = [double(sprintf("%i0",iRewsize)) double(sprintf("%i%i",iRewsize,iRewsize))] 
            these_trials = find(m_RX == trial_type);
            smoothed_licks_trialtype = cat(1,mouse_smoothed_licks(these_trials));
            t_lens = cellfun(@length,smoothed_licks_trialtype);
            max_t_len = max(t_lens);
            smoothed_licks_trialtype = arrayfun(@(i) [smoothed_licks_trialtype{i} nan(1,max_t_len-t_lens(i))],1:length(t_lens),'un',0)';
            smoothed_licks_trialtype = cat(1,smoothed_licks_trialtype{:});
            plot(nanmean(smoothed_licks_trialtype),'color',RX_colors(counter,:))
            counter = counter + 1;
            xlim([0 100])
            ylim([0 12])  
            if mIdx == 1 
                ylabel("Lick Rate (Hz)")
            end
        end
    end
end

RXX_colors = [0.5 0.5 0.5 ; 0.5 .75 .75 ; .25 1.0 1.0 ; 0.0 1.0 1.0 ; ... 
              0.5 0.5 0.5 ; 0.5 0.5 0.7 ; .75 .75 1.0 ; 0.5 0.5 1.0 ; 
              0.5 0.5 0.5 ; 1.0 .75 1.0 ; 1.0 0.5 1.0 ; 1.0 0.0 1.0];
figure() 
for mIdx = 1:5
    mouse_smoothed_licks = cat(1,licks_smoothed_trials{mIdx}{:}); 
    m_RX = cat(1,mouse_RXX{mIdx}{:});
    
    counter = 1;
    for iRewsize = [1 2 4]
        subplot(3,numel(mouse_grps),mIdx + numel(mouse_grps) * (min(3,iRewsize)-1)); hold on
        for trial_type = [double(sprintf("%i00",iRewsize)) double(sprintf("%i%i0",iRewsize,iRewsize)) double(sprintf("%i0%i",iRewsize,iRewsize)) double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize))] 
            these_trials = find(m_RX == trial_type);
            smoothed_licks_trialtype = cat(1,mouse_smoothed_licks(these_trials));
            t_lens = cellfun(@length,smoothed_licks_trialtype);
            max_t_len = max(t_lens);
            smoothed_licks_trialtype = arrayfun(@(i) [smoothed_licks_trialtype{i} nan(1,max_t_len-t_lens(i))],1:length(t_lens),'un',0)';
            smoothed_licks_trialtype = cat(1,smoothed_licks_trialtype{:});
            plot(nanmean(smoothed_licks_trialtype),'color',RXX_colors(counter,:))
            counter = counter + 1;
            xlim([0 150])
            ylim([0 12])  
            if mIdx == 1 
                ylabel("Lick Rate (Hz)")
            end
        end
    end
end

%% 2) Is licking predictive of patch leave?  
%   - Start with RNil -> PRT, then move to time post last reward  

cool3 = cool(3); 
n = 5;
quintiles_cmap = cell(3,1); 
for r = 1:3
    quintiles_cmap{r} = [linspace(.5,cool3(r,1),n)' linspace(.5,cool3(r,2),n)' linspace(.5,cool3(r,3),n)'];
end

max_timepoint = 100;

for mIdx = 1:5 
    mouse_smoothed_licks = cat(1,licks_smoothed_trials{mIdx}{:}); 
    m_RNil = cat(1,mouse_RNil{mIdx}{:}); 
    m_PRTs = cat(1,mouse_prts{mIdx}{:});
    
    r_prt = nan(3,max_timepoint); 
    p_prt = nan(3,max_timepoint); 
    for iRewsize = [1 2 4]
        subplot(4,numel(mouse_grps),mIdx + numel(mouse_grps) * (min(3,iRewsize)-1)); hold on
        these_trials = m_RNil == iRewsize; 
        these_prts = m_PRTs(these_trials);
        [~,~,prt_quintile] = histcounts(these_prts,quantile(these_prts,0:.2:1));  
        tt_smoothed_licks = mouse_smoothed_licks(these_trials); 
        
        t_lens = cellfun(@length,tt_smoothed_licks);
        max_t_len = max(t_lens);  
        tt_smoothed_licks = arrayfun(@(i) [tt_smoothed_licks{i} nan(1,max_t_len-t_lens(i))],1:length(t_lens),'un',0)';
        tt_smoothed_licks = cat(1,tt_smoothed_licks{:}); % now this is an array 
        lick_traces_means = nan(n,max_t_len); 
        for quintile = 1:5 
            lick_traces_means(quintile,:) = nanmean(tt_smoothed_licks(prt_quintile == quintile,:));
            plot(lick_traces_means(quintile,:),'color',quintiles_cmap{min(3,iRewsize)}(quintile,:))
        end 
        xlim([0 max_timepoint])
        xticks(0:25:max_timepoint) 
        xticklabels((0:25:max_timepoint)*tbin_sec)
        ylim([0 13])     
        if mIdx == 1
            ylabel("Lick Rate (Hz)")  
        end
        
        if iRewsize == 1
            title(mouse_names(mIdx))
        end
        
        % Now calculate significance per timepoint and plot
        subplot(4,numel(mouse_grps),mIdx + 3 * numel(mouse_grps))  ;hold on
        for t = 1:max_timepoint 
            timepoint_licks = tt_smoothed_licks(:,t); 
            [r,p] = corrcoef(timepoint_licks(~isnan(timepoint_licks)),these_prts(~isnan(timepoint_licks))); 
            r_prt(min(3,iRewsize),t) = r(2); 
            p_prt(min(3,iRewsize),t) = p(2);  
            if t > 1
                if p_prt(min(3,iRewsize),t-1) < .01
                    plot([t-1 t],r_prt(min(3,iRewsize),t-1:t),'-','color',cool3(min(3,iRewsize),:),'linewidth',1.5)
                else 
                    plot([t-1 t],r_prt(min(3,iRewsize),t-1:t),':','color',cool3(min(3,iRewsize),:))
                end 
            end
        end 
        ylim([-1 1])
        if mIdx == 1
            ylabel("Pearson Correlation")  
        end
        xticks(0:25:max_timepoint) 
        xticklabels((0:25:max_timepoint)*tbin_sec) 
        xlabel("Time Since Patch Stop (sec)")
    end
end

