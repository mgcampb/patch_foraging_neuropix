%% Some behavior analysis around the proximity cue 
%  1) Does proportion of patch engagement change throughout the session? 
%  2) What does velocity look like on patches that were engaged/not engaged w/? 
%  3) Patch history effects: does previous trial type affect P(stop)

paths = struct; 
% just use recording days
paths.neuro_data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% use all sessions w/ behavioral data
paths.beh_data = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

sessions = dir(fullfile(paths.beh_data,'*.mat'));
sessions = {sessions.name};   

tbin_sec = 0.02; % for indexing vel later

mice = ["75","76","78","79","80"];  
mouse_names = ["m75","m76","m78","m79","m80"];  
% to use all sessions w/ behavior  
mouse_grps = cell(length(mice),1); 
for m = 1:numel(mice) 
    mouse_grps{m} = find(cellfun(@(x) strcmp(x(1:2),mice(m)),sessions,'un',1));
end 

% to pare down to just recording sessions
recording_sessions = dir(fullfile(paths.neuro_data,'*.mat'));
recording_sessions = {recording_sessions.name};
% to just use recording sessions
recording_session_bool = cellfun(@(x) ismember(x,recording_sessions),sessions);

%% Get trial-type z-scored prts, velocity trace per prox cue, patchStop bool

prts_zscored = cell(numel(mouse_grps),1); % PRTs zscored within trial type
patchStop_bool = cell(numel(mouse_grps),1); 
vel_cue_trials = cell(numel(mouse_grps),1);  
recording_session_bool_mice = cell(numel(mouse_grps),1); 
prev_trialtype = cell(numel(mouse_grps),1); 
next_patchStop_bool = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps) 
    prts_zscored{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    patchStop_bool{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    vel_cue_trials{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    recording_session_bool_mice{mIdx} = false(numel(mouse_grps{mIdx}),1);   
    prev_trialtype{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    next_patchStop_bool{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    for i = 1:numel(mouse_grps{mIdx}) % iterate over sessions
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.beh_data,session));    
        recording_session_bool_mice{mIdx}(i) = recording_session_bool(sIdx); 
        
        % get patch information
        rewsize = mod(data.patches(:,2),10);   
        N0 = round(mod(data.patches(:,2),100)/10);
        N0(N0 == 3) = .125;
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5;   
        
        % get behavior timing information for stop trials
        patchcue_hit_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2); 
        patchleave_hit_sec = data.patchCSL(:,3);   
        prts = patchleave_hit_sec - patchstop_sec;
        
        % zscore PRTs within trial type
        nTrials = length(prts); 
        session_prts_zscored = nan(nTrials,1);   
        session_prev_trialtype = nan(nTrials,1);  
        counter = 1; 
        for iRewsize = [1 2 4]
            for iN0 = [.125 .25 .5]
                session_prts_zscored(rewsize == iRewsize & N0 == iN0) = zscore(prts(rewsize == iRewsize & N0 == iN0));  
                session_prev_trialtype(rewsize == iRewsize & N0 == iN0) = counter; 
                counter = counter + 1; 
            end
        end 
        session_prev_trialtype(end) = []; 
        
        % get behavior timing information for no stop trials 
        if ~isempty(data.patchCL) % make sure we had some miss trials
            patchcue_miss_sec = data.patchCL(:,1);
            patchleave_miss_sec = data.patchCL(:,2); 
        else 
            patchcue_miss_sec = []; 
            patchleave_miss_sec = []; 
        end
        
        % collect velocity trace per cue
        velocity = data.vel;  
        nCues = length(patchcue_hit_sec) + length(patchcue_miss_sec);  
        nHits = length(patchcue_hit_sec); 
        nMiss = length(patchcue_miss_sec); 
        session_patchStop_bool = nan(nCues,1); 
        session_vel_cue_trials = cell(nCues,1); 
        pre_cue_sec = 1; % save velocity to .5 seconds before cue start
        post_cue_sec = 0; % save velocity to .5 seconds after cue end  
        % vector ix that we will update to track where we are hit/miss cues
        hit_ix = 1; 
        if ~isempty(patchcue_miss_sec)
            miss_ix = 1;  
        else 
            miss_ix = NaN; 
        end
        % Now collect velocity traces and patchstop bool 
        for iCue = 1:nCues 
            if ~any(isnan([hit_ix miss_ix]))
                if patchcue_hit_sec(hit_ix) < patchcue_miss_sec(miss_ix) % next hit cue came before miss
                    session_patchStop_bool(iCue) = 1;  
                    cue_start_ix = round((patchcue_hit_sec(hit_ix) - pre_cue_sec) / tbin_sec);
                    cue_end_ix = round((patchstop_sec(hit_ix) + post_cue_sec) / tbin_sec);
                    hit_ix = hit_ix + 1; 
                    if hit_ix > nHits 
                        hit_ix = NaN; 
                    end
                elseif patchcue_miss_sec(miss_ix) < patchcue_hit_sec(hit_ix) % next miss cue came before hit
                    session_patchStop_bool(iCue) = 0;   
                    cue_start_ix = round((patchcue_miss_sec(miss_ix) - pre_cue_sec) / tbin_sec);
                    cue_end_ix = round((patchleave_miss_sec(miss_ix) + post_cue_sec) / tbin_sec);
                    miss_ix = miss_ix + 1; 
                    if miss_ix > nMiss 
                        miss_ix = NaN;
                    end
                end
            else % we are out of hit or miss trials
                if isnan(miss_ix) 
                    session_patchStop_bool(iCue) = 1;  
                    cue_start_ix = round((patchcue_hit_sec(hit_ix) - pre_cue_sec) / tbin_sec);
                    cue_end_ix = round((patchstop_sec(hit_ix) + post_cue_sec) / tbin_sec);
                    hit_ix = hit_ix + 1; 
                else 
                    session_patchStop_bool(iCue) = 0;   
                    cue_start_ix = round((patchcue_miss_sec(miss_ix) - pre_cue_sec) / tbin_sec);
                    cue_end_ix = round((patchleave_miss_sec(miss_ix) + post_cue_sec) / tbin_sec);
                    miss_ix = miss_ix + 1; 
                end
            end
            % add to velocity traces
            session_vel_cue_trials{iCue} = velocity(cue_start_ix:min(length(velocity),cue_end_ix)); 
        end
        
        session_next_patchStop_bool = nan(nTrials-1,1); 
        hit_counter = 1;
        for iCue = 1:(nCues-1)
            if session_patchStop_bool(iCue) == 1 
                session_next_patchStop_bool(hit_counter) = session_patchStop_bool(iCue+1); 
                hit_counter = hit_counter + 1;  
                if hit_counter == nTrials 
                    break
                end
            end
        end

        % log data 
        prts_zscored{mIdx}{i} = session_prts_zscored; 
        patchStop_bool{mIdx}{i} = session_patchStop_bool;
        vel_cue_trials{mIdx}{i} = session_vel_cue_trials; 
        prev_trialtype{mIdx}{i} = session_prev_trialtype;  
        next_patchStop_bool{mIdx}{i} = session_next_patchStop_bool; 
        
        if length(session_prev_trialtype) ~= length(session_next_patchStop_bool)
            disp(length(session_prev_trialtype) - length(session_next_patchStop_bool)) 
        end
    end
end 

%% 1) Does proportion of patch engagement change throughout the session?   
close all
figure();
% bool scatterplot 
for mIdx = 1:numel(mouse_grps) 
    cmap = copper(numel(mouse_grps{mIdx}));
    s_nCues = cellfun(@length,patchStop_bool{mIdx}); 
    max_nCues = max(s_nCues); 
    pooled_patchStop_bool = nan(max_nCues,numel(mouse_grps{mIdx}));
    subplot(2,numel(mouse_grps),mIdx);hold on;colormap(copper)
    for i = 1:numel(mouse_grps{mIdx}) % floor(numel(mouse_grps{mIdx})/3):numel(mouse_grps{mIdx})   
        scatter(1:s_nCues(i),.05*randn(s_nCues(i),1) + patchStop_bool{mIdx}{i},1.5,cmap(i,:))
        pooled_patchStop_bool(:,i) = [patchStop_bool{mIdx}{i} ; nan(max_nCues - s_nCues(i),1)]; 
    end 
%     colorbar
    xlim([0 200])
    ylim([-.25 1.25])  
    yticks([0 1]) 
    if mIdx == 1
        yticklabels(["Miss","Hit"])  
    else 
        yticklabels([])
    end 
    
    sessions_third = floor(numel(mouse_grps{mIdx})/3);
    sessions_half = floor(numel(mouse_grps{mIdx})/2);
    
    prop_engage_early = nanmean(pooled_patchStop_bool(:,1:sessions_third),2);  
    prop_engage_mid = nanmean(pooled_patchStop_bool(:,sessions_third:2*sessions_third),2);  
    prop_engage_late = nanmean(pooled_patchStop_bool(:,2*sessions_third:end),2);  
    prop_engage_half1 = nanmean(pooled_patchStop_bool(:,1:sessions_half),2);  
    prop_engage_half2 = nanmean(pooled_patchStop_bool(:,sessions_half+1:end),2);   
    prop_engage_recording = nanmean(pooled_patchStop_bool(:,recording_session_bool_mice{mIdx}),2);   
    prop_engage = nanmean(pooled_patchStop_bool,2);
    title(mouse_names(mIdx));
    
    subplot(2,numel(mouse_grps),numel(mouse_grps) + mIdx);hold on
%     plot(smoothdata(prop_engage_early,'gaussian',20),'color',[.5 .5 .5],'linewidth',1.5) 
%     plot(smoothdata(prop_engage_mid,'gaussian',20),'color',[.25 .25 .25],'linewidth',1.5) 
%     plot(smoothdata(prop_engage_late,'gaussian',20),'color',[0 0 0],'linewidth',1.5)   
    plot(smoothdata(prop_engage_half1,'gaussian',20),'color',cmap(sessions_third,:),'linewidth',1.5) 
    plot(smoothdata(prop_engage_half2,'gaussian',20),'color',cmap(end-sessions_third,:),'linewidth',1.5) 
%     plot(smoothdata(prop_engage_recording,'gaussian',20),'color',[0 .5 0 ],'linewidth',1.5) 
%     plot(smoothdata(prop_engage,'gaussian',20),'color',[0 0 0],'linewidth',1.5)  
    xlim([0 200]) 
    ylim([0 1])  
    xlabel("Trial Number") 
    ylabel("Proportion Hit Trials") 
    if mIdx == 1 
        legend("First half of training","Second half of training") % ,"Recording Sessions")
    end
end 

%% 2a) What does velocity look like on hit / miss patches? 
%   i. color traces by hit / miss 
%   ii. shade hit trials by zscored PRT 
%  - probably going to need to restrict this to recording sessions... tons of lines otherwise
close all 
for mIdx = 1:numel(mouse_grps) 
    figure()
    mouse_recording_sessions = find(recording_session_bool_mice{mIdx}); 
    s_nCues = cellfun(@length,patchStop_bool{mIdx}); 
    for r_i = 1:2 % numel(mouse_recording_sessions)
        i = mouse_recording_sessions(r_i); 
        subplot(1,2,r_i); hold on % numel(find(recording_session_bool_mice{mIdx})),r_i);hold on
        
        hit_cues = find(patchStop_bool{mIdx}{i}); 
        miss_cues = find(~patchStop_bool{mIdx}{i}); 
        
        n_vis_cues = 20;  
        vis_hit_cues = hit_cues(randi(length(hit_cues),n_vis_cues,1));
        vis_miss_cues = miss_cues(randi(length(miss_cues),n_vis_cues,1));
        
        for i_cue = 1:n_vis_cues 
%             plot(smoothdata(vel_cue_trials{mIdx}{i}{vis_hit_cues(i_cue)},'gaussian',10),'color',[0 .7 0],'linewidth',.5) 
            plot(smoothdata(vel_cue_trials{mIdx}{i}{vis_miss_cues(i_cue)},'gaussian',10),'color',[.7 0 0],'linewidth',.5) 
        end 
        
        xlim([0 200]) 
        
        prt_cmap = cbrewer('seq','Greens',numel(hit_cues));
        [~,zscore_sort] = sort(prts_zscored{mIdx}{i}); 
        for i_cue = 1:numel(hit_cues) 
            iCue = hit_cues(i_cue); 
            plot(smoothdata(vel_cue_trials{mIdx}{i}{iCue},'gaussian',10),'color',prt_cmap(zscore_sort(i_cue),:),'linewidth',.5) 
        end
        
%         for iCue = 1:s_nCues(i) 
%             if patchStop_bool{mIdx}{i}(iCue) == true && hit_cue_counter < n_vis_cues 
%                 plot(smoothdata(vel_cue_trials{mIdx}{i}{iCue},'gaussian',10),'color',[0 .7 0],'linewidth',.5) 
%                 hit_cue_counter = hit_cue_counter + 1; 
%             elseif patchStop_bool{mIdx}{i}(iCue) == false && miss_cue_counter < n_vis_cues
%                 plot(smoothdata(vel_cue_trials{mIdx}{i}{iCue},'gaussian',10),'color',[.7 0 0],'linewidth',.5) 
%                 miss_cue_counter = miss_cue_counter + 1;
%             end
%         end
    end
end

%% 2b) Across sessions/mice, show some miss trials and hit trials colored by trialtype-zscored PRT 
close all
greens10 = .75 * cbrewer('seq','Greens',5); 
max_timepoint = 100; 
cue_ix = round(pre_cue_sec/tbin_sec);
figure()
for mIdx = 1:numel(mouse_grps) 
    subplot(3,5,mIdx);hold on
    mouse_prts_zscored = cat(1,prts_zscored{mIdx}{:});  
    [~,~,zscored_prt_decile] = histcounts(mouse_prts_zscored,quantile(mouse_prts_zscored,0:.2:1));
    mouse_patchStop_bool = cat(1,patchStop_bool{mIdx}{:}); 
    vel_traces_cell = cat(1,vel_cue_trials{mIdx}{:});
    mouse_vel_traces_hit = cat(1,vel_traces_cell(logical(mouse_patchStop_bool)));  
    
    cue_lens = cellfun(@length,mouse_vel_traces_hit);
    max_cue_len = max(cue_lens);  
    mouse_vel_traces_hit = arrayfun(@(i) [mouse_vel_traces_hit{i} nan(1,max_cue_len-cue_lens(i))],1:length(cue_lens),'un',0)';
    mouse_vel_traces_hit = cat(1,mouse_vel_traces_hit{:}); % now this is an array 
    vel_traces_means = nan(10,max_cue_len); 
    for decile = 1:5 
        vel_traces_means(decile,:) = nanmean(mouse_vel_traces_hit(zscored_prt_decile == decile,:));
    end 

%     figure();hold on
    for decile = 1:5
        plot(vel_traces_means(decile,:),'color',greens10(decile,:))
    end 
%     
%     miss_trials = find(~mouse_patchStop_bool); 
%     for i_miss_trial = 1:3  
%         iMiss_trial = miss_trials(i_miss_trial); 
%         plot(smoothdata(vel_traces_cell{iMiss_trial},'gaussian',10),'linewidth',.25,'color',[.7 0 0])
%     end 
    
    mouse_vel_traces_miss = cat(1,vel_traces_cell(~logical(mouse_patchStop_bool)));  
    cue_lens = cellfun(@length,mouse_vel_traces_miss);
    max_cue_len = max(cue_lens);  
    mouse_vel_traces_miss = arrayfun(@(i) [mouse_vel_traces_miss{i} nan(1,max_cue_len-cue_lens(i))],1:length(cue_lens),'un',0)';
    mouse_vel_traces_miss = cat(1,mouse_vel_traces_miss{:}); % now this is an array 
    mean_vel_miss = nanmean(mouse_vel_traces_miss);
    plot(mean_vel_miss,'color',[.7 0 0],'linewidth',1.5) 
    xlim([0 max_timepoint])   
    xticks(0:25:max_timepoint) 
    xticklabels((-max_timepoint/2:25:max_timepoint/2) * tbin_sec) 
    yl = ylim();
    
    if mIdx == 1 
        ylabel("Velocity (a.u.)") 
    end 
    v2 = [cue_ix yl(1);cue_ix yl(2);max_timepoint yl(2);max_timepoint yl(1)];
    f2 = [1 2 3 4];
    patch('Faces',f2,'Vertices',v2,'FaceColor',[1 1 0],'FaceAlpha',.2,'LineStyle','none'); 
    title(mouse_names(mIdx))

    % Now investigate correlations between velocity and zscored PRT @ diff timepoints 
    % Stopping is trivially correlated w/ speed
%     mouse_vel_traces_all = cat(1,vel_traces_cell); 
    mouse_vel_traces_all = cat(1,vel_traces_cell(true(length(mouse_patchStop_bool),1)));
    cue_lens = cellfun(@length,mouse_vel_traces_all);
    max_cue_len = max(cue_lens);   
    mouse_vel_traces_all = arrayfun(@(i) [mouse_vel_traces_all{i} nan(1,max_cue_len-cue_lens(i))],1:length(cue_lens),'un',0)';  
    mouse_vel_traces_all = cat(1,mouse_vel_traces_all{:});  
    r_prt = nan(max_timepoint,1); 
    p_prt = nan(max_timepoint,1); 
    r_stop = nan(max_timepoint,1); 
    p_stop = nan(max_timepoint,1); 
    for t = 1:max_timepoint 
        vel_hit = mouse_vel_traces_hit(:,t);
        [tr_prt,tp_prt] = corrcoef(vel_hit(~isnan(vel_hit)),mouse_prts_zscored(~isnan(vel_hit)));  
        r_prt(t) = tr_prt(2); p_prt(t) = tp_prt(2);
        vel_all = mouse_vel_traces_all(:,t);
        [tr_stop,tp_stop] = corrcoef(vel_all(~isnan(vel_all)),double(mouse_patchStop_bool(~isnan(vel_all))));  
        r_stop(t) = tr_stop(2); p_stop(t) = tp_stop(2);  
        if p_stop(t) < .01 
            subplot(3,5,numel(mouse_grps) + mIdx);hold on
            text(t,-.05,'*','HorizontalAlignment','center') 
        end
        if p_prt(t) < .01
            subplot(3,5,2*numel(mouse_grps) + mIdx);hold on
            text(t,.1,'*','HorizontalAlignment','center') 
        end
    end
    
    % Visualize significance 
    subplot(3,5,numel(mouse_grps) + mIdx) 
    plot(1:round(pre_cue_sec/tbin_sec),r_stop(1:round(pre_cue_sec/tbin_sec)),'r','linewidth',1.5);hold on  
    plot(round(pre_cue_sec/tbin_sec):length(r_stop)-1,r_stop(round(pre_cue_sec/tbin_sec)+1:end),'r--','linewidth',1.5);hold on  
    ylim([-1 0])  
    xlim([0 max_timepoint])   
    % note some hard coding here for now
    xticks(0:25:max_timepoint) 
    xticklabels((-max_timepoint/2:25:max_timepoint/2) * tbin_sec) 
    v2 = [cue_ix -1;cue_ix 0 ;max_timepoint 0;max_timepoint -1];
    f2 = [1 2 3 4];
    patch('Faces',f2,'Vertices',v2,'FaceColor',[1 1 0],'FaceAlpha',.2,'LineStyle','none'); 
    if mIdx == 1
        ylabel("Pearson Correlation Stop")
    end
    
    subplot(3,5,2*numel(mouse_grps) + mIdx)
    plot(r_prt,'linewidth',1.5);hold on
    ylim([-.2 .15])  
    xlim([0 max_timepoint])   
    xticks(0:25:max_timepoint) 
    xticklabels((-cue_ix:25:max_timepoint/2) * tbin_sec) 
    v2 = [cue_ix -.2;cue_ix .15;max_timepoint .15;max_timepoint -.2];
    f2 = [1 2 3 4];
    patch('Faces',f2,'Vertices',v2,'FaceColor',[1 1 0],'FaceAlpha',.2,'LineStyle','none'); 
    if mIdx == 1
        ylabel("Pearson Correlation PRT")
    end 
    xlabel(sprintf("Time Since Proximity Cue Onset (sec)")) 
    
end

%% 3) Patch history effects: does previous trial type affect P(stop)
% this is a null result
figure()
for mIdx = 1:5
%     sessions_half = floor(numel(mouse_grps{mIdx})/2);
    
    mouse_prev_trialtype = cat(1,prev_trialtype{mIdx}{:}); 
    onehot_prev_trialtype = zeros(length(mouse_prev_trialtype),9); 
    for iTrial = 1:length(mouse_prev_trialtype) 
        onehot_prev_trialtype(iTrial,mouse_prev_trialtype(iTrial)) = 1;
    end
    mouse_next_patchStop_bool = cat(1,next_patchStop_bool{mIdx}{:});
    
    p = kruskalwallis(mouse_next_patchStop_bool,mouse_prev_trialtype,'off');
    
    p_stay = nan(9,1); 
    p_stay_std = nan(9,1); 
    for tt = 1:9 
        p_stay(tt) = mean(mouse_next_patchStop_bool(mouse_prev_trialtype == tt)); 
        p_stay_std(tt) = std(mouse_next_patchStop_bool(mouse_prev_trialtype == tt)); 
    end
    
    subplot(1,5,mIdx) 
    bar(p_stay) 
%     scatter(.05*randn(length(mouse_next_patchStop_bool),1) + mouse_prev_trialtype,.01*randn(length(mouse_next_patchStop_bool),1) + mouse_next_patchStop_bool)
end

