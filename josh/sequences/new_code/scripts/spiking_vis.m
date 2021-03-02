%% Plot spiking activity sorted by driscoll transient discovery 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
addpath(genpath('/Users/joshstern/Documents/UchidaLab_neuralData')); 

% make these from cue_reward_responsive.m
load('./structs/transients_table.mat');
load('./structs/taskvar_peth.mat');

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

%% Make new datstructures to look at spiking 
% singleCell_trialed_sts: cell array of nNeurons matrices w/ trial # and st relative
%                         to trial start
% trial_rasters: cell array of nTrials matrices w/ neuron # and st relative
%                to trial start

for mIdx = 5 % 1:numel(mouse_grps)
    for i = 2 % 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i); 
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        % load in data
        data = load(fullfile(paths.data,session));   
        
        % Neuron information
        good_cells = data.sp.cids(data.sp.cgs==2); % going to need to map back to this
        nNeurons = numel(good_cells);   
        session_transients_table = transients_table(transients_table.Session == session_title,:);  
        pos_Rew0_peaks = session_transients_table.Rew0_peak_pos; 
        pos_Rew1plus_peaks = session_transients_table.Rew1plus_peak_pos; 
        region = session_transients_table.Region;   
        
        % subselect brain region
        ROI = "PFC";  
        good_cells_roi = good_cells(region == ROI);  
        pos_Rew0_peaks_roi = pos_Rew0_peaks(region == ROI);  
        pos_Rew1plus_peaks_roi = pos_Rew1plus_peaks(region == ROI); 
        
        % Load trial information
        nTrials = length(data.patchCSL); 
        rewsize = mod(data.patches(:,2),10);   
        patchcue_sec = data.patchCSL(:,1);
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);   
        cue_prt_ms = 1000 * (patchleave_sec - patchcue_sec); 
        rew_sec = data.rew_ts;   

        [~,rew0_sort] = sort(pos_Rew0_peaks_roi);
        [~,rew1plus_sort] = sort(pos_Rew1plus_peaks_roi);
        rew0_sorted_cellIDs = good_cells_roi(rew0_sort(ismember(rew0_sort,find(~isnan(pos_Rew0_peaks_roi))))); % get rid of non significant cells
        rew1plus_sorted_cellIDs = good_cells_roi(rew1plus_sort(ismember(rew1plus_sort,find(~isnan(pos_Rew1plus_peaks_roi))))); % get rid of non significant cells
        
        pre_cue_sec = 1; % time to include before cue in seconds
        post_leave_sec = 1; % time to include after leave in seconds
        
        % get PETH for visualization
        session_peth_rew0 = taskvar_peth{2}(transients_table.Session == session_title,:);  
        session_peth_rew1plus = taskvar_peth{3}(transients_table.Session == session_title,:);  
        % select for ROI 
        session_peth_rew0 = session_peth_rew0(region == ROI,:); 
        session_peth_rew1plus = session_peth_rew1plus(region == ROI,:); 
        % sort
        session_peth_rew0 = session_peth_rew0(rew0_sort(ismember(rew0_sort,find(~isnan(pos_Rew0_peaks_roi)))),:); 
        session_peth_rew1plus = session_peth_rew1plus(rew1plus_sort(ismember(rew1plus_sort,find(~isnan(pos_Rew1plus_peaks_roi)))),:); 
        
        these_sorted_IDs = rew1plus_sorted_cellIDs; 
        nNeurons_sig = length(these_sorted_IDs); 
        % Collect sorted spikes in trial rasters cell array
        trial_rasters = cell(nTrials,1); 
        vel_trials = cell(nTrials,1); 
        pos_trials = cell(nTrials,1);  
        t_trials = cell(nTrials,1); 
        for iTrial = 1:nTrials
            trial_spikes = cell(nNeurons_sig,1); % make cell array, then concat all in one step for speed
            trial_sts = data.sp.st(data.sp.st > (patchcue_sec(iTrial)-pre_cue_sec) & data.sp.st < patchleave_sec(iTrial) + post_leave_sec) - patchcue_sec(iTrial) + pre_cue_sec;
            trial_clu = data.sp.clu(data.sp.st > (patchcue_sec(iTrial)-pre_cue_sec) & data.sp.st < patchleave_sec(iTrial) + post_leave_sec);
            for iNeuron = 1:nNeurons_sig
                cellID_ix = these_sorted_IDs(iNeuron);
                i_spike_ts = unique(trial_sts(trial_clu == cellID_ix));
                trial_spikes{iNeuron} = [iNeuron + zeros(numel(i_spike_ts),1) i_spike_ts];
            end 
            vel_trials{iTrial} = data.vel(data.velt > (patchcue_sec(iTrial)-pre_cue_sec) & data.velt < (patchleave_sec(iTrial) + post_leave_sec));
            pos_trials{iTrial} = data.patch_pos(data.velt > (patchcue_sec(iTrial)-pre_cue_sec) & data.velt < (patchleave_sec(iTrial) + post_leave_sec));
            t_trials{iTrial} = data.velt(data.velt > (patchcue_sec(iTrial)-pre_cue_sec) & data.velt < (patchleave_sec(iTrial) + post_leave_sec)) - patchstop_sec(iTrial); %  + pre_cue_sec; 
            
            % add to our data structure
            trial_rasters{iTrial} = cat(1,trial_spikes{:}); % concatenate single neurons
            if mod(iTrial,10) == 0
                fprintf("Finished trial %i \n",iTrial)
            end 
        end
    end
end

%% Visualize single trials 
colors = cool(3); 
rew_vis_time = 250; 
% close all
for iTrial = 1
    iTrial_rew_ts = [1000 * (patchstop_sec(iTrial) - patchcue_sec(iTrial) + pre_cue_sec) ; 1000 * ((patchstop_sec(iTrial) - patchcue_sec(iTrial) + pre_cue_sec) +  rew_sec(rew_sec > patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial))]; 
    n_rews = numel(iTrial_rew_ts);
    figure() 
    subplot(1,10,1:8)
    scatter(trial_rasters{iTrial}(:,2) * 1000,trial_rasters{iTrial}(:,1),1.5,'k.') 
    title(sprintf("Trial %i Raster",iTrial)) 
    xlim([0,max(trial_rasters{iTrial}(:,2) * 1000)]);ylim([-25,max(trial_rasters{iTrial}(:,1))]) 
    xlabel("Time (msec)") 
    ylabel("Spiking Activity")   
    % plot rewards 
    hold on
%     plot([iTrial_rew_ts iTrial_rew_ts]',repmat([1 nNeurons],[n_rews,1])','--','color',colors(3,:),'linewidth',1)  
    for r = 1:n_rews
        v2 = [iTrial_rew_ts(r) -25 ;iTrial_rew_ts(r) 0 ;iTrial_rew_ts(r)+rew_vis_time 0 ;iTrial_rew_ts(r)+rew_vis_time -25 ];
        f2 = [1 2 3 4];
        patch('Faces',f2,'Vertices',v2,'FaceColor',colors(min(3,rewsize(iTrial)),:),'FaceAlpha',.4);  
        if r == 1
            xline(iTrial_rew_ts(r),'-','color',colors(min(3,rewsize(iTrial)),:),'linewidth',1) 
        else 
            xline(iTrial_rew_ts(r),'--','color',colors(min(3,rewsize(iTrial)),:),'linewidth',1) 
        end
    end
    v2 = [1000*pre_cue_sec -25 ;1000 * pre_cue_sec 0 ;1000 * pre_cue_sec+rew_vis_time 0 ;1000 * pre_cue_sec+rew_vis_time -25 ];
    f2 = [1 2 3 4];
    patch('Faces',f2,'Vertices',v2,'FaceColor',[1 1 0],'FaceAlpha',.4);
    xline(pre_cue_sec * 1000,'k-','linewidth',1) 
    xline(pre_cue_sec * 1000 + cue_prt_ms(iTrial),'r-','linewidth',1)  
    
    % add PETH
    subplot(1,10,9:10)  
    imagesc(flipud(zscore(session_peth_rew1plus,[],2))) 
    ylim([0 size(session_peth_rew1plus,1) + 25])
    caxis([-3 3])
    yticks([]) 
    xticks([])
    set(gca,'Visible','off')
end

%% Concatenate and visualize multiple trials using Patch to show rewards and cue w/ patch

vis_trials = 21;
trial_edges = [0 ; cumsum(cue_prt_ms(vis_trials) + 1000 * pre_cue_sec + 1000 * post_leave_sec)];
figure()
% subplot(1,10,1:8);hold on
for i_trial = 1:numel(vis_trials)
    iTrial = vis_trials(i_trial); 
    iTrial_rew_ts = [1000 * (patchstop_sec(iTrial) - patchcue_sec(iTrial) + pre_cue_sec) ; 1000 * ((patchstop_sec(iTrial) - patchcue_sec(iTrial) + pre_cue_sec) +  rew_sec(rew_sec > patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial))];
    n_rews = numel(iTrial_rew_ts); 
    subplot(8,10,[1:8 11:18 21:28 31:38 41:48 51:58]);hold on
    scatter(trial_edges(i_trial) + trial_rasters{iTrial}(:,2) * 1000,trial_rasters{iTrial}(:,1),1,'ko') 
    ylabel("Spiking Activity")   
    % plot rewards 
    hold on
    v2 = [trial_edges(i_trial)+1000*pre_cue_sec -25 ;trial_edges(i_trial) + 1000 * pre_cue_sec 0 ;trial_edges(i_trial) + 1000 * pre_cue_sec+rew_vis_time 0 ;trial_edges(i_trial) + 1000 * pre_cue_sec+rew_vis_time -25 ];
    f2 = [1 2 3 4];
    subplot(8,10,61:68)
    patch('Faces',f2,'Vertices',v2,'FaceColor',[1 1 0],'FaceAlpha',.4); 
    
%     plot([iTrial_rew_ts iTrial_rew_ts]',repmat([1 nNeurons],[n_rews,1])','--','color',colors(3,:),'linewidth',1)  
    for r = 1:n_rews 
        subplot(8,10,61:68);hold on
        v2 = [trial_edges(i_trial) + iTrial_rew_ts(r) -25 ;trial_edges(i_trial) + iTrial_rew_ts(r) 0 ;trial_edges(i_trial)+iTrial_rew_ts(r)+rew_vis_time 0 ;trial_edges(i_trial)+iTrial_rew_ts(r)+rew_vis_time -25 ];
        f2 = [1 2 3 4];
        patch('Faces',f2,'Vertices',v2,'FaceColor',colors(min(3,rewsize(iTrial)),:),'FaceAlpha',.4);   
        subplot(8,10,[1:8 11:18 21:28 31:38 41:48 51:58])
        if r == 1
            xline(trial_edges(i_trial) + iTrial_rew_ts(r),'-','color',colors(min(3,rewsize(iTrial)),:),'linewidth',1) 
        else 
            xline(trial_edges(i_trial) + iTrial_rew_ts(r),'--','color',colors(min(3,rewsize(iTrial)),:),'linewidth',1) 
        end
    end 
    
    subplot(8,10,[1:8 11:18 21:28 31:38 41:48 51:58])
    xline(trial_edges(i_trial) + pre_cue_sec * 1000,'k-','linewidth',1) 
    xline(trial_edges(i_trial) + pre_cue_sec * 1000 + cue_prt_ms(iTrial),'r-','linewidth',1)   
    xline(trial_edges(i_trial),'k-','linewidth',2)     

%     Add velocity and position 
    subplot(8,10,71:78);hold on
    plot(trial_edges(i_trial)/1000 + t_trials{iTrial},vel_trials{iTrial},'g','linewidth',1.5) 
    plot(trial_edges(i_trial)/1000 + t_trials{iTrial},pos_trials{iTrial},'b','linewidth',1.5) 
    yline(0,'k--')

end  

subplot(8,10,[1:8 11:18 21:28 31:38 41:48 51:58])
title(sprintf("%s Trial %i",session_title,iTrial)) 
xlim([0 trial_edges(end)]) 
ylim([0 size(session_peth_rew1plus,1)]) 
xticks([])  
yticks([])
subplot(8,10,61:68)
xlim([0 trial_edges(end)]) 
yticks([])  
xticks([]) 
ylabel("Events")   
legend(["Proximity Cue",sprintf("%i uL Reward",rewsize(iTrial))])
% set(get(gca,'ylabel'),'rotation',0)
subplot(8,10,71:78) 
xlim([min(t_trials{vis_trials(1)}) max(t_trials{vis_trials(1)})])%  max(t_trials{vis_trials(1)})])   
% xticks(1:1/diff(data.velt(1:2)):length(t_trials{1}))
% xticklabels(1:20:length(t_trials{1}))
ylabel("Behavior")   
yticks([])
legend(["Velocity","Patch Position"]) 
xlabel("Time On Patch (sec)")

% ylim([0 size(session_peth_rew1plus,1)]) 

% add PETH
subplot(8,10,[9:10 19:20 29:30 39:40 49:50 59:60])
imagesc(flipud(zscore(session_peth_rew1plus,[],2)))
ylim([0 size(session_peth_rew1plus,1)])
caxis([-3 3])
yticks([])
xticks([]) 
set(gca,'Visible','off')  

%         % Collect trial reward timings... could be useful for cute vis
%         rew_locs_cell = cell(nTrials,1); 
%         for iTrial = 1:nTrials
%             rew_locs_cell{iTrial} = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial));
%         end

%%  Old Single cell code   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% now make singleCell_trialed_sts
singleCell_trialed_sts = {nNeurons};
singleCell_trialed_smoothed = {nNeurons};
sec_before = 0;
sec_after = 2000 / 1000;
for iNeuron = 1:nNeurons
    neuron_spikes = {numel(prt_withinRew_sort)}; % make cell array, then concat all in one step for speed
    smoothed = {numel(prt_withinRew_sort)};
    good_cells_ix = good_cells_sorted(iNeuron);
    neuron_sts = data.sp.st(data.sp.clu == good_cells_ix);
    neuron_clu = data.sp.clu(data.sp.clu == good_cells_ix);
    for j = 1:numel(prt_withinRew_sort)
        iTrial = prt_withinRew_sort(j);
        start = patchstop_sec(iTrial) - sec_before;
        finish = patchstop_sec(iTrial) + sec_after;
        i_spike_ts = unique(neuron_sts(neuron_sts > start & neuron_sts < finish)) - patchstop_sec(iTrial);
        neuron_spikes{j} = [j + zeros(numel(i_spike_ts),1) i_spike_ts];
        
        smoothed{j} = FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx}(iNeuron),1:round(sec_after * 1000 / tbin_ms));
    end
    % add to our data structure
    singleCell_trialed_sts{iNeuron} = cat(1,neuron_spikes{:});
    singleCell_trialed_smoothed{iNeuron} = cat(1,smoothed{:});
    if mod(iNeuron,50) == 0
        fprintf("Finished neuron %i \n",iNeuron)
    end
end

%% visualize some single cells   

colors3rew = [0 1 1 ;.5 .5 1;1 0 1;.5 1 1 ; .75 .75 1 ; 1 .5 1];
rewtrials = fliplr(cumsum([1 numel(find(rewsize == 1)) numel(find(rewsize == 2)) numel(find(rewsize == 4))])); 
trials10x = find(rew_barcode(:,1) == 1 & rew_barcode(:,2) < 0 & prts > 2.55);
trials20x = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) < 0 & prts > 2.55);
trials40x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) < 0 & prts > 2.55);
trials11x = find(rew_barcode(:,1) == 1 & rew_barcode(:,2) == 1 & prts > 2.55);
trials22x = find(rew_barcode(:,1) == 2 & rew_barcode(:,2) == 2 & prts > 2.55);
trials44x = find(rew_barcode(:,1) == 4 & rew_barcode(:,2) == 4 & prts > 2.55);
r0trials = fliplr(cumsum([1 numel(trials10x) numel(trials20x) numel(trials40x) numel(trials11x) numel(trials22x) numel(trials44x)]));
close all  
for iNeuron = 60:70
    figure()  
    subplot(2,1,1)
    scatter(singleCell_trialed_sts{iNeuron}(:,2) * 1000,singleCell_trialed_sts{iNeuron}(:,1),10,'k.')   
    title(sprintf("Neuron %i Spiking Activity",iNeuron))
    xlabel("Time (msec)")    
    ylabel("Trials") 
%     yticks([])  
    hold on
    % shade in raster by reward size  
    for rewcounter = 1:6
        v = [1 r0trials(rewcounter); 1 r0trials(rewcounter+1);2000 r0trials(rewcounter+1); 2000 r0trials(rewcounter)];
        f = [1 2 3 4];
        patch('Faces',f,'Vertices',v,'FaceColor',colors3rew(rewcounter,:),'FaceAlpha',.1,'EdgeColor','none');  
    end 
    xlim([1 2000]) 
    ylim([1 numel(prt_withinRew_sort)])
    subplot(2,1,2) 
    imagesc(flipud(zscore(singleCell_trialed_smoothed{iNeuron},[],2)));colormap('jet')  
    title(sprintf("Neuron %i Z-Scored Activity",iNeuron))
    xlabel("Time (msec)")  
    xticks([12 25 37 50 62 75 87 100]);xticklabels([250 500 750 1000 1250 1500 1750 2000]) 
%     yticks([]) 
    ylabel("Trials") 
    colorbar()
end 