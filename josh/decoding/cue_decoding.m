%% Perform LDA to determine whether mouse will stop for patch 
%  - is timecourse of decoding different from velocity? 
%  - LDA is mathematically almost equivalent to CD from Inagaki & Chen 2020
%    - One difference is the use of the scatter matrix 
%  - another alternative approach would be a regression-based analysis
%  - are different parts of the population involved in discrimination at
%    diff timepoints?  
%  - Key question: are we finding a fixed discriminant or something dynamic?
%  - can we predict PRT? 


%% Set paths
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
load(paths.sig_cells);  
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table.mat';
load(paths.transients_table);  
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData')); 

% analysis options
calcFR_opt = struct;
calcFR_opt.tstart = 0;
calcFR_opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calcFR_opt.smoothSigma_time = 0; 
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

% How much time to keep before cue
pre_cue_sec = 0.5; 
post_cue_sec = 0; 

% init data structures
fr_mat_cues = cell(numel(mouse_grps),1); % neural data per cue
vel_cues = cell(numel(mouse_grps),1); % velocity per cue
patchstop_bool_tsteps = cell(numel(mouse_grps),1); % label timesteps
patchstop_bool_cues = cell(numel(mouse_grps),1); % label cues
timesince_cue = cell(numel(mouse_grps),1); % time since cue onset 
zscored_prts = cell(numel(mouse_grps),1); 
qrts = cell(numel(mouse_grps),1); 
% neuron information in nice form
GLM_cluster = cell(numel(mouse_grps),1); 
brain_region = cell(numel(mouse_grps),1); 
transient_peak = cell(numel(mouse_grps),1); 

% log whether there are any miss trials
any_misses = cell(numel(mouse_grps),1); 

for mIdx = 1:numel(mouse_grps) 
    fr_mat_cues{mIdx} = cell(numel(mouse_grps{mIdx}),1);   
    vel_cues{mIdx} = cell(numel(mouse_grps{mIdx}),1);   
    patchstop_bool_tsteps{mIdx} = cell(numel(mouse_grps{mIdx}),1);   
    patchstop_bool_cues{mIdx} = cell(numel(mouse_grps{mIdx}),1);    
    timesince_cue{mIdx} = cell(numel(mouse_grps{mIdx}),1);   
    zscored_prts{mIdx} = cell(numel(mouse_grps{mIdx}),1);    
    qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    GLM_cluster{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    brain_region{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    transient_peak{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    any_misses{mIdx} = nan(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        data = load(fullfile(paths.data,sessions{sIdx}));  
        
        % grab all neurons... later pare down using cell labels
        good_cells = data.sp.cids(data.sp.cgs==2);   
        calcFR_opt.tend = max(data.sp.st); 
        fr_mat = calcFRVsTime(good_cells,data,calcFR_opt);  
        
        % get patch type information
        rewsize = mod(data.patches(:,2),10);   
        N0 = round(mod(data.patches(:,2),100)/10);
        N0(N0 == 3) = .125;
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5;   

        % gather patch timing information
        patchcue_hit_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2);
        patchleave_sec = data.patchCSL(:,3);
        i_prts = patchleave_sec - patchstop_sec; 
        
        % zscore PRTs within trial type
        nTrials = length(i_prts); 
        i_prts_zscored = nan(nTrials,1);   
        counter = 1; 
        for iRewsize = [1 2 4]
            for iN0 = [.125 .25 .5]
                i_prts_zscored(rewsize == iRewsize & N0 == iN0) = zscore(i_prts(rewsize == iRewsize & N0 == iN0));  
            end
        end 
        
        % get behavior timing information for miss trials 
        if ~isempty(data.patchCL) % make sure we had some miss trials
            patchcue_miss_sec = data.patchCL(:,1);
            patchleave_miss_sec = data.patchCL(:,2); 
        else 
            patchcue_miss_sec = []; 
            patchleave_miss_sec = []; 
        end
        
        % Gather data per cue presentation 
        velocity = data.vel;  
        nCues = length(patchcue_hit_sec) + length(patchcue_miss_sec);  
        nHits = length(patchcue_hit_sec); 
        nMiss = length(patchcue_miss_sec); 
        i_patchstop_bool_tsteps = cell(nCues,1); % vector of bools per cue
        i_patchstop_bool_cues = nan(nCues,1); % bool per cue 
        i_vel_cues = cell(nCues,1); 
        i_fr_mat_cues = cell(nCues,1); 
        i_timesince_cue = cell(nCues,1); 
        
        % vector ix that we will update to track where we hit/miss cues
        hit_ix = 1; 
        if ~isempty(patchcue_miss_sec)
            miss_ix = 1;  
            any_misses{mIdx}(i) = true; 
        else % if we don't have any misses, then log this
            miss_ix = NaN; 
            any_misses{mIdx}(i) = false; 
        end 
        
        % for now, just don't log data from session with no miss trials
        % later, can add this if we want to get into z-scored PRT analysis
        if any_misses{mIdx}(i) == true
            % Now collect velocity traces, neural data, and patchstop bool 
            for iCue = 1:nCues 
                if ~any(isnan([hit_ix miss_ix]))
                    if patchcue_hit_sec(hit_ix) < patchcue_miss_sec(miss_ix) % next hit cue came before miss
                        i_patchstop_bool_cues(iCue) = 1;  
                        cue_start_ix = round((patchcue_hit_sec(hit_ix) - pre_cue_sec) / tbin_sec);
                        cue_end_ix = round((patchstop_sec(hit_ix) + post_cue_sec) / tbin_sec);
                        hit_ix = hit_ix + 1; 
                        if hit_ix > nHits 
                            hit_ix = NaN; 
                        end
                    elseif patchcue_miss_sec(miss_ix) < patchcue_hit_sec(hit_ix) % next miss cue came before hit
                        i_patchstop_bool_cues(iCue) = 0;   
                        cue_start_ix = round((patchcue_miss_sec(miss_ix) - pre_cue_sec) / tbin_sec);
                        cue_end_ix = round((patchleave_miss_sec(miss_ix) + post_cue_sec) / tbin_sec);
                        miss_ix = miss_ix + 1; 
                        if miss_ix > nMiss 
                            miss_ix = NaN;
                        end
                    end
                % need to change logic a bit; we are out of hit or miss trials
                else 
                    if isnan(miss_ix) 
                        i_patchstop_bool_cues(iCue) = 1;  
                        cue_start_ix = round((patchcue_hit_sec(hit_ix) - pre_cue_sec) / tbin_sec);
                        cue_end_ix = round((patchstop_sec(hit_ix) + post_cue_sec) / tbin_sec);
                        hit_ix = hit_ix + 1; 
                    else 
                        i_patchstop_bool_cues(iCue) = 0;   
                        cue_start_ix = round((patchcue_miss_sec(miss_ix) - pre_cue_sec) / tbin_sec);
                        cue_end_ix = round((patchleave_miss_sec(miss_ix) + post_cue_sec) / tbin_sec);
                        miss_ix = miss_ix + 1; 
                    end
                end
                
                % Log velocity, neural data, and time since cue
                i_fr_mat_cues{iCue} = fr_mat(:,cue_start_ix:cue_end_ix); 
                i_vel_cues{iCue} = velocity(cue_start_ix:cue_end_ix); 
                i_patchstop_bool_tsteps{iCue} = i_patchstop_bool_cues(iCue) + zeros(length(i_vel_cues{iCue}),1);
                i_timesince_cue{iCue} = (1:(length(i_vel_cues{iCue}))) * tbin_sec - pre_cue_sec;
            end
        end

        % log data to cell arrays 
        % neural data
        fr_mat_cues{mIdx}{i} = i_fr_mat_cues;
        vel_cues{mIdx}{i} = i_vel_cues;
        patchstop_bool_tsteps{mIdx}{i} = i_patchstop_bool_tsteps;
        patchstop_bool_cues{mIdx}{i} = i_patchstop_bool_cues; 
        timesince_cue{mIdx}{i} = i_timesince_cue;
        
        % zscored prts and qrts
        zscored_prts{mIdx}{i} = i_prts_zscored; 
        qrts{mIdx}{i} = cellfun(@(x) size(x,2),fr_mat_cues{mIdx}{i}) * tbin_sec - pre_cue_sec;
        % neuron information 
        session_table = transients_table(strcmp(transients_table.Session,session_title),:); 
        GLM_cluster{mIdx}{i} = session_table.GLM_Cluster;
        brain_region{mIdx}{i} = session_table.Region;
        transient_peak{mIdx}{i} = [session_table.Rew0_peak_pos session_table.Rew1plus_peak_pos];
    end
end 

%% Brief behavioral analysis 
%  - Structure of quits across course of session
%  - Which sessions seem usable? 

% bool scatterplot within mice 
for mIdx = 1:numel(mouse_grps) 
    cmap = copper(numel(mouse_grps{mIdx}));
    s_nCues = cellfun(@length,patchstop_bool_cues{mIdx}); 
    max_nCues = max(s_nCues); 
    pooled_patchStop_bool = nan(max_nCues,numel(mouse_grps{mIdx}));
    
    for i = 1:numel(mouse_grps{mIdx})
        figure(1)
        subplot(2,numel(mouse_grps),mIdx);hold on;colormap(copper)
        scatter(1:s_nCues(i),.05*randn(s_nCues(i),1) + patchstop_bool_cues{mIdx}{i},1.5,cmap(i,:))
        pooled_patchStop_bool(:,i) = [patchstop_bool_cues{mIdx}{i} ; nan(max_nCues - s_nCues(i),1)]; 
        
        figure(mIdx + 1); hold on
        subplot(1,numel(mouse_grps{mIdx}),i)
        scatter(1:s_nCues(i),.05*randn(s_nCues(i),1) + patchstop_bool_cues{mIdx}{i},10,cmap(i,:))
        ylim([-.25 1.25])  
        yticks([0 1]) 
        if i == 1
            yticklabels(["Miss","Hit"])  
        else 
            yticklabels([])
        end  
        xlabel("Trial Number")  
        sIdx = mouse_grps{mIdx}(i);   
        session = sessions{sIdx}(1:end-4);
        session_title = session([1:2 end-2:end]);
        title(session_title)
    end 
    figure(1)
    ylim([-.25 1.25])  
    yticks([0 1]) 
    if mIdx == 1
        yticklabels(["Miss","Hit"])  
    else 
        yticklabels([])
    end 
    
    prop_engage = nanmean(pooled_patchStop_bool,2);
    figure(1)
    title(mouse_names(mIdx));
    subplot(2,numel(mouse_grps),numel(mouse_grps) + mIdx);hold on
    plot(smoothdata(prop_engage,'gaussian',20),'color',[0 0 0],'linewidth',1.5)  
    ylim([0 1])  
    xlabel("Trial Number") 
    ylabel("Proportion Hit Trials") 
end 

%% Visualize distribution of QRTs across sessions 

bins = 0:.2:max(qrts_pooled); 
% visualize distn within mice 
figure()
for mIdx = 1:numel(mouse_grps) 
    mouse_qrts = cat(1,qrts{mIdx}{:}); 
    mouse_patchstop_bool = cat(1,patchstop_bool_cues{mIdx}{:}); 
    subplot(1,numel(mouse_grps),mIdx);hold on
    histogram(mouse_qrts(mouse_patchstop_bool == true),bins);
    histogram(mouse_qrts(mouse_patchstop_bool == false),bins);
    if mIdx == 1
        legend(["Hit","Miss"]) 
    end
    xlabel("QRT (sec)") 
end

bins = 0:.1:max(qrts_pooled); 
% visualize across mice 
qrts_pooled = cat(1,qrts{:}); 
qrts_pooled = cat(1,qrts_pooled{:}); 

patchstop_bool_pooled = cat(1,patchstop_bool_cues{:}); 
patchstop_bool_pooled = cat(1,patchstop_bool_pooled{:}); 
figure();hold on
histogram(qrts_pooled(patchstop_bool_pooled == true),bins);
histogram(qrts_pooled(patchstop_bool_pooled == false),bins);
xlabel("QRT (sec)") 
legend(["Hit","Miss"]) 

%% Calculate selectivity vectors 

w = cell(numel(mouse_grps),1); 
r_hit = cell(numel(mouse_grps),1); 
r_miss = cell(numel(mouse_grps),1); 
w_vel = cell(numel(mouse_grps),1); 
vel_hit = cell(numel(mouse_grps),1); 
vel_miss = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)  
    w{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    r_hit{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    r_miss{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    w_vel{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    vel_hit{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    vel_miss{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx}) 
        if any_misses{mIdx}(i) == true
            % Concatenate session data for easy indexing 
            i_fr_mat_full = cat(2,fr_mat_cues{mIdx}{i}{:}); 
            i_vel_full = cat(2,vel_cues{mIdx}{i}{:});
            i_patchstop_bool_full = cat(1,patchstop_bool_tsteps{mIdx}{i}{:})';
            i_timesince_cue_full = cat(2,timesince_cue{mIdx}{i}{:});
            
            % load neuron information
            i_glm_cluster = GLM_cluster{mIdx}{i};
            i_brain_region = brain_region{mIdx}{i};
            i_transient_peak = transient_peak{mIdx}{i}; 
            
            % subselect population of interest 
%             keep_neurons = strcmp(i_brain_region,"PFC");  
            keep_neurons = ~isnan(i_glm_cluster);  
%             keep_neurons = i_glm_cluster == 3; 
            i_fr_mat_roi = i_fr_mat_full(keep_neurons,:); 
            nNeurons = length(find(keep_neurons)); 

            % now bin timesince cue according to some discretization
            binsize = .05; 
            timesince_cue_bins = -pre_cue_sec:binsize:1;
            [~,~,timesince_cue_bin_full] = histcounts(i_timesince_cue_full,timesince_cue_bins); 
            timesince_cue_bin_full(timesince_cue_bin_full == 0) = nan; % ignore unbinned data
            max_bin = max(timesince_cue_bin_full);
            
            % Calculate population selectivity vector per timestep 
            i_r_hit = nan(nNeurons,max_bin); 
            i_r_miss = nan(nNeurons,max_bin); 
            i_vel_hit = nan(max_bin,1); 
            i_vel_miss = nan(max_bin,1); 
            for t = 1:max_bin
                i_r_hit(:,t) = mean(i_fr_mat_roi(:,i_patchstop_bool_full == true & timesince_cue_bin_full == t),2);
                i_r_miss(:,t) = mean(i_fr_mat_roi(:,i_patchstop_bool_full == false & timesince_cue_bin_full == t),2);
                
                i_vel_hit(t) = mean(i_vel_full(i_patchstop_bool_full == true & timesince_cue_bin_full == t));
                i_vel_miss(t) = mean(i_vel_full(i_patchstop_bool_full == false & timesince_cue_bin_full == t));
            end  
            i_CD = i_r_hit - i_r_miss;
            i_CD_vel = i_vel_hit - i_vel_miss; 
            
            % save to datastructure
            w{mIdx}{i} = i_CD; 
            r_hit{mIdx}{i} = i_r_hit;
            r_miss{mIdx}{i} = i_r_miss;
            w_vel{mIdx}{i} = i_CD_vel; 
            vel_hit{mIdx}{i} = i_vel_hit; 
            vel_miss{mIdx}{i} = i_vel_miss; 
        end
    end
end

%% Now visualize structure in selectivity vector over time 
close all 
vis_mice = [2 4 5];
corr_vars = {w,r_hit,r_miss}; 
era_edges = [1 12.5 16.5 28]; 
corr_var_names = ["Selectivity Vector","r_{hit}","r_{miss}"]; 

for iCorr_var = 1:numel(corr_vars) 
    corr_var = corr_vars{iCorr_var}; 
    all_corr_var = corr_var(vis_mice);
    all_corr_var = cat(1,all_corr_var{:});
    all_corr_var = cat(1,all_corr_var{:});
    figure()
    subplot(1,numel(vis_mice)+1,1)
    imagesc(corrcoef(all_corr_var));hold on
    
    highlight_square(era_edges)
    xticks(1:10:max_bin)
    xticklabels(timesince_cue_bins(1:10:max_bin))
    yticks(1:10:max_bin)
    yticklabels(timesince_cue_bins(1:10:max_bin))
    caxis([-1,1])
    title(strjoin(mouse_names(vis_mice)))
    xlabel("Time since cue onset (sec)") 
    ylabel("Time since cue onset (sec)") 
    xlim([1 max_bin]) 
    ylim([1 max_bin])
    colorbar()
    for m_ix = 1:numel(vis_mice) 
        mIdx = vis_mice(m_ix);
        subplot(1,numel(vis_mice)+1,m_ix+1)
        mouse_corr_var = cat(1,corr_var{mIdx}{:});
        imagesc(corrcoef(zscore(mouse_corr_var,[],1) )); % ,[],2))) 
        highlight_square(era_edges)
        xticks(1:10:max_bin)
        xticklabels(timesince_cue_bins(1:10:max_bin))
        caxis([-1,1])
        title(mouse_names(mIdx))
        xlabel("Time since cue onset (sec)") 
        yticks([])
        xlim([1 max_bin]) 
        ylim([1 max_bin]) 
    end
    suptitle(sprintf("%s Pearson Correlation",corr_var_names(iCorr_var)))
end 

% Now check out velocity 
pooled_vel_hit = vel_hit(vis_mice);  
pooled_vel_hit = cat(1,pooled_vel_hit{:});  
pooled_vel_hit = cat(2,pooled_vel_hit{:});  
pooled_vel_miss = vel_miss(vis_mice);  
pooled_vel_miss = cat(1,pooled_vel_miss{:});  
pooled_vel_miss = cat(2,pooled_vel_miss{:});  

figure() 
subplot(2,numel(vis_mice)+1,1);hold on
plot(pooled_vel_hit,'color',.1+[.2 .7 .2],'linewidth',.25)
plot(mean(pooled_vel_hit,2),'color',[.2 .7 .2],'linewidth',2)
plot(pooled_vel_miss,'color',.1+[.7 .2 .2],'linewidth',.25)
plot(mean(pooled_vel_miss,2),'color',[.7 .2 .2],'linewidth',2)

for e = 2:(length(era_edges)-1)
    xline(era_edges(e),'linewidth',1.5) 
end

title(strjoin(mouse_names(vis_mice))) 
xticks(1:10:max_bin)
xticklabels(timesince_cue_bins(1:10:max_bin))
xlabel("Time since cue onset (sec)") 
ylabel("Velocity (A.U.)") 
yl = ylim();
for m_ix = 1:numel(vis_mice) 
    mIdx = vis_mice(m_ix);
    subplot(2,numel(vis_mice)+1,m_ix + 1) ;hold on
    mouse_vel_hit = cat(2,vel_hit{mIdx}{:});
    mouse_vel_miss = cat(2,vel_miss{mIdx}{:});
    for e = 2:(length(era_edges)-1)
        xline(era_edges(e),'linewidth',1.5)
    end
    plot(mean(mouse_vel_hit,2),'color',[.2 .7 .2],'linewidth',2) 
    plot(mouse_vel_hit,'color',.1+[.2 .7 .2],'linewidth',.5)
    plot(mean(mouse_vel_miss,2),'color',[.7 .2 .2],'linewidth',2) 
    plot(mouse_vel_miss,'color',.1+[.7 .2 .2],'linewidth',.5)
    xticks(1:10:max_bin)
    xticklabels(timesince_cue_bins(1:10:max_bin))
    title(mouse_names(mIdx))
    xlabel("Time since cue onset (sec)")
    yticks([])
    ylim(yl)
end 

subplot(2,numel(vis_mice)+1,numel(vis_mice) + 2);hold on
plot(gradient(pooled_vel_hit),'color',.1+[.2 .7 .2],'linewidth',.25)
plot(gradient(mean(pooled_vel_hit,2)),'color',[.2 .7 .2],'linewidth',2)
plot(gradient(pooled_vel_miss),'color',.1+[.7 .2 .2],'linewidth',.25)
plot(gradient(mean(pooled_vel_miss,2)),'color',[.7 .2 .2],'linewidth',2)

for e = 2:(length(era_edges)-1)
    xline(era_edges(e),'linewidth',1.5) 
end

title(strjoin(mouse_names(vis_mice))) 
xticks(1:10:max_bin)
xticklabels(timesince_cue_bins(1:10:max_bin))
xlabel("Time since cue onset (sec)") 
ylabel("Acceleration (A.U.)") 
yl = ylim();
for m_ix = 1:numel(vis_mice) 
    mIdx = vis_mice(m_ix);
    subplot(2,numel(vis_mice)+1,numel(vis_mice) + m_ix + 2) ;hold on
    mouse_vel_hit = cat(2,vel_hit{mIdx}{:});
    mouse_vel_miss = cat(2,vel_miss{mIdx}{:});
    for e = 2:(length(era_edges)-1)
        xline(era_edges(e),'linewidth',1.5)
    end
    plot(gradient(mean(mouse_vel_hit,2)),'color',[.2 .7 .2],'linewidth',2) 
    plot(gradient(mouse_vel_hit),'color',.1+[.2 .7 .2],'linewidth',.5)
    plot(gradient(mean(mouse_vel_miss,2)),'color',[.7 .2 .2],'linewidth',2) 
    plot(gradient(mouse_vel_miss),'color',.1+[.7 .2 .2],'linewidth',.5)
    xticks(1:10:max_bin)
    xticklabels(timesince_cue_bins(1:10:max_bin))
    title(mouse_names(mIdx))
    xlabel("Time since cue onset (sec)")
    yticks([])
    ylim(yl)
end 

%% Visualize r_hit sorted by GLM cluster 
close all
vis_mice = [2 4 5];
activity_vars = {r_hit}; % {w r_hit r_miss};
activity_var_names = ["Selectivity Vector","r_{hit}","r_{miss}"]; 

for iActivity_var = 1:numel(activity_vars) 
    activity_var = activity_vars{iActivity_var}; 

    all_activity_var = activity_var(vis_mice);
    all_activity_var = cat(1,all_activity_var{:});
    all_activity_var = cat(1,all_activity_var{:});
    all_glm_cluster = cat(1,GLM_cluster(vis_mice));
    all_glm_cluster = cat(1,all_glm_cluster{:});
    all_glm_cluster = cat(1,all_glm_cluster{:});
    all_glm_cluster = all_glm_cluster(~isnan(all_glm_cluster));
    [~,cluster_sort] = sort(all_glm_cluster);  

    figure()
    subplot(1,numel(vis_mice)+1,1) ;hold on
    imagesc(zscore(all_activity_var(cluster_sort,:),[],2))  
    xticks(1:10:max_bin)
    xticklabels(timesince_cue_bins(1:10:max_bin))
    gscatter(zeros(numel(cluster_sort),1),1:numel(cluster_sort),all_glm_cluster(cluster_sort))
    xlim([0 max_bin])
    xticks(1:10:max_bin)
    xticklabels(timesince_cue_bins(1:10:max_bin))
    ylim([1 numel(cluster_sort)])
    title(strjoin(mouse_names(vis_mice))) 
    xlabel("Time since cue onset (sec)")
    legend("Cluster 1","Cluster 2","Cluster 3") 
    ylabel("Sorted Neurons")

    for m_ix = 1:numel(vis_mice) 
        mIdx = vis_mice(m_ix);
        subplot(1,numel(vis_mice)+1,m_ix + 1);hold on
        mouse_r_hit = cat(1,activity_var{mIdx}{:}); 
        mouse_glm_cluster = cat(1,GLM_cluster{mIdx}{:});
        mouse_glm_cluster = mouse_glm_cluster(~isnan(mouse_glm_cluster));
        [~,cluster_sort] = sort(mouse_glm_cluster);  
        imagesc(zscore(mouse_r_hit(cluster_sort,:),[],2))  
        xticks(1:10:max_bin)
        xticklabels(timesince_cue_bins(1:10:max_bin))
        gscatter(zeros(numel(cluster_sort),1),1:numel(cluster_sort),mouse_glm_cluster(cluster_sort))
        xlim([0 max_bin]) 
        ylim([1 numel(cluster_sort)]) 
        title(mouse_names(mIdx))
        xlabel("Time since cue onset (sec)") 
        f = gca(); legend(f,'off');
    end 
    suptitle(activity_var_names(iActivity_var))
end

%% Ok, now we have our eras. let's define 3 choice directions and analyze projection dynamics
era_edges_ix = floor(era_edges);
era_edges_ix(1:end-1) = era_edges_ix(1:end-1) - 1; 

%  This is back at session level
CD = cell(numel(mouse_grps),1); 
projections = cell(numel(mouse_grps),1); 
for mIdx = 5 % 1:numel(mouse_grps)  
    CD{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    projections{mIdx} = cell(numel(mouse_grps{mIdx}),1);
    for i = 2 % 1:numel(mouse_grps{mIdx})
        % neuron information
        i_glm_cluster = GLM_cluster{mIdx}{i};
        i_brain_region = brain_region{mIdx}{i};
        i_transient_peak = transient_peak{mIdx}{i};
        % select population of interest (must be same as before... this isn't great coding practice)
        % keep_neurons = strcmp(i_brain_region,"PFC");
        keep_neurons = ~isnan(i_glm_cluster);
        
        nNeurons = length(w{mIdx}{i});
        CD{mIdx}{i} = nan(nNeurons,3); 
        for e = 1:3 
            CD{mIdx}{i}(:,e) = mean(w{mIdx}{i}(:,(era_edges_ix(e)+1):era_edges_ix(e+1)),2);  
            CD{mIdx}{i}(:,e) = CD{mIdx}{i}(:,e) ./ norm(CD{mIdx}{i}(:,e));
        end 
%         CD{mIdx}{i} = orth(CD{mIdx}{i}); % orthonormalize
        
        % now project trials onto discriminant axes 
        nCues = length(fr_mat_cues{mIdx}{i}); 
        projections{mIdx}{i} = cell(nCues,1); 
        for iCue = 1:nCues
            norm_trial = fr_mat_cues{mIdx}{i}{iCue}(keep_neurons,:) ./ vecnorm(fr_mat_cues{mIdx}{i}{iCue},2); 
            projections{mIdx}{i}{iCue} = CD{mIdx}{i}' * norm_trial;
        end
    end
end

%% Visualize  

for mIdx = 5 
    for i = 2  
        patchstop_bool = patchstop_bool_cues{mIdx}{i};
        max_cue_len = max(cellfun(@length,projections{mIdx}{i}));
        
        % pad projected trials
        nCues = length(fr_mat_cues{mIdx}{i}); 
        i_padded_projections = cell(nCues,1); 
        for iCue = 1:nCues  
            iCue_len = size(projections{mIdx}{i}{iCue},2); 
            i_padded_projections{iCue} = [projections{mIdx}{i}{iCue} nan(3,max_cue_len - iCue_len)]; 
        end
        
        hit_projections = cat(1,i_padded_projections(patchstop_bool == 1)); 
        mean_hit_projections = nanmean(cat(3,hit_projections{:}),3); 
        norm_mean_hit_projections = mean_hit_projections - mean_hit_projections(:,1); 
%         norm_mean_hit_projections = mean_hit_projections ./ max(mean_hit_projections,2); 
        sem_hit_projections = nanstd(cat(3,hit_projections{:}),[],3) / sqrt(length(find(patchstop_bool==1))); 
        
        miss_projections = cat(1,i_padded_projections(patchstop_bool == 0)); 
        mean_miss_projections = nanmean(cat(3,miss_projections{:}),3); 
        norm_mean_miss_projections = mean_miss_projections - mean_miss_projections(:,1);
        sem_miss_projections = nanstd(cat(3,miss_projections{:}),[],3) / sqrt(length(find(patchstop_bool==0)));
        
%         figure() ; hold on
%         for e = 1:3 
%             subplot(3,1,e);hold on
%             shadedErrorBar((1:max_cue_len) * tbin_sec - pre_cue_sec,norm_mean_hit_projections(e,:),sem_hit_projections(e,:),'lineprops',{'color',[.2 .7 .2]}) 
%             shadedErrorBar((1:max_cue_len) * tbin_sec - pre_cue_sec,norm_mean_miss_projections(e,:),sem_miss_projections(e,:),'lineprops',{'color',[1 .3 .3]}) 
%             xlim([-pre_cue_sec 1]) 
%         end 
        
        % Plot how AUC changes over timepoints   
        max_t = 75;
        auc = nan(3,max_t); 
        for e = 1:3 
            e_hit_projections = cat(1,cellfun(@(x) x(e,:),hit_projections,'UniformOutput',false));
            e_hit_projections = cat(1,e_hit_projections{:}); 
            e_miss_projections = cat(1,cellfun(@(x) x(e,:),miss_projections,'UniformOutput',false)); 
            e_miss_projections = cat(1,e_miss_projections{:}); 
            
            e_projections = [e_hit_projections ; e_miss_projections]; 
            labels = sort(patchstop_bool,'descend');
            
            for t = 1:max_t
                [x,y,thresh,auc(e,t)] = perfcurve(labels,e_projections(:,t),0,'XCrit','prec','TVals',linspace(min(e_projections(:)),max(e_projections(:)),50)); 
            end
        end
%         
        figure();plot(auc')
        xticks(1:10:max_t)
        xticklabels(tbin_sec * (0:10:max_t) - pre_cue_sec)
    end
end

%% Functions 

function highlight_square(edges)  
    for i = 1:(numel(edges)-1)
        patch('Faces',1:4,'Vertices',[edges(i) edges(i); edges(i) edges(i+1); edges(i+1) edges(i+1) ; edges(i+1) edges(i)], ... 
              'FaceAlpha',0,'EdgeColor','w','linewidth',1.5)
    end
end
