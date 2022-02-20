%% Given trialed naive bayes decoding results, analyze correlation of decoded elapsed time and PRT 
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

clu_pool_sessions = nb_results.clu_pool_sessions;

%% 0) Reformat decoded time, load rewsize, RXNil, prts, and postRew_prts
%   Cell format: timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature}{iTrial}
%   Cell format: timeSince_sameRewsize_hat{mIdx}{i}{i_feature}{iTrial}

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
prts = cell(nMice,1); 
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
            last_rew_sec(iTrial) = rew_indices(end)-1;
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
        
        % get number of cells per cluster
        session_sigcells = sig_cells(strcmp(sig_cells.Session,session),:); 
        glm_clusters_session = session_sigcells.GMM_cluster; 
        n_cells{mIdx}{i_i} = [arrayfun(@(x) length(find(glm_clusters_session == x)),(1:4)) length(glm_clusters_session)];
        
        % log trial/behavior information
        rewsize{mIdx}{i_i} = session_rewsize;
        RXNil{mIdx}{i_i} = session_RXNil; 
        prts{mIdx}{i_i} = session_prts; 
        postRew_prts{mIdx}{i_i} = session_postRew_prts; 
        last_rew_ix{mIdx}{i_i} = i_last_rew_ix;
    end
end 
clear y_hat % we now have this in an easier form to work with

%% 1) Decoded time vs PRT on RXNil trials
var_bins = nb_results.var_bins; 
smoothing_sigma = 1; 
trialtypes = [40]; % RXNil trialtypes to look at 
var_dt = diff(nb_results.var_bins{1}{1}{1}(1:2));
analyze_ix = round([3000 3000 5000 3000 3000] / tbin_ms); 
vis_features = [5]; 

r_prt = cell(numel(analysis_mice),1);
p_prt = cell(numel(analysis_mice),1);
for m_ix = 1:numel(analysis_mice)
    for i_feature = 1:numel(vis_features)
        r_prt{m_ix}{i_feature} = nan(numel(trialtypes),analyze_ix(mIdx));
        p_prt{m_ix}{i_feature} = nan(numel(trialtypes),analyze_ix(mIdx));
    end
end 

cell_threshold = true; 
% close all
for i_feature = 1:numel(vis_features)
    iFeature = vis_features(i_feature); 
%     figure();hold on
    for m_ix = 1:numel(analysis_mice)
        mIdx = analysis_mice(m_ix); 
        % load RXNil for pooled sessions 
        if iFeature <= 3 && cell_threshold == true % introduce # cell session inclusion criterion
            mouse_include_sessions = ismember(pool_sessions{mIdx},clu_pool_sessions{iFeature}{mIdx}); 
            mouse_RXNil = RXNil{mIdx}(mouse_include_sessions); 
            mouse_RXNil = cat(1,mouse_RXNil{:}); 
            mouse_prts = prts{mIdx}(mouse_include_sessions); 
            mouse_prts = cat(1,mouse_prts{:}); 
            % gather decoded time variable from i_feature
            mouse_timePatch = cellfun(@(x) x{iFeature}, timePatch_hat{mIdx}(mouse_include_sessions),'un',0);
            mouse_timePatch = cat(1,mouse_timePatch{:});
        else
            mouse_RXNil = cat(1,RXNil{mIdx}{:});
            % load PRTs for pooled sessions
            mouse_prts = cat(1,prts{mIdx}{:});
            % gather decoded time variable from i_feature
            mouse_timePatch = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx},'un',0);
            mouse_timePatch = cat(1,mouse_timePatch{:});
        end
        
        if ~isempty(mouse_timePatch)
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_timePatch = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_timePatch,'un',0);
            mouse_timePatch = var_dt  * cat(1,pad_timePatch{:}); % now this is a matrix
            for i_tt = 1:numel(trialtypes)
                these_trials = find(mouse_RXNil == trialtypes(i_tt));
                this_rewsize = floor(trialtypes(i_tt)/10);
                prts_tt = mouse_prts(these_trials);
                nTrials_tt = length(these_trials); 
                if nTrials_tt > 0
                    cmap = cbrewer('div',"RdBu",nTrials_tt);

                    % grab decoded variable
                    time_on_patch_tt = mouse_timePatch(these_trials,:);
                    [~,prt_sort] = sort(mouse_prts(these_trials));
                    prt_sorted_these_trials = these_trials(prt_sort);
                    time_on_patch_tt_prt_sort = time_on_patch_tt(prt_sort,:);
                    % get corrcoef per timepoint
                    for i_time = 1:analyze_ix(mIdx)
                        i_time_decoding = time_on_patch_tt(:,i_time);
                        if length(find(~isnan(i_time_decoding))) > 1
                            [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_tt(~isnan(i_time_decoding)));

                            r_prt{m_ix}{i_feature}(i_tt,i_time) = r(2);
                            p_prt{m_ix}{i_feature}(i_tt,i_time) = p(2);
                        end
                    end

                    subplot(numel(trialtypes),numel(analysis_mice),numel(analysis_mice) * (i_tt - 1) + m_ix);hold on

                    %  single trials visualization
                    %             for iTrial = 1:nTrials_tt
                    %                 plot((1:analyze_ix(mIdx))*tbin_ms/1000,gauss_smoothing(time_on_patch_tt_prt_sort(iTrial,:),1),'color',cmap(iTrial,:),'linewidth',.25)
                    %             end

                    % terciles visualization
                    terciles = [0 quantile(prts_tt,2) max(prts_tt)];
                    [~,~,tercile_bin] = histcounts(prts_tt,terciles);
                    for i_tercile = 1:max(tercile_bin)
                        if length(find(tercile_bin == i_tercile)) > 1
                            tt_tercile_mean = nanmean(time_on_patch_tt(tercile_bin == i_tercile,:));
                            tt_tercile_sem = nanstd(time_on_patch_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile)));
%                             shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
                        end
                    end

                    star_yloc = max(var_bins{mIdx}{this_rewsize}{1})+.05;
                    for i_time = 1:max(var_bins{mIdx}{this_rewsize}{1})*1000/tbin_ms
                        if p_prt{m_ix}{i_feature}(i_tt,i_time) < .05
                            % text(i_time*tbin_ms/1000,max(var_bins{mIdx}{this_rewsize}{1})+.05,'*','HorizontalAlignment','center')
                            plot([i_time i_time+1]*tbin_ms/1000,[star_yloc star_yloc],'k-','linewidth',2)
                        end
                    end 
                end
                
                ylim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
                xlim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
                xticks(0:1:max(var_bins{mIdx}{this_rewsize}{1}))
                yticks(0:1:max(var_bins{mIdx}{this_rewsize}{1}))
                set(gca,'fontsize',13)
                
                if i_tt == 1
                    title(mouse_names(mIdx))
                end
                if i_tt == numel(trialtypes)
                    xlabel("True Time (sec)");
                end
                if m_ix == 1
                    ylabel(sprintf("%s \n Decoded Time (sec)",[num2str(trialtypes(i_tt)) 'Nil']));
                end
            end
        end
    end
    suptitle(feature_names(iFeature))
end

%% 1b) Pool RXNil across mice
%   show across features
cell_threshold = true; 
pool_analyze_ix = round(3000/tbin_ms); 
vis_features = [5];
trialtypes = [10 11 20 22 40 44];
% close all
p_prt = cell(length(vis_features),1);
p_prt_direct = cell(length(vis_features),1);
figure();hold on 
for i_feature = 1:numel(vis_features)
    iFeature = vis_features(i_feature); 
    xMouse_timePatch = cell(numel(analysis_mice),1); 
    xMouse_RXNil = cell(numel(analysis_mice),1); 
    xMouse_prts = cell(numel(analysis_mice),1); 
    for m_ix = 1:numel(analysis_mice)
        mIdx = analysis_mice(m_ix); 
        % load RXNil for pooled sessions 
        if iFeature <= 3 && cell_threshold == true % introduce # cell session inclusion criterion
            mouse_include_sessions = ismember(pool_sessions{mIdx},clu_pool_sessions{iFeature}{mIdx}); 
            mouse_RXNil = RXNil{mIdx}(mouse_include_sessions); 
            mouse_RXNil = cat(1,mouse_RXNil{:}); 
            mouse_prts = prts{mIdx}(mouse_include_sessions); 
            mouse_prts = cat(1,mouse_prts{:}); 
            % gather decoded time variable from i_feature
            mouse_timePatch = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx}(mouse_include_sessions),'un',0);
            mouse_timePatch = cat(1,mouse_timePatch{:});
        else
            mouse_RXNil = cat(1,RXNil{mIdx}{:});
            % load PRTs for pooled sessions
            mouse_prts = cat(1,prts{mIdx}{:});
            % gather decoded time variable from i_feature
            mouse_timePatch = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx},'un',0);
            mouse_timePatch = cat(1,mouse_timePatch{:});
        end  
        if ~isempty(mouse_timePatch)
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_timePatch = cellfun(@(x) [x(1:min(length(x),pool_analyze_ix)) ; nan(max(0,pool_analyze_ix - length(x)),1)]',mouse_timePatch,'un',0);
            mouse_timePatch = var_dt  * cat(1,pad_timePatch{:}); % now this is a matrix
            xMouse_timePatch{m_ix} = mouse_timePatch; 
            xMouse_RXNil{m_ix} = mouse_RXNil; 
            xMouse_prts{m_ix} = mouse_prts; 
        end 
%         disp([mean(mouse_prts(mouse_RXNil == 10)) mean(mouse_prts(mouse_RXNil == 20)) mean(mouse_prts(mouse_RXNil == 40))])
    end
    % concatenate across mice
    xMouse_timePatch_full = cat(1,xMouse_timePatch{:}); 
    xMouse_RXNil_full = cat(1,xMouse_RXNil{:}); 
    xMouse_prts_full = cat(1,xMouse_prts{:}); 
    
    for i_tt = 1:numel(trialtypes)
        these_trials = find(xMouse_RXNil_full == trialtypes(i_tt));
        this_rewsize = floor(trialtypes(i_tt)/10);
        prts_tt = xMouse_prts_full(these_trials);
        nTrials_tt = length(these_trials);
        cmap = cbrewer('div',"RdBu",nTrials_tt);
        
        % grab decoded variable
        time_on_patch_tt = xMouse_timePatch_full(these_trials,:);
        [~,prt_sort] = sort(xMouse_prts_full(these_trials));
        prt_sorted_these_trials = these_trials(prt_sort);
        time_on_patch_tt_prt_sort = time_on_patch_tt(prt_sort,:);
        terciles = [0 quantile(prts_tt,2) max(prts_tt)];
        [~,~,tercile_bin] = histcounts(prts_tt,terciles);
        % get corrcoef per timepoint
        for i_time = 1:pool_analyze_ix
            i_time_decoding = time_on_patch_tt(:,i_time);
            if length(find(~isnan(i_time_decoding))) > 1
                % for pearson r
%                 [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_tt(~isnan(i_time_decoding)));
%                 r_prt{m_ix}{i_feature}(i_tt,i_time) = r(2);
%                 p_prt{m_ix}{i_feature}(i_tt,i_time) = p(2);
                p_prt{i_feature}(i_tt,i_time) = kruskalwallis(i_time_decoding(~isnan(i_time_decoding)),tercile_bin(~isnan(i_time_decoding)),'off'); 
                [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_tt(~isnan(i_time_decoding)));
                p_prt_direct{i_feature}(i_tt,i_time) = p(2); 
            end
        end
        
        subplot(numel(trialtypes),numel(vis_features),numel(vis_features) * (i_tt - 1) + i_feature);hold on

        % terciles visualization
        for i_tercile = 1:max(tercile_bin)
            if length(find(tercile_bin == i_tercile)) > 1
                tt_tercile_mean = nanmedian(time_on_patch_tt(tercile_bin == i_tercile,:));
                tt_tercile_sem = nanstd(time_on_patch_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile)));
                shadedErrorBar((1:pool_analyze_ix)*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:),'linewidth',1.5})
            end
        end
        
        star_yloc = max(var_bins{mIdx}{this_rewsize}{1})+.55;
        for i_time = 1:max(var_bins{mIdx}{this_rewsize}{1})*1000/tbin_ms
            if p_prt{i_feature}(i_tt,i_time) < .05
                % text(i_time*tbin_ms/1000,max(var_bins{mIdx}{this_rewsize}{1})+.05,'*','HorizontalAlignment','center')
                plot([i_time i_time+1]*tbin_ms/1000,[star_yloc star_yloc],'k-','linewidth',2)
            end
        end
        
        ylim([0 pool_analyze_ix*tbin_ms/1000+.65])
        xlim([0 pool_analyze_ix*tbin_ms/1000])
%         xlim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
        xticks(0:1:pool_analyze_ix*tbin_ms/1000)
        yticks(0:1:pool_analyze_ix*tbin_ms/1000)
        set(gca,'fontsize',13)
        
        if i_tt == 1
            title(feature_names(iFeature))
            legend(["Short PRT","Middle PRT","Long PRT"])
        end
        if i_tt == numel(trialtypes)
            xlabel("True Time (sec)");
        end
        if i_feature == 1
            ylabel(sprintf("%s \n Decoded Time (sec)",[num2str(trialtypes(i_tt)) 'Nil']));
        end
    end
end
    
%% 2) Decoded time since rew vs PRT after last rew 
analyze_ix = round([3000 3000 5000 3000 3000] / tbin_ms); 

trialtypes = [10 20 40];
smoothing_sigma = 1; 
vis_rewsizes = [1 2 4];
vis_features = [1 2 5]; 

r_prt = cell(numel(analysis_mice),1);
p_prt = cell(numel(analysis_mice),1);
for m_ix = 1:numel(analysis_mice)
    for i_feature = 1:numel(vis_features)
        r_prt{m_ix}{i_feature} = nan(numel(trialtypes),analyze_ix(mIdx));
        p_prt{m_ix}{i_feature} = nan(numel(trialtypes),analyze_ix(mIdx));
    end
end

cell_threshold = true; 

for i_feature = 1:numel(vis_features) 
    iFeature = vis_features(i_feature); 
    figure();hold on
    for m_ix = 1:numel(analysis_mice)
        mIdx = analysis_mice(m_ix); 
        
        % load RXNil for pooled sessions 
        if iFeature <= 3 && cell_threshold == true % introduce # cell session inclusion criterion
            mouse_include_sessions = ismember(pool_sessions{mIdx},clu_pool_sessions{iFeature}{mIdx}); 
            mouse_last_rew_ix = last_rew_ix{mIdx}(mouse_include_sessions); 
            mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:}); 
            mouse_rewsize = rewsize{mIdx}(mouse_include_sessions); 
            mouse_rewsize = cat(1,mouse_rewsize{:});
            mouse_postRew_prts = postRew_prts{mIdx}(mouse_include_sessions); 
            mouse_postRew_prts = cat(1,mouse_postRew_prts{:}); 
            nTrials = length(mouse_postRew_prts);
            % gather decoded time variable from i_feature
            mouse_timeSince = cellfun(@(x) x{iFeature}, timeUntil_hat{mIdx}(mouse_include_sessions),'un',0);
            mouse_timeSince = cat(1,mouse_timeSince{:});
        else
            % load last reward ix for pooled sessions
            mouse_last_rew_ix = cat(1,last_rew_ix{mIdx}{:});
            % load post rew residence times
            mouse_postRew_prts = cat(1,postRew_prts{mIdx}{:});
            nTrials = length(mouse_postRew_prts);
            % load reward size
            mouse_rewsize = cat(1,rewsize{mIdx}{:});
            
            % gather decoded time variable from i_feature
            mouse_timeSince = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx},'un',0);
            mouse_timeSince = cat(1,mouse_timeSince{:});
        end
        
        if ~isempty(mouse_timeSince)
            % now just pull off decoded timesince after last reward
            mouse_timeSince = arrayfun(@(iTrial) mouse_timeSince{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0);
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_timeSince = cellfun(@(x) [x(1:min(length(x),analyze_ix(mIdx))) ; nan(max(0,analyze_ix(mIdx) - length(x)),1)]',mouse_timeSince,'un',0);
            mouse_timeSince = var_dt * cat(1,pad_timeSince{:}); % now this is a matrix
            
            for i_rewsize = 1:numel(vis_rewsizes)
                iRewsize = vis_rewsizes(i_rewsize);
                these_trials = find(mouse_rewsize == iRewsize);
                nTrials_tt = length(these_trials);
                cmap = cbrewer('div',"RdBu",nTrials_tt);
                
                % grab decoded variable
                timesince_tt = mouse_timeSince(these_trials,:);
                prts_these_trials = mouse_postRew_prts(these_trials);
                
                for i_time = 1:analyze_ix(mIdx)
                    i_time_decoding = timesince_tt(:,i_time);
                    if length(find(~isnan(i_time_decoding))) > 1
                        [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_these_trials(~isnan(i_time_decoding)));
                        
                        r_prt{m_ix}{i_feature}(i_rewsize,i_time) = r(2);
                        p_prt{m_ix}{i_feature}(i_rewsize,i_time) = p(2);
                    end
                end
                
                subplot(numel(trialtypes),numel(analysis_mice),numel(analysis_mice) * (i_rewsize - 1) + m_ix);hold on
                
                % terciles visualization
                terciles = [0 quantile(prts_these_trials,2) max(prts_these_trials)];
                [~,~,tercile_bin] = histcounts(prts_these_trials,terciles);
                for i_tercile = 1:max(tercile_bin)
                    tt_tercile_mean = nanmean(timesince_tt(tercile_bin == i_tercile,:));
                    tt_tercile_sem = nanstd(timesince_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile)));
                    shadedErrorBar((1:analyze_ix(mIdx))*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
                end
                
                if i_rewsize == 1
                    title(mouse_names(mIdx))
                end
                if i_rewsize == numel(vis_rewsizes)
                    xlabel(sprintf("True Time \n Since Last Rew (sec)"));
                end
                if m_ix == 1
                    ylabel(sprintf("%i uL \n Decoded Time (sec)",iRewsize));
                end
                
                % add significance stars
                star_yloc = max(var_bins{mIdx}{this_rewsize}{1})+.05;
                for i_time = 1:max(var_bins{mIdx}{this_rewsize}{1})*1000/tbin_ms
                    if p_prt{m_ix}{i_feature}(i_tt,i_time) < .05
                        %                     text(i_time*tbin_ms/1000,max(var_bins{mIdx}{this_rewsize}{1})+.05,'*','HorizontalAlignment','center')
                        plot([i_time i_time+1]*tbin_ms/1000,[star_yloc star_yloc],'k-','linewidth',2)
                    end
                end
                
                ylim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
                xlim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
                xticks(0:1:max(var_bins{mIdx}{this_rewsize}{1}))
                yticks(0:1:max(var_bins{mIdx}{this_rewsize}{1}))
                set(gca,'fontsize',13)
            end
        end
    end
    suptitle(feature_names(iFeature))
    legend(["Shortest PRT Post Rew","Moderate PRT Post Rew","Longest PRT Post Rew"])
end

%% 2b) Pool post last rew across mice
%   show across features
cell_threshold = true; 
pool_analyze_ix = round(3000/tbin_ms); 
vis_features = [1 2 5];
vis_rewsizes = [4]; 
% close all
figure();hold on 

mouse_last_rew_ix = last_rew_ix{mIdx}(mouse_include_sessions);
mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:});
mouse_rewsize = rewsize{mIdx}(mouse_include_sessions);
mouse_rewsize = cat(1,mouse_rewsize{:});
mouse_postRew_prts = postRew_prts{mIdx}(mouse_include_sessions);
mouse_postRew_prts = cat(1,mouse_postRew_prts{:});

for i_feature = 1:numel(vis_features)
    iFeature = vis_features(i_feature); 
    xMouse_last_rew_ix = cell(numel(analysis_mice),1); 
    xMouse_postRew_prts = cell(numel(analysis_mice),1); 
    xMouse_rewsize = cell(numel(analysis_mice),1); 
    xMouse_timeSince = cell(numel(analysis_mice),1); 
    for m_ix = [1 3 4] % 3:numel(analysis_mice)
        mIdx = analysis_mice(m_ix); 
        % load RXNil for pooled sessions 
        if iFeature <= 3 && cell_threshold == true % introduce # cell session inclusion criterion
            mouse_include_sessions = ismember(pool_sessions{mIdx},clu_pool_sessions{iFeature}{mIdx}); 
            mouse_last_rew_ix = last_rew_ix{mIdx}(mouse_include_sessions); 
            mouse_last_rew_ix = cat(1,mouse_last_rew_ix{:}); 
            mouse_rewsize = rewsize{mIdx}(mouse_include_sessions); 
            mouse_rewsize= cat(1,mouse_rewsize{:}); 
            mouse_postRew_prts = postRew_prts{mIdx}(mouse_include_sessions);
            mouse_postRew_prts = cat(1,mouse_postRew_prts{:});
            % gather decoded time variable from i_feature
            mouse_timeSince = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx}(mouse_include_sessions),'un',0);
            mouse_timeSince = cat(1,mouse_timeSince{:});
        else
            mouse_last_rew_ix = cat(1,last_rew_ix{mIdx}{:});
            % load PRTs for pooled sessions
            mouse_postRew_prts = cat(1,postRew_prts{mIdx}{:});
            mouse_rewsize = cat(1,rewsize{mIdx}{:}); 
            % gather decoded time variable from i_feature
            mouse_timeSince = cellfun(@(x) x{iFeature}, timeSince_hat{mIdx},'un',0);
            mouse_timeSince= cat(1,mouse_timeSince{:});
        end  
        if ~isempty(mouse_timeSince)
            nTrials = length(mouse_last_rew_ix);
            mouse_timeSince = arrayfun(@(iTrial) mouse_timeSince{iTrial}(mouse_last_rew_ix(iTrial):end),(1:nTrials)','un',0);
            % concatenate and pad to make [nTrials x analyze_ix] sized matrix that will be nice to work with
            pad_timeSince = cellfun(@(x) [x(1:min(length(x),pool_analyze_ix)) ; nan(max(0,pool_analyze_ix - length(x)),1)]',mouse_timeSince,'un',0);
            mouse_timeSince = var_dt * cat(1,pad_timeSince{:}); % now this is a matrix
            xMouse_last_rew_ix{m_ix} = mouse_last_rew_ix; 
            xMouse_postRew_prts{m_ix} = mouse_postRew_prts; 
            xMouse_rewsize{m_ix} = mouse_rewsize; 
            xMouse_timeSince{m_ix} = mouse_timeSince; 
        end
    end
    
    % concatenate across mice
    xMouse_last_rew_ix_full = cat(1,xMouse_last_rew_ix{:}); 
    xMouse_postRew_prts_full = cat(1,xMouse_postRew_prts{:}); 
    xMouse_rewsize_full = cat(1,xMouse_rewsize{:}); 
    xMouse_timeSince_full = cat(1,xMouse_timeSince{:}); 
    
    for i_rewsize = 1:numel(vis_rewsizes)
        iRewsize = vis_rewsizes(i_rewsize);
        these_trials = find(xMouse_rewsize_full == iRewsize);
        nTrials_tt = length(these_trials);
        cmap = cbrewer('div',"RdBu",nTrials_tt);
        
        % grab decoded variable
        timesince_tt = xMouse_timeSince_full(these_trials,:);
        prts_tt = xMouse_postRew_prts_full(these_trials);
        
        % grab decoded variable
        [~,prt_sort] = sort(prts_tt);
        prt_sorted_these_trials = these_trials(prt_sort);
        time_on_patch_tt_prt_sort = timesince_tt(prt_sort,:);
        terciles = [0 quantile(prts_tt,2) max(prts_tt)];
        [~,~,tercile_bin] = histcounts(prts_tt,terciles);
        % get corrcoef per timepoint
        for i_time = 1:pool_analyze_ix
            i_time_decoding = timesince_tt(:,i_time);
            if length(find(~isnan(i_time_decoding))) > 1
                [r,p] = corrcoef(i_time_decoding(~isnan(i_time_decoding)),prts_tt(~isnan(i_time_decoding)));
                
                r_prt{m_ix}{i_feature}(i_rewsize,i_time) = r(2);
                p_prt{m_ix}{i_feature}(i_rewsize,i_time) = p(2);
%                 p_prt{m_ix}{i_feature}(i_tt,i_time) = kruskalwallis(i_time_decoding(~isnan(i_time_decoding)),tercile_bin(~isnan(i_time_decoding)),'off'); 
            end
        end
        
        subplot(numel(vis_rewsizes),numel(vis_features),numel(vis_features) * (i_rewsize - 1) + i_feature);hold on
        % terciles visualization
        for i_tercile = 1:max(tercile_bin)
            if length(find(tercile_bin == i_tercile)) > 1
                tt_tercile_mean = nanmean(timesince_tt(tercile_bin == i_tercile,:));
                tt_tercile_sem = nanstd(timesince_tt(tercile_bin == i_tercile,:)) / sqrt(length(find(tercile_bin == i_tercile)));
                shadedErrorBar((1:pool_analyze_ix)*tbin_ms/1000,tt_tercile_mean,tt_tercile_sem,'lineProps',{'color',rdbu3(i_tercile,:)})
            end
        end
        
        star_yloc = max(var_bins{mIdx}{this_rewsize}{1})+1.05;
        for i_time = 1:max(var_bins{mIdx}{this_rewsize}{1})*1000/tbin_ms
            if p_prt{m_ix}{i_feature}(i_rewsize,i_time) < .05
                % text(i_time*tbin_ms/1000,max(var_bins{mIdx}{this_rewsize}{1})+.05,'*','HorizontalAlignment','center')
                plot([i_time i_time+1]*tbin_ms/1000,[star_yloc star_yloc],'k-','linewidth',2)
            end
        end
        
        ylim([0 pool_analyze_ix*tbin_ms/1000+1.15])
        xlim([0 pool_analyze_ix*tbin_ms/1000])
%         xlim([min(var_bins{mIdx}{this_rewsize}{1}) max(var_bins{mIdx}{this_rewsize}{1})+.15])
        xticks(0:1:pool_analyze_ix*tbin_ms/1000)
        yticks(0:1:pool_analyze_ix*tbin_ms/1000)
        set(gca,'fontsize',13)
        
        if i_rewsize == 1
            title(feature_names(iFeature))
        end

        if i_feature == 1
            ylabel(sprintf("%i uL \n Decoded Time (sec)",iRewsize));
        end
        
        if i_rewsize == numel(vis_rewsizes)
            xlabel(sprintf("True Time \n Since Last Rew (sec)"));
        end
        
    end
end
