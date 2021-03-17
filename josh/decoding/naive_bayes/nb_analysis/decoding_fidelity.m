%% A script to assess time decoding fidelity across sessions 
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
dataset_opt = nb_results.dataset_opt;
feature_names = cellfun(@(x) x.name, nb_results.dataset_opt.features{5}{1});

%% 0.i) Collect predictions to reformat s.t. easier to pool trials
%   Cell format: timeSince_hat{mIdx}{i}{trained_rewsize}{i_feature}{iTrial}
%   Cell format: timeSince_sameRewsize_hat{mIdx}{i}{i_feature}{iTrial}

y_hat = nb_results.y_hat;
y_true = nb_results.y_true; 
nMice = numel(mouse_grps); 

timeSince_hat = cell(nMice,1); 
timePatch_hat = cell(nMice,1); 
timeUntil_hat = cell(nMice,1);  
timeUntil_hat_postRew = cell(nMice,1);  
totalRew_hat = cell(nMice,1);  

% to load from decoding directly
timeSince_ix = 1; 
timePatch_ix = 2; 
timeUntil_ix = 3; 
totalRew_ix = 4; 

% these are just from the pop_decoding_session_sep
rewsizes = [1 2 4];
rewsize = cell(nMice,1);
trial_decoding_features = 1:6;

n_cells = cell(nMice,1); 

for mIdx = 1:nMice
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);  
        session = sessions{sIdx}(1:end-4); 
        data = load(fullfile(paths.beh_data,sessions{sIdx}));  
        session_title = session([1:2 end-2:end]);
        session_rewsize = mod(data.patches(:,2),10); 
        rewsize{mIdx}{i} = session_rewsize;
        nTrials = length(session_rewsize);
        post_rew_ix = cellfun(@(x) size(x,2),y_true{mIdx}{i,timeUntil_ix});
        
        for i_feature = 1:numel(trial_decoding_features) 
            iFeature = trial_decoding_features(i_feature); 
            timeSince_hat{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{i}{iTrial}{timeSince_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            timePatch_hat{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{i}{iTrial}{timePatch_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            timeUntil_hat{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{i}{iTrial}{timeUntil_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
            if ~isempty(timeUntil_hat{mIdx}{i}{i_feature}{1})
                timeUntil_hat_postRew{mIdx}{i}{i_feature} = arrayfun(@(iTrial) timeUntil_hat{mIdx}{i}{i_feature}{iTrial}(end-post_rew_ix(iTrial)+1:end),(1:nTrials)','un',0);
            else 
                timeUntil_hat_postRew{mIdx}{i}{i_feature} = timeUntil_hat{mIdx}{i}{i_feature};
            end
            totalRew_hat{mIdx}{i}{i_feature} = arrayfun(@(iTrial) y_hat{mIdx}{i}{iTrial}{totalRew_ix}{min(3,session_rewsize(iTrial))}{iFeature},(1:nTrials)','un',0);
        end

        session_sigcells = sig_cells(strcmp(sig_cells.Session,session),:); 
        glm_clusters_session = session_sigcells.GMM_cluster; 
        n_cells{mIdx}{i} = [arrayfun(@(x) length(find(glm_clusters_session == x)),(1:4)) length(glm_clusters_session)];
    end
end 
clear y_hat % we now have this in an easier form

%% 1) Confusion matrix visualization (time since reward)

y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;

i_feature = 1;
vis_features = 1:6;
rewsizes = 4;
for i_rewsize = 1:numel(rewsizes)
    for mIdx = 1:5
        figure()
        for i = 1:numel(timeSince_hat{mIdx})
            session_rewsize = rewsize{mIdx}{i};
            n_sessions = numel(mouse_grps{mIdx});
            
            iRewsize = rewsizes(i_rewsize);
            session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
            var_dt = diff(session_var_bins(1:2));

            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature);
                
                timeSince_hat_full = timeSince_hat{mIdx}{i}{i_feature}(session_rewsize == iRewsize);
                timeSince_hat_full = var_dt * cat(1,timeSince_hat_full{:}); 
                if ~isempty(timeSince_hat_full)
                    timeSince_true_full = y_true{mIdx}{i,timeSince_ix}(session_rewsize == iRewsize);
                    timeSince_true_full = var_dt * cat(2,timeSince_true_full{:})';
                    rmse = sqrt(nanmean((timeSince_hat_full - timeSince_true_full).^2));
                    this_confusionmat = confusionmat(timeSince_true_full,timeSince_hat_full);
                    
                    subplot(numel(vis_features),n_sessions,i + n_sessions * (i_feature-1))
                    imagesc(flipud(this_confusionmat));
                    if iFeature <= 5
                    n = n_cells{mIdx}{i}(iFeature); 
                    end
                    n_bins = length(session_var_bins);
                    % title
                    if i_feature == 1
                        title(sprintf("%s \n N = %i MI = %.3f RMSE: %.3f",session_titles{mIdx}{i},n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                    else
                        title(sprintf("N = %i MI = %.3f RMSE: %.3f",n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                    end
%                     end
                    % ylabel
                    if i == 1
                        ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                        yticks(1:10:(length(session_var_bins)-1));
                        yticklabels(fliplr(session_var_bins(1:10:end)))
                    else
                        yticks([])
                    end
                    % xlabel
                    if i_feature == numel(vis_features)
                        xlabel("Predicted Time (sec)")
                        xticks(1:10:(length(session_var_bins)-1));
                        xticklabels(session_var_bins(1:10:end))
                    else
                        xticks([])
                    end
                end
            end
        end
    end
end

%% 2) Plot # neurons vs normalized mutual information and RMSE
cool3 = cool(3); 

iFeature = 5; % all GLM significant cells
n_cells_threshold = 30; 
figure()
for iRewsize = [1 2 4]
    rmse = nan(length(mPFC_sessions),1);
    mi = nan(length(mPFC_sessions),1);
    n_these_cells = nan(length(mPFC_sessions),1);
    counter = 1;
    for mIdx = 1:5
        for i = 1:numel(timeSince_hat{mIdx})
            n_bins = length(var_bins{mIdx}{iRewsize}{timeSince_ix});
            total_time = max(var_bins{mIdx}{iRewsize}{1}); 
            timeSince_hat_full = timeSince_hat{mIdx}{i}{iFeature}(rewsize{mIdx}{i} == iRewsize);
            timeSince_hat_full = var_dt * cat(1,timeSince_hat_full{:});
            if ~isempty(timeSince_hat_full)
                timeSince_true_full = y_true{mIdx}{i,timeSince_ix}(rewsize{mIdx}{i} == iRewsize);
                timeSince_true_full = var_dt * cat(2,timeSince_true_full{:})';
                rmse(counter) = sqrt(nanmean((timeSince_hat_full/total_time - timeSince_true_full/total_time).^2));
                mi(counter) = MI_confusionmat(confusionmat(timeSince_true_full,timeSince_hat_full)) / log(n_bins);
                n_these_cells(counter) = n_cells{mIdx}{i}(iFeature);
            end
            counter = counter + 1;
        end
    end
    subplot(1,2,1);hold on
    scatter(n_these_cells,mi,200,cool3(min(iRewsize,3),:),'.')
    scatter(n_these_cells,mi,[],'k') 
    xlabel("# Cells")
    ylabel("Mutual Information (Normalized)")
    set(gca,'fontsize',14) 
    xline(n_cells_threshold,'k--','linewidth',1.5)
    w = polyfit(log(n_these_cells),mi,1);
    xl = xlim();
    plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',cool3(min(iRewsize,3),:),'linewidth',1.5)
    
    subplot(1,2,2);hold on
    scatter(n_these_cells,rmse,200,cool3(min(iRewsize,3),:),'.')
    scatter(n_these_cells,rmse,[],'k')
    xlabel("# Cells") 
    ylabel("RMSE (Normalized)") 
    set(gca,'fontsize',14)  
    xline(n_cells_threshold,'k--','linewidth',1.5)
    
    w = polyfit(log(n_these_cells),rmse,1);
    xl = xlim();
    plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',cool3(min(iRewsize,3),:),'linewidth',1.5)
end
suptitle("Time Since Reward Decoding Fidelity")

%% 2.5) Choose cell threshold of 30 now pool sessions based on this 

all_cells_feature_ix = 5;
pool_sessions = arrayfun(@(i_mouse) find(cellfun(@(x) x(all_cells_feature_ix),n_cells{i_mouse}) > n_cells_threshold),(1:nMice)','un',0);
% only use sessions w/ 
cluster_n_cells_threshold = 15; 
clu_pool_sessions{1} = arrayfun(@(i_mouse) find(cellfun(@(x) all(x(1) > cluster_n_cells_threshold),n_cells{i_mouse})),(1:nMice)','un',0);
clu_pool_sessions{2} = arrayfun(@(i_mouse) find(cellfun(@(x) all(x(2) > cluster_n_cells_threshold),n_cells{i_mouse})),(1:nMice)','un',0);
clu_pool_sessions{3} = arrayfun(@(i_mouse) find(cellfun(@(x) all(x(3) > cluster_n_cells_threshold),n_cells{i_mouse})),(1:nMice)','un',0);

%% 3) Visualize confusion matrices in n_cell thresholded sessions 

y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;
close all
i_feature = 1;
vis_features = 1:6;
rewsizes = 4; 
good_confusion_mats = cell(nMice,1); 
good_errors = cell(nMice,1); 
good_y_true = cell(nMice,1); 
for i_rewsize = 1:numel(rewsizes)
    for mIdx = 1:5
        good_confusion_mats{mIdx} = cell(numel(pool_sessions{mIdx}),1); 
        figure()
        for i_i = 1:numel(pool_sessions{mIdx})
            i = pool_sessions{mIdx}(i_i); 
            session_rewsize = rewsize{mIdx}{i};
            n_pool_sessions = numel(pool_sessions{mIdx});
            
            iRewsize = rewsizes(i_rewsize);
            session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
            var_dt = diff(session_var_bins(1:2));
            
            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature);
                if iFeature > 3 || iFeature <=3 && ismember(i,clu_pool_sessions{iFeature}{mIdx}) % only add if enough cells for this cluster
                    timeSince_hat_full = timeSince_hat{mIdx}{i}{i_feature}(session_rewsize == iRewsize);
                    timeSince_hat_full = var_dt * cat(1,timeSince_hat_full{:});
                    if ~isempty(timeSince_hat_full)
                        timeSince_true_full = y_true{mIdx}{i,timeSince_ix}(session_rewsize == iRewsize);
                        timeSince_true_full = var_dt * cat(2,timeSince_true_full{:})';
                        rmse = sqrt(nanmean((timeSince_hat_full - timeSince_true_full).^2));
                        this_confusionmat = confusionmat(timeSince_true_full,timeSince_hat_full);
                        good_confusion_mats{mIdx}{i_i}{i_feature} = confusionmat(timeSince_true_full,timeSince_hat_full);
                        good_errors{mIdx}{i_i}{i_feature} = timeSince_hat_full - timeSince_true_full;
                        good_y_true{mIdx}{i_i} = timeSince_true_full;
                        
                        subplot(numel(vis_features),n_pool_sessions,i_i + n_pool_sessions * (i_feature-1))
                        imagesc(flipud(this_confusionmat));
                        if iFeature <= 5
                            n = n_cells{mIdx}{i}(iFeature);
                            n_bins = length(session_var_bins);
                            % title
                            if i_feature == 1
                                title(sprintf("%s \n N = %i MI = %.3f RMSE: %.3f",session_titles{mIdx}{i},n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                            else
                                title(sprintf("N = %i MI = %.3f RMSE: %.3f",n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                            end
                        end
                        % ylabel
                        if i == 1
                            ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                            yticks(1:10:(length(session_var_bins)-1));
                            yticklabels(fliplr(session_var_bins(1:10:end)))
                        else
                            yticks([])
                        end
                        % xlabel
                        if i_feature == numel(vis_features)
                            xlabel("Predicted Time (sec)")
                            xticks(1:10:(length(session_var_bins)-1));
                            xticklabels(session_var_bins(1:10:end))
                        else
                            xticks([])
                        end
                    end
                end
            end
        end
    end
end

%% 4) Visualize confusion matrices pooling sessions over mice 
vis_mice = find(cellfun(@length, pool_sessions) > 0);
figure() 
for m_ix = 1:numel(vis_mice) 
    mIdx = vis_mice(m_ix); 
    session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
    var_dt = diff(session_var_bins(1:2));
    
    for i_feature = 1:numel(vis_features)
        iFeature = vis_features(i_feature);
        
        feature_confusionmat_tmp = cellfun(@(x) x{i_feature},good_confusion_mats{mIdx},'un',0);
        feature_confusionmat_pooled = sum(cat(3,feature_confusionmat_tmp{:}),3); 
        
        subplot(numel(vis_features),numel(vis_mice),m_ix + numel(vis_mice) * (i_feature-1))
        imagesc(flipud(feature_confusionmat_pooled));
        if iFeature <= 5
            n = n_cells{mIdx}{i}(iFeature);
            n_bins = length(session_var_bins);
            % title
            if i_feature == 1
                title(sprintf("%s \n MI = %.3f",mouse_names(mIdx),MI_confusionmat(this_confusionmat)/log(n_bins)))
            else
                title(sprintf("MI = %.3f",MI_confusionmat(feature_confusionmat_pooled)/log(n_bins)))
            end
        end
        % ylabel
        if m_ix == 1
            ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
            yticks(1:10:(length(session_var_bins)-1));
            yticklabels(fliplr(session_var_bins(1:10:end)))
        else
            yticks([])
        end
        
        % xlabel
        if i_feature == numel(vis_features)
            xlabel("Predicted Time (sec)")
            xticks(1:10:(length(session_var_bins)-1));
            xticklabels(session_var_bins(1:10:end))
        else
            xticks([])
        end 
        set(gca,'fontsize',13)
    end
end

%% 5) Visualize confusion mats pooled across mice 
pool_mice = 2:5; 
pool_var_bins = var_bins{2}{4}{1};
colors = [lines(4);0 .9 .9 ; .2 .2 .2]; 
n_bins = length(pool_var_bins) - 1;
%  Do not include mouse 78 here
xMouse_goodConfusion_mats = good_confusion_mats(pool_mice);
xMouse_good_errors = good_errors(pool_mice);
xMouse_good_y_true = good_y_true(pool_mice);
close all

for i_feature = 1:numel(vis_features) 
    confusionmat_tmp = cell(numel(pool_mice),1);
    for m_ix = 1:numel(pool_mice)
        for i = 1:numel(xMouse_goodConfusion_mats{m_ix}) 
            if ~isempty(xMouse_goodConfusion_mats{m_ix}{i}{i_feature})
                confusionmat_tmp{m_ix}{i} = xMouse_goodConfusion_mats{m_ix}{i}{i_feature}(1:n_bins,1:n_bins);
            end
        end
    end 
    confusionmat_tmp(cellfun(@isempty,confusionmat_tmp)) = []; 
    confusionmat_cat = cellfun(@(x) cat(3,x{:}),confusionmat_tmp,'un',0);
    confusionmat_cat = cat(3,confusionmat_cat{:}); 
    confusionmat_pooled = sum(confusionmat_cat,3);
    
    figure(1)
    subplot(2,numel(vis_features),i_feature)
    imagesc(flipud(confusionmat_pooled)) 
    
    % ylabel
    if i_feature == 1
        ylabel("True Time Since Reward (sec)")
        yticks(1:10:(length(pool_var_bins)-1));
        yticklabels(fliplr(pool_var_bins(1:10:end)))
    else
        yticks([])
    end
    
    xlabel("Decoded T. Since Rew (sec)") 
    xticks(1:10:(length(pool_var_bins)-1));
    xticklabels(pool_var_bins(1:10:end))
    title(dataset_opt.features{5}{2}{i_feature}.name)
    set(gca,'fontsize',12)

    errors_tmp = cell(numel(pool_mice),1); 
    ytrue_tmp = cell(numel(pool_mice),1);  
    rmse_mean_mice = cell(numel(pool_mice),1); 
    for m_ix = 1:numel(pool_mice)  
        errors_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1); 
        ytrue_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1);
        for i = 1:numel(xMouse_goodConfusion_mats{m_ix}) 
            if ~isempty(xMouse_goodConfusion_mats{m_ix}{i}{i_feature})
                errors_tmp{m_ix}{i} = xMouse_good_errors{m_ix}{i}{i_feature};
                ytrue_tmp{m_ix}{i} = xMouse_good_y_true{m_ix}{i};
            end
        end
        % get mean in mouse to plot over overall avg 
        rmse_mean_mice{m_ix} = nan(numel(pool_var_bins)-1,1); 
        mouse_errors_full = cat(1,errors_tmp{m_ix}{:});
        mouse_ytrue_full = cat(1,ytrue_tmp{m_ix}{:});
        for i_time = 1:numel(pool_var_bins) 
            iTime = pool_var_bins(i_time);
            n = length(find((abs(mouse_ytrue_full - iTime) < .001)));
            rmse_mean_mice{m_ix}(i_time) = sqrt(nanmean(mouse_errors_full(abs(mouse_ytrue_full - iTime) < .01).^2));
        end 
    end 
    errors_cat = cellfun(@(x) cat(1,x{:}),errors_tmp,'un',0);
    ytrue_cat = cellfun(@(x) cat(1,x{:}),ytrue_tmp,'un',0);
    errors_full = cat(1,errors_cat{:}); 
    ytrue_full = cat(1,ytrue_cat{:}); 
    
    rmse_mean = nan(numel(pool_var_bins),1); 
    rmse_sem = nan(numel(pool_var_bins),1);
    for i_time = 1:numel(pool_var_bins) 
        iTime = pool_var_bins(i_time); 
        n = length(find((abs(ytrue_full - iTime) < .001)));
        rmse_mean(i_time) = sqrt(nanmean(errors_full(abs(ytrue_full - iTime) < .01).^2)); 
        rmse_sem(i_time) = 1.96*nanstd(errors_full(abs(ytrue_full - iTime) < .01).^2) / sqrt(n); 
    end
    subplot(2,numel(vis_features),i_feature + numel(vis_features))
    shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
    ylim([0 2]) 
    xlabel("True Time (sec)") 
    if i_feature == 1
        ylabel("RMSE (sec)")
    end 
    for m_ix = 1:numel(pool_mice) 
        plot(pool_var_bins,rmse_mean_mice{m_ix},'color',colors(i_feature,:))
    end
    set(gca,'fontsize',13) 
    
    % now add same to other figure 
    if ismember(i_feature,[1 2 4 5])
        figure(2) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names([1 2 4 5])) 
        title("Decoding Timecourse Across Sub-Populations") 
        ylabel("RMSE (sec)")
    end 
    
    if ismember(i_feature,[1 2 4]) || i_feature == 6
        figure(3) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names([1 2 4 6])) 
        title("Decoding Timecourse Compared To Velocity") 
        ylabel("RMSE (sec)")
    end
    
    if i_feature == 5 || i_feature == 6
        figure(4) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names(5:6)) 
        title("Decoding Timecourse Compared To Velocity")  
        ylabel("RMSE (sec)")
    end 
end 

%% 5.5) Add session pooling information and save
nb_results.pool_sessions = pool_sessions;
save(paths.nb_results,'nb_results');

%% 5.75) Now select for sessions with 20+ cluster 1 and cluster 2 cells for noise correlation analysis
clu12_n_cells_threshold = 20; 
clu12_pool_sessions = arrayfun(@(i_mouse) find(cellfun(@(x) all(x([1 2]) > clu12_n_cells_threshold),n_cells{i_mouse})),(1:nMice)','un',0);
% and save
nb_results.clu12_pool_sessions = clu12_pool_sessions;
save(paths.nb_results,'nb_results');

%% 6) Now analyze total reward decoding fidelity
%  First, confusion matrix 
y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;
close all
vis_features = [1:3 5];
rewsizes = 4;
for i_rewsize = 1:numel(rewsizes)
    for mIdx = 1:5
        figure()
        for i_i = 1:numel(pool_sessions{mIdx})
            i = pool_sessions{mIdx}(i_i); 
            n_pool_sessions = numel(pool_sessions{mIdx});
            session_rewsize = rewsize{mIdx}{i};
            n_sessions = numel(mouse_grps{mIdx});
            
            iRewsize = rewsizes(i_rewsize);
            session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
            var_dt = diff(session_var_bins(1:2));

            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature);
                
                totalRew_hat_full = totalRew_hat{mIdx}{i}{i_feature}(session_rewsize == iRewsize);
                totalRew_hat_full = var_dt * cat(1,totalRew_hat_full{:}); 
                if ~isempty(totalRew_hat_full)
                    totalRew_true_full = y_true{mIdx}{i,totalRew_ix}(session_rewsize == iRewsize);
                    totalRew_true_full = var_dt * cat(2,totalRew_true_full{:})';
                    this_confusionmat = confusionmat(totalRew_true_full,totalRew_hat_full);
                    
                    subplot(numel(vis_features),n_pool_sessions,i_i + n_pool_sessions * (i_feature-1))
                    imagesc(flipud(this_confusionmat(1:4,1:4)));
                    if iFeature <= 5
                        n = n_cells{mIdx}{i}(iFeature);
                        n_bins = size(this_confusionmat,2);
                        % title
                        if i_feature == 1
                            title(sprintf("%s \n N = %i MI = %.3f",session_titles{mIdx}{i},n,MI_confusionmat(this_confusionmat)/log(n_bins)))
                        else
                            title(sprintf("N = %i MI = %.3f",n,MI_confusionmat(this_confusionmat)/log(n_bins)))
                        end
                    end  
%                     xlim([1 4])
%                     ylim([1 4])
%                     ylabel
                    if i == 1
                        ylabel(sprintf("%s \n True Total Rew",dataset_opt.features{mIdx}{i}{iFeature}.name))
                        yticks(1:10:(length(session_var_bins)-1));
                        yticklabels(fliplr(session_var_bins(1:10:end)))
                    else
                        yticks([])
                    end
                    % xlabel
                    if i_feature == numel(vis_features)
                        xlabel("Predicted Total Rew (sec)")
                        xticks(1:10:(length(session_var_bins)-1));
                        xticklabels(session_var_bins(1:10:end))
                    else
                        xticks([])
                    end
                    
                    set(gca, 'YDir','reverse')
                end
            end
        end
    end
end

%% 7) Assess time until leave decoding fidelity

y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;

vis_features = 1:5;
rewsizes = 1;
for i_rewsize = 1:numel(rewsizes)
    for mIdx = 1:5
        figure()
        for i = 1:numel(timeSince_hat{mIdx})
            session_rewsize = rewsize{mIdx}{i};
            n_sessions = numel(mouse_grps{mIdx});
            
            iRewsize = rewsizes(i_rewsize);
            session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
            var_dt = diff(session_var_bins(1:2));

            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature);
                
                
                timeUntil_hat_full = timeUntil_hat_postRew{mIdx}{i}{i_feature}(session_rewsize == iRewsize);
                timeUntil_hat_full = var_dt * cat(1,timeUntil_hat_full{:});
                if ~isempty(timeUntil_hat_full)
                    timeUntil_true_full = y_true{mIdx}{i,timeUntil_ix}(session_rewsize == iRewsize);
                    timeUntil_true_full = var_dt * cat(2,timeUntil_true_full{:})';
                    rmse = sqrt(nanmean((timeUntil_hat_full - timeUntil_true_full).^2));
                    this_confusionmat = confusionmat(timeUntil_true_full,timeUntil_hat_full);
                    
                    subplot(numel(vis_features),n_sessions,i + n_sessions * (i_feature-1))
                    imagesc(flipud(this_confusionmat));
                    if iFeature <= 5
                        n = n_cells{mIdx}{i}(iFeature);
                        n_bins = length(session_var_bins);
                        % title
                        if i_feature == 1
                            title(sprintf("%s \n N = %i MI = %.3f RMSE: %.3f",session_titles{mIdx}{i},n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                        else
                            title(sprintf("N = %i MI = %.3f RMSE: %.3f",n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                        end
                    end
                    % ylabel
                    if i == 1
                        ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                        yticks(1:10:(length(session_var_bins)-1));
                        yticklabels(fliplr(session_var_bins(1:10:end)))
                    else
                        yticks([])
                    end
                    % xlabel
                    if i_feature == numel(vis_features)
                        xlabel("Predicted Time (sec)")
                        xticks(1:10:(length(session_var_bins)-1));
                        xticklabels(session_var_bins(1:10:end))
                    else
                        xticks([])
                    end
                    
                end
            end
        end
    end
end

%% 8) number of neurons vs time until leave decoding fidelity

cool3 = cool(3); 

iFeature = 5; % all GLM significant cells
n_cells_threshold = 30; 
figure()
for iRewsize = [1 2 4]
    rmse = nan(length(mPFC_sessions),1);
    mi = nan(length(mPFC_sessions),1);
    n_these_cells = nan(length(mPFC_sessions),1);
    counter = 1;
    for mIdx = 1:5
        for i = 1:numel(timeSince_hat{mIdx})
            n_bins = length(var_bins{mIdx}{iRewsize}{timeSince_ix});
            total_time = max(var_bins{mIdx}{iRewsize}{1}); 
            timeUntil_hat_full = timeUntil_hat_postRew{mIdx}{i}{i_feature}(rewsize{mIdx}{i} == iRewsize);
            timeUntil_hat_full = var_dt * cat(1,timeUntil_hat_full{:});
            if ~isempty(timeSince_hat_full)
                timeUntil_true_full = y_true{mIdx}{i,timeUntil_ix}(rewsize{mIdx}{i} == iRewsize);
                timeUntil_true_full = var_dt * cat(2,timeUntil_true_full{:})';
                rmse(counter) = sqrt(nanmean((timeUntil_hat_full/total_time - timeUntil_true_full/total_time).^2));
                mi(counter) = MI_confusionmat(confusionmat(timeUntil_true_full,timeUntil_hat_full)) / log(n_bins);
                n_these_cells(counter) = n_cells{mIdx}{i}(iFeature);
            end
            counter = counter + 1;
        end
    end
    subplot(1,2,1);hold on
    scatter(n_these_cells,mi,200,cool3(min(iRewsize,3),:),'.')
    scatter(n_these_cells,mi,[],'k') 
    xlabel("# Cells")
    ylabel("Mutual Information (Normalized)")
    set(gca,'fontsize',14) 
    xline(n_cells_threshold,'k--','linewidth',1.5)
    w = polyfit(log(n_these_cells),mi,1);
    xl = xlim();
    plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',cool3(min(iRewsize,3),:),'linewidth',1.5)
    
    subplot(1,2,2);hold on
    scatter(n_these_cells,rmse,200,cool3(min(iRewsize,3),:),'.')
    scatter(n_these_cells,rmse,[],'k')
    xlabel("# Cells") 
    ylabel("RMSE (Normalized)") 
    set(gca,'fontsize',14)  
    xline(n_cells_threshold,'k--','linewidth',1.5)
    
    w = polyfit(log(n_these_cells),rmse,1);
    xl = xlim();
    plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',cool3(min(iRewsize,3),:),'linewidth',1.5)
end
suptitle("Time Until Leave Decoding Fidelity")

%% 9) Visualize confusion matrices in n_cell thresholded sessions 

y_true = nb_results.y_true; 
var_bins = nb_results.var_bins;
close all
vis_features = 1:6;
rewsizes = 4; 
good_confusion_mats = cell(nMice,1); 
good_errors = cell(nMice,1); 
good_y_true = cell(nMice,1); 
for i_rewsize = 1:numel(rewsizes)
    for mIdx = 1:5
        good_confusion_mats{mIdx} = cell(numel(pool_sessions{mIdx}),1); 
        figure()
        for i_i = 1:numel(pool_sessions{mIdx})
            i = pool_sessions{mIdx}(i_i); 
            session_rewsize = rewsize{mIdx}{i};
            n_pool_sessions = numel(pool_sessions{mIdx});
            
            iRewsize = rewsizes(i_rewsize);
            session_var_bins = var_bins{mIdx}{iRewsize}{timeSince_ix};
            var_dt = diff(session_var_bins(1:2));

            for i_feature = 1:numel(vis_features)
                iFeature = vis_features(i_feature);
                if iFeature > 3 || iFeature <=3 && ismember(i,clu_pool_sessions{iFeature}{mIdx}) % only add if enough cells for this cluster
                    timeUntil_hat_full = timeUntil_hat_postRew{mIdx}{i}{i_feature}(session_rewsize == iRewsize);
                    timeUntil_hat_full = var_dt * cat(1,timeUntil_hat_full{:});
                    if ~isempty(timeUntil_hat_full)
                        timeUntil_true_full = y_true{mIdx}{i,timeUntil_ix}(session_rewsize == iRewsize);
                        timeUntil_true_full = var_dt * cat(2,timeUntil_true_full{:})';
                        rmse = sqrt(nanmean((timeUntil_hat_full - timeUntil_true_full).^2));
                        this_confusionmat = confusionmat(timeUntil_true_full,timeUntil_hat_full);
                        good_confusion_mats{mIdx}{i_i}{i_feature} = confusionmat(timeUntil_true_full,timeUntil_hat_full);
                        good_errors{mIdx}{i_i}{i_feature} = timeUntil_hat_full - timeUntil_true_full;
                        good_y_true{mIdx}{i_i} = timeUntil_true_full;
                        
                        subplot(numel(vis_features),n_pool_sessions,i_i + n_pool_sessions * (i_feature-1))
                        imagesc(flipud(this_confusionmat));
                        if iFeature <= 5
                            n = n_cells{mIdx}{i}(iFeature);
                            n_bins = length(session_var_bins);
                            % title
                            if i_feature == 1
                                title(sprintf("%s \n N = %i MI = %.3f RMSE: %.3f",session_titles{mIdx}{i},n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                            else
                                title(sprintf("N = %i MI = %.3f RMSE: %.3f",n,MI_confusionmat(this_confusionmat)/log(n_bins),rmse))
                            end
                        end
                        % ylabel
                        if i == 1
                            ylabel(sprintf("%s \n True Time (sec)",dataset_opt.features{mIdx}{i}{iFeature}.name))
                            yticks(1:10:(length(session_var_bins)-1));
                            yticklabels(fliplr(session_var_bins(1:10:end)))
                        else
                            yticks([])
                        end
                        % xlabel
                        if i_feature == numel(vis_features)
                            xlabel("Predicted Time (sec)")
                            xticks(1:10:(length(session_var_bins)-1));
                            xticklabels(session_var_bins(1:10:end))
                        else
                            xticks([])
                        end
                    end
                end
            end
        end
    end
end
%% 10) Last, visualize time until leave confusion mats pooled across mice 

pool_mice = [3 5]; 
pool_var_bins = var_bins{2}{4}{1};
colors = [lines(4);0 .9 .9 ; .2 .2 .2]; 
n_bins = length(pool_var_bins) - 1;
%  Do not include mouse 78 here
xMouse_goodConfusion_mats = good_confusion_mats(pool_mice);
xMouse_good_errors = good_errors(pool_mice);
xMouse_good_y_true = good_y_true(pool_mice);
close all

for i_feature = 1:numel(vis_features) 
    confusionmat_tmp = cell(numel(pool_mice),1);
    for m_ix = 1:numel(pool_mice)
        for i = 1:numel(xMouse_goodConfusion_mats{m_ix}) 
            if ~isempty(xMouse_goodConfusion_mats{m_ix}{i}{i_feature})
                confusionmat_tmp{m_ix}{i} = xMouse_goodConfusion_mats{m_ix}{i}{i_feature}(1:n_bins,1:n_bins);
            end
        end
    end 
    confusionmat_tmp(cellfun(@isempty,confusionmat_tmp)) = []; 
    confusionmat_cat = cellfun(@(x) cat(3,x{:}),confusionmat_tmp,'un',0);
    confusionmat_cat = cat(3,confusionmat_cat{:}); 
    confusionmat_pooled = sum(confusionmat_cat,3);
    
    figure(1)
    subplot(2,numel(vis_features),i_feature)
    imagesc(flipud(confusionmat_pooled)) 
    
    % ylabel
    if i_feature == 1
        ylabel("True Time Until Leave (sec)")
        yticks(1:10:(length(pool_var_bins)-1));
        yticklabels(fliplr(pool_var_bins(1:10:end)))
    else
        yticks([])
    end
    
    xlabel("Decoded T. Until Leave (sec)") 
    xticks(1:10:(length(pool_var_bins)-1));
    xticklabels(pool_var_bins(1:10:end))
    title(dataset_opt.features{5}{2}{i_feature}.name)
    set(gca,'fontsize',12)

    errors_tmp = cell(numel(pool_mice),1); 
    ytrue_tmp = cell(numel(pool_mice),1);  
    rmse_mean_mice = cell(numel(pool_mice),1); 
    for m_ix = 1:numel(pool_mice)  
        errors_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1); 
        ytrue_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1);
        for i = 1:numel(xMouse_goodConfusion_mats{m_ix}) 
            if ~isempty(xMouse_goodConfusion_mats{m_ix}{i}{i_feature})
                errors_tmp{m_ix}{i} = xMouse_good_errors{m_ix}{i}{i_feature};
                ytrue_tmp{m_ix}{i} = xMouse_good_y_true{m_ix}{i};
            end
        end
        % get mean in mouse to plot over overall avg 
        rmse_mean_mice{m_ix} = nan(numel(pool_var_bins)-1,1); 
        mouse_errors_full = cat(1,errors_tmp{m_ix}{:});
        mouse_ytrue_full = cat(1,ytrue_tmp{m_ix}{:});
        for i_time = 1:numel(pool_var_bins) 
            iTime = pool_var_bins(i_time);
            n = length(find((abs(mouse_ytrue_full - iTime) < .001)));
            rmse_mean_mice{m_ix}(i_time) = sqrt(nanmean(mouse_errors_full(abs(mouse_ytrue_full - iTime) < .01).^2));
        end 
    end 
    errors_cat = cellfun(@(x) cat(1,x{:}),errors_tmp,'un',0);
    ytrue_cat = cellfun(@(x) cat(1,x{:}),ytrue_tmp,'un',0);
    errors_full = cat(1,errors_cat{:}); 
    ytrue_full = cat(1,ytrue_cat{:}); 
    
    rmse_mean = nan(numel(pool_var_bins),1); 
    rmse_sem = nan(numel(pool_var_bins),1);
    for i_time = 1:numel(pool_var_bins) 
        iTime = pool_var_bins(i_time); 
        n = length(find((abs(ytrue_full - iTime) < .001)));
        rmse_mean(i_time) = sqrt(nanmean(errors_full(abs(ytrue_full - iTime) < .01).^2)); 
        rmse_sem(i_time) = 1.96*nanstd(errors_full(abs(ytrue_full - iTime) < .01).^2) / sqrt(n); 
    end
    subplot(2,numel(vis_features),i_feature + numel(vis_features))
    shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
    ylim([0 2]) 
    xlabel("True Time (sec)") 
    if i_feature == 1
        ylabel("RMSE (sec)")
    end 
    for m_ix = 1:numel(pool_mice) 
        plot(pool_var_bins,rmse_mean_mice{m_ix},'color',colors(i_feature,:))
    end
    set(gca,'fontsize',13) 
    
    % now add same to other figure 
    if ismember(i_feature,[1 2 4])
        figure(2) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names([1 2 4 5])) 
        title("Decoding Timecourse Across Sub-Populations") 
        ylabel("RMSE (sec)")
    end 
    
    if ismember(i_feature,[1 2 4]) || i_feature == 6
        figure(3) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names([1 2 4 6])) 
        title("Decoding Timecourse Compared To Velocity") 
        ylabel("RMSE (sec)")
    end
    
    if i_feature == 5 || i_feature == 6
        figure(4) 
        shadedErrorBar(pool_var_bins,rmse_mean,rmse_sem,'lineprops',{'linewidth',1.5,'color',colors(i_feature,:)}) ;hold on
        ylim([0 2]) 
        xlabel("True Time (sec)") 
        set(gca,'fontsize',13)  
        legend(feature_names(5:6)) 
        title("Decoding Timecourse Compared To Velocity")  
        ylabel("RMSE (sec)")
    end 
end 

%% 11) RMSE in 500 msec bins visualization, across features
% close all
pool_mice = [3 5]; 
pool_var_bins = var_bins{5}{1}{1};
n_bins = length(pool_var_bins) - 1;
%  Do not include mouse 78 here
xMouse_goodConfusion_mats = good_confusion_mats(pool_mice);
xMouse_good_errors = good_errors(pool_mice);
xMouse_good_y_true = good_y_true(pool_mice);
vis_features = [1 2 3 4 5 6];
bins = 0:.25:1; 
binlabels = arrayfun(@(x) sprintf("%.2f - %.2f Until Leave",bins(x),bins(x+1)),(1:length(bins)-1));
rmse_mean = nan(numel(vis_features),length(bins)-1);
rmse_sem = nan(numel(vis_features),length(bins)-1);

figure()
for i_feature = 1:numel(vis_features)  
    iFeature = vis_features(i_feature); 
    errors_tmp = cell(numel(pool_mice),1); 
    ytrue_tmp = cell(numel(pool_mice),1);  
    rmse_mean_mice = cell(numel(pool_mice),1); 
    for m_ix = 1:numel(pool_mice)  
        errors_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1); 
        ytrue_tmp{m_ix} = cell(numel(xMouse_goodConfusion_mats{m_ix}) ,1);
        for i = 1:numel(xMouse_goodConfusion_mats{m_ix}) 
            if ~isempty(xMouse_goodConfusion_mats{m_ix}{i}{i_feature})
                errors_tmp{m_ix}{i} = xMouse_good_errors{m_ix}{i}{i_feature};
                ytrue_tmp{m_ix}{i} = xMouse_good_y_true{m_ix}{i};
            end
        end
        % get mean in mouse to plot over overall avg 
        rmse_mean_mice{m_ix} = nan(numel(pool_var_bins)-1,1); 
        mouse_errors_full = cat(1,errors_tmp{m_ix}{:});
        mouse_ytrue_full = cat(1,ytrue_tmp{m_ix}{:});
        for i_time = 1:numel(pool_var_bins) 
            iTime = pool_var_bins(i_time);
            n = length(find((abs(mouse_ytrue_full - iTime) < .001)));
            rmse_mean_mice{m_ix}(i_time) = sqrt(nanmean(mouse_errors_full(abs(mouse_ytrue_full - iTime) < .01).^2));
        end 
    end 
    errors_cat = cellfun(@(x) cat(1,x{:}),errors_tmp,'un',0);
    ytrue_cat = cellfun(@(x) cat(1,x{:}),ytrue_tmp,'un',0);
    errors_full = cat(1,errors_cat{:}); 
    ytrue_full = cat(1,ytrue_cat{:}); 
    
    [~,~,ytrue_bins] = histcounts(ytrue_full,bins);
    
    for i_bin = 1:max(ytrue_bins)
        n = length(find((abs(ytrue_full - iTime) < .001)));
        rmse_mean(i_feature,i_bin) = sqrt(nanmean(errors_full(ytrue_bins == i_bin).^2)); 
        rmse_sem(i_feature,i_bin) = 1.96*nanstd(errors_full(ytrue_bins == i_bin).^2) / sqrt(n); 
    end
end

colororder([lines(4);0 .9 .9 ; .4 .4 .4])
b = bar(rmse_mean');
x = arrayfun(@(i_feature) b(i_feature).XEndPoints,1:numel(vis_features),'un',0);
x = cat(2,x{:});
hold on;
rmse_mean= rmse_mean'; 
rmse_sem = rmse_sem';
errorbar(x,rmse_mean(:),rmse_sem(:),'k.','linewidth',1.5)
legend(feature_names(vis_features))

xticks(b(1).XData)
xticklabels(binlabels)

ylabel("RMSE (sec)") 
ylim([0 2.5])
title("m78 and m80")
set(gca,'fontsize',13)

%% 12a) Preview of one-by-one adding: Time Since Reward
%  number of cells per feature vs MI / RMSE in terms of time since reward decoding fidelity
lines4 = lines(4); 
iRewsize = 4; % 4 uL trials
n_cells_threshold = 30; 
figure()
for iFeature = 1:4 % each GLM cluster
    rmse = nan(length(mPFC_sessions),1);
    mi = nan(length(mPFC_sessions),1);
    n_these_cells = nan(length(mPFC_sessions),1);
    counter = 1;
    for mIdx = 1:5
        for i = 1:numel(timeSince_hat{mIdx})
            n_bins = length(var_bins{mIdx}{iRewsize}{timeSince_ix});
            timeSince_hat_full = timeSince_hat{mIdx}{i}{iFeature}(rewsize{mIdx}{i} == iRewsize);
            timeSince_hat_full = var_dt * cat(1,timeSince_hat_full{:});
            if ~isempty(timeSince_hat_full)
                total_time = max(var_bins{mIdx}{iRewsize}{1});
                timeSince_true_full = y_true{mIdx}{i,timeSince_ix}(rewsize{mIdx}{i} == iRewsize);
                timeSince_true_full = var_dt * cat(2,timeSince_true_full{:})';
                rmse(counter) = sqrt(nanmean((timeSince_hat_full/total_time - timeSince_true_full/total_time).^2));
                mi(counter) = MI_confusionmat(confusionmat(timeSince_true_full,timeSince_hat_full)) / log(n_bins);
                n_these_cells(counter) = n_cells{mIdx}{i}(iFeature);
            end
            counter = counter + 1;
        end
    end
    subplot(1,2,1);hold on
    scatter(n_these_cells,mi,200,lines4(iFeature,:),'.')
    scatter(n_these_cells,mi,[],'k') 
    xlabel("# Cells")
    ylabel("Mutual Information (Normalized)")
    set(gca,'fontsize',14) 
%     xline(n_cells_threshold,'k--','linewidth',1.5)
    w = polyfit(log(n_these_cells(n_these_cells > 0)),mi(n_these_cells > 0),1);
    xl = xlim();
    p1(iFeature) = plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',lines4(iFeature,:),'linewidth',1.5);
    
    subplot(1,2,2);hold on
    scatter(n_these_cells,rmse,200,lines4(iFeature,:),'.')
    scatter(n_these_cells,rmse,[],'k')
    xlabel("# Cells") 
    ylabel("RMSE (Normalized)") 
    set(gca,'fontsize',14)  
%     xline(n_cells_threshold,'k--','linewidth',1.5)
    
    w = polyfit(log(n_these_cells(n_these_cells > 0)),rmse(n_these_cells > 0),1);
    xl = xlim();
    p2(iFeature) = plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',lines4(iFeature,:),'linewidth',1.5);
end 
subplot(1,2,1)
legend(p1,feature_names(1:4)) 
subplot(1,2,2)
legend(p2,feature_names(1:4)) 
suptitle("Time Since Reward Decoding Fidelity")

%% 12b) Preview of one-by-one adding: Time Until Leave

lines4 = lines(4); 
iRewsize = 4; % 4 uL trials
n_cells_threshold = 30; 
figure()
for iFeature = 1:4 % each GLM cluster
    rmse = nan(length(mPFC_sessions),1);
    mi = nan(length(mPFC_sessions),1);
    n_these_cells = nan(length(mPFC_sessions),1);
    counter = 1;
    for mIdx = 1:5
        for i = 1:numel(timeSince_hat{mIdx})
            n_bins = length(var_bins{mIdx}{iRewsize}{timeUntil_ix});
            total_time = max(var_bins{mIdx}{iRewsize}{1}); 
            timeUntil_hat_full = timeUntil_hat_postRew{mIdx}{i}{iFeature}(rewsize{mIdx}{i} == iRewsize);
            timeUntil_hat_full = var_dt * cat(1,timeUntil_hat_full{:});
            if ~isempty(timeUntil_hat_full)
                timeUntil_true_full = y_true{mIdx}{i,timeUntil_ix}(rewsize{mIdx}{i} == iRewsize);
                timeUntil_true_full = var_dt * cat(2,timeUntil_true_full{:})';
                rmse(counter) = sqrt(nanmean((timeUntil_hat_full/total_time - timeUntil_true_full/total_time).^2));
                mi(counter) = MI_confusionmat(confusionmat(timeUntil_true_full,timeUntil_hat_full)) / log(n_bins);
                n_these_cells(counter) = n_cells{mIdx}{i}(iFeature);
            end
            counter = counter + 1;
        end
    end
    subplot(1,2,1);hold on
    scatter(n_these_cells,mi,200,lines4(iFeature,:),'.')
    scatter(n_these_cells,mi,[],'k') 
    xlabel("# Cells")
    ylabel("Mutual Information (Normalized)")
    set(gca,'fontsize',14) 
%     xline(n_cells_threshold,'k--','linewidth',1.5)
    w = polyfit(log(n_these_cells(n_these_cells > 0)),mi(n_these_cells > 0),1);
    xl = xlim();
    p1(iFeature) = plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',lines4(iFeature,:),'linewidth',1.5);
    
    subplot(1,2,2);hold on
    scatter(n_these_cells,rmse,200,lines4(iFeature,:),'.')
    scatter(n_these_cells,rmse,[],'k')
    xlabel("# Cells") 
    ylabel("RMSE (Normalized)") 
    set(gca,'fontsize',14)  
%     xline(n_cells_threshold,'k--','linewidth',1.5)
    
    w = polyfit(log(n_these_cells(n_these_cells > 0)),rmse(n_these_cells > 0),1);
    xl = xlim();
    p2(iFeature) = plot(1:xl(2),w(1) * log(1:xl(2)) + w(2),'color',lines4(iFeature,:),'linewidth',1.5);
end 
subplot(1,2,1)
legend(p1,feature_names(1:4)) 
subplot(1,2,2)
legend(p2,feature_names(1:4)) 
suptitle("Time Until Leave Decoding Fidelity")
