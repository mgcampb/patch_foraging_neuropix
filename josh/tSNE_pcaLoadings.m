%% Upon Paul Masset's suggestion, performing tSNE in PC Loading space
% Assumes use of Malcolm Campbell's exploratory_analysis.m to create PSTH
% struct in data path, which in turn uses HGK's plot_psth function

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_matlab/neuroPixelsData/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_matlab/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_matlab'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

% load PSTH struct from data path
data = load(fullfile(paths.data,'/PSTH/psth.mat'));
psth = data.psth;

%% PCA on PSTHs
reset_figs;

loadings = {};
n_PCs = 2; % number of PCs to use in clustering

for sIdx = 1:numel(sessions)
    session = sessions{sIdx}(1:end-4);
    fprintf('PCA on PSTHs for session %d/%d: %s...\n',sIdx,numel(sessions),session);
    
    psth_by_alignment = {psth{sIdx}.psth_cue,...
        psth{sIdx}.psth_stop,...
        psth{sIdx}.psth_leave,...
        psth{sIdx}.psth_rew};
    t_window_by_alignment = {psth{sIdx}.t_window_cue,...
        psth{sIdx}.t_window_stop,...
        psth{sIdx}.t_window_leave,...
        psth{sIdx}.t_window_rew};
    psth_label = {'cue','stop','leave','rew'};
    
    % initialize array
    loadings{sIdx} = zeros(size(psth{sIdx}.psth_cue,1),length(psth_label) * n_PCs);
    
    for aIdx = 1:numel(psth_label) % different alignments (cue, stop, leave, rew)
        fig_counter = fig_counter+1;
        hfig(fig_counter) = figure('Position',[200 200 800 700]);
        hfig(fig_counter).Name = sprintf('%s - pca on psths - %s',session,psth_label{aIdx});

        % get psth for this alignment, and exclude nans
        psth_this = psth_by_alignment{aIdx};
        t_this = t_window_by_alignment{aIdx};
        t_this = t_this(1:10:end);
        t_this = t_this(~all(isnan(psth_this)));
        psth_this = psth_this(:,~all(isnan(psth_this)));
        
        % z score      
        psth_zscore = my_zscore(psth_this);

        % pca
        [coeffs,score,~,~,expl] = pca(psth_zscore);

        % eigenvalue plot
        subplot(2,2,1); hold on; set_fig_prefs;
        my_scatter(1:numel(expl),expl,'r',0.3);
        my_scatter(1:numel(expl),cumsum(expl),'b',0.3);
        ylim([0 100]);
        legend({'Indiv.','Cum.'});
        xlabel('Num. PCs');
        ylabel('% explained variance');
        xlim([-10 size(coeffs,2)+10]);
        title(sprintf('SESSION: %s',session),'Interpreter','none');

        % top two principal components
        subplot(2,2,2); hold on; set_fig_prefs;
        plot(t_this,coeffs(:,1));
        plot(t_this,coeffs(:,2));    
        xlabel(sprintf('Time relative to %s (sec)',psth_label{aIdx}));
        ylabel('PC coefficient');
        legend({'PC1','PC2'},'Location','northwest');
        plot([0 0],ylim,'k:','HandleVisibility','off');

        % cells projected into top two principal components
        subplot(2,2,3); hold on; set_fig_prefs;
        my_scatter(score(:,1),score(:,2),'k',0.3);
        xlabel('PC1 loading');
        ylabel('PC2 loading');
        title(sprintf('%d single units',size(psth_zscore,1)));
        axis equal;
        plot([0 0],ylim,'k:');
        plot(xlim,[0 0],'k:');

        % histogram of component scores
        subplot(2,2,4); hold on; set_fig_prefs;
        [ks1,x1] = ksdensity(score(:,1),'Bandwidth',3);
        [ks2,x2] = ksdensity(score(:,2),'Bandwidth',3);
        plot(x1,ks1);
        plot(x2,ks2);
        plot([0 0],ylim,'k:','HandleVisibility','off');
        legend({'PC1','PC2'},'location','northwest');
        xlabel('PC loading');
        ylabel('Density');
        
        % add loadings to rows of matrix to cluster 
        loadings{sIdx}(:,(aIdx-1)*n_PCs + 1:(aIdx-1)*n_PCs + n_PCs) = score(:,1:n_PCs);
        
    end
end
save_figs(fullfile(paths.figs,'pca_on_psths'),hfig,'png');

%% Now cluster the 8-dimensional space using t-SNE
reset_figs;
for sIdx = 1:numel(sessions)
    session = erase(sessions{sIdx}(1:end-4),'_');
    % perform t-SNE on the PC loadings
    figure()
    reducedLoadings = tsne(loadings{sIdx});
    scatter(reducedLoadings(:,1),reducedLoadings(:,2))
    title(sprintf('%s t-SNE Clustering of PC Space',session))
end
