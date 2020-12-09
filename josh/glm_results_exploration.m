%% Get familiar with GLM results format 
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
paths.glm_results = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results'; 
paths.sig_cells = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/sig_cells/sig_cells_mb_cohort_PFC.mat';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath('/Users/joshstern/Documents/UchidaLab_NeuralData');

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = .5; % in seconds; only takes within patch times up to this amount before patch leave
opt.min_fr = 1; % minimum firing rate (on patch, excluding buffer) to keep neurons 
opt.cortex_only = true;
tbin_ms = opt.tbin*1000;
sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]}; 
mPFC_sessions = [1:8 10:13 15:18 23 25]; 

%% Load mPFC results 

load(paths.sig_cells); 

%% Plot RX PSTH per cell cluster  
% code from malcolm!
opt = struct;
opt.data_set = 'mb';
opt.brain_region = 'PFC';
opt.min_stop = -2;
opt.max_stop = 5;
opt.rew_size = [1 2 4];
opt.plot_col = cool(3);

opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1;

hfig = figure;
hfig.Name = sprintf('PSTH patch stop sig cells RR vs R0 %s cohort %s',opt.data_set,opt.brain_region);

clear meanRRa 
clear meanR0a
ctr = 1;
for clustIdx = 1:3

    psth_clust = psth_all(sig_cells.KMeansCluster==clustIdx);
   
    miny = 100;
    maxy = 0;

    for rIdx = 1:numel(opt.rew_size)
        rew_size = opt.rew_size(rIdx);
        meanRRa{rIdx}{clustIdx} = [];
        meanR0a{rIdx}{clustIdx} = [];
        t = opt.min_stop:0.01:opt.max_stop;
        for cIdx = 1:numel(psth_clust)
            keep_trial = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3)>-1;
            meanRRa{rIdx}{clustIdx} = [meanRRa{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(keep_trial,:),1)];

            keep_trial = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3)>-1;
            meanR0a{rIdx}{clustIdx} = [meanR0a{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(keep_trial,:),1)];
        end

        subplot(3,numel(opt.rew_size),ctr); hold on;
        shadedErrorBar(t,nanmean(meanRRa{rIdx}{clustIdx}),nanstd(meanRRa{rIdx}{clustIdx})/sqrt(size(meanRRa{rIdx}{clustIdx},1)),'lineprops',{'Color',opt.plot_col(rIdx,:)});
        shadedErrorBar(t,nanmean(meanR0a{rIdx}{clustIdx}),nanstd(meanR0a{rIdx}{clustIdx})/sqrt(size(meanR0a{rIdx}{clustIdx},1)),'lineprops','k');
        xlim([0 2]);
        miny = min(miny,min(ylim));
        maxy = max(maxy,max(ylim));
       
        title(sprintf('Cluster%d, %d%d vs %d0',clustIdx,rew_size,rew_size,rew_size));
       
        if rIdx==1
            ylabel('Firing rate');
        end
       
        ctr = ctr+1;
    end
   
    subplot(3,numel(opt.rew_size),ctr-3); ylim([miny maxy]);
    subplot(3,numel(opt.rew_size),ctr-2); ylim([miny maxy]);
    subplot(3,numel(opt.rew_size),ctr-1); ylim([miny maxy]);
end

subplot(3,numel(opt.rew_size),ctr-3); xlabel('Time from patch stop (s)');
subplot(3,numel(opt.rew_size),ctr-2); xlabel('Time from patch stop (s)');
subplot(3,numel(opt.rew_size),ctr-1); xlabel('Time from patch stop (s)');

%% RX peaksort PETH to get sense of structure of coding in significant cells 
%  sort based on peak of response in first 2 seconds of R0 activity 
clustIdx = 3; % which clusters to show in PETH  
clusters_name = "Cluster 3";
psth_clust = psth_all(ismember(sig_cells.KMeansCluster,clustIdx)); % which clusters to use
clear meanRRa 
clear meanR0a
for rIdx = 1:numel(opt.rew_size)
    rew_size = opt.rew_size(rIdx);
    meanRRa{rIdx} = [];
    meanR0a{rIdx} = [];
    t = opt.min_stop:0.01:opt.max_stop;
    for cIdx = 1:numel(psth_clust)
        r0trials = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3) == 0;
        if isempty(find(isnan(nanmean(psth_clust{cIdx}.psth_stop(r0trials,t > 0 & t < 3),1)),1))
            meanR0a{rIdx} = [meanR0a{rIdx}; nanmean(psth_clust{cIdx}.psth_stop(r0trials,t > 0 & t < 3),1)];
        end
        
        rrtrials = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3) == 0;
        if isempty(find(isnan(nanmean(psth_clust{cIdx}.psth_stop(rrtrials,t > 0 & t < 3),1)),1))
            meanRRa{rIdx} = [meanRRa{rIdx}; nanmean(psth_clust{cIdx}.psth_stop(rrtrials,t > 0 & t < 3),1)];  
        end
    end
    
    % this is the step where we don't average anymore; instead peaksort PETH 
    [~,peak_ix_r0] = max(meanR0a{rIdx},[],2); 
    [~,peak_sort_r0] = sort(peak_ix_r0); 
    subplot(2,numel(opt.rew_size),rIdx); 
    imagesc(flipud(zscore(meanR0a{rIdx}(peak_sort_r0,:),[],2))) 
    title(sprintf("%s %i0",clusters_name,rew_size))
    subplot(2,numel(opt.rew_size),rIdx+3);
    imagesc(flipud(zscore(meanRRa{rIdx}(peak_sort_r0,:),[],2))) 
    title(sprintf("%s %i%i",clusters_name,rew_size,rew_size)) 
    
    ctr = ctr+1;
end 

%% RXX trials 
% code from malcolm!
opt = struct;
opt.data_set = 'mb';
opt.brain_region = 'PFC';
opt.min_stop = -2;
opt.max_stop = 5;
opt.rew_size = [1 2 4]; 
opt.plot_col = cat(3,[zeros(4,1) linspace(0,1,4)' linspace(0,1,4)'], ... 
                     [linspace(0,.5,4)' linspace(0,.5,4)' linspace(0,1,4)'], ... 
                     [linspace(0,1,4)' zeros(4,1) linspace(0,1,4)']);

opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1;

hfig = figure;
hfig.Name = sprintf('PSTH patch stop sig cells RR vs R0 %s cohort %s',opt.data_set,opt.brain_region);

ctr = 1;
for clustIdx = 1:3

    psth_clust = psth_all(sig_cells.KMeansCluster==clustIdx);
   
    miny = 100;
    maxy = 0;

    for rIdx = 1:numel(opt.rew_size)
        rew_size = opt.rew_size(rIdx);
        meanR00a{rIdx}{clustIdx} = [];
        meanRR0a{rIdx}{clustIdx} = [];
        meanR0Ra{rIdx}{clustIdx} = [];
        meanRRRa{rIdx}{clustIdx} = [];
        t = opt.min_stop:0.01:opt.max_stop;
        for cIdx = 1:numel(psth_clust)
            trialsR00x = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3)==0 & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsRR0x = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3)==0 & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsR0Rx = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3)==rew_size & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsRRRx = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3)==rew_size & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            
            meanR00a{rIdx}{clustIdx} = [meanR00a{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsR00x,:),1)];
            meanRR0a{rIdx}{clustIdx} = [meanRR0a{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsRR0x,:),1)];
            meanR0Ra{rIdx}{clustIdx} = [meanR0Ra{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsR0Rx,:),1)];
            meanRRRa{rIdx}{clustIdx} = [meanRRRa{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsRRRx,:),1)];
        end

        subplot(3,numel(opt.rew_size),ctr); hold on;
        shadedErrorBar(t,nanmean(meanR00a{rIdx}{clustIdx}),nanstd(meanR00a{rIdx}{clustIdx})/sqrt(size(meanR00a{rIdx}{clustIdx},1)),'lineprops',{'Color',opt.plot_col(1,:,rIdx)});
        shadedErrorBar(t,nanmean(meanRR0a{rIdx}{clustIdx}),nanstd(meanRR0a{rIdx}{clustIdx})/sqrt(size(meanRR0a{rIdx}{clustIdx},1)),'lineprops',{'Color',opt.plot_col(2,:,rIdx)});
        shadedErrorBar(t,nanmean(meanR0Ra{rIdx}{clustIdx}),nanstd(meanR0Ra{rIdx}{clustIdx})/sqrt(size(meanR0Ra{rIdx}{clustIdx},1)),'lineprops',{'Color',opt.plot_col(3,:,rIdx)});
        shadedErrorBar(t,nanmean(meanRRRa{rIdx}{clustIdx}),nanstd(meanRRRa{rIdx}{clustIdx})/sqrt(size(meanRRRa{rIdx}{clustIdx},1)),'lineprops',{'Color',opt.plot_col(4,:,rIdx)});

        xlim([0 3]);
        miny = min(miny,min(ylim));
        maxy = max(maxy,max(ylim));
       
        title(sprintf('Cluster%d, %d%d vs %d0',clustIdx,rew_size,rew_size,rew_size));
       
        if rIdx==1
            ylabel('Firing rate');
        end
       
        ctr = ctr+1;
    end
   
    subplot(3,numel(opt.rew_size),ctr-3); ylim([miny maxy]);
    subplot(3,numel(opt.rew_size),ctr-2); ylim([miny maxy]);
    subplot(3,numel(opt.rew_size),ctr-1); ylim([miny maxy]);
end

subplot(3,numel(opt.rew_size),ctr-3); xlabel('Time from patch stop (s)');
subplot(3,numel(opt.rew_size),ctr-2); xlabel('Time from patch stop (s)');
subplot(3,numel(opt.rew_size),ctr-1); xlabel('Time from patch stop (s)');


%% Cluster 3 vs derivative of cluster 2 neurons 

opt = struct;
opt.data_set = 'mb';
opt.brain_region = 'PFC';
opt.min_stop = -2;
opt.max_stop = 5;
opt.rew_size = [2 4]; 
opt.plot_col = cool(3);

opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1;

ctr = 1;

for rIdx = 1:numel(opt.rew_size) 
    rew_size = opt.rew_size(rIdx);
    hfig = figure;
    hfig.Name = sprintf('PSTH patch stop sig cells %iXX %s cohort %s',rew_size,opt.data_set,opt.brain_region);
    
    
    miny = 100;
    maxy = 0; 
    
    psth_clust2 = psth_all(sig_cells.KMeansCluster==2);  
    
    meanR00_clust2 = [];
    meanRR0_clust2 = [];
    meanR0R_clust2 = [];
    meanRRR_clust2 = [];
    t = opt.min_stop:0.01:opt.max_stop;
    for cIdx = 1:numel(psth_clust2)
        trialsR00x = psth_clust2{cIdx}.rew_barcode(:,1)==rew_size & psth_clust2{cIdx}.rew_barcode(:,2)==0 & psth_clust2{cIdx}.rew_barcode(:,3)==0 & psth_clust2{cIdx}.rew_barcode(:,4) > -1;
        trialsRR0x = psth_clust2{cIdx}.rew_barcode(:,1)==rew_size & psth_clust2{cIdx}.rew_barcode(:,2)==rew_size & psth_clust2{cIdx}.rew_barcode(:,3)==0 & psth_clust2{cIdx}.rew_barcode(:,4) > -1;
        trialsR0Rx = psth_clust2{cIdx}.rew_barcode(:,1)==rew_size & psth_clust2{cIdx}.rew_barcode(:,2)==0 & psth_clust2{cIdx}.rew_barcode(:,3)==rew_size & psth_clust2{cIdx}.rew_barcode(:,4) > -1;
        trialsRRRx = psth_clust2{cIdx}.rew_barcode(:,1)==rew_size & psth_clust2{cIdx}.rew_barcode(:,2)==rew_size & psth_clust2{cIdx}.rew_barcode(:,3)==rew_size & psth_clust2{cIdx}.rew_barcode(:,4) > -1;
        
        meanR00_clust2 = [meanR00_clust2; nanmean(psth_clust2{cIdx}.psth_stop(trialsR00x,:),1)];
        meanRR0_clust2 = [meanRR0_clust2; nanmean(psth_clust2{cIdx}.psth_stop(trialsRR0x,:),1)];
        meanR0R_clust2 = [meanR0R_clust2; nanmean(psth_clust2{cIdx}.psth_stop(trialsR0Rx,:),1)];
        meanRRR_clust2 = [meanRRR_clust2; nanmean(psth_clust2{cIdx}.psth_stop(trialsRRRx,:),1)];
    end  
    meanR00_clust2 = gauss_smoothing(nanmean(meanR00_clust2),opt.smoothSigma_time / opt.tbin); 
    meanRR0_clust2 = gauss_smoothing(nanmean(meanRR0_clust2),opt.smoothSigma_time / opt.tbin); 
    meanR0R_clust2 = gauss_smoothing(nanmean(meanR0R_clust2),opt.smoothSigma_time / opt.tbin); 
    meanRRR_clust2 = gauss_smoothing(nanmean(meanRRR_clust2),opt.smoothSigma_time / opt.tbin); 
    
    meanR00_clust2Grad = zscore(gradient(meanR00_clust2));  
%     meanR00_clust2Grad_norm = meanR00_clust2Grad - min(meanR00_clust2Grad(200:500)); 
%     meanR00_clust2Grad_norm = meanR00_clust2Grad_norm ./ max(meanR00_clust2Grad_norm(200:500));
    meanRR0_clust2Grad = zscore(gradient(meanRR0_clust2));  
%     meanRR0_clust2Grad_norm = meanRR0_clust2Grad - min(meanRR0_clust2Grad(200:500)); 
%     meanRR0_clust2Grad_norm = meanRR0_clust2Grad_norm ./ max(meanRR0_clust2Grad_norm(200:500));
    meanR0R_clust2Grad = zscore(gradient(meanR0R_clust2));  
%     meanR0R_clust2Grad_norm = meanR0R_clust2Grad - min(meanR0R_clust2Grad(200:500)); 
%     meanR0R_clust2Grad_norm = meanR0R_clust2Grad_norm ./ max(meanR0R_clust2Grad_norm(200:500));
    meanRRR_clust2Grad = zscore(gradient(meanRRR_clust2));  
%     meanRRR_clust2Grad_norm = meanRRR_clust2Grad - min(meanRRR_clust2Grad(200:500)); 
%     meanRRR_clust2Grad_norm = meanRRR_clust2Grad_norm ./ max(meanRRR_clust2Grad_norm(200:500));
    
    % Plot avgs
    subplot(3,4,5); hold on;
    plot(t,meanR00_clust2,'linewidth',1.5)
    subplot(3,4,6); hold on;
    plot(t,meanRR0_clust2,'linewidth',1.5)
    subplot(3,4,7); hold on;
    plot(t,meanR0R_clust2,'linewidth',1.5)
    subplot(3,4,8); hold on;
    plot(t,meanRRR_clust2,'linewidth',1.5)

    % compare other two clusters to derivative of cluster 2
    for clustIdx = [1 3]
        psth_clust = psth_all(sig_cells.KMeansCluster==clustIdx);
        
        meanR00a{rIdx}{clustIdx} = [];
        meanRR0a{rIdx}{clustIdx} = [];
        meanR0Ra{rIdx}{clustIdx} = [];
        meanRRRa{rIdx}{clustIdx} = [];
        for cIdx = 1:numel(psth_clust)
            trialsR00x = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3)==0 & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsRR0x = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3)==0 & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsR0Rx = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==0 & psth_clust{cIdx}.rew_barcode(:,3)==rew_size & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            trialsRRRx = psth_clust{cIdx}.rew_barcode(:,1)==rew_size & psth_clust{cIdx}.rew_barcode(:,2)==rew_size & psth_clust{cIdx}.rew_barcode(:,3)==rew_size & psth_clust{cIdx}.rew_barcode(:,4) > -1;
            
            meanR00a{rIdx}{clustIdx} = [meanR00a{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsR00x,:),1)];
            meanRR0a{rIdx}{clustIdx} = [meanRR0a{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsRR0x,:),1)];
            meanR0Ra{rIdx}{clustIdx} = [meanR0Ra{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsR0Rx,:),1)];
            meanRRRa{rIdx}{clustIdx} = [meanRRRa{rIdx}{clustIdx}; nanmean(psth_clust{cIdx}.psth_stop(trialsRRRx,:),1)];
        end 
        
%         plot vs derivative of ramping cluster  
        meanR00a_norm = zscore(nanmean(meanR00a{rIdx}{clustIdx})); %  - min(nanmean(meanR00a{rIdx}{clustIdx}(200:500)));  
%         meanR00a_norm = meanR00a_norm ./ max(meanR00a_norm(200:500)); 
        meanRR0a_norm = zscore(nanmean(meanRR0a{rIdx}{clustIdx})); % - min(nanmean(meanRR0a{rIdx}{clustIdx}));  
%         meanRR0a_norm = meanRR0a_norm ./ max(meanRR0a_norm(200:500)); 
        meanR0Ra_norm = zscore(nanmean(meanR0Ra{rIdx}{clustIdx})); % - min(nanmean(meanR0Ra{rIdx}{clustIdx}));  
%         meanR0Ra_norm = meanR0Ra_norm ./ max(meanR0Ra_norm(200:500)); 
        meanRRRa_norm = zscore(nanmean(meanRRRa{rIdx}{clustIdx})); %  - min(nanmean(meanRRRa{rIdx}{clustIdx}));  
%         meanRRRa_norm = meanRRRa_norm ./ max(meanRRRa_norm(200:500)); 
        
        if clustIdx == 1 
            sign = -1; 
        else
            sign = 1; 
        end
        
        subplot(3,4,1 + 4 * (clustIdx - 1)); hold on;
        plot(t,meanR00a_norm,'linewidth',1.5) 
        plot(t,sign*meanR00_clust2Grad,'linewidth',1.5)
        subplot(3,4,2 + 4 * (clustIdx - 1)); hold on;
        plot(t,meanRR0a_norm,'linewidth',1.5)
        plot(t,sign*meanRR0_clust2Grad,'-','linewidth',1.5)
        subplot(3,4,3 + 4 * (clustIdx - 1)); hold on;
        plot(t,meanR0Ra_norm,'linewidth',1.5)
        plot(t,sign*meanR0R_clust2Grad,'-','linewidth',1.5)
        subplot(3,4,4 + 4 * (clustIdx - 1)); hold on;
        plot(t,meanRRRa_norm,'linewidth',1.5)
        plot(t,sign*meanRRR_clust2Grad,'-','linewidth',1.5)
        
        subplot(3,4,5); hold on;
        plot(t,nanmean(meanR00a{rIdx}{clustIdx}),'linewidth',1.5)
        subplot(3,4,6); hold on;
        plot(t,nanmean(meanRR0a{rIdx}{clustIdx}),'linewidth',1.5)
        subplot(3,4,7); hold on;
        plot(t,nanmean(meanR0Ra{rIdx}{clustIdx}),'linewidth',1.5)
        subplot(3,4,8); hold on;
        plot(t,nanmean(meanRRRa{rIdx}{clustIdx}),'linewidth',1.5)
        
%         xlim([-1 3]);
%         miny = min(miny,min(ylim));
%         maxy = max(maxy,max(ylim));
%        
%         title(sprintf('Cluster%d, %d%d vs %d0',clustIdx,rew_size,rew_size,rew_size));
%        
%         if rIdx==1
%             ylabel('Firing rate');
%         end
%
%         ctr = ctr+1;
    end
    
    %     subplot(3,numel(opt.rew_size),ctr-3); ylim([miny maxy]);
    %     subplot(3,numel(opt.rew_size),ctr-2); ylim([miny maxy]);
    %     subplot(3,numel(opt.rew_size),ctr-1); ylim([miny maxy]);
    for sp = 1:12
        subplot(3,4,sp)
        xlim([0 4])
    end
end

% subplot(3,numel(opt.rew_size),ctr-3); xlabel('Time from patch stop (s)');
% subplot(3,numel(opt.rew_size),ctr-2); xlabel('Time from patch stop (s)');
% subplot(3,numel(opt.rew_size),ctr-1); xlabel('Time from patch stop (s)');

%% Practice grabbing the cellIDs from a session for creeping stuff  
num_cells = nan(numel(mPFC_sessions),1);  
sig_cell_sessions = sig_cells.Session;
for i = 1:18
    sIdx = mPFC_sessions(i);  
    sCellIDs = sig_cells.CellID(strcmp(sig_cell_sessions,sessions{sIdx}(1:end-4))); 
    num_cells(i) = numel(sCellIDs);  
    fprintf("%s Sig Cells: %i \n",sessions{sIdx}(1:end-4),numel(sCellIDs))
end