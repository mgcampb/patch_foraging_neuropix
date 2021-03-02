addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = 'C:\data\patch_foraging_neuropix\mua_psth\';
if ~isfolder(paths.results)
    mkdir(paths.results);
end
paths.figs = 'C:\figs\patch_foraging_neuropix\psth_mua';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';

% PSTH windows
opt.min_stop = -2;
opt.max_stop = 5;
opt.min_leave = -5;
opt.max_leave = 2;

%% get sessions

session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'));

%%
psth_all = {};
counter = 1;
for sIdx = 1:numel(session_all)
    
    fprintf('Session %d/%d: %s\n',sIdx,numel(session_all),session_all{sIdx});
    
    dat = load(fullfile(paths.data,session_all{sIdx})); 
    
    if ~isfield(dat,'brain_region_rough')
        continue
    end
    
    good_cells = dat.sp.cids(dat.sp.cgs==2);
    good_cells = good_cells(strcmp(dat.brain_region_rough,opt.brain_region));
    
    % combine all spikes to get mua
    spiket = dat.sp.st(ismember(dat.sp.clu,good_cells));
    spiket_ms = spiket*1000;
    
    % make reward barcode matrix
    nTimesteps = 15;
    patchstop_sec = dat.patchCSL(:,2);
    patchleave_sec = dat.patchCSL(:,3);
    prts = patchleave_sec - patchstop_sec;
    floor_prts = floor(prts);
    rewsize = mod(dat.patches(:,2),10);
    rew_barcode = zeros(length(dat.patchCSL), nTimesteps);
    for iTrial = 1:size(dat.patchCSL,1)
        rew_indices = round(dat.rew_ts(dat.rew_ts >= patchstop_sec(iTrial) & dat.rew_ts < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        rew_barcode(iTrial, (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial, rew_indices) = rewsize(iTrial);
    end 
    
    % behavioral events to align PSTHs to
    patchcue_ms = dat.patchCSL(:,1)*1000;
    patchstop_ms = dat.patchCSL(:,2)*1000;
    patchleave_ms = dat.patchCSL(:,3)*1000;
    
    % create psth's using Hyunggoo's plot_timecourse

    % what to compute
    psth_this = struct;
    psth_this.session = session_all{sIdx};
    psth_this.rew_barcode = rew_barcode;    

    % patch stop
    t_align = patchstop_ms;
    t_start = max(patchcue_ms,patchstop_ms+opt.min_stop*1000); 
    t_end = min(patchleave_ms,patchstop_ms+opt.max_stop*1000);
    [~,~,z]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
    % pad with NaNs to make all PSTHs the same size
    rate_rsp = z.rate_rsp/numel(good_cells);
    if min(z.x)>opt.min_stop
        numpad = (min(z.x)-opt.min_stop)*100;
        rate_rsp = [nan(size(rate_rsp,1),round(numpad)) rate_rsp];
    end
    if max(z.x)<opt.min_stop
        numpad = (opt.max_stop-max(z.x))*100;
        rate_rsp = [rate_rsp nan(size(rate_rsp,1),round(numpad))];
    end
    psth_this.psth_stop = rate_rsp;
    psth_this.t_stop = opt.min_stop:0.01:opt.max_stop;

    % patch leave
    t_align = patchleave_ms;
    t_start = max(patchstop_ms,patchleave_ms+opt.min_leave*1000);
    t_end = patchleave_ms+opt.max_leave*1000;
    [~,~,z,t]=plot_timecourse('timestamp',spiket_ms,t_align,t_start,t_end);
    % pad with NaNs to make all PSTHs the same size
    rate_rsp = z.rate_rsp/numel(good_cells);
    if min(z.x)>opt.min_leave
        numpad = (min(z.x)-opt.min_leave)*100;
        rate_rsp = [nan(size(rate_rsp,1),round(numpad)) rate_rsp];
    end
    if max(z.x)<opt.min_leave
        numpad = (opt.max_leave-max(z.x))*100;
        rate_rsp = [rate_rsp nan(size(rate_rsp,1),round(numpad))];
    end
    psth_this.psth_leave = rate_rsp;
    psth_this.t_leave = opt.min_leave:0.01:opt.max_leave;    

    psth_all{counter} = psth_this;
    counter = counter+1;

end

save(fullfile(paths.results,sprintf('mua_psth_%s_cohort_%s',opt.data_set,opt.brain_region)),'psth_all');

%% avg psth around patch stop: RX trials

opt.rew_size = [1 2 4];
plot_col = cool(3);

hfig = figure;
hfig.Name = sprintf('PSTH patch stop MUA RR vs R0 %s cohort %s',opt.data_set,opt.brain_region);


miny = 100;
maxy = 0;

for rIdx = 1:numel(opt.rew_size)
    rew_size = opt.rew_size(rIdx);
    meanRR = [];
    meanR0 = [];
    t = opt.min_stop:0.01:opt.max_stop;
    for sIdx = 1:numel(psth_all)
        keep_trial = psth_all{sIdx}.rew_barcode(:,1)==rew_size & psth_all{sIdx}.rew_barcode(:,2)==rew_size & psth_all{sIdx}.rew_barcode(:,3)>-1;
        meanRR = [meanRR; nanmean(psth_all{sIdx}.psth_stop(keep_trial,:),1)];

        keep_trial = psth_all{sIdx}.rew_barcode(:,1)==rew_size & psth_all{sIdx}.rew_barcode(:,2)==0 & psth_all{sIdx}.rew_barcode(:,3)>-1;
        meanR0 = [meanR0; nanmean(psth_all{sIdx}.psth_stop(keep_trial,:),1)];
    end

    subplot(1,numel(opt.rew_size),rIdx); hold on;
    shadedErrorBar(t,nanmean(meanRR,1),nanstd(meanRR,[],1)/sqrt(size(meanRR,1)),'lineprops',{'Color',plot_col(rIdx,:)});
    shadedErrorBar(t,nanmean(meanR0,1),nanstd(meanR0,[],1)/sqrt(size(meanR0,1)),'lineprops','k');
    xlim([0 2]);
    miny = min(miny,min(ylim));
    maxy = max(maxy,max(ylim));

    title(sprintf('%d%d vs %d0',rew_size,rew_size,rew_size));

    if rIdx==1
        ylabel('Firing rate');
    end

end

subplot(1,numel(opt.rew_size),1); ylim([miny maxy]);
subplot(1,numel(opt.rew_size),2); ylim([miny maxy]);
subplot(1,numel(opt.rew_size),3); ylim([miny maxy]);


subplot(1,numel(opt.rew_size),1); xlabel('Time from patch stop (s)');
subplot(1,numel(opt.rew_size),2); xlabel('Time from patch stop (s)');
subplot(1,numel(opt.rew_size),3); xlabel('Time from patch stop (s)');

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

% %% avg psth around patch stop: RXX trials
% 
% hfig = figure;
% hfig.Name = sprintf('PSTH patch stop sig cells RXX %s cohort %s',opt.data_set,opt.brain_region);
% opt.rew_size = [1 2 4];
% plot_col = cool(3);
% 
% ctr = 1;
% for clustIdx = 1:3
% 
%     psth_clust = psth_all(sig_cells.KMeansCluster==clustIdx);
%     
%     miny = 100;
%     maxy = 0;
% 
%     for rIdx = 1:numel(opt.rew_size)
%         rew_size = opt.rew_size(rIdx);
%         meanR00 = [];
%         meanRR0 = [];
%         meanR0R = [];
%         meanRRR = [];
%         t = opt.min_stop:0.01:opt.max_stop;
%         for sIdx = 1:numel(psth_clust)
%             
%             barcode = psth_clust{sIdx}.rew_barcode;
%             
%             keep_trial = barcode(:,1)==rew_size & barcode(:,2)==0 & barcode(:,3)==0 & barcode(:,4)>-1;
%             meanR00 = [meanR00; nanmean(psth_clust{sIdx}.psth_stop(keep_trial,:),1)];
% 
%             keep_trial = barcode(:,1)==rew_size & barcode(:,2)==rew_size & barcode(:,3)==0 & barcode(:,4)>-1;
%             meanRR0 = [meanRR0; nanmean(psth_clust{sIdx}.psth_stop(keep_trial,:),1)];
%             
%             keep_trial = barcode(:,1)==rew_size & barcode(:,2)==0 & barcode(:,3)==rew_size & barcode(:,4)>-1;
%             meanR0R = [meanR0R; nanmean(psth_clust{sIdx}.psth_stop(keep_trial,:),1)];
%             
%             keep_trial = barcode(:,1)==rew_size & barcode(:,2)==rew_size & barcode(:,3)==rew_size & barcode(:,4)>-1;
%             meanRRR = [meanRRR; nanmean(psth_clust{sIdx}.psth_stop(keep_trial,:),1)];
%         end
% 
%         subplot(3,numel(opt.rew_size),ctr); hold on;
%         shadedErrorBar(t,nanmean(meanRRR),nanstd(meanRRR)/sqrt(size(meanRRR,1)),'lineprops',{'Color',plot_col(rIdx,:)});
%         shadedErrorBar(t,nanmean(meanRR0),nanstd(meanRR0)/sqrt(size(meanRR0,1)),'lineprops',{'Color',plot_col(rIdx,:)*0.7});
%         shadedErrorBar(t,nanmean(meanR0R),nanstd(meanR0R)/sqrt(size(meanR0R,1)),'lineprops',{'Color',plot_col(rIdx,:)*0.3});
%         shadedErrorBar(t,nanmean(meanR00),nanstd(meanR00)/sqrt(size(meanR00,1)),'lineprops','k');
%         xlim([0 3]);
%         xticks(0:3);
%         miny = min(miny,min(ylim));
%         maxy = max(maxy,max(ylim));
%         
%         title(sprintf('Cluster%d, %duL RXX',clustIdx,rew_size));
%         
%         if rIdx==1
%             ylabel('Firing rate');
%         end
%         
%         ctr = ctr+1;
%     end
%     
%     subplot(3,numel(opt.rew_size),ctr-3); ylim([miny maxy]);
%     subplot(3,numel(opt.rew_size),ctr-2); ylim([miny maxy]);
%     subplot(3,numel(opt.rew_size),ctr-1); ylim([miny maxy]);
% end
% 
% subplot(3,numel(opt.rew_size),ctr-3); xlabel('Time from patch stop (s)');
% subplot(3,numel(opt.rew_size),ctr-2); xlabel('Time from patch stop (s)');
% subplot(3,numel(opt.rew_size),ctr-1); xlabel('Time from patch stop (s)');
% 
% saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
% 
% %% avg psth around patch leave
% 
% hfig = figure('Position',[200 200 300 600]);
% hfig.Name = sprintf('PSTH patch leave sig cells %s cohort %s',opt.data_set,opt.brain_region);
% opt.rew_size = [1 2 4];
% plot_col = cool(3);
% 
% for clustIdx = 1:3
% 
%     psth_clust = psth_all(sig_cells.KMeansCluster==clustIdx);
%     
% 
%     subplot(3,1,clustIdx); hold on;
%     
%     for rIdx = 1:numel(opt.rew_size)
%         rew_size = opt.rew_size(rIdx);
%         meanFR = [];
%         t = opt.min_leave:0.01:opt.max_leave;
%         for sIdx = 1:numel(psth_clust)
%             keep_trial = psth_clust{sIdx}.rew_barcode(:,1)==rew_size;
%             meanFR = [meanFR; nanmean(psth_clust{sIdx}.psth_leave(keep_trial,:),1)];
%         end   
%         
%         shadedErrorBar(t,nanmean(meanFR),nanstd(meanFR)/sqrt(size(meanFR,1)),'lineprops',{'Color',plot_col(rIdx,:)});
% 
%     end
%     xlim([-3 1]);
%     plot([0 0],ylim(),'k:');
%     title(sprintf('Cluster%d',clustIdx));
%     ylabel('Firing rate');
%     if clustIdx==1
%         legend({'1uL','2uL','4uL'});
%     end
%     
% end
% xlabel('Time from patch leave (s)');
% 
% saveas(hfig,fullfile(paths.figs,hfig.Name),'png');