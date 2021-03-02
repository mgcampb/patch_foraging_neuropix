% script to get PSTH around patch stop and patch leave for sig cells from GLM analysis
% 11/17/2020 MGC

% cluster on these PSTHs directly (disregarding kmeans label)
% 12/3/2020 MGC

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.sig_cells = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201114_all_sessions_model_comparison\sig_cells';
paths.figs = 'C:\figs\patch_foraging_neuropix\psth_sig_cells';
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

% for firing rate matrix
opt.tbin = 0.001; % 1 ms timebins for easier compatibility with HyungGoo's code
opt.smoothSigma_time = 0.1;

%% load sig_cells

load(fullfile(paths.sig_cells,sprintf('sig_cells_%s_cohort_%s',opt.data_set,opt.brain_region)));
session_all = unique(sig_cells.Session);

%% make psth with z-scored firing rate
psth_all_zscore = {};
counter = 1;
for sIdx = 1:numel(session_all)
    
    fprintf('Session %d/%d: %s\n',sIdx,numel(session_all),session_all{sIdx});
    
    dat = load(fullfile(paths.data,session_all{sIdx}));   
    good_cells = sig_cells.CellID(strcmp(sig_cells.Session,session_all{sIdx}));
    
    % create z-scored firing rate matrix    
    opt.tstart = 0;
    opt.tend = max(dat.velt);
    frMat = calcFRVsTime(good_cells,dat,opt);
    frMat = zscore(frMat,[],2);
    
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
    for cIdx = 1:numel(good_cells)
        fprintf('\tcell %d/%d...\n',cIdx,numel(good_cells));
        
        % what to compute
        psth_this = struct;
        psth_this.session = session_all{sIdx};
        psth_this.cid = good_cells(cIdx);
        psth_this.rew_barcode = rew_barcode;
        
        % convert spike times to ms
        fr_this = frMat(cIdx,:);
        
        % patch stop
        t_align = patchstop_ms;
        t_start = max(patchcue_ms,patchstop_ms+opt.min_stop*1000); 
        t_end = min(patchleave_ms,patchstop_ms+opt.max_stop*1000);
        [~,~,z]=plot_timecourse('stream',fr_this,t_align,t_start,t_end);
        % pad with NaNs to make all PSTHs the same size
        rate_rsp = z.rate_rsp;
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
        [~,~,z]=plot_timecourse('stream',fr_this,t_align,t_start,t_end);
        % pad with NaNs to make all PSTHs the same size
        rate_rsp = z.rate_rsp;
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
        
        psth_all_zscore{counter} = psth_this;
        counter = counter+1;
        
    end

end

save(fullfile(paths.sig_cells,sprintf('sig_cells_%s_cohort_%s',opt.data_set,opt.brain_region)),'psth_all_zscore','-append');

%% avg psth around patch stop: RX trials

opt.rew_size = [1 2 4];
plot_col = cool(3);

psth_concat = [];
for rIdx = 1:numel(opt.rew_size)
    rew_size = opt.rew_size(rIdx);
    meanRR = [];
    meanR0 = [];
    t = opt.min_stop:0.01:opt.max_stop;
    for cIdx = 1:numel(psth_all_zscore)
        keep_trial = psth_all_zscore{cIdx}.rew_barcode(:,1)==rew_size & psth_all_zscore{cIdx}.rew_barcode(:,2)==rew_size & psth_all_zscore{cIdx}.rew_barcode(:,3)>-1;
        meanRR = [meanRR; nanmean(psth_all_zscore{cIdx}.psth_stop(keep_trial,:),1)];

        keep_trial = psth_all_zscore{cIdx}.rew_barcode(:,1)==rew_size & psth_all_zscore{cIdx}.rew_barcode(:,2)==0 & psth_all_zscore{cIdx}.rew_barcode(:,3)>-1;
        meanR0 = [meanR0; nanmean(psth_all_zscore{cIdx}.psth_stop(keep_trial,:),1)];
    end
    psth_concat = [psth_concat meanRR(:,201:400) meanR0(:,201:400)];
end
    

%% cluster psth's

keep = all(~isnan(psth_concat),2);
sig_cells_filt = sig_cells(keep,:);
psth_concat_filt = psth_concat(keep,:);

psth_concat_filt_zscore = zscore(psth_concat_filt,[],2);

[coeff,score,~,~,expl] = pca(psth_concat_filt_zscore);


num_clust = 3;

kidx = kmeans(score(:,1:4),num_clust);

figure; hold on;
t = 0.01:0.01:12;
plot_col = cool(3);
for i = 1:num_clust
    subplot(num_clust,1,i); hold on;
    plot_idx = 1:200;
    for j = 1:6
        plot(t(plot_idx),mean(psth_concat_filt_zscore(kidx==i,plot_idx)),'Color',plot_col(floor((j-1)/2)+1,:));
        plot_idx = plot_idx + 200;
    end
    title(sprintf('PSTH Clust%d',i));
    ylabel('z-scored activity');
end

tab = crosstab(kidx,sig_cells_filt.KMeansCluster);
figure;
bar3(tab);
xticklabels({'GLM Clust1','GLM Clust2','GLM Clust3'});
yticklabels({'PSTH Clust1','PSTH Clust2'});