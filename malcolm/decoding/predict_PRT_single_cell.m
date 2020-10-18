%% script to plot neural activity (PC1 and example neuron)
% for large reward patches with just one reward at t=0
% to see if there's a correlation with leave times
% MGC 6/23/2020

% ** TO DO: add firing rate cutoff ** 

paths = struct;
paths.data = 'C:\Users\malcg\Dropbox (Personal)\UchidaLab\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\decoding\single_cell'; % where to save figs

addpath(genpath('C:\code\HGRK_analysis_tools'));
addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

% analysis options
opt = struct;
opt.session = '80_20200317'; % session to analyze
opt.cellid = [368 426]; % example cells to plot
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

% for concatenating patches
opt.patch_type = 'R-nil'; % options: 'R-nil' or 'RR-nil'
opt.rew_size = 1;

% ADD: firing rate cutoff (opt.min_fr)
paths.figs = fullfile(paths.figs,opt.session,sprintf('%duL',opt.rew_size),opt.patch_type);

%% load data
dat = load(fullfile(paths.data,opt.session));
good_cells = dat.sp.cids(dat.sp.cgs==2);

% time bins
opt.tstart = 0;
opt.tend = max(dat.sp.st);
tbinedge = opt.tstart:opt.tbin:opt.tend;
tbincent = tbinedge(1:end-1)+opt.tbin/2;

%% compute firing rate for selected cell
fr_mat = calcFRVsTime(opt.cellid,dat,opt);

%% extract in-patch times
in_patch = false(size(tbincent));
for i = 1:size(dat.patchCSL,1)
    in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
end
fr_mat_in_patch = fr_mat(:,in_patch);
tbincent = tbincent(in_patch);

%% extract patches of the correct type

% get patch num for each patch
patch_num = nan(size(tbincent));
for i = 1:size(dat.patchCSL,1)
    patch_num(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = i;
end

% reward sequence in each patch
max_prt = ceil(max(dat.patches(:,5)))+1; % add 1 sec buffer
rew_seq = nan(size(dat.patchCSL,1),max_prt+1); % add 1 for zero space
rew_bin_edges = -0.5:1:max_prt+0.5;
for i = 1:size(rew_seq,1)
    rewt_this = dat.rew_ts(dat.rew_ts>dat.patchCSL(i,1) & dat.rew_ts<dat.patchCSL(i,3)+0.5);
    rewt_this = rewt_this-dat.patchCSL(i,2);
    rew_seq(i,:) = histcounts(rewt_this,rew_bin_edges);
end

if strcmp(opt.patch_type,'R-nil')
    patches_to_analyze = find(sum(rew_seq,2)==1 & mod(dat.patches(:,2),10)==opt.rew_size);
elseif strcmp(opt.patch_type,'RR-nil')
    patches_to_analyze = find(sum(rew_seq,2)==2 & rew_seq(:,1)==1 & rew_seq(:,2)==1 & mod(dat.patches(:,2),10)==opt.rew_size);
end
prt = dat.patches(patches_to_analyze,5);

%% plot example cell sfor these patches

% get firing rate trace for this cell for these patches
T = ceil(max(prt));
t = 0:opt.tbin:T;
N = numel(t);

for cIdx = 1:numel(opt.cellid)
    fr_selected_patches = nan(numel(patches_to_analyze),N);
    for i = 1:numel(patches_to_analyze)
        fr_this = fr_mat_in_patch(cIdx,patch_num==patches_to_analyze(i));
        fr_selected_patches(i,1:numel(fr_this)) = fr_this;
    end

    % sort by prt
    [~,sort_idx] = sort(prt);
    fr_selected_patches_sort = fr_selected_patches(sort_idx,:);

    % plot
    hfig = figure('Position',[300 300 600 500]); hold on;
    hfig.Name = sprintf('%s_c%d_%duL_%s',opt.session,opt.cellid(cIdx),opt.rew_size,opt.patch_type);
    plot_col = winter(numel(patches_to_analyze));
    for i = 1:numel(patches_to_analyze)
        y_this = fr_selected_patches_sort(i,:);
        lh = plot(t,y_this);
        lh.Color = [plot_col(i,:) 0.8];
        max_idx = find(~isnan(y_this),1,'last');
        my_scatter(t(max_idx),y_this(max_idx),plot_col(i,:),0.5);
    end
    xlabel('Time on patch (sec)');
    ylabel('Firing rate (Hz)')
    title(sprintf('%s\nCELL %d, %d uL patches, %s',opt.session,opt.cellid(cIdx),opt.rew_size,opt.patch_type),'Interpreter','none')
    plot([min(prt) min(prt)],ylim,'k--');

    % plot significance
    for i = 1:size(fr_selected_patches,2)
        keep = ~isnan(fr_selected_patches(:,i));
        y_this = fr_selected_patches(keep,i);
        prt_this = prt(keep);
        if numel(y_this)>1
            [r,p] = corrcoef(y_this,prt_this);
            if p(1,2)<0.05
                my_scatter(t(i),max(ylim)-2,'k',1);
            end
        end
    end

    % plot rewards
    if strcmp(opt.patch_type,'R-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
    elseif strcmp(opt.patch_type,'RR-nil')
        plot(0,max(ylim),'bv','MarkerFaceColor','b');
        plot(1,max(ylim),'bv','MarkerFaceColor','b');
    end

    save_figs(paths.figs,hfig,'png');
end
