paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\IPI_analysis';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));
addpath(genpath('C:\code\HGK_analysis_tools'));

opt = struct;
opt.tbin = 0.02;
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
opt.patch_leave_buffer = 0.5;

opt.min_fr = 1;

opt.session = '78_20200312';

opt.pc = 1;


% % all sessions to analyze:
% session_all = dir(fullfile(paths.data,'*.mat'));
% session_all = {session_all.name}';
% for i = 1:numel(session_all)
%     session_all{i} = session_all{i}(1:end-4);
% end




%%

dat = load(fullfile(paths.data,opt.session));
good_cells_all = dat.sp.cids(dat.sp.cgs==2);

%% time bins
opt.tstart = 0;
opt.tend = max(dat.velt);
tbinedge = dat.velt;
tbincent = tbinedge(1:end-1)+opt.tbin/2;


%% extract in-patch times
in_patch = false(size(tbincent));
in_patch_buff = false(size(tbincent)); % add buffer for pca
for i = 1:size(dat.patchCSL,1)
    in_patch(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)) = true;
    in_patch_buff(tbincent>=dat.patchCSL(i,2) & tbincent<=dat.patchCSL(i,3)-opt.patch_leave_buffer) = true;
end

%% remove cells that don't pass minimum firing rate cutoff

% compute binned spikecounts for each cell  
t = dat.velt;
spikecounts_whole_session = nan(numel(t),numel(good_cells_all));
for cIdx = 1:numel(good_cells_all)
    spike_t = dat.sp.st(dat.sp.clu==good_cells_all(cIdx));
    spikecounts_whole_session(:,cIdx) = histc(spike_t,t);
end

% filter spikecounts to only include in patch times (excluding buffer)
spikecounts = spikecounts_whole_session(in_patch_buff,:);

% apply firing rate cutoff
T = size(spikecounts,1)*opt.tbin;
N = sum(spikecounts);
fr = N/T;
good_cells = good_cells_all(fr>opt.min_fr);


%% compute PCA

% compute firing rate mat
fr_mat = calcFRVsTime(good_cells,dat,opt);

% take zscore
fr_mat_zscore = zscore(fr_mat,[],2); % z-score is across whole session including out-of-patch times - is this weird??

% pca on firing rate matrix, only in patches with buffer before patch leave
coeffs = pca(fr_mat_zscore(:,in_patch_buff)');

% project full session onto these PCs
score = coeffs'*fr_mat_zscore;

%%

vel = dat.vel(1:end-1);
vel_out_patch = vel;
vel_out_patch(in_patch)=nan;
v_thresh = vel_out_patch<0.2;

% find contiguous times when velocity drops below threshold
flag = 0;
start_idx = [];
end_idx = [];
counter = 1;
for idx = 1:numel(v_thresh)
    if v_thresh(idx) && ~flag
        start_idx(counter) = idx;
        flag = 1;
    elseif ~v_thresh(idx) && flag
        end_idx(counter) = idx;
        flag = 0;
        counter = counter+1;
    end
end
if numel(end_idx)<numel(start_idx)
    end_idx = [end_idx start_idx(end)];
end
        
stop_length = (end_idx-start_idx)*opt.tbin;


%%

keep = stop_length>1.5;
t_align = start_idx(stop_length>1.5);
t_start = start_idx(stop_length>1.5)-1000/20;
t_end = start_idx(stop_length>1.5)+5000/20;

hfig = figure('Position',[200 200 1700 400]);
hfig.Name = sprintf('Stops during IPI %s PC%d',opt.session,opt.pc);

subplot(1,4,1);
[p1,~,z1,t1] = plot_timecourse('stream',vel_out_patch,t_align',t_start',t_end',[],'resample_bin',1);
p1(2).XTickLabel = -1:5;
p1(2).XLabel.String = 'Time from stop (sec)';
p1(2).YLabel.String = 'Speed (a.u.)';
title('Velocity')
subplot(1,4,2);
[p2,~,z2,t2] = plot_timecourse('stream',score(opt.pc,:),t_align',t_start',t_end',[],'resample_bin',1);
p2(2).XTickLabel = -1:5;
p2(2).XLabel.String = 'Time from stop (sec)';
p2(2).YLabel.String = 'PC val';
title(sprintf('On-patch PC%d',opt.pc));

% subplot(1,3,3);
% [p3,~,z3,t3] = plot_timecourse('stream',fr_mat(good_cells==368,:),t_align',t_start',t_end',[],'resample_bin',1);
% p3(2).XTickLabel = -1:5;
% p3(2).XLabel.String = 'Time from stop (sec)';
% p3(2).YLabel.String = 'Firing rate';
% title('m80 0317, Cell 368')

subplot(1,4,3);
num_lags = 50;
t_xcorr = opt.tbin*(-num_lags:num_lags);
xcorr_all = nan(size(z1.rate_rsp,1),num_lags*2+1);
for i = 1:size(xcorr_all,1)
    x1 = z1.rate_rsp(i,:)-mean(z1.rate_rsp(i,:));
    x2 = z2.rate_rsp(i,:)-mean(z2.rate_rsp(i,:));
    xcorr_all(i,:) = xcorr(x2,x1,num_lags,'coeff');
end
plot(t_xcorr,nanmean(xcorr_all));
grid on
xlabel('lag (seconds)');
ylabel('xcorr');
title('around spontaneous stops');

% overall correlation across session
subplot(1,4,4);
xcorr_whole_sesh = xcorr(score(opt.pc,:)-mean(score(opt.pc,:)),vel-mean(vel),num_lags,'coeff');
plot(t_xcorr,xcorr_whole_sesh);
grid on
xlabel('lag (seconds)');
ylabel('xcorr');
title('whole session');

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

%%

% figure; hold on;
% plot(tbincent,score(1,:),'k.');
% plot(tbincent(in_patch),score(1,in_patch),'r.');