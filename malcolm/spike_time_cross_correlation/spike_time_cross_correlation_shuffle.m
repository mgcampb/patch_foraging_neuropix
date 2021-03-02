paths = struct;

paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation';
if ~isfolder(paths.results)
    mkdir(paths.results);
end
paths.figs = 'C:\figs\patch_foraging_neuropix\spike_time_cross_correlation';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.num_shuf = 1000;
session = '80_20200317';

%%
dat = load(fullfile(paths.data,session));

%%
good_cells = dat.sp.cids(dat.sp.cgs==2);

%%

minT = 0;
maxT = max(dat.sp.st);
binsize = 0.001;
num_lags = 200;
N_pairs = numel(good_cells)*(numel(good_cells)-1)/2;

xcorr_shuf = nan(opt.num_shuf,num_lags*2+1);

pairs_idx = combntns(1:numel(good_cells),2);

tic

pair_idx_this = 74675;
    
fprintf('computing xcorr for pair %d/%d\n',pair_idx_this,N_pairs);

cell1 = good_cells(pairs_idx(pair_idx_this,1));
cell2 = good_cells(pairs_idx(pair_idx_this,2));

st1 = dat.sp.st(dat.sp.clu==cell1);
st2 = dat.sp.st(dat.sp.clu==cell2);

st1_bin = histcounts(st1,minT:binsize:maxT);
st2_bin = histcounts(st2,minT:binsize:maxT);

st_xcorr = xcorr(st1_bin,st2_bin,num_lags);
xcorr_this = st_xcorr;

tic
pb = ParforProgressbar(opt.num_shuf);
parfor shuf_idx = 1:opt.num_shuf
    st1_shuf = mod(st1+unifrnd(20,maxT-20),maxT);
    st1_shuf_bin = histcounts(st1_shuf,minT:binsize:maxT);
    xcorr_shuf(shuf_idx,:) = xcorr(st1_shuf_bin,st2_bin,num_lags);
    pb.increment();
end
toc

%%
hfig = figure('Position',[200 200 1000 400]); hold on;
hfig.Name = sprintf('%s c%d vs c%d with shuffle',session,cell1,cell2);
stairs(-num_lags:num_lags,xcorr_this,'b');
% plot(-num_lags:num_lags,prctile(xcorr_shuf,1),'r');
plot(-num_lags:num_lags,prctile(xcorr_shuf,99),'r');
plot([0 0],ylim,'k--');
ylim([0 max(ylim)]);
legend({'xcorr','99% shuffle'});
xlabel('lag (ms)');
ylabel('xcorr (un-normalized)');
title(sprintf('%s, c%d vs c%d',session,cell1,cell2),'Interpreter','none');
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');