paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation';
if ~isfolder(paths.results)
    mkdir(paths.results);
end

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
xcorr_all = nan(N_pairs,num_lags*2+1);

pairs_idx = combntns(1:numel(good_cells),2);

tic
pb = ParforProgressbar(N_pairs);
parfor i = 1:N_pairs
    
    fprintf('computing xcorr for pair %d/%d\n',i,N_pairs);
    
    cell1 = good_cells(pairs_idx(i,1));
    cell2 = good_cells(pairs_idx(i,2));
    
    st1 = dat.sp.st(dat.sp.clu==cell1);
    st2 = dat.sp.st(dat.sp.clu==cell2);

    st1_bin = histcounts(st1,minT:binsize:maxT);
    st2_bin = histcounts(st2,minT:binsize:maxT);

    st_xcorr = xcorr(st1_bin,st2_bin,num_lags);
    st_xcorr(num_lags+1)=0;
    xcorr_all(i,:) = st_xcorr;
    
    pb.increment();

end
toc

%%
save(fullfile(paths.results,session),'xcorr_all','pairs_idx','session','good_cells');