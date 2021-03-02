paths.results = 'C:\data\patch_foraging_neuropix\spike_time_cross_correlation';
paths.figs = 'C:\figs\patch_foraging_neuropix\spike_time_cross_correlation';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

baseline = mean(xcorr_all(:,[1:100 end-100:end]),2);

ex_pair = [];
for lag = 2:10
    ex_pair = [ex_pair; find(xcorr_all(:,201+lag)>(8*baseline) & baseline>0.5);...
        find(xcorr_all(:,201-lag)>(8*baseline) & baseline>0.5)];
    ex_pair = unique(ex_pair);
end

hfig = figure('Position',[200 200 1400 1000]);
hfig.Name = sprintf('%s',session);
for i = 1:numel(ex_pair)
    cell1 = good_cells(pairs_idx(ex_pair(i),1));
    cell2 = good_cells(pairs_idx(ex_pair(i),2));
    subplot(3,3,i); hold on;
    stairs(-200:200,xcorr_all(ex_pair(i),:));
    ylim([0 max(ylim)]);
    xlabel('lag (ms)');
    ylabel('xcorr (unscaled)');
    title(sprintf('%s, c%d vs c%d',session,cell1,cell2),'Interpreter','none');
end
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');