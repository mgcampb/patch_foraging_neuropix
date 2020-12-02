data_dir = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data\';
sessions = dir(sprintf('%s*.mat',data_dir));
sessions = {sessions.name}';
sessions = sessions(~contains(sessions,'mc'));
sessions = sessions(~contains(sessions,'79_202003')); % these didn't target PFC

figure; hold on;
scale_factor_all = nan(numel(sessions),1);
for i = 1:numel(sessions)
    clear anatomy
    fprintf('session %d/%d: %s\n',i,numel(sessions),sessions{i});
    load(fullfile(data_dir,sessions{i}),'anatomy')
    if exist('anatomy','var')
        plot(anatomy.scale_factor,anatomy.probe_tip/anatomy.insertion_depth,'ko');
        scale_factor_all(i) = anatomy.scale_factor;
    end 
end
refline(1,0);

figure;
histogram(scale_factor_all);