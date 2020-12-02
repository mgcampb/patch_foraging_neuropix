% script to add "cortex" label to cells based on histology

data_dir = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data\';
sessions = dir(sprintf('%s*.mat',data_dir));
sessions = {sessions.name}';
sessions = sessions(~contains(sessions,'mc'));
sessions = sessions(~contains(sessions,'79_202003')); % these didn't target PFC

brain_region_all = {};
for i = 1:numel(sessions)
    clear anatomy
    load(fullfile(data_dir,sessions{i}),'anatomy')
    if exist('anatomy','var')
        brain_region_all = unique([brain_region_all; anatomy.brain_region']);
    end
end

%%
keep = contains(brain_region_all,'ACA') | ...
    contains(brain_region_all,'ILA') | ...
    contains(brain_region_all,'ORB') | ...
    contains(brain_region_all,'MOs') | ...
    contains(brain_region_all,'PL');
cortex = brain_region_all(keep);
non_cortex = brain_region_all(~keep);

%%
for i = numel(sessions)
    clear anatomy
    fprintf('session %d/%d: %s\n',i,numel(sessions),sessions{i});
    load(fullfile(data_dir,sessions{i}),'anatomy')
    if exist('anatomy','var')
        cell_labels_new = anatomy.cell_labels;
        cell_labels_new.Cortex = ismember(cell_labels_new.BrainRegion,cortex);
        anatomy.cell_labels = cell_labels_new;
        save(fullfile(data_dir,sessions{i}),'anatomy','-append');
    end
end