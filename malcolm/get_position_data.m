% script to add position data to processed data files
% MGC 11/5/2020

%% load data

paths = struct;
paths.data_np = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.data_behav = 'Z:\Malcolm\patch_foraging_behavior_data\';

sessions = dir(fullfile(paths.data_np,'*.mat'));
sessions = {sessions.name}';
sessions = sessions(contains(sessions,'mc'));

sessions = {'mc2_20201021.mat'};

%% 
for sesh_idx = 1:numel(sessions)
    
    fprintf('session %d/%d: %s\n',sesh_idx,numel(sessions),sessions{sesh_idx});
    
    % load data
    if contains(sessions{sesh_idx},'mc')
        mouse = sessions{sesh_idx}(3);
        date = sessions{sesh_idx}(5:end-4);
        behav_dir = fullfile(paths.data_behav,mouse,date);
        behav_file = dir(fullfile(behav_dir,sprintf('EventPosVel%s.mat',date(5:end))));
        behav_file = behav_file.name;
    else        
        mouse = sessions{sesh_idx}(1:2);
        date = sessions{sesh_idx}(8:end-4);
        behav_dir = fullfile(paths.data_behav,mouse);
        behav_file = dir(fullfile(behav_dir,sprintf('EventPosVel%s.mat',date)));
        behav_file = behav_file.name;
    end    

    np_dat = load(fullfile(paths.data_np,sessions{sesh_idx}));
    behav_dat = load(fullfile(behav_dir,behav_file));
    
    % get position in patch

    [~,patchCue_idx_np] = min(abs(np_dat.patchCSL(:,1)-np_dat.velt),[],2);
    [~,patchStop_idx_np] = min(abs(np_dat.patchCSL(:,2)-np_dat.velt),[],2);
    [~,patchLeave_idx_np] = min(abs(np_dat.patchCSL(:,3)-np_dat.velt+0.5),[],2); % 0.5 seconds after patch leave

    nonzero_idx = find(behav_dat.now_eventPosVel(:,2)~=0);
    idx_and_val = [nonzero_idx behav_dat.now_eventPosVel(nonzero_idx,2)];
    idx1 = find(idx_and_val(1:end-1,2)>700 & idx_and_val(2:end,2)>0); % cue
    idx2 = idx1+1; % stop
    idx3 = find(idx_and_val(2:end,2)==-4 & idx_and_val(1:end-1,2)<700)+1; % 0.5 seconds after leave
    patchCue_idx_virmen = idx_and_val(idx1,1);
    patchStop_idx_virmen = idx_and_val(idx2,1);
    patchLeave_idx_virmen = idx_and_val(idx3,1);

    patch_pos = nan(size(np_dat.vel));

    for i = 1:numel(patchCue_idx_np)
        idx_np = patchStop_idx_np(i):patchLeave_idx_np(i);
        idx_virmen = patchStop_idx_virmen(i):patchLeave_idx_virmen(i);
        t_virmen = linspace(np_dat.velt(idx_np(1)),np_dat.velt(idx_np(end)),numel(idx_virmen));
        patch_pos(idx_np) = interp1(t_virmen,behav_dat.now_eventPosVel(idx_virmen,3),np_dat.velt(idx_np));
    end
    
    save(fullfile(paths.data_np,sessions{sesh_idx}),'patch_pos','-append');
end