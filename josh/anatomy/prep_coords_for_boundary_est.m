%% Prepare coordinates for distance to boundary estimation in python
%  Go through and save anatomy3d tables as csv files that can be read by
%  pandas

data_path = 'D:\patchforaging_data\processed_neural_data'; % where our data is 
save_path = 'D:\patchforaging_data\anatomy3d'; % where we want to save csv files

session_all = dir(fullfile(data_path,'*.mat'));
session_all = {session_all.name}';

%% load data
for i_session = 1:numel(session_all)
    dat = load(fullfile(data_path,session_all{i_session}));
    if isfield(dat,'anatomy3d')
        fprintf('Reformatting anatomy3d data for Session %d/%d: %s\n',i_session,numel(session_all),session_all{i_session});
        csv_fname = [session_all{i_session}(1:end-4) '_anatomy3dCoords.csv']; 
        writetable(dat.anatomy3d.Coords, fullfile(save_path,csv_fname))
    else
        fprintf('Didnt find anatomy3d data for Session %d/%d: %s\n',i_session,numel(session_all),session_all{i_session});
    end
end