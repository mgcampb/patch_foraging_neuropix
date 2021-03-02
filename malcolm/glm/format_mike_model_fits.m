% reformat mike's model fits into the form taken by my glm code
% MGC 2/4/21

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.mike_model_fits = 'C:\data\patch_foraging_behavior\mike_model_fits';
addpath(paths.mike_model_fits);

opt = struct;
opt.rew_size = [1 2 4];
opt.tbin = 0.02;

% all sessions to analyze:
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name}';
for pIdx = 1:numel(session_all)
    session_all{pIdx} = session_all{pIdx}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'));

%%
for sIdx = 1:numel(session_all)
    opt.session = session_all{sIdx};
    fprintf('session %d/%d: %s\n',sIdx,numel(session_all),opt.session);
    
    %% struct to hold decision variables
    DVs = struct;
    DVs.mod1 = cell(numel(opt.rew_size),1);
    DVs.mod2 = cell(numel(opt.rew_size),1);
    DVs.mod3 = cell(numel(opt.rew_size),1);
    DVs.modH = cell(numel(opt.rew_size),1);
    
    %% load data
    dat = load(fullfile(paths.data,opt.session)); 
    t = dat.velt; % time stamps for each bin
    rew_size_all = mod(dat.patches(:,2),10); % reward size of each patch

    %% get patch num for each patch
    
    patch_num = nan(size(t));
    for pIdx = 1:size(dat.patchCSL,1)
        % include one time bin before patch stop to catch the first reward
        patch_num(t>=(dat.patchCSL(pIdx,2)-opt.tbin) & t<=dat.patchCSL(pIdx,3)) = pIdx;
    end
    in_patch = ~isnan(patch_num);

    %% Get Mike's model fits for each reward size
    for rIdx = 1:numel(opt.rew_size)
        
        % load model fits for proper reward size
        if rIdx==1 
            mike_model_fits = load(fullfile(paths.mike_model_fits,'bmsResults12.mat'));
        elseif rIdx==2
            mike_model_fits = load(fullfile(paths.mike_model_fits,'bmsResults11.mat'));
        elseif rIdx==3
            mike_model_fits = load(fullfile(paths.mike_model_fits,'bmsResults10.mat'));
        end
        
        tmp1 = [rew_size_all; 0];
        tmp2 = patch_num; tmp2(isnan(tmp2))=numel(tmp1);
        in_patch_this = in_patch' & tmp1(tmp2)==opt.rew_size(rIdx); 

        % get reward size for each patch
        rew_size_patch = mod(dat.patches(:,2),10);

        % only keep large rewards for now
        keep = rew_size_patch == opt.rew_size(rIdx); % only large reward sizes for now
        patchCSL = dat.patchCSL(keep,:);

        % create vector of reward times for each patch
        RTs = cell(size(patchCSL,1),1);
        for pIdx = 1:size(patchCSL,1)
            keep_rew = dat.rew_ts>patchCSL(pIdx,1) & dat.rew_ts<=(patchCSL(pIdx,3)+0.5);
            RTs{pIdx} = round(dat.rew_ts(keep_rew)-patchCSL(pIdx,2));
        end

        % create inputs for Mike's patch_output functions
        mike_input_data = struct;
        mike_input_data.RTs = RTs;
        mike_input_data.PRT = patchCSL(:,3)-patchCSL(:,2); % patch residence times
        opt_mike = struct;
        opt_mike.delta_t = opt.tbin;
        mouse_num = str2num(opt.session(1:2));
        mouse_idx = find(mike_model_fits.miceGroup==mouse_num);
        
        % get decision variables (DVs) for all models
        DV_this = cell(4,1);
        t_mike = cell(4,1); % session time of mike's in-patch times
        
        % model 1
        theta = mike_model_fits.results(1).x(mouse_idx,:);
        [~, bins] = mod1_pF_output(theta, mike_input_data, opt_mike);
        DV_this{1} = [bins.x];
        x_cell = {bins.x};
        for pIdx = 1:numel(x_cell)
            t_mike{1} = [t_mike{1} (patchCSL(pIdx,2) + opt.tbin*(0:numel(x_cell{pIdx})-1))];
        end
        
        % model 2
        theta = mike_model_fits.results(2).x(mouse_idx,:);
        [~, bins] = mod2_pF_output(theta, mike_input_data, opt_mike);
        DV_this{2} = [bins.x];
        x_cell = {bins.x};
        for pIdx = 1:numel(x_cell)
            t_mike{2} = [t_mike{2} (patchCSL(pIdx,2) + opt.tbin*(0:numel(x_cell{pIdx})-1))];
        end
        
        % model 3
        theta = mike_model_fits.results(3).x(mouse_idx,:);
        [~, bins] = mod3_pF_output(theta, mike_input_data, opt_mike);
        DV_this{3} = [bins.x];
        x_cell = {bins.x};
        for pIdx = 1:numel(x_cell)
            t_mike{3} = [t_mike{3} (patchCSL(pIdx,2) + opt.tbin*(0:numel(x_cell{pIdx})-1))];
        end
        
        % model H
        theta = mike_model_fits.results(4).x(mouse_idx,:);
        [~, bins] = modH_pF_output(theta, mike_input_data, opt_mike);
        DV_this{4} = [bins.x];
        x_cell = {bins.x};
        for pIdx = 1:numel(x_cell)
            t_mike{4} = [t_mike{4} (patchCSL(pIdx,2) + opt.tbin*(0:numel(x_cell{pIdx})-1))];
        end

        for mIdx = 1:4
            DV_aligned = interp1(t_mike{mIdx},DV_this{mIdx},t);
            DV_aligned(isnan(DV_aligned)) = 0;
            DV_aligned(~in_patch_this) = 0;
            DV_this{mIdx} = DV_aligned;
        end
        
        DVs.mod1{rIdx} = DV_this{1};
        DVs.mod2{rIdx} = DV_this{2};
        DVs.mod3{rIdx} = DV_this{3};
        DVs.modH{rIdx} = DV_this{4};
                
    end
    
    % save re-formatted data
    save(fullfile(paths.mike_model_fits,'sessions',sprintf('%s_DVs',opt.session)),'DVs');

end