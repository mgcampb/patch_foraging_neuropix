% plot PSTHs of behavioral variables on R0 and RR trials by reward size for
% each session, mouse, and then combined across all

% MGC 5/28/2021

paths = struct;
paths.data = 'G:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.session_list = 'C:\data\patch_foraging_neuropix\GLM_output\20210514_accel';
paths.malcolm_functions = 'C:\code\patch_foraging_neuropix\malcolm\functions';
addpath(genpath(paths.malcolm_functions));
paths.hgrk = 'C:\code\HGRK_analysis_tools';
addpath(genpath(paths.hgrk));
paths.figs = 'C:\figs\patch_foraging_neuropix\behavior\PSTH_RR_R0\GLM_sessions';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.rew_size = [1 2 4];
opt.tbin = 0.02;
opt.smooth_sigma_vel = 0.1;
opt.smooth_sigma_lickrate = 0.1;

%% load sessions
session_all = dir(fullfile(paths.session_list,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
session_all = session_all(~contains(session_all,'mc'))';

%% Generate PSTHs
z_all = cell(numel(session_all),1);
for sIdx = 1:numel(session_all)

    fprintf('Session %d/%d: %s\n',sIdx,numel(session_all),session_all{sIdx});
    
    session = session_all{sIdx};
    dat = load(fullfile(paths.data,session));

    % reinitialize patch timing vectors
    patchstop_sec = dat.patchCSL(:,2);
    patchleave_sec = dat.patchCSL(:,3);
    rew_ms = dat.rew_ts;

    % Trial level features
    patchCSL = dat.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(dat.patches(:,2),10);

    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_sec(iTrial) & rew_ms < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end 
    
    % prepare behavioral variables to plot:
    % position
    pos_ms = interp1(dat.velt,dat.patch_pos,0:0.001:max(dat.velt)); 
    % speed
    speed_ms = interp1(dat.velt,dat.vel,0:0.001:max(dat.velt)); 
    % acceleration
    vel_smooth = gauss_smoothing(dat.vel,opt.smooth_sigma_vel/opt.tbin);
    accel = diff(vel_smooth)/opt.tbin;
    accel = [accel accel(end)];
    accel_ms = interp1(dat.velt,accel,0:0.001:max(dat.velt)); 
    % lick rate
    lickcounts = histc(dat.lick_ts,dat.velt)/opt.tbin;
    lickrate = gauss_smoothing(lickcounts,opt.smooth_sigma_lickrate/opt.tbin);
    lickrate_ms = interp1(dat.velt,lickrate,0:0.001:max(dat.velt)); 
    
    % gather into a list
    behav_var = cell(4,1);
    behav_var{1} = pos_ms;
    behav_var{2} = speed_ms;
    behav_var{3} = accel_ms;
    behav_var{4} = lickrate_ms;
    % var names:
    behav_var_name = cell(4,1);
    behav_var_name{1} = 'Position';
    behav_var_name{2} = 'Speed';
    behav_var_name{3} = 'Acceleration';
    behav_var_name{4} = 'Lick Rate';
    
    hfig = figure('Position',[200 200 600 800]);
    hfig.Name = sprintf('PSTH position speed %s',session_all{sIdx});  
    ylims = nan(numel(behav_var),2);
    clims = nan(numel(behav_var),2);
    ax = cell(numel(behav_var),numel(opt.rew_size),2);
    z = cell(numel(behav_var),numel(opt.rew_size));
    
    for rIdx = 1:numel(opt.rew_size)
        % select trials
        keep = rew_barcode(:,1)==opt.rew_size(rIdx) & rew_barcode(:,2)>=0;
        rew_barcode_filt = rew_barcode(keep,:);
        
        % group into RR and R0 trials
        RR = rew_barcode_filt(:,2)>0;
        
        % align time points (ms)
        t_align = 1000*patchCSL(keep,2);
        t_start = t_align;
        t_end = t_align+2000;
        
        % plot each behavioral variable
        for vIdx = 1:numel(behav_var)
            subplot(numel(behav_var), numel(opt.rew_size), rIdx + (vIdx-1)*numel(opt.rew_size));
            
            [ax_this,~,z_this] = plot_timecourse('stream',behav_var{vIdx},t_align,t_start,t_end,RR);
            ax{vIdx,rIdx,1} = ax_this(1);
            ax{vIdx,rIdx,2} = ax_this(2);
            z{vIdx,rIdx} = z_this;
            ax_this(2).Legend.Visible = 'off';
            if vIdx==1
                ax_this(1).Title.String = sprintf('%duL\n%s',opt.rew_size(rIdx),behav_var_name{vIdx});
            else
                ax_this(1).Title.String = behav_var_name{vIdx};
            end
            if rIdx>1
                ax_this(1).YLabel.String = '';
            end
            if vIdx==numel(behav_var)
                ax_this(2).XLabel.String = 'Time from patch stop (sec)';
            end
            clims(vIdx,1) = min(clims(vIdx,1),min(ax_this(1).CLim));
            clims(vIdx,2) = max(clims(vIdx,2),max(ax_this(1).CLim));
            ylims(vIdx,1) = min(ylims(vIdx,2),min(ax_this(2).YLim));
            ylims(vIdx,2) = max(ylims(vIdx,2),max(ax_this(2).YLim));
        end
    end
    
    % set color scale and ylims to be consistent across plots
    for rIdx = 1:numel(opt.rew_size)
        for vIdx = 1:4
            ax{vIdx,rIdx,1}.CLim = clims(vIdx,:);
            ax{vIdx,rIdx,2}.YLim = ylims(vIdx,:);
        end
    end

    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
    close(hfig);
    
    z_all{sIdx} = z;

end

%% average PSTH over sessions

t = z_all{1}{1,1}.x;
mean_all = cell(numel(behav_var),numel(opt.rew_size));
for i = 1:numel(behav_var)
    for j = 1:numel(opt.rew_size)
        mean_this = nan(numel(session_all),2,numel(t));
        for k = 1:numel(session_all)
            mean_this(k,:,:) = z_all{k}{i,j}.mean;
        end
        mean_all{i,j} = mean_this;
    end
end

% make plot
hfig = figure('Position',[200 200 600 800]);
hfig.Name = sprintf('PSTH position speed averaged over sessions'); 
for rIdx = 1:numel(opt.rew_size)
    for vIdx = 1:numel(behav_var)
        subplot(numel(behav_var), numel(opt.rew_size), rIdx + (vIdx-1)*numel(opt.rew_size)); hold on;
        mean_this = squeeze(mean(mean_all{vIdx,rIdx}));
        sem_this = squeeze(std(mean_all{vIdx,rIdx}))/sqrt(numel(session_all));
        shadedErrorBar(t,mean_this(1,:),sem_this(1,:),'lineprops','r-');
        shadedErrorBar(t,mean_this(2,:),sem_this(2,:),'lineprops','b-');
        if vIdx==1
            ylim([-4 -3]);
            title(sprintf('%d uL',opt.rew_size(rIdx)));
            if rIdx==1
                ylabel('Position');
            elseif rIdx==3
                legend({'R0','RR'});
            end
        elseif vIdx==2
            ylim([-0.5 1.5]);
            if rIdx==1
                ylabel('Speed');
            end
        elseif vIdx==3
            ylim([-2 2]);
            if rIdx==1
                ylabel('Acceleration');
            end
        elseif vIdx==4
            ylim([4 10]);
            if rIdx==1
                ylabel('Lick Rate');
            end
            xlabel('Time from patch stop (sec)');
        end
        lh = plot([1 1],ylim,'k--');
        lh.HandleVisibility = 'off';
    end
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');