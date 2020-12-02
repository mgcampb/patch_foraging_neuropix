addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.data = 'H:\My Drive\UchidaLab\PatchForaging\processed_neuropix_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\psth_RXX';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';

%%
session_all = dir(fullfile(paths.data,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

%% Generate "reward barcodes" to average firing rates  

rew_barcodes = cell(numel(session_all),1);
for sIdx = 1:numel(session_all)

    session = session_all{sIdx};
    data = load(fullfile(paths.data,session));

    % reinitialize patch timing vectors
    patchstop_sec = data.patchCSL(:,2);
    patchleave_sec = data.patchCSL(:,3);
    rew_ms = data.rew_ts;

    % Trial level features
    patchCSL = data.patchCSL;
    prts = patchCSL(:,3) - patchCSL(:,2);
    floor_prts = floor(prts);
    rewsize = mod(data.patches(:,2),10);

    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_sec(iTrial) & rew_ms < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -1; % set part of patch after leave = -1
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end 
    rew_barcodes{sIdx} = rew_barcode;
    
    keyboard;

end



for pIdx = 1:6
    
    subplot(2,3,pIdx);
    [plot_PCsTime_4XX,~,psth_PCsTime_4XX(pIdx)] = plot_timecourse('stream', score(pIdx,:), t_align{aIdx}/tbin_ms, t_start{aIdx}/tbin_ms, t_end{aIdx}{maxTime}/tbin_ms, gr.rews_4XX, 'resample_bin',1);
    
    plot_PCsTime_4XX(2).XTick = [0 .05 .1 .15];
    plot_PCsTime_4XX(2).XTickLabel = {[0 1 2 3]};
    
    plot_PCsTime_4XX(2).XLabel.String = 'time since patch stop (s)';
    
    plot_PCsTime_4XX(2).Legend.String = {['400'] ['440'] ['404'] ['444']};
    
    title(['PC' num2str(pIdx)], 'FontSize', 18);
    
end