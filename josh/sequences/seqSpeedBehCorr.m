%% A refactored script to analyze how sequence progression speed correlates with behavior across sessions
%  start by trying to use mid_resp neurons from the gaussian selection method
%   - would be nice to have single way of selecting sequence neurons 

%% Basic path stuff + load midresponsive neurons struct

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
calc_frOpt = struct;
calc_frOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec) 
tbin_ms = calc_frOpt.tbin * 1000;
calc_frOpt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec) 
buffer = 500; % cut off 500 ms

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

load('midresp_struct.mat') % struct from gaussian-exp regression  

%% Now load in data and pull out just the mid-responsive neurons  
%  need to do a sort to index (make sure that it is the same protocol as
%  used to make midresp_struct)
FR_decVar = struct; 
index_sort_all = {numel(sessions)};
for sIdx = 1:24
    new_structs = true; 
    if new_structs == true
        [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,calc_frOpt,sIdx,buffer);
    end 
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    
    % Perform PETH/sorting 
    nBins = 40;
    decVar_bins = linspace(0,2,nBins+1);
    opt.norm = "zscore";
    opt.trials = 'all'; 
    opt.suppressVis = true;
    dvar = "timesince";
    [sorted_peth,order,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,opt); 
    nNeurons = size(sorted_peth,1);  
    index_sort_all{sIdx} = order;
    
    figure();colormap("jet")
    imagesc(flipud(sorted_peth(midresp_struct(sIdx).mid_resp_ix,:))) 
    title("Sorted PETH for Mid-Responsive Neurons")
end  

%% Now perform sequence progression speed analysis
close all

mid_resp_mPFC_sessions = [2:4 6:12 14:18 22:24]; 

r_seqProg = nan(numel(mid_resp_mPFC_sessions),1);  
p_seqProg = nan(numel(mid_resp_mPFC_sessions),1);
r_meanMid = nan(numel(mid_resp_mPFC_sessions),1); 
p_meanMid = nan(numel(mid_resp_mPFC_sessions),1);

for i = 1:numel(mid_resp_mPFC_sessions)
    sIdx = mid_resp_mPFC_sessions(i);
    session = sessions{sIdx}(1:end-4);  
    session_title = sessions{sIdx}([1:2 end-6:end-4]);
    data = load(fullfile(paths.data,session));
    session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing
    
    % reinitialize ms vectors to make barcode matrix
    rew_ms = data.rew_ts;
    patchCSL = data.patchCSL;
    patches = data.patches;
    prts = patchCSL(:,3) - patchCSL(:,2);
    patchstop_ms = patchCSL(:,2);
    patchleave_ms = patchCSL(:,3);
    floor_prts = floor(prts);
    patchType = patches(:,2);
    rewsize = mod(patchType,10);
    nTrials = length(patchType); 
    nNeurons = size(FR_decVar(sIdx).fr_mat{1},1);
    
    % make barcode matrices
    nTimesteps = 15;
    rew_barcode = zeros(length(patchCSL) , nTimesteps);
    rew_ix_cell = {length(patchCSL)};
    last_rew_ix = nan(length(patchCSL),1);
    for iTrial = 1:length(patchCSL)
        rew_indices = round(rew_ms(rew_ms >= patchstop_ms(iTrial) & rew_ms < patchleave_ms(iTrial)) - patchstop_ms(iTrial)) + 1;
        last_rew_ix(iTrial) = max(rew_indices);
        rew_ix_cell{iTrial} = (rew_indices(rew_indices > 1) - 1) * 1000 / tbin_ms;
        rew_barcode(iTrial , (last_rew_ix(iTrial) + 1):end) = -1; % set part of patch after last rew_ix = -1
        rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
        rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
    end
    
    % from barcode matrix, get the trials w/ only rew at t = 0
    one_rew_trials = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1);
    lg1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 4);
    md1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 2);
    sm1rew = find(rew_barcode(:,1) > 0 & rew_barcode(:,2) == -1 & rewsize == 1);
    
    shuffle = false;
    
    % select population for analysis
    mid_resp = midresp_struct(sIdx).mid_resp_ix; 
    mid_resp_labels = ismember(1:nNeurons,mid_resp);

    % data structures
    prog_slopes = nan(nTrials,1);
    mean_seq = nan(nTrials,1);
    
    for j = 1:numel(one_rew_trials)
        iTrial = one_rew_trials(j); 
        fr_mat_iTrial = FR_decVar(sIdx).fr_mat{iTrial}(index_sort_all{sIdx},:);
        norm_fr_mat_iTrial = zscore(fr_mat_iTrial(mid_resp,:),[],2); % just sequence bois now
        [times,neurons] = find(norm_fr_mat_iTrial(:,1:50) > 0);
        activity = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,1:50) > 0);
        
        % weighted linear regression on first second sequence
        mdl = fitlm(neurons,times,'Intercept',false,'Weights',activity);
        prog_slopes(iTrial) = mdl.Coefficients.Estimate;
        
%         mean_seq(iTrial) = mean(mean(norm_fr_mat_iTrial(:,1:50)));
        mean_seq(iTrial) = mean(mean(fr_mat_iTrial(mid_resp,1:50)));

%   visualize regression on sequential activition
%         figure();
%         subplot(1,2,1)
%         colormap('jet')
%         imagesc(flipud(zscore(fr_mat_iTrial,[],2)))
%         hold on
%         gscatter(zeros(nNeurons,1),1:nNeurons,fliplr(mid_resp_labels),[0 0 0; 1 0 0]) 
%         legend("Non Mid-responsive","Mid-responsive")
%         title(sprintf("%i uL Trial %i",rewsize(iTrial),iTrial))
%         xlabel("Time on Patch (ms)")
%         if size(norm_fr_mat_iTrial,2) < 100
%             xticks([0 25 50 75])
%             xticklabels([0 500 1000 1500])
%         else
%             xticks([0 25 50 75 100])
%             xticklabels([0 500 1000 1500 2000]) 
%         end 
%         xlim([0 100])
%         subplot(1,2,2)
%         colormap('jet')
%         imagesc(flipud(norm_fr_mat_iTrial(:,1:50)))
%         colorbar()
%         title(sprintf('Trial %i',iTrial))
%         xlabel('Time')
%         % draw lines to indicate reward delivery
%         subplot(1,2,2); hold on
%         title("Data for Weighted Linear Regression")
%         scatter(neurons,numel(mid_resp) - times,activity,'kx')
%         xlabel("Time on Patch (ms)")
%         xticks([0 25 50])
%         xticklabels([0 500 1000])
%         plot(neurons,numel(mid_resp) - mdl.Fitted,'linewidth',2)
    end
    
%     colors = [0 0 0; 0 1 1;0 0 1;0 0 1];
    % Seq-PRT
    [r0,p0] = corrcoef(prog_slopes(one_rew_trials),prts(one_rew_trials));
    [r1,p1] = corrcoef(prog_slopes(sm1rew),prts(sm1rew));
    [r2,p2] = corrcoef(prog_slopes(md1rew),prts(md1rew));
    [r3,p3] = corrcoef(prog_slopes(lg1rew),prts(lg1rew));
%     colors = cool(3);
%     figure();hold on
%     gscatter(prog_slopes(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
%     xlabel("Slope of sequence progression")
%     ylabel("PRT")
%     title(sprintf("%s Slope of sequence progression vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",session_title,p0(2),p1(2),p2(2),p3(2)))
%     legend("1 uL","2 uL","4 uL") 
    r_seqProg(i) = r0(2); 
    p_seqProg(i) = p0(2);
    
    % Mean value of sequence-PRT correlation
    [r0,p0] = corrcoef(mean_seq(one_rew_trials),prts(one_rew_trials));
    [r1,p1] = corrcoef(mean_seq(sm1rew),prts(sm1rew));
    [r2,p2] = corrcoef(mean_seq(md1rew),prts(md1rew));
    [r3,p3] = corrcoef(mean_seq(lg1rew),prts(lg1rew));
%     figure();hold on;colormap('cool')
%     gscatter(mean_seq(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
%     title(sprintf("%s Mean Mid-Responsive Neuron Activity vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",session_title,p0(2),p1(2),p2(2),p3(2)))
%     ylabel("PRT")
%     xlabel("Mean Mid-Responsive Neuron Activity") 
    r_meanMid(i) = r0(2); 
    p_meanMid(i) = p0(2);
end 

%% Visualize cross session results 
close all
significance_labels = zeros(numel(mid_resp_mPFC_sessions),1); 
significance_labels(p_seqProg < .05 & p_meanMid > .05) = 1; 
significance_labels(p_seqProg > .05 & p_meanMid < .05) = 2; 
significance_labels(p_seqProg < .05 & p_meanMid < .05) = 3;  
colors = [0 0 0;1 0 0; 0 0 1; .8 .2 .8];

figure()  
yline(0,'linewidth',1.5) 
xline(0,'linewidth',1.5)  
hold on 
p = gscatter(r_seqProg,r_meanMid,significance_labels,colors);
legend([p(1:4)],"No significance", "Progression Speed p < .05","Mean Activity p < .05","Progression Speed and Mean Activity p < .05")
grid()
xlim([-1,1]) 
ylim([-1,1]) 
xlabel("Progression Speed vs PRT Pearson Corr")
ylabel("Mean Activity vs PRT Pearson Corr")
title("Mid Responsive Neuron Behavioral Correlations Across Sessions")  



