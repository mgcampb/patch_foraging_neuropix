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
peakSortOpt.preLeave_buffer = 500;  
peakSortOpt.cortex_only = true;

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Now load in data and pull out just the mid-responsive neurons  
%  need to do a sort to index (make sure that it is the same protocol as
%  used to make midresp_struct)
FR_decVar = struct; 
peakSort_midresp = cell(numel(sessions),1); 
midresp = cell(numel(sessions),1);
for sIdx = 1:25 
    session = sessions{sIdx};
    dat = load(fullfile(paths.data,session));  
    
    if isfield(dat,'anatomy')
        [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,calc_frOpt,sIdx);
        FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
        FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
        FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew; 
        
        % find mid-responsive neurons
        nTrials = length(FR_decVar(sIdx).fr_mat);  
        nBins = 40;
        decVar_bins = linspace(0,2,nBins+1);
        transient_opt = struct; 
        transient_opt.visualization = false; 
        transient_opt.nShuffles = 500;
        transient_opt.preRew_buffer = round(calc_frOpt.smoothSigma_time * 3 * 1000 / tbin_ms); 
        transient_opt.postStop_buffer = NaN; % allow first reward 
        trial_selection = 1:nTrials;

        transients_struct_tmp = driscoll_transient_discovery(FR_decVar(sIdx),trial_selection,decVar_bins,tbin_ms,transient_opt); 
        midresp{sIdx} = transients_struct_tmp.midresp;
        
        % Perform PETH/sorting
        peakSortOpt.norm = "zscore";
        peakSortOpt.trials = 'all'; 
        peakSortOpt.neurons = midresp{sIdx}; % just use the mid-responsive neurons
        peakSortOpt.suppressVis = false; 
        peakSortOpt.preRew_buffer = round(calc_frOpt.smoothSigma_time * 3 * 1000 / tbin_ms); 
        peakSortOpt.tbin_ms = tbin_ms;
        dvar = "timesince";
        [~,peaksort,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,peakSortOpt);
        peakSort_midresp{sIdx} = peaksort;

    end
end

%% Now perform sequence progression speed analysis
close all

mid_resp_mPFC_sessions = [(1:8) (10:18) 23 25];  
mouse_groups = {1:2,3:8,9:12,13:17,18:19}; % group mPFC sessions by animal 

% log regressions per session
r_seqProg = nan(numel(mid_resp_mPFC_sessions),1);  
p_seqProg = nan(numel(mid_resp_mPFC_sessions),1);
r_meanMid = nan(numel(mid_resp_mPFC_sessions),1); 
p_meanMid = nan(numel(mid_resp_mPFC_sessions),1); 

r2uL_seqProg = nan(numel(mid_resp_mPFC_sessions),1);  
p2uL_seqProg = nan(numel(mid_resp_mPFC_sessions),1);
r2uL_meanMid = nan(numel(mid_resp_mPFC_sessions),1); 
p2uL_meanMid = nan(numel(mid_resp_mPFC_sessions),1);

r4uL_seqProg = nan(numel(mid_resp_mPFC_sessions),1);  
p4uL_seqProg = nan(numel(mid_resp_mPFC_sessions),1);
r4uL_meanMid = nan(numel(mid_resp_mPFC_sessions),1); 
p4uL_meanMid = nan(numel(mid_resp_mPFC_sessions),1);  

% pool within reward size between sessions within mice  
prt1uL_pooled = cell(numel(mouse_groups),1); 
prt2uL_pooled = cell(numel(mouse_groups),1); 
prt4uL_pooled = cell(numel(mouse_groups),1); 
seqProg1uL_pooled = cell(numel(mouse_groups),1); 
seqProg2uL_pooled = cell(numel(mouse_groups),1); 
seqProg4uL_pooled = cell(numel(mouse_groups),1);  
meanMid1uL_pooled = cell(numel(mouse_groups),1);  
meanMid2uL_pooled = cell(numel(mouse_groups),1);  
meanMid4uL_pooled = cell(numel(mouse_groups),1);  
dayLabels1uL = cell(numel(mouse_groups),1);
dayLabels2uL = cell(numel(mouse_groups),1);
dayLabels4uL = cell(numel(mouse_groups),1); 
session_titles_grp = cell(numel(mouse_groups),1);

% log pooled within rewsize regressions per mouse 
r1uL_seqProg_pooled = nan(numel(mouse_groups),1);  
p1uL_seqProg_pooled = nan(numel(mouse_groups),1);
r1uL_meanMid_pooled = nan(numel(mouse_groups),1); 
p1uL_meanMid_pooled = nan(numel(mouse_groups),1);

r2uL_seqProg_pooled = nan(numel(mouse_groups),1);  
p2uL_seqProg_pooled = nan(numel(mouse_groups),1);
r2uL_meanMid_pooled = nan(numel(mouse_groups),1); 
p2uL_meanMid_pooled = nan(numel(mouse_groups),1);

r4uL_seqProg_pooled = nan(numel(mouse_groups),1);  
p4uL_seqProg_pooled = nan(numel(mouse_groups),1);
r4uL_meanMid_pooled = nan(numel(mouse_groups),1); 
p4uL_meanMid_pooled = nan(numel(mouse_groups),1);  

for m = 1:numel(mouse_groups)
    prt1uL_pooled{m} = [];
    prt2uL_pooled{m} = [];
    prt4uL_pooled{m} = [];
    seqProg1uL_pooled{m} = [];
    seqProg2uL_pooled{m} = [];
    seqProg4uL_pooled{m} = [];
    meanMid1uL_pooled{m} = [];
    meanMid2uL_pooled{m} = [];
    meanMid4uL_pooled{m} = []; 
    dayLabels1uL{m} = []; 
    dayLabels2uL{m} = []; 
    dayLabels4uL{m} = []; 
    session_titles_grp{m} = cell(numel(mouse_groups{m}),1);
    for i = 1:numel(mouse_groups{m}) % use this to index day vector 
        mi = mouse_groups{m}(i);
        sIdx = mid_resp_mPFC_sessions(mi);
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
        %     % select population for analysis
        %     mid_resp = midresp_struct(sIdx).mid_resp_ix;
        mid_resp_labels = ismember(1:nNeurons,midresp{sIdx});
        
        % data structures
        prog_slopes = nan(nTrials,1);
        mean_seq = nan(nTrials,1);
        
        for j = 1:numel(one_rew_trials)
            iTrial = one_rew_trials(j);
            fr_mat_iTrial = FR_decVar(sIdx).fr_mat{iTrial}(midresp{sIdx},:);
            norm_fr_mat_iTrial = zscore(fr_mat_iTrial(peakSort_midresp{sIdx},:),[],2); % just sequence bois now
            [times,neurons] = find(norm_fr_mat_iTrial(:,1:50) > 0);
            activity = norm_fr_mat_iTrial(norm_fr_mat_iTrial(:,1:50) > 0);
            
            % weighted linear regression on first second sequence
            mdl = fitlm(neurons,times,'Intercept',false,'Weights',activity);
            prog_slopes(iTrial) = mdl.Coefficients.Estimate / numel(midresp{sIdx}); % normalized progression speed
            
            mean_seq(iTrial) = mean(mean(norm_fr_mat_iTrial(:,1:50)));
            %         mean_seq(iTrial) = mean(mean(fr_mat_iTrial(:,1:50)));
            
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
            %         scatter(neurons,numel(midresp{sIdx}) - times,activity,'kx')
            %         xlabel("Time on Patch (ms)")
            %         xticks([0 25 50])
            %         xticklabels([0 500 1000])
            %         plot(neurons,numel(midresp{sIdx}) - mdl.Fitted,'linewidth',2)
        end
        
        % Seq-PRT
        [r0,p0] = corrcoef(prog_slopes(one_rew_trials),prts(one_rew_trials));
        [~,p1] = corrcoef(prog_slopes(sm1rew),prts(sm1rew));
        [r2,p2] = corrcoef(prog_slopes(md1rew),prts(md1rew));
        [r3,p3] = corrcoef(prog_slopes(lg1rew),prts(lg1rew));
%         colors = cool(3);
%         figure();hold on
%         gscatter(prog_slopes(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
%         xlabel("Slope of sequence progression")
%         ylabel("PRT")
%         title(sprintf("%s Slope of sequence progression vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",session_title,p0(2),p1(2),p2(2),p3(2)))
%         legend("1 uL","2 uL","4 uL")
        r_seqProg(mi) = r0(2);
        p_seqProg(mi) = p0(2);
        r2uL_seqProg(mi) = r2(2);
        p2uL_seqProg(mi) = p2(2);
        r4uL_seqProg(mi) = r3(2);
        p4uL_seqProg(mi) = p3(2);
        
        % Mean value of sequence-PRT correlation
        [r0,p0] = corrcoef(mean_seq(one_rew_trials),prts(one_rew_trials));
        [~,p1] = corrcoef(mean_seq(sm1rew),prts(sm1rew));
        [r2,p2] = corrcoef(mean_seq(md1rew),prts(md1rew));
        [r3,p3] = corrcoef(mean_seq(lg1rew),prts(lg1rew));
%         figure();hold on;colormap('cool')
%         gscatter(mean_seq(one_rew_trials),prts(one_rew_trials),rewsize(one_rew_trials),colors,'.')
%         title(sprintf("%s Mean Mid-Responsive Neuron Activity vs PRT (overall p = %f, 1uL p = %f, 2 uL p = %f, 4 uL p = %f)",session_title,p0(2),p1(2),p2(2),p3(2)))
%         ylabel("PRT")
%         xlabel("Mean Mid-Responsive Neuron Activity")
        r_meanMid(mi) = r0(2);
        p_meanMid(mi) = p0(2);
        r2uL_meanMid(mi) = r2(2);
        p2uL_meanMid(mi) = p2(2);
        r4uL_meanMid(mi) = r3(2);
        p4uL_meanMid(mi) = p3(2);
        
        % collect trials together to pool across sessions 
        prt1uL_pooled{m} = [prt1uL_pooled{m};prts(sm1rew)];
        prt2uL_pooled{m} = [prt2uL_pooled{m};prts(md1rew)];
        prt4uL_pooled{m} = [prt4uL_pooled{m};prts(lg1rew)];
        seqProg1uL_pooled{m} = [seqProg1uL_pooled{m};prog_slopes(sm1rew)];
        seqProg2uL_pooled{m} = [seqProg2uL_pooled{m};prog_slopes(md1rew)];
        seqProg4uL_pooled{m} = [seqProg4uL_pooled{m};prog_slopes(lg1rew)];
        meanMid1uL_pooled{m} = [meanMid1uL_pooled{m};mean_seq(sm1rew)];
        meanMid2uL_pooled{m} = [meanMid2uL_pooled{m};mean_seq(md1rew)];
        meanMid4uL_pooled{m} = [meanMid4uL_pooled{m};mean_seq(lg1rew)]; 
        dayLabels1uL{m} = [dayLabels1uL{m};i+zeros(length(prts(sm1rew)),1)];
        dayLabels2uL{m} = [dayLabels2uL{m};i+zeros(length(prts(md1rew)),1)];
        dayLabels4uL{m} = [dayLabels4uL{m};i+zeros(length(prts(lg1rew)),1)]; 
        session_titles_grp{m}{i} = session_title;
    end 
    
    % 1 uL pooling
    [r1_progSlope,p1_progSlope] = corrcoef(seqProg1uL_pooled{m},prt1uL_pooled{m});
    [r1_meanMid,p1_meanMid] = corrcoef(meanMid1uL_pooled{m},prt1uL_pooled{m});
%     colors1uL = [linspace(.5,0,numel(mouse_groups{m}))' ones(numel(mouse_groups{m}),1) ones(numel(mouse_groups{m}),1)];
%     figure();hold on
%     gscatter(seqProg1uL_pooled{m},prt1uL_pooled{m},dayLabels1uL{m},colors1uL,'.')
%     scatter(seqProg1uL_pooled{m},prt1uL_pooled{m},'ko')
%     xlabel("Slope of sequence progression")
%     ylabel("PRT")
%     mouseName = session_titles_grp{m}{1}(1:2);
%     title(sprintf("m%s 1 uL Slope of sequence progression vs PRT (p = %f,r = %f)",mouseName,p1_progSlope(2),r1_progSlope(2)))
%     legend(session_titles_grp{m}) % days from this mice 
    % log regression results
    r1uL_seqProg_pooled(m) = r1_progSlope(2);
    p1uL_seqProg_pooled(m) = p1_progSlope(2);
    r1uL_meanMid_pooled(m) = r1_meanMid(2);
    p1uL_meanMid_pooled(m) = p1_meanMid(2);
    
    [r2_progSlope,p2_progSlope] = corrcoef(seqProg2uL_pooled{m},prt2uL_pooled{m});  
    [r2_meanMid,p2_meanMid] = corrcoef(meanMid2uL_pooled{m},prt2uL_pooled{m});
%     colors2uL = [linspace(.5,0,numel(mouse_groups{m}))' linspace(.5,0,numel(mouse_groups{m}))' ones(numel(mouse_groups{m}),1)];
%     figure();hold on
%     gscatter(seqProg2uL_pooled{m},prt2uL_pooled{m},dayLabels2uL{m},colors2uL,'.') 
%     scatter(seqProg2uL_pooled{m},prt2uL_pooled{m},'ko')
%     xlabel("Slope of sequence progression")
%     ylabel("PRT") 
%     mouseName = session_titles_grp{m}{1}(1:2);
%     title(sprintf("m%s 2 uL Slope of sequence progression vs PRT (p = %f,r = %f)",mouseName,p2_progSlope(2),r2_progSlope(2)))
%     legend(session_titles_grp{m}) % days from this mice 
    % log regression results
    r2uL_seqProg_pooled(m) = r2_progSlope(2);
    p2uL_seqProg_pooled(m) = p2_progSlope(2);
    r2uL_meanMid_pooled(m) = r2_meanMid(2);
    p2uL_meanMid_pooled(m) = p2_meanMid(2);
    
    [r3_progSlope,p3_progSlope] = corrcoef(seqProg4uL_pooled{m},prt4uL_pooled{m}); 
    [r3_meanMid,p3_meanMid] = corrcoef(meanMid4uL_pooled{m},prt4uL_pooled{m});
%     colors4uL = [ones(numel(mouse_groups{m}),1) linspace(.5,0,numel(mouse_groups{m}))' ones(numel(mouse_groups{m}),1)];
%     figure();hold on
%     gscatter(seqProg4uL_pooled{m},prt4uL_pooled{m},dayLabels4uL{m},colors4uL,'.') 
%     scatter(seqProg4uL_pooled{m},prt4uL_pooled{m},'ko')
%     xlabel("Slope of sequence progression")
%     ylabel("PRT") 
%     mouseName = session_titles_grp{m}{1}(1:2);
%     title(sprintf("m%s 4 uL Slope of sequence progression vs PRT (p = %f,r = %f)",mouseName,p3(2),r3(2)))
%     legend(session_titles_grp{m}) % days from this mice 
    % log regression results
    r4uL_seqProg_pooled(m) = r3_progSlope(2);
    p4uL_seqProg_pooled(m) = p3_progSlope(2);
    r4uL_meanMid_pooled(m) = r3_meanMid(2);
    p4uL_meanMid_pooled(m) = p3_meanMid(2);
end

%% Visualize cross session results 
close all 

% results for progression speed vs meanMid across all rewsizes
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
legend(p(1:4),"No significance", "Progression Speed p < .05","Mean Activity p < .05","Progression Speed and Mean Activity p < .05")
grid()
xlim([-1,1]) 
ylim([-1,1]) 
xlabel("Progression Speed vs PRT Pearson Corr")
ylabel("Mean Activity vs PRT Pearson Corr")
title("Mid Responsive Neuron Behavioral Correlations Across Sessions")  

% Plot pooled within reward size results across mice   
% First progression speed
figure(); 
colors = cool(3);
b = bar([r1uL_seqProg_pooled r2uL_seqProg_pooled r4uL_seqProg_pooled],'FaceColor','flat');
for k = 1:3
    b(k).CData = colors(k,:);
end 

% plot pooled within reward size results across mice  
figure(); 
colors = cool(3);
b = bar([r1uL_meanMid_pooled r2uL_meanMid_pooled r4uL_meanMid_pooled],'FaceColor','flat');
for k = 1:3
    b(k).CData = colors(k,:);
end 



% Plot within reward size within single days
% results for progSpeed all rewsizes vs progSpeed 4 uL
% significance_labels = zeros(numel(mid_resp_mPFC_sessions),1); 
% significance_labels(p_seqProg < .05 & p4uL_seqProg > .05) = 1; 
% significance_labels(p_seqProg > .05 & p4uL_seqProg < .05) = 2; 
% significance_labels(p_seqProg < .05 & p4uL_seqProg < .05) = 3;  
% colors = [0 0 0;1 0 0; 0 0 1; .8 .2 .8];
% figure()  
% yline(0,'linewidth',1.5) 
% xline(0,'linewidth',1.5)  
% hold on 
% p = gscatter(r_seqProg,r4uL_seqProg,significance_labels,colors);
% legend(p(1:4),"No significance", "All Rewsizes Progression Speed p < .05","4 uL Progression Speed p < .05","All Rewsizes and 4 uL Progression Speed p < .05")
% grid()
% xlim([-1,1]) 
% ylim([-1,1]) 
% xlabel("All Rewsizes Progression Speed vs PRT Pearson Corr")
% ylabel("4 uL Progression Speed vs PRT Pearson Corr")
% title("Mid Responsive Neuron Behavioral Correlations Across Sessions")   
% 
% % results for progSpeed all rewsizes vs progSpeed 2 uL
% significance_labels = zeros(numel(mid_resp_mPFC_sessions),1); 
% significance_labels(p_seqProg < .05 & p2uL_seqProg > .05) = 1; 
% significance_labels(p_seqProg > .05 & p2uL_seqProg < .05) = 2; 
% significance_labels(p_seqProg < .05 & p2uL_seqProg < .05) = 3;  
% colors = [0 0 0;1 0 0; 0 0 1; .8 .2 .8];
% 
% figure()  
% yline(0,'linewidth',1.5) 
% xline(0,'linewidth',1.5)  
% hold on 
% p = gscatter(r_seqProg,r2uL_seqProg,significance_labels,colors);
% legend(p(1:4),"No significance", "All Rewsizes Progression Speed p < .05","2 uL Progression Speed p < .05","All Rewsizes and 2 uL Progression Speed p < .05")
% grid()
% xlim([-1,1]) 
% ylim([-1,1]) 
% xlabel("All Rewsizes Progression Speed vs PRT Pearson Corr")
% ylabel("2 uL Progression Speed vs PRT Pearson Corr")
% title("Mid Responsive Neuron Behavioral Correlations Across Sessions")  
