%% Quantify sequentiality across days using the sequentiality index (SI) from Orhan and Ma 2019 

% SI = entropy of peak time distn + mean log ridge to background ratio
% Accounts for 1) active periods of neurons should tile uniformly and 
%              2) each neuron should only be active in short interval 

%% Starting on simulated data
% show ramp, chaos, and sequence
close all
nNeurons = 100;
time = linspace(.1,10,40);

% Sequence of gaussian activations
peaks = linspace(0,10,nNeurons);
ideal_seq = flipud(normpdf(repmat(time,[nNeurons,1])',peaks,ones(1,nNeurons))'); 

% Chaotic activity (Random) 
ideal_chaos = rand(nNeurons,length(time)); 

% Ramping activity (exponential with varying tau) 
tau_min = .5; 
tau_scaling = .1;
taus = tau_min + tau_scaling * rand(nNeurons,1);
ideal_ramp = exp(time .* taus); 

% Mix of sequence and ramping activity 
half = round(nNeurons / 2);
seq_ramp_mix = zscore([ideal_ramp(half+1:end,:);ideal_seq(half+1:end,:)],[],2);

% now calculate SI 
opt = struct; 
ridgeWidth = 2;
[SI_seq,SI_seq_norm,log_r2b_seq,entropy_seq] = calculateSI(ideal_seq,ridgeWidth,opt);
[SI_chaos,SI_chaos_norm,log_r2b_chaos,entropy_chaos] = calculateSI(ideal_chaos,ridgeWidth,opt);
[SI_mix,SI_mix_norm,log_r2b_mix,entropy_mix] = calculateSI(seq_ramp_mix,ridgeWidth,opt);
[SI_ramp,SI_ramp_norm,log_r2b_ramp,entropy_ramp] = calculateSI(ideal_ramp,ridgeWidth,opt);

% visualize example activity and title with SI
figure();colormap('jet') 
subplot(2,4,1)
imagesc(ideal_ramp) 
ylabel("Neurons")
xlabel("Time")
xticks([])
title(sprintf("Ramp \n (SI = %.2f, normSI = %.2f)",SI_ramp,max(0,SI_ramp_norm)))
subplot(4,4,9) 
plot(ideal_ramp(1:round(nNeurons/5):nNeurons,:)','linewidth',2) 
title("Ramp PSTHs") 
ylabel("FR")
xlabel("Time")
yticks([])
xticks([])
subplot(2,4,2)
imagesc(ideal_chaos) 
ylabel("Neurons")
xlabel("Time")
xticks([])
title(sprintf("Chaotic \n (SI = %.2f, normSI = %.2f)",SI_chaos,SI_chaos_norm))
subplot(4,4,10) 
plot(ideal_chaos(1:2,:)','linewidth',2) 
title("Chaotic PSTHs")
ylabel("FR")
xlabel("Time")
yticks([])
xticks([])
subplot(2,4,3)
imagesc(seq_ramp_mix) 
ylabel("Neurons")
xlabel("Time")
xticks([])
title(sprintf("Seq/Ramp Mix \n (SI = %.2f, normSI = %.2f)",SI_mix,SI_mix_norm))
subplot(4,4,11) 
plot(seq_ramp_mix(1:round(nNeurons/5):nNeurons,:)','linewidth',2) 
title("Seq/Ramp Mix PSTHs")
yticks([])
ylabel("FR")
xlabel("Time")
xticks([])
subplot(2,4,4)
imagesc(ideal_seq)  
ylabel("Neurons")
xlabel("Time")
xticks([])
title(sprintf("Sequential \n (SI = %.2f, normSI = %.2f)",SI_seq,SI_seq_norm))
subplot(4,4,12) 
plot(ideal_seq(1:round(nNeurons/5):nNeurons,:)','linewidth',2)  
title("Sequential PSTHs")
yticks([])
ylabel("FR") 
xlabel("Time")
xticks([])

% visualize entropy and r2b 
figure() 
grid()
hold on
scatter(mean(log_r2b_seq),entropy_seq,300,'.') 
scatter(mean(log_r2b_chaos),entropy_chaos,300,'.')  
scatter(mean(log_r2b_mix),entropy_mix,300,'.')  
scatter(mean(log_r2b_ramp),entropy_ramp,300,'.')  
legend("Sequence","Chaos","Seq/Ramp Mix","Ramp")
xlabel("Log Ridge to Background Ratio")
ylabel("Entropy of Peak Distribution (Nat)")
title("Components of SI Between Idealized Activity Patterns")

%% Now try for single session of peak sorted PETH 
% Extract FR matrices and timing information 

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/78';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
calc_frOpt = struct;
calc_frOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calc_frOpt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
buffer = 500; % how much to trim off end of trial

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

FR_decVar = struct; 
FRandTimes = struct;
index_sort_all = {numel(sessions)};

for sIdx = 2:2
    [FR_decVar_tmp,FRandTimes_tmp] = genSeqStructs(paths,sessions,calc_frOpt,sIdx,buffer);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
    FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
    FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix;

    % Perform PETH/sorting
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,~,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    
    % Same for shuffled data 
    shuffle_opt.norm = "zscore";
    shuffle_opt.trials = 'all';
    shuffle_opt.suppressVis = true; 
    shuffle_opt.shuffle = true;
    dvar = "timesince";
    [shuffled_sorted_peth,~,~] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,shuffle_opt);
    
    % Perform SI calculation 
    opt = struct; 
    opt.norm = "prezscore";
    ridgeWidth = 4;
    [SI_317,SI_317_norm,log_r2b_317,entropy_317] = calculateSI(sorted_peth,ridgeWidth,opt);
    [SI_317_shuff,SI_317_norm_shuff,log_r2b_317_shuff,entropy_317_shuff] = calculateSI(shuffled_sorted_peth,ridgeWidth,opt);
    
    % visualize entropy and r2b 
    figure() 
    hold on
    scatter(mean(log_r2b_seq),entropy_seq,300,'.') 
    scatter(mean(log_r2b_chaos),entropy_chaos,300,'.')  
    scatter(mean(log_r2b_mix),entropy_mix,300,'.')  
    scatter(mean(log_r2b_ramp),entropy_ramp,300,'.')  
    scatter(mean(log_r2b_317),entropy_317,300,'.')  
    scatter(mean(log_r2b_317_shuff),entropy_317_shuff,300,'.')  
    legend("Sequence","Chaos","Seq/Ramp Mix","Ramp","m80 3/17","shuffled m80 3/17")
    xlabel("Log Ridge to Background Ratio")
    ylabel("Entropy of Peak Distribution (Nat)")
    title("Components of SI Between Activity Patterns")
end

%% Now iterate across sessions!
outerPath = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/';
session_names = {}; 
entropies = []; 
log_r2bs = []; 
norm_SIs = [];
session_counts = [];

for mouse = {'75','76','78','79','80'}
    paths = struct;
    paths.data = [outerPath mouse{:}];

    % analysis options
    calc_frOpt = struct;
    calc_frOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
    calc_frOpt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)
    buffer = 500; % how much to trim time in ms to trim off end of trial

    sessions = dir(fullfile(paths.data,'*.mat'));
    sessions = {sessions.name}; 
    session_counts = [session_counts numel(sessions)];

    FR_decVar = struct; 
    FRandTimes = struct;

    for sIdx = 1:numel(sessions) 
        session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing 
        session_name = ['m' session(1:2) ' ' session(end-2) '/' session(end-1:end)]; 
        session_names = [session_names session_name];
        [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,calc_frOpt,sIdx,buffer);

        % Perform PETH/sorting
        decVar_bins = linspace(0,2,41);
        opt.norm = "zscore";
        opt.trials = 'all';
        opt.suppressVis = true;
        dvar = "timesince";
        [sorted_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,opt);

        % Same for shuffled data 
        shuffle_opt.norm = "zscore";
        shuffle_opt.trials = 'all';
        shuffle_opt.suppressVis = true; 
        shuffle_opt.shuffle = true;
        dvar = "timesince";
        [shuffled_sorted_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,shuffle_opt);

        % Perform SI calculation 
        opt = struct; 
        opt.norm = "prezscore";
        ridgeWidth = 4;
        [SI_session,SI_session_norm,log_r2b_session,entropy_session] = calculateSI(sorted_peth,ridgeWidth,opt);
%         [SI_session_shuff,SI_session_norm_shuff,log_r2b_session_shuff,entropy_session_shuff] = calculateSI(shuffled_sorted_peth,ridgeWidth,opt);
        
        % add to pooled datastructures
        log_r2bs = [log_r2bs mean(log_r2b_session)]; 
        entropies = [entropies entropy_session]; 
        norm_SIs = [norm_SIs SI_session_norm];
    end
end

%% Visualize cross-session SI results
% visualize entropy and r2b  
close all
colors = cool(5);
figure() 
hold on
% synthetic data
scatter(mean(log_r2b_seq),entropy_seq,300,'.') 
scatter(mean(log_r2b_chaos),entropy_chaos,300,'.')  
scatter(mean(log_r2b_mix),entropy_mix,300,'.')  
scatter(mean(log_r2b_ramp),entropy_ramp,300,'.')  
% mouse data
cumulative_session_counts = cumsum([0 session_counts]);
for i = 1:(numel(cumulative_session_counts)-1)
    mouse_sessions = cumulative_session_counts(i)+1:cumulative_session_counts(i+1);
    scatter(log_r2bs(mouse_sessions),entropies(mouse_sessions),300,colors(i,:),'marker','.') 
end
legend_names = ["Sequence","Chaos","Seq/Ramp Mix","Ramp", {'75','76','78','79','80'}];
legend(legend_names)
xlabel("Log Ridge to Background Ratio")
ylabel("Entropy of Peak Distribution (Nat)")
title("Components of SI Across Sessions")

figure() 
b = bar(norm_SIs); 
b.FaceColor = 'flat';
for i = 1:(numel(cumulative_session_counts)-1)
    mouse_sessions = cumulative_session_counts(i)+1:cumulative_session_counts(i+1);
    b.CData(mouse_sessions,:) = repmat(colors(i,:),[session_counts(i),1]); 
end 
hold on 
yline(SI_chaos_norm,'--',"Chaos",'linewidth',1.5)
yline(SI_mix_norm,'--',"Mix",'linewidth',1.5)
ylim([0,1]) 
xticklabels(session_names)  
ylabel("Normalized SI")  
title("Normalized SI Across Sessions")




