%% Quantify sequentiality across days using the sequentiality index (SI) from Orhan and Ma 2019 

% SI = entropy of peak time distn + mean log ridge to background ratio
% Accounts for 1) active periods of neurons should tile uniformly and 
%              2) each neuron should only be active in short interval 

%% Starting on simulated data
% show ramp, chaos, and sequence
close all
nNeurons = 500;
time = linspace(.1,10,100);

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
title(sprintf("Ramping \n (SI = %.2f, normSI = %.2f)",SI_ramp,SI_ramp_norm))
subplot(4,4,9) 
plot(ideal_ramp(1:round(nNeurons/5):nNeurons,:)','linewidth',2) 
title("Ramp PSTHs") 
yticks([])
ylabel("FR (A.U.)")
subplot(2,4,2)
imagesc(ideal_chaos) 
title(sprintf("Chaotic \n (SI = %.2f, normSI = %.2f)",SI_chaos,SI_chaos_norm))
subplot(4,4,10) 
plot(ideal_chaos(1:2,:)','linewidth',2) 
title("Chaotic PSTHs")
yticks([])
subplot(2,4,3)
imagesc(seq_ramp_mix) 
title(sprintf("Seq/Ramp Mix \n (SI = %.2f, normSI = %.2f)",SI_mix,SI_mix_norm))
subplot(4,4,11) 
plot(seq_ramp_mix(1:round(nNeurons/5):nNeurons,:)','linewidth',2) 
title("Seq/Ramp Mix PSTHs")
yticks([])
ylabel("FR (A.U.)")
subplot(2,4,4)
imagesc(ideal_seq)  
title(sprintf("Sequential \n (SI = %.2f, normSI = %.2f)",SI_seq,SI_seq_norm))
subplot(4,4,12) 
plot(ideal_seq(1:round(nNeurons/5):nNeurons,:)','linewidth',2)  
title("Sequential PSTHs")
yticks([])
ylabel("FR (A.U.)") 

% visualize entropy and r2b 
figure() 
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
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/80';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
opt = struct;
opt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
opt.smoothSigma_time = 0.1; % gauss smoothing sigma for rate matrix (in sec)

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name};

%% Extract FR matrices and timing information 
FR_decVar = struct; 
FRandTimes = struct;
index_sort_all = {numel(sessions)};
for sIdx = 3:3 
    buffer = 500;
    [FR_decVar_tmp,FRandTimes_tmp] = genSeqStructs(paths,sessions,opt,sIdx,buffer);
    % assign to sIdx
    FR_decVar(sIdx).fr_mat = FR_decVar_tmp.fr_mat;
    FR_decVar(sIdx).decVarTime = FR_decVar_tmp.decVarTime;
    FR_decVar(sIdx).decVarTimeSinceRew = FR_decVar_tmp.decVarTimeSinceRew;
    FRandTimes(sIdx).fr_mat = FRandTimes_tmp.fr_mat;
    FRandTimes(sIdx).stop_leave_ms = FRandTimes_tmp.stop_leave_ms;
    FRandTimes(sIdx).stop_leave_ix = FRandTimes_tmp.stop_leave_ix;

    % Sort by all trials to get ordering
    decVar_bins = linspace(0,2,41);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar(sIdx),dvar,decVar_bins,opt);
    index_sort_all{sIdx} = neuron_order; 
    
    % Perform SI calculation 
    opt = struct; 
    opt.norm = "prezscore";
    ridgeWidth = 4;
    [SI_317,SI_317_norm,log_r2b_317,entropy_317] = calculateSI(sorted_peth,ridgeWidth,opt);
    
    % visualize entropy and r2b 
    figure() 
    hold on
    scatter(mean(log_r2b_seq),entropy_seq,300,'.') 
    scatter(mean(log_r2b_chaos),entropy_chaos,300,'.')  
    scatter(mean(log_r2b_mix),entropy_mix,300,'.')  
    scatter(mean(log_r2b_ramp),entropy_ramp,300,'.')  
    scatter(mean(log_r2b_317),entropy_317,300,'.')  
    legend("Sequence","Chaos","Seq/Ramp Mix","Ramp","m80 3/17")
    xlabel("Log Ridge to Background Ratio")
    ylabel("Entropy of Peak Distribution (Nat)")
    title("Components of SI Between Idealized Activity Patterns")
    
end


