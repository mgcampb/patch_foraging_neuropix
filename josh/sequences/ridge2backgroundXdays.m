%% Calculate ridge to background ratio for ridge responsive neurons across days
%  just look at time since reward as this is the main focus
%  replace linear with exponential to allow for monotonic nonlinearity

%% Start testing with single session to check that fits are working
paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
paths.figs = '/Users/joshstern/Documents/UchidaLab_NeuralData/neural_data_figs'; % where to save figs

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData/HGK_analysis_tools'));
addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

% analysis options
calc_frOpt = struct;
calc_frOpt.tbin = 0.02; % time bin for whole session rate matrix (in sec)
calc_frOpt.smoothSigma_time = 0.100; % gauss smoothing sigma for rate matrix (in sec)
buffer = 500; % how much to trim off end of trial

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
midresp_struct = struct; % a struct to store the mid-responsive neurons we pull out w/ gaussian selection

close all
for sIdx = 24:24
    new_structs = true; 
    if new_structs == true
        [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,calc_frOpt,sIdx,buffer);
    end
    
    % Perform PETH/sorting 
    nBins = 40;
    decVar_bins = linspace(0,2,nBins+1);
    opt.norm = "zscore";
    opt.trials = 'all';
    opt.suppressVis = false;
    dvar = "timesince";
    [sorted_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,opt); 
    nNeurons = size(sorted_peth,1);
    
%     sorted_peth = sorted_peth(1:5:end,:); % downsample for testing 
%     figure();colormap('jet')
%     imagesc(sorted_peth)
    % Subselect neurons with non-monotonic responsivity
    tic
    new_regressions = true;
    if new_regressions == true
        % dependent variable for fits
        x = 1:size(sorted_peth,2); 
        
        % linear fit datastructures
        peth_linFit = nan(size(sorted_peth));
        peth_linResid = nan(size(sorted_peth));
        linear_r2 = nan(size(sorted_peth,1),1);
        slope_fit = nan(size(sorted_peth,1),1);
        intercept_fit = nan(size(sorted_peth,1),1);
        
        % exponential fit datastructures
        peth_expFit = nan(size(sorted_peth));
        peth_expResid = nan(size(sorted_peth));
        exp_r2 = nan(size(sorted_peth,1),1);
        
        % gaussian fit datastructures
        peth_gaussFit = nan(size(sorted_peth));
        peth_gaussResid = nan(size(sorted_peth));
        gauss_r2 = nan(size(sorted_peth,1),1);
        mu_fit = nan(size(sorted_peth,1),1);
        
        % exp and gaussian models
        expModel = 'c + alpha * exp(beta * x)'; % note that beta can be negative
        gaussModel = 'alpha * exp(-(x - mu).^2 / (2*sigma.^2))';
        
        for neuron = 1:nNeurons
            % fit linear model
            [linear_fit,linear_gof] = fit(x',sorted_peth(neuron,:)','poly1');
            slope_fit(neuron) = linear_fit.p1;
            intercept_fit(neuron) = linear_fit.p2;
            linear_fit = linear_fit.p1 * x' + linear_fit.p2;
            linear_r2(neuron) = linear_gof.rsquare;
            peth_linResid(neuron,:) = sorted_peth(neuron,:) - linear_fit';
            peth_linFit(neuron,:) = linear_fit;  
            
            % fit exponential model 
            [exp_fit,exp_gof] = fit(x',sorted_peth(neuron,:)',expModel,'StartPoint',[0,0.1,0.1]); 
            exp_fit = exp_fit.c + exp_fit.alpha * exp(exp_fit.beta * x)';
            exp_r2(neuron) = exp_gof.rsquare; 
            peth_expResid(neuron,:) = sorted_peth(neuron,:) - exp_fit';
            peth_expFit(neuron,:) = exp_fit;
            
            % fit gaussian constrained s.t. mean between 250 and 1750 msec
            [gauss_fit,gauss_gof] = fit(x',sorted_peth(neuron,:)',gaussModel,'StartPoint',[1,20,20],'Lower',[.5,5,.5],'Upper',[20,35,10]);
            mu_fit(neuron) = gauss_fit.mu;
            gauss_fit = gauss_fit.alpha * exp(-(x - gauss_fit.mu).^2 / (2*gauss_fit.sigma.^2))';
            gauss_r2(neuron) = gauss_gof.rsquare;
            peth_gaussResid(neuron,:) = sorted_peth(neuron,:) - gauss_fit';
            peth_gaussFit(neuron,:) = gauss_fit;
        end
    end
    toc
    
    visualization = true;
    if visualization == true 
        label = "Time Since Reward (msec)";
        % visualize fits
        threePaneFitPlot(sorted_peth,peth_linResid,peth_linFit,decVar_bins,label,"Linear")
        threePaneFitPlot(sorted_peth,peth_gaussResid,peth_gaussFit,decVar_bins,label,"Gaussian") 
        threePaneFitPlot(sorted_peth,peth_expResid,peth_expFit,decVar_bins,label,"Exp")
        % Visualize improvement of fit with gaussian
        figure()  
        subplot(1,3,1)
        subtr = exp_r2 - linear_r2; 
        labels = subtr > 0;
        gscatter(1:numel(subtr),subtr,labels)
        title("Exp r^2 - linear r^2")
        hold on 
        yline(0,'--','linewidth',1.5)
        subplot(1,3,2)
        subtr = gauss_r2 - linear_r2;  
        labels = subtr > 0;
        gscatter(1:numel(subtr),subtr,labels)
        title("Gaussian r^2 - linear r^2")
        hold on 
        yline(0,'--','linewidth',1.5)
        subplot(1,3,3)
        gauss_exp_subtr = gauss_r2 - exp_r2; 
        gauss_exp_labels = gauss_exp_subtr > 0;
        gscatter(1:numel(gauss_exp_subtr),gauss_exp_subtr,gauss_exp_labels) 
        yline(0,'--','linewidth',1.5)
        title("Gaussian r^2 - Exp r^2")
        hFig=findall(0,'type','figure');
        hLeg=findobj(hFig(1,1),'type','legend');
        set(hLeg,'visible','off') 
    end
    
    % Pull off mid-responsive neurons 
    eps = .01; 
    mid_resp = find(gauss_exp_labels == 1 & mu_fit > (5+eps) & mu_fit < (35-eps));
    midresp_struct(sIdx).mid_resp_ix = mid_resp;
    midResp_peth = sorted_peth(mid_resp,:);  
    
    if visualization == true
        % Show mid-responsive neurons pulled off 
        figure();colormap('jet')
        imagesc(flipud(midResp_peth))
        colorbar()
        title("Activity of selected mid-responsive neurons") 
        xlabel("Time Since Reward (msec)") 
        xticks([1 10 20 30 40]) 
        xticklabels([0 500 1000 1500 2000]) 
        ylabel("Sorted mid-responsive neurons") 
    end
    
    % calculate ridge to background ratio for mid-responsive neurons 
    ridgeWidth = 2; % +/- 100 msec
    mean_r2b = calcR2B(midResp_peth,ridgeWidth,"zscore");
    
    % now repeat for shuffled data 
    nShuffles = 0; 
    shuffle_opt.norm = "zscore";
    shuffle_opt.trials = 'all'; 
    shuffle_opt.neurons = mid_resp; % pull off mid-responsive
    shuffle_opt.suppressVis = true; 
    shuffle_opt.shuffle = true;
    dvar = "timesince"; 
    mean_r2b_shuffled = nan(nShuffles,1); 
    tic 
    for shuffle = 1:nShuffles
        [shuffled_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,shuffle_opt);  
        mean_r2b_shuffled(shuffle) = calcR2B(shuffled_peth,ridgeWidth,"zscore");
    end 
    toc 
    
    % calculate a p value
    p = numel(find(mean_r2b_shuffled > mean_r2b)) / nShuffles;
end

%% Now repeat across sessions and mice 

outerPath = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/'; 
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
session_names = {}; 
session_counts = [];
r2b_xSession = []; 
p_xSession = [];
nMid_xSession = [];
proportionMid_xSession = [];  
new_regs = true;
if new_regs == true
    midresp_struct = struct;  
else
    load('./midresp_struct.mat') % struct from gaussian-exp regression  
end

session_counter = 1;

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
    
    for sIdx = 1:numel(sessions)
        session = erase(sessions{sIdx}(1:end-4),'_'); % latex thing 
        session_name = ['m' session(1:2) ' ' session(end-2) '/' session(end-1:end)]; 
        session_names = [session_names session_name]; % for later visualization
        
        % Make structs
        [FR_decVar_tmp,~] = genSeqStructs(paths,sessions,calc_frOpt,sIdx,buffer);
        
        % Create PETH/perform sort
        nBins = 40;
        decVar_bins = linspace(0,2,nBins+1);
        opt.norm = "zscore";
        opt.trials = 'all';
        opt.suppressVis = true;
        dvar = "timesince";
        [sorted_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,opt);
        nNeurons = size(sorted_peth,1);
        
        if new_regs == true 
            % NOTE THAT THIS ONLY WORKS IF WE DO LOOP OVER ALL SESSIONS
            % Subselect neurons with non-monotonic responsivity
            x = 1:size(sorted_peth,2); % dependent variable for fits
            
            % exponential fit datastructures
            peth_expFit = nan(size(sorted_peth));
            peth_expResid = nan(size(sorted_peth));
            exp_r2 = nan(size(sorted_peth,1),1);
            
            % gaussian fit datastructures
            peth_gaussFit = nan(size(sorted_peth));
            peth_gaussResid = nan(size(sorted_peth));
            gauss_r2 = nan(size(sorted_peth,1),1);
            mu_fit = nan(size(sorted_peth,1),1);
            
            % exp and gaussian models
            expModel = 'c + alpha * exp(beta * x)'; % note that beta can be negative
            gaussModel = 'alpha * exp(-(x - mu).^2 / (2*sigma.^2))';
            
            parfor neuron = 1:nNeurons
                % fit exponential model
                [exp_fit,exp_gof] = fit(x',sorted_peth(neuron,:)',expModel,'StartPoint',[0,0.1,0.1]);
                exp_fit = exp_fit.c + exp_fit.alpha * exp(exp_fit.beta * x)';
                exp_r2(neuron) = exp_gof.rsquare;
                peth_expResid(neuron,:) = sorted_peth(neuron,:) - exp_fit';
                peth_expFit(neuron,:) = exp_fit;
                
                % fit gaussian constrained s.t. mean between 250 and 1750 msec
                [gauss_fit,gauss_gof] = fit(x',sorted_peth(neuron,:)',gaussModel,'StartPoint',[1,20,20],'Lower',[.5,5,.5],'Upper',[20,35,10]);
                mu_fit(neuron) = gauss_fit.mu;
                gauss_fit = gauss_fit.alpha * exp(-(x - gauss_fit.mu).^2 / (2*gauss_fit.sigma.^2))';
                gauss_r2(neuron) = gauss_gof.rsquare;
                peth_gaussResid(neuron,:) = sorted_peth(neuron,:) - gauss_fit';
                peth_gaussFit(neuron,:) = gauss_fit;
            end
            fprintf("Completed fitting for Session %s \n",session_name)
            % Calculate non-monotonic improvement in fit
            gauss_exp_subtr = gauss_r2 - exp_r2;
            gauss_exp_labels = gauss_exp_subtr > 0;
            
            % Pull off mid-responsive neurons
            eps = .01;
            mid_resp = find(gauss_exp_labels == 1 & mu_fit > (5+eps) & mu_fit < (35-eps));
            midresp_struct(sIdx).mid_resp_ix = mid_resp;
        else
            mid_resp = midresp_struct(sIdx).mid_resp_ix;
        end
        
        midResp_peth = sorted_peth(mid_resp,:);
        
        % calculate ridge to background ratio for mid-responsive neurons
        ridgeWidth = 2; % +/- 100 msec
        mean_r2b = calcR2B(midResp_peth,ridgeWidth,"zscore");
        
        % now repeat for shuffled data
        nShuffles = 500;
        shuffle_opt.norm = "zscore";
        shuffle_opt.trials = 'all';
        shuffle_opt.neurons = mid_resp; % pull off mid-responsive
        shuffle_opt.suppressVis = true;
        shuffle_opt.shuffle = true;
        dvar = "timesince";
        mean_r2b_shuffled = nan(nShuffles,1);
        tic
        parfor shuffle = 1:nShuffles
            [shuffled_peth,~,~] = peakSortPETH(FR_decVar_tmp,dvar,decVar_bins,shuffle_opt);
            mean_r2b_shuffled(shuffle) = calcR2B(shuffled_peth,ridgeWidth,"zscore");
        end
        toc
        
        % calculate a p value
        p = numel(find(mean_r2b_shuffled > mean_r2b)) / nShuffles; 
        nMid = numel(mid_resp); 
        prop_mid = nMid / nNeurons;
        
        % record data
        p_xSession = [p_xSession p];
        r2b_xSession = [r2b_xSession mean_r2b];  
        nMid_xSession = [nMid_xSession nMid]; 
        proportionMid_xSession = [proportionMid_xSession prop_mid]; 
        
        fprintf("### Session %s Results ### \n",session_name)
        fprintf("mean r2b: %f \t r2b pValue: %f \n",mean_r2b,p) 
        fprintf("nMid: %i \t proportionMid: %f \n",nMid,prop_mid)
        
        fprintf("### Session %s Complete ### \n",session_name) 
        session_counter = session_counter + 1;
    end 
end

%% Visualize cross session results

close all
colors = cool(5);
% barplot r2b ratio
figure() 
b = bar(r2b_xSession); 
b.FaceColor = 'flat';
cumulative_session_counts = cumsum([0 session_counts]);
for i = 1:(numel(cumulative_session_counts)-1)
    mouse_sessions = cumulative_session_counts(i)+1:cumulative_session_counts(i+1);
    b.CData(mouse_sessions,:) = repmat(colors(i,:),[session_counts(i),1]); 
end 
hold on 
xticks(cumulative_session_counts+1)
xticklabels(session_names(min(cumulative_session_counts(1:end-1)+1,length(session_names))))  
ylabel("Mean Ridge to Background Ratio")  
title("Ridge to Background Ratio Across Sessions")

labels = []; % Just dont feel like figuring out how to do this a smart way
for i = 1:numel(nMid_xSession) 
    labels = [labels sprintf("%i Mid \n(%.2f %%)",nMid_xSession(i),100*proportionMid_xSession(i))];
end

% add nMid and propMid labels
x = b(1).XEndPoints;
y = b(1).YEndPoints;
text(x,y,labels,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom') 

% add stars to indicate significance 
sig3 = find(p_xSession < .001); 
sig2 = find(p_xSession < .01 & p_xSession > .001); 
sig1 = find(p_xSession < .05 & p_xSession > .01); 

text(sig3,y(sig3) + .05,"***",'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
text(sig2,y(sig2) + .05,"**",'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')
text(sig1,y(sig1) + .05,"*",'HorizontalAlignment','center',...
    'VerticalAlignment','bottom') 
ylim([1,1.75])


