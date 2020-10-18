%% Fit CP Tensor Decomposition
%% Format NP data into tensor format 

session = '80_20200316.mat';

if ~exist('FR','var')
    load(session);
    getTimestamps_goodunits;
    dat = load(session);
    [FR,ts] = calcFRVsTime(goodIDs,dat); % fit gaussian kernel to good units
    ts = ts .* 1000 ; % put ts in ms units
    % behavioral timepoints
    patchStop_ts = round(patchStop_ts .* 1000);
    patchLeave_ts = round(patchLeave_ts .* 1000);
    rewTimes_ts = round(rew_ts .* 1000);
end

event_ts = [patchStop_ts ; patchLeave_ts ; rewTimes_ts];

time_pre = 750; % look at x seconds before event
time_post = 750; % look at x seconds before event

event_ts = [event_ts - time_pre , event_ts + time_post];

resolution = 20;
N = length(goodIDs);  % neurons
T = round((time_pre + time_post)/resolution); % resolution in ms
K = length(event_ts);  % trials

event_idx = interp1(ts,1:length(ts),event_ts);

% change this into two columns!
event_idx = zeros(length(event_ts),2);
event_idx(:,1) = interp1(ts,1:length(ts),event_ts(:,1));
event_idx(:,2) = interp1(ts,1:length(ts),event_ts(:,2));

data = zeros(N,T,K);

for event = 1:length(event_ts)
    pre_ts = max(1,round(event_idx(event,1)));
    post_ts = min(round(event_idx(event,2)),size(FR,2));
    display([pre_ts,post_ts]);
    data(:,:,event) = FR(:,pre_ts:post_ts-1); 
end

imagesc(permute(data(1,:,:),[2,3,1]));

%% Perform TCA
% these commands require that you download Sandia Labs' tensor toolbox:
% code from ahwilla github
% convert data to a tensor object
data = tensor(data);

ranks = [1:50];
n_fits = 1; % number of initializations to use per rank
n_ranks = length(ranks);

err = zeros(n_fits,n_ranks);

figure(); hold on
for r = 1:length(ranks)
    R = ranks(r);
    for n = 1:n_fits
        % fit model
        est_factors = cp_als(tensor(data),R);

        % store error
        err(n,r) = norm(full(est_factors) - data)/norm(data);
    end
    plot(r + .01 .* randn(n_fits,1), err(:,r), 'ob')
end

ylim([0 1.0])
ylabel('model error')

%% Visualize tca for interesting rank 

est_factors = cp_als(tensor(data),10);

% visualize fit for first several fits
    % score aligns the cp decompositions
%         [sc, est_factors] = score(est_factors, true_factors);

% plot the estimated factors
viz_ktensor(est_factors, ... 
    'Plottype', {'bar', 'line', 'scatter'}, ...
    'Modetitles', {'neurons', 'time', 'trials'})
set(gcf, 'Name', ['estimated factors - fit #' num2str(n)])
