function [] = create_glm_coupling(data_path,windows)

%%%%% BEFORE RUNNING THIS, CHECK THE BEST MODELS STUFF - CHANGE FOR CELL
%%%%% COUPLING

% data_path = '/Users/kiah/Dropbox/Rat data/open field sessions/Harlan 16_2017-06-20_18-17-20';
% data_path = 'C:\Users\khardcas\Dropbox\Rat data\open field
% sessions\Harlan 16_2017-06-20_18-17-20'
boxSize = 150;
if windows
    mat_files = dir([data_path,'\*.mat']);
else
    mat_files = dir([data_path,'/*.mat']);
end


mat_files = {mat_files.name}';
if windows
    split = strsplit(data_path,'\');
else
    split = strsplit(data_path,'/');
end

behav_file = split{end};
cell_files =  find(~cellfun(@isempty,strfind(mat_files,'TT')) == 1);
csc_files = find(~cellfun(@isempty,strfind(mat_files,'CSC')) == 1);
numCell = numel(cell_files);

% find the tetrodes that have cells
tetrodes_cells = nan(numel(cell_files),1);
for k = 1:numel(cell_files)
    cell_file_k = mat_files{cell_files(k)};
    tt = strfind(cell_file_k,'TT'); c = strfind(cell_file_k,'c');
    tetrodes_cells(k) = str2double(cell_file_k(tt+2:c-1));
end
tetrodes = unique(tetrodes_cells);

% get order of csc's
tetrodes_csc = nan(numel(csc_files),1);
for k = 1:numel(csc_files)
    csc_file_k = mat_files{csc_files(k)};
    tt = strfind(csc_file_k,'CSC'); c = strfind(cell_file_k,'mat');
    tetrodes_csc(k) = str2double(csc_file_k(tt+3:c-2));
end

% load behavioral data into a behav_data struct
% Timestamps = time stamps (in microseconds)
% angle = head direction angle (every 33 ms or so)
% x,y = x and y position of the animal (every 33 ms or so)

fprintf('Loading data\n');
if windows
    behav = load([data_path,'\',behav_file,'.mat']);
else
    behav = load([data_path,'/',behav_file,'.mat']);
end

%% Do some pre-processing on the behavioral data
% Fill in the non-detects with interpolated values
x = behav.x;
y = behav.y;
hd = mod(behav.angle,360)/180*pi;
hd_uw = unwrap(hd);
ts = behav.Timestamps;

non_detect = x == 0;
%%%% Note to self: want to investigate whether or not this is really
%%%% messing with stuff
x(non_detect) = interp1(ts(~non_detect),x(~non_detect),ts(non_detect),'linear');
y(non_detect) = interp1(ts(~non_detect),y(~non_detect),ts(non_detect),'linear');
hd_uw(non_detect) = interp1(ts(~non_detect),hd_uw(~non_detect),ts(non_detect),'linear');

% Re-scale x and y values to be the size of the box
x = (x - min(x))/range(x)*boxSize;
y = (y - min(y))/range(y)*boxSize;

% Compute the running speed
speed = nan(size(x));
for i = 1:numel(x)
    if i == 1
        speed(i) = sqrt((x(i) - x(i+1)).^2 + (y(i) - y(i+1)).^2)/(ts(i+1) - ts(i))*1e6;
    elseif i == numel(x)
        speed(i) = sqrt((x(i-1) - x(i)).^2 + (y(i-1) - y(i)).^2)/(ts(i) - ts(i-1))*1e6;
    else
        speed(i) = sqrt((x(i-1) - x(i+1)).^2 + (y(i-1) - y(i+1)).^2)/(ts(i+1) - ts(i-1))*1e6;
    end
end
speed = conv(speed,gausswin(5)/sum(gausswin(5)),'same');


%% Set the dt, and upsample behavioral data to match that dt

dt = 0.003; %3 ms
dt_micro = dt*1e6;
ts_us = ts(1):dt_micro:ts(numel(ts)); % upsampled timestamps

x_us = interp1(ts,x,ts_us);
y_us = interp1(ts,y,ts_us);
hd_us = mod(interp1(ts,hd_uw,ts_us),2*pi);
speed_us = interp1(ts,speed,ts_us);
maxSpeed = 100;
speed_us(speed_us > maxSpeed) = maxSpeed;

% load the cell data and make firing rate matrix
S = nan(numCell,numel(ts_us)); % num neurons x time
fprintf('Loading spikes\n');
for k = 1:numCell
    if windows
        load([data_path,'\',mat_files{cell_files(k)}])
    else
        load([data_path,'/',mat_files{cell_files(k)}])
    end
    spikes = hist(cellTS,ts_us);
    S(k,:) = spikes;
end

% from here on out, the ts_us will be in milliseconds, and start at 0
ts_us0 = (ts_us-ts_us(1))./1e3;

%% Compute the input matrix for the GLM: navigational variables
% These matrices will be the same for all neurons in the session

% POSITION
fprintf('Making position spline\n');
bin_p = 9; s = 0.5;
posVec = linspace(0,boxSize,bin_p); posVec(1) = -0.01;
[posgrid,~] = spline_2d(x_us,y_us,posVec,s);

A{1} = posgrid;

% HEAD DIRECTION
fprintf('Making head direction spline\n');
bin_h = 8;
hdVec = linspace(0,2*pi,bin_h+1); hdVec = hdVec(1:end-1);
s = 0.5;
[hdgrid] = spline_1d_circ(hd_us,hdVec,s);

A{2} = hdgrid;

% SPEED
fprintf('Making speed spline\n');
spdVec = [0:10:50 60:20:100]; spdVec(1) = -0.1;
s = 0.5;
[speedgrid,~] = spline_1d(speed_us,spdVec,s);

A{3} = speedgrid;

%% Compute the input matrix for neuron-specific variables: theta locking, spike history, and coupling

% set binning for theta and slow and fast gamma locking
bin_t = 8; s = 0.5;
thetaVec = linspace(0,2*pi,bin_t+1); thetaVec = thetaVec(1:end-1);
tetrode_num = 12; % number of tetrodes on this drive
thetagrid_all = cell(tetrode_num,1);
slowgammagrid_all = cell(tetrode_num,1);
fastgammagrid_all = cell(tetrode_num,1);
fprintf('Making theta and gamma phase splines \n');
for k = 1:tetrode_num
    
    % load the csc data FROM THAT TETRODE
    load([data_path,'/',mat_files{csc_files(tetrodes_csc == k)}])
    lfp_fs = 1/((lfp_ts(2) - lfp_ts(1))/1e6);
    index = knnsearch(lfp_ts,ts_us');
    
    % filter for theta
    [b,a] = butter(3,[6 10]/(lfp_fs/2)); %bandpass between 6 and 10 Hz
    filt_lfp = filtfilt(b,a,lfp);
    hilb_lfp = hilbert(filt_lfp); % compute hilbert transform
    phase_hifs = atan2(imag(hilb_lfp),real(hilb_lfp)); %inverse tangent (-pi to pi)
    ind = phase_hifs <0; phase_hifs(ind) = phase_hifs(ind)+2*pi; % from 0 to 2*pi
    phase_us = phase_hifs(index);
    thetagrid_all{k} = spline_1d_circ(phase_us,thetaVec,s);
    
    % SAME FOR SLOW GAMMA
    [b,a] = butter(3,[30 50]/(lfp_fs/2)); %bandpass between 6 and 10 Hz
    filt_lfp = filtfilt(b,a,lfp);
    hilb_lfp = hilbert(filt_lfp); % compute hilbert transform
    phase_hifs = atan2(imag(hilb_lfp),real(hilb_lfp)); %inverse tangent (-pi to pi)
    ind = phase_hifs <0; phase_hifs(ind) = phase_hifs(ind)+2*pi; % from 0 to 2*pi
    phase_us = phase_hifs(index);
    slowgammagrid_all{k} = spline_1d_circ(phase_us,thetaVec,s);
    
    % SAME FOR FAST GAMMA
    [b,a] = butter(3,[60 100]/(lfp_fs/2)); %bandpass between 6 and 10 Hz
    filt_lfp = filtfilt(b,a,lfp);
    hilb_lfp = hilbert(filt_lfp); % compute hilbert transform
    phase_hifs = atan2(imag(hilb_lfp),real(hilb_lfp)); %inverse tangent (-pi to pi)
    ind = phase_hifs <0; phase_hifs(ind) = phase_hifs(ind)+2*pi; % from 0 to 2*pi
    phase_us = phase_hifs(index);
    fastgammagrid_all{k} = spline_1d_circ(phase_us,thetaVec,s);
end

% set control points for spike history (or coupling)
spikeHistoryVec = [0:3 5 10 20 35 50]; %in bins
spikeCouplingVec = [-fliplr(spikeHistoryVec) spikeHistoryVec(2:end)];
shgrid_all = cell(numCell,1);
coupgrid_all = cell(numCell,1);

for n = 1:numCell
    % SPIKE HISTORY / COUPLING
    fprintf('Making spike history spline for cell # %d of %d\n', n, numCell);
    spiketrain = S(n,:); s = 0.5;
    [shgrid,~] = spline_spike_hist(spiketrain,spikeHistoryVec,s);
    shgrid_all{n} = shgrid;
    
    [coupgrid,~] = spline_spike_coupling(spiketrain,spikeCouplingVec,s);
    coupgrid_all{n} = coupgrid;
end

ctl_pts_all{1} = posVec;
ctl_pts_all{2} = hdVec;
ctl_pts_all{3} = spdVec;
ctl_pts_all{4} = thetaVec;
ctl_pts_all{5} = thetaVec;
ctl_pts_all{6} = thetaVec;
ctl_pts_all{7} = spikeHistoryVec;
ctl_pts_all{8} = spikeCouplingVec;


% calculate train and testing indices
% define the test and train indices
numFolds = 10;
T = numel(ts_us); numPts = 3*round(1/dt); % 10 seconds
numSeg = ceil(T/(numFolds*numPts));
oneSeg = ones(numPts,1)*(1:numFolds);
new_ind = repmat(oneSeg(:),numSeg,1);
new_ind = new_ind(1:T);

test_ind = cell(numFolds,1);
train_ind = cell(numFolds,1);
for k = 1:numFolds
    test_ind{k} = find(new_ind == k);
    train_ind{k} = setdiff(1:T,test_ind{k});
end

%% Now, run the glm for all cells in the session
for n = 9:numCell
    fprintf('Starting forward search for cell # %d\n', n);
    
    A{4} = thetagrid_all{tetrodes_cells(n)};
    
    A{5} = slowgammagrid_all{tetrodes_cells(n)};
    
    A{6} = fastgammagrid_all{tetrodes_cells(n)};
    
    A{7} = shgrid_all{n};
    
    A(6:end) = [];
    otherCells = find(tetrodes_cells ~= tetrodes_cells(n));
    for m = 1:numel(otherCells)
        A{m+7} = coupgrid_all{otherCells(m)};
    end
    %}
    spiketrain = S(n,:);
    
    tic
    [testFit_all, trainFit_all, bestModels, bestModel_testfit, parameters, pvals, final_pval] = forward_search_kfold(A,spiketrain,train_ind,test_ind);
    toc
    
    % need to re-set the bestModels stuff for the other cells... 
    param_mean = mean(parameters);
    plotfig = 0;
    [glm_tuning_curves] = plot_tuning(A,bestModels,param_mean,ctl_pts_all,s,plotfig,dt);
    
    
    pos_tuning = compute_2d_tuning_curve(x_us,y_us,spiketrain/dt,20,0,150);
    hd_tuning = compute_1d_tuning_curve(hd_us,spiketrain/dt,18,0,2*pi);
    spd_tuning = compute_1d_tuning_curve(speed_us,spiketrain/dt,18,0,100);
    
    % create an output structure
    output(n).testFit_all = testFit_all;
    output(n).trainFit_all = trainFit_all;
    output(n).bestModels = bestModels;
    output(n).bestModel_testfit = bestModel_testfit;
    output(n).parameters = parameters;
    output(n).glm_tuning = glm_tuning_curves;
    output(n).pvals = pvals;
    output(n).final_pval = final_pval;
    output(n).pos_tuning = pos_tuning;
    output(n).hd_tuning = hd_tuning;
    output(n).spd_tuning = spd_tuning;
    
    temp = strsplit(behav_file,'_');

    glm_name = strcat(temp{1},'_',temp{2:end},'_output_ver3_part2');
    save(glm_name,'output')
end


return




