%% perform jPCA on synthetic data to show where rotations will/won't appear

% first just look at within reward size
path = 'dv2.mat';
load(path)
close all;
jPCA_data = struct;

large_rew = double(fr(:,1601:end)); % take out just the large reward trials
small_rew = double(fr(:,1:1600));

times = 1:400;

for i = 1:4
    jPCA_data(i).A = zscore(large_rew(:,1 + ((i-1) * 400):(400 * i)),[],2)';
    jPCA_data(i).times = times;
end

% now perform jPCA
jPCA_params.numPCs = 6;  % default anyway, but best to be specific
jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
[Projection, Summary] = jPCA(jPCA_data, times, jPCA_params);

plot_params.substRawPCs = false;
plot_params.colors = {[1 0 1],[.75 .25 .75],[.25 .5 1],[0 1 1]};
phaseSpace(Projection, Summary,plot_params);  % makes the plot

conditions = 1:4;
labels = {"400","440","404","444"};
PETH_PCA_jPCAGrid(conditions,jPCA_data,Projection,labels,"Synthetic DV2 Ramping")

%% just sine and cosine baby
close all;
fr_rot = ([sin(linspace(0,1.5 * pi,50)); cos(linspace(0,1.5 * pi,50))] + .2 * rand(2,50))';

[coeffs,score,~,~,expl] = pca(zscore(fr_rot));

figure() 
subplot(1,2,1)
imagesc(fr_rot');colormap('jet')
title("Synthetic Rotations PETH") 
xlabel("Time")
ylabel("Neuron")
subplot(2,2,2)
plot(score,'linewidth',2)
title("PCs over time")
subplot(2,2,4)
plot(score(:,1),score(:,2),'linewidth',2)
title("PCA Trajectory")

jPCA_data = struct;
jPCA_data(1).A = repmat(fr_rot,[1,3]);
[~,index] = max(jPCA_data(1).A);
[~,index_sort] = sort(index);
jPCA_data(1).A = jPCA_data(1).A(:,index_sort);
jPCA_data(1).times = 1:size(fr_rot,1);
jPCA_data(2).A = repmat(fr_rot,[1,3]);
jPCA_data(2).times = 1:size(fr_rot,1);

% now perform jPCA
jPCA_params.numPCs = 4; 
jPCA_params.meanSubtract = false; % looks better w/o mean subtraction
[Projection, Summary] = jPCA(jPCA_data, times, jPCA_params);

PETH_PCA_jPCAGrid(1,jPCA_data,Projection,"","3 sine 3 cosine")

%% Now show churchland data 

% loading data
load exampleData
jPCA_params.softenNorm = 5;  % how each neuron's rate is normized, see below
jPCA_params.suppressBWrosettes = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressHistograms = true;  % these are useful sanity plots, but lets ignore them for now

times = -50:10:150;  % 50 ms before 'neural movement onset' until 150 ms after
jPCA_params.numPCs = 6;  % default anyway, but best to be specific
jPCA_params.meanSubtract = true;
jPCA_params.substRawPCs = false;

[~,ix] = max( zscore(Data(1).A,[],1),[],1);
[~,index_sort] = sort(ix);
Data(1).A = Data(1).A(:,index_sort);
Data(50).A = Data(50).A(:,index_sort);
Data(75).A = Data(75).A(:,index_sort);

[Projection, Summary] = jPCA(Data, times, jPCA_params);

phaseSpace(Projection, Summary,jPCA_params);  % makes the plot

conditions = [1 50 75];

labels = {}; 
labels{1} = "Condition 1";
labels{50} = "Condition 50";
labels{75} = "Condition 75";

PETH_PCA_jPCAGrid(conditions,Data,Projection,labels,"Churchland 2012")

%% Now make synthetic data that actually works
close all
% start w/ just sequential activation of gaussians
x = .1:.1:10;
peaks = linspace(0,10,50);
y = normpdf(repmat(x,[length(peaks),1])',peaks,ones(1,length(peaks)));
y2 = repmat(y(1:50,:),[2,1]); % a repeated sequence!

y = y + .05 * rand(size(y));
y2 = y2 + .05 * rand(size(y2));

% prep for jPCA
times = 1:size(y,1);
times = 1:100;
jPCA_data(1).A = y; 
jPCA_data(1).times = times;
jPCA_data(2).A = y2; 
jPCA_data(2).times = times;

% perform jPCA
jPCA_params.suppressBWrosettes = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressHistograms = true;  % these are useful sanity plots, but lets ignore them for now

jPCA_params.numPCs = 4; 
jPCA_params.meanSubtract = false;
jPCA_params.substRawPCs = false;
[Projection, Summary] = jPCA(jPCA_data, 1:100, jPCA_params);
PETH_PCA_jPCAGrid(1:2,jPCA_data,Projection,{"1","2"},"Gaussian sequence")

%% Now make synthetic data that works and is something like a decision variable and reward
close all
% start with ramping population
nRamp = 25;
x = .1:.1:10;
peaks = 9 * ones(nRamp,1);
rampers = normpdf(repmat(x,[nRamp,1]),peaks,ones(nRamp,1))';
% now stretch to varying degrees
gain_diverse_rampers = normpdf(repmat(x,[nRamp,1]),peaks,ones(nRamp,1))';
gain_factor = randi([10,90],[1,nRamp]);
for i = 1:nRamp
    gain_diverse_rampers(1:90,i) = imresize(gain_diverse_rampers(gain_factor(i):90,i),[90 1],'nearest');
end
[~,ix] = sort(gain_factor);
gain_diverse_rampers = fliplr(gain_diverse_rampers(:,ix)); % sort
% reward-responsive
nRew = 25;
peaks = 1 * ones(nRew,1);
reward_responsive = normpdf(repmat(x,[nRew,1]),peaks,ones(nRew,1))';
% add latency
peaks_latent = 5 * rand([nRew,1]);
latent_reward_responsive = normpdf(repmat(x,[nRew,1]),peaks_latent,ones(nRew,1))';
[~,ix] = sort(peaks_latent);
latent_reward_responsive = latent_reward_responsive(:,ix);

noise_level = 0.05;
% homogenous rampers + rew population
homog_rampsRews = [reward_responsive rampers]; % combine
homog_rampsRews2 = repmat(homog_rampsRews(1:50,:),[2,1]); % a repeated sequence!
% add noise
homog_rampsRews = homog_rampsRews + noise_level * rand(size(homog_rampsRews));
homog_rampsRews2 = homog_rampsRews2 + noise_level * rand(size(homog_rampsRews2));

% non homogenous rampers + rew responsive population
nonhomog_rampsRews = [reward_responsive gain_diverse_rampers]; % combine
nonhomog_rampsRews2 = repmat(nonhomog_rampsRews(1:50,:),[2,1]); % a repeated sequence!
% noise
nonhomog_rampsRews = nonhomog_rampsRews + noise_level * rand(size(nonhomog_rampsRews));
nonhomog_rampsRews2 = nonhomog_rampsRews2 + noise_level * rand(size(nonhomog_rampsRews2));

% homogenous rampers + latent responsive rew responsive population
homog_rampsLatentRews = [latent_reward_responsive rampers]; % combine
homog_rampsLatentRews2 = repmat(homog_rampsLatentRews(1:50,:),[2,1]); % a repeated sequence!
% noise
homog_rampsLatentRews = homog_rampsLatentRews + noise_level * rand(size(homog_rampsLatentRews));
homog_rampsLatentRews2 = homog_rampsLatentRews2 + noise_level * rand(size(homog_rampsLatentRews2));

% Last, non-homogenous rampers + latent responsive rew responsive population
nonhomog_rampsLatentRews = [latent_reward_responsive gain_diverse_rampers]; % combine
nonhomog_rampsLatentRews2 = repmat(nonhomog_rampsLatentRews(1:50,:),[2,1]); % a repeated sequence!
% noise
nonhomog_rampsLatentRews = nonhomog_rampsLatentRews + noise_level * rand(size(nonhomog_rampsLatentRews));
nonhomog_rampsLatentRews2 = nonhomog_rampsLatentRews2 + noise_level * rand(size(nonhomog_rampsLatentRews2));

% jPCA for homogenous population
times = 1:100:size(homog_rampsRews,1) * 100;
jPCA_data(1).A = homog_rampsRews; 
jPCA_data(1).times = times;
jPCA_data(2).A = homog_rampsRews2; 
jPCA_data(2).times = times;

% perform jPCA
jPCA_params.suppressBWrosettes = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressHistograms = true;  % these are useful sanity plots, but lets ignore them for now
jPCA_params.suppressText = false;
jPCA_params.numPCs = 4; 
jPCA_params.meanSubtract = false;
jPCA_params.substRawPCs = false;

% basic reward responsivity and ramping
[Projection, Summary] = jPCA(jPCA_data,times, jPCA_params);
PETH_PCA_jPCAGrid(1:2,jPCA_data,Projection,{"1","2"},"Homogenous Ramp and Reward")

% non-homogenous ramping
jPCA_data(1).A = nonhomog_rampsRews; 
jPCA_data(2).A = nonhomog_rampsRews2; 
% perform jPCA
[Projection, Summary] = jPCA(jPCA_data,times, jPCA_params);
PETH_PCA_jPCAGrid(1:2,jPCA_data,Projection,{"1","2"},"Non-homogenous Ramp and Reward")

% latent reward response 
jPCA_data(1).A = homog_rampsLatentRews; 
jPCA_data(2).A = homog_rampsLatentRews2; 
% perform jPCA
[Projection, Summary] = jPCA(jPCA_data,times, jPCA_params);
PETH_PCA_jPCAGrid(1:2,jPCA_data,Projection,{"1","2"},"Homogenous Ramp and Reward Latency")

% latent reward response 
jPCA_data(1).A = nonhomog_rampsLatentRews; 
jPCA_data(2).A = nonhomog_rampsLatentRews2; 
% perform jPCA
[Projection, Summary] = jPCA(jPCA_data,times, jPCA_params);
PETH_PCA_jPCAGrid(1:2,jPCA_data,Projection,{"1","2"},"Non-Homogenous Ramp and Reward Latency")

