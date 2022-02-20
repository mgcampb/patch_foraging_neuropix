%% Dimensionality demonstration 
%  Show how we can use either high-D or low-D to get representation of time
%  since reward 

% eye() [everyone diff] vs 1D code of ramp

% n_timepoints = 15; 
% n_neurons = n_timepoints;
% highD_code = eye(n_timepoints) + .1 * randn(n_timepoints); 
% lowD_code = repmat((1:n_timepoints),[n_neurons,1]) + randn(n_timepoints); 

n_neurons = 50; 
phi_highD = linspace(0,10,n_neurons); 
phi_lowD = 10 + zeros(n_neurons,1); 
G = 10;
sigma_low = 20;
sigma_high = 1;
n_trials = 20;
S = repmat((1:.01:10)',[1 n_trials])';
X_lowEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_lowD,G,sigma_low,S); 
X_highEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S); 

lowD_code = mean(cat(3,X_lowEmbedding),3);
highD_code = mean(cat(3,X_highEmbedding),3);
n_timepoints = size(lowD_code,2);

figure()
subplot(1,2,1)
imagesc(flipud(mean(cat(3,X_lowEmbedding),3)))
subplot(1,2,2)
imagesc(flipud(mean(cat(3,X_highEmbedding),3)))

% perform PCA
[coeff_lowD,score_lowD,~,~,expl_lowD] = pca(zscore(lowD_code')); 
[coeff_highD,score_highD,~,~,expl_highD] = pca(zscore(highD_code'));

% find peaks
[~,peak_lowD] = max(lowD_code,[],2); 
[~,peak_highD] = max(highD_code,[],2); 
shannonH_lowD = calc_shannonH(peak_lowD,1:25:n_timepoints);
shannonH_highD = calc_shannonH(peak_highD,1:25:n_timepoints);

figure() 
subplot(2,2,1)
imagesc(lowD_code)
xlabel("Time (A.U.)",'fontsize',14)
ylabel("Neurons",'fontsize',14)
title("Low Embedding Dimensionality",'fontsize',14)
set(gca,'fontsize',16)
subplot(2,2,2) 
imagesc(highD_code)
xlabel("Time (A.U.)",'fontsize',14)
ylabel("Neurons",'fontsize',14)
title("High Embedding Dimensionality",'fontsize',14)

set(gca,'fontsize',16)

% colormap(flipud(cbrewer('div','Spectral',100)))
subplot(2,2,3) ; hold on
histogram(peak_lowD,1:25:n_timepoints);
histogram(peak_highD,1:25:n_timepoints); 
xlabel("Time-bin (A.U.)",'fontsize',14)
ylabel("Neurons with Peak Time",'fontsize',14)
legend([sprintf("Low-D Embedding: H = %.3f",shannonH_lowD),sprintf("High-D Embedding: H = %.3f",shannonH_highD)],'fontsize',14)

set(gca,'fontsize',16)

subplot(2,2,4) ; hold on 
plot(cumsum(expl_lowD(1:10)),'linewidth',3)
plot(cumsum(expl_highD(1:10)),'linewidth',3)
ylim([0,100])
ylabel("Cumulative Variance Explained",'fontsize',14)
xlabel("Principal Component",'fontsize',14)
legend(["Low-D Embedding","High-D Embedding"],'fontsize',14)
yticks([0,100])

set(gca,'fontsize',16)

%% Now intrinsic dimensionality

n_neurons = 50; 
phi_highD = linspace(0,10,n_neurons); 
phi_lowD = 10 + zeros(n_neurons,1); 
G = 10;
sigma_low = 20;
sigma_high = 1;
n_trials = 20;
S1 = repmat((1:.01:10)',[1 n_trials])';
S2 = repmat((1:.02:10)',[1 n_trials])';
X_highEmbedding_S1 = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S1); 
X_highEmbedding_S2 = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S2);  

X_S1 = mean(cat(3,X_highEmbedding_S1),3);
X_S2 = mean(cat(3,X_highEmbedding_S2),3);

X_lowIntrinsic = [X_S2 X_S1];

tt_gain = 2;
tt_axis = 2 * randn(n_neurons,1); 
tt_embedding = [(-1 + zeros(length((1:.02:10)'),1)) ; (1 + zeros(length((1:.01:10)'),1))];
X_tt_embedding = tt_axis * tt_embedding';

tt_1_len = length((1:.02:10)');
tt_2_len = length((1:.01:10)');

X_highIntrinsic = [X_S2 X_S1] + X_tt_embedding;

% perform PCA
[~,score_lowIntrinsic] = pca(zscore(X_lowIntrinsic')); 
[~,score_highIntrinsic] = pca(zscore(X_highIntrinsic'));

RdBu9 = cbrewer('div','RdBu',9);
tt1_color = RdBu9(7,:);
tt1_tickColor = RdBu9(8,:);
tt2_color = RdBu9(3,:);
tt2_tickColor = RdBu9(2,:);
%%
figure() 
subplot(2,2,1)
imagesc(X_lowIntrinsic);hold on
xlabel("Time (A.U.)",'fontsize',14)
ylabel("Neurons",'fontsize',14)
title(sprintf("High Embedding Dimensionality \n Intrinsic Dimensionality = 1"))
plot(1:tt_1_len,1 + n_neurons + zeros(tt_1_len,1),'color',tt1_tickColor,'linewidth',8)
plot((tt_1_len+1):(tt_1_len + tt_2_len+1),1 + n_neurons + zeros(tt_2_len+1,1),'color',tt2_tickColor,'linewidth',8)
ylim([1 n_neurons + 1])
xticks([tt_1_len / 2 tt_1_len + (tt_2_len / 2)])
xticklabels(["Trial Type 1" "Trial Type 2"])
set(gca,'fontsize',15)
subplot(2,2,2) 
imagesc(X_highIntrinsic);hold on
xlabel("Time (A.U.)",'fontsize',14)
ylabel("Neurons",'fontsize',14)
title(sprintf("High Embedding Dimensionality \n Intrinsic Dimensionality = 2"))
plot(1:tt_1_len,1 + n_neurons + zeros(tt_1_len,1),'color',tt1_tickColor,'linewidth',8)
plot((tt_1_len+1):(tt_1_len + tt_2_len+1),1 + n_neurons + zeros(tt_2_len+1,1),'color',tt2_tickColor,'linewidth',8)
ylim([1 n_neurons + 1])
xticks([tt_1_len / 2 tt_1_len + (tt_2_len / 2)])
xticklabels(["Trial Type 1" "Trial Type 2"])
set(gca,'fontsize',15)

subplot(2,2,3);hold on
plot3(score_lowIntrinsic(1:tt_1_len,1),score_lowIntrinsic(1:tt_1_len,2),score_lowIntrinsic(1:tt_1_len,3),'linewidth',3,'color',tt1_color)
plot3(score_lowIntrinsic(tt_1_len+1:end,1),score_lowIntrinsic(tt_1_len+1:end,2),score_lowIntrinsic(tt_1_len+1:end,3),'linewidth',3,'color',tt2_color)
xticks([])

tick_interval = 50;  

% add time ticks
time_ticks = 1:tick_interval:tt_1_len;
plot3(score_lowIntrinsic(time_ticks,1),score_lowIntrinsic(time_ticks,2),score_lowIntrinsic(time_ticks,3), ... 
      'ko', 'markerSize', 8, 'markerFaceColor',tt1_tickColor); 
  
time_ticks = (tt_1_len+1):tick_interval:(tt_1_len + tt_2_len);
plot3(score_lowIntrinsic(time_ticks,1),score_lowIntrinsic(time_ticks,2),score_lowIntrinsic(time_ticks,3), ... 
      'ko', 'markerSize', 8, 'markerFaceColor',tt2_tickColor); 
grid()
view(-135,30)
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
title(sprintf("High Embedding Dimensionality \n Intrinsic Dimensionality = 1"))

set(gca,'fontsize',16)

subplot(2,2,4);hold on

plot3(score_highIntrinsic(1:tt_1_len,1),score_highIntrinsic(1:tt_1_len,2),score_highIntrinsic(1:tt_1_len,3),'linewidth',3,'color',tt1_color)
plot3(score_highIntrinsic(tt_1_len+1:end,1),score_highIntrinsic(tt_1_len+1:end,2),score_highIntrinsic(tt_1_len+1:end,3),'linewidth',3,'color',tt2_color)
title(sprintf("High Embedding Dimensionality \n Intrinsic Dimensionality = 2"))
time_ticks = 1:tick_interval:tt_1_len;
plot3(score_highIntrinsic(time_ticks,1),score_highIntrinsic(time_ticks,2),score_highIntrinsic(time_ticks,3), ... 
      'ko', 'markerSize', 8, 'markerFaceColor',tt1_tickColor); 
  
time_ticks = (tt_1_len+1):tick_interval:(tt_1_len + tt_2_len);
plot3(score_highIntrinsic(time_ticks,1),score_highIntrinsic(time_ticks,2),score_highIntrinsic(time_ticks,3), ... 
      'ko', 'markerSize', 8, 'markerFaceColor',tt2_tickColor); 
grid()
view(-135,30)
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
set(gca,'fontsize',16)


%% functions
function R = gen_manifold_embedding_dataset(n_neurons,phi,G,sigma,S) 
    % Generate neural population response from neurons w/ gaussian
    % responsivity
    % ----
    % Arguments
    % phi: stimulus preference per neuron [n_neurons x 1]
    % G: gain
    % S: 1D stimulus [n_trials x t_len]
    % ----
    % Returns
    % R: [n_neurons x t_len x n_trials]
    
    [n_trials,t_len] = size(S); 
    
    R = nan(n_neurons,t_len,n_trials); 
    
    % iterate over trials
    for i_trial = 1:n_trials
        for i_timestep = 1:t_len
            R(:,i_timestep,i_trial) = G * exp(-((S(i_trial,i_timestep) - phi) / (sqrt(2.0 * sigma))).^2);
            R(:,i_timestep,i_trial) = poissrnd(R(:,i_timestep,i_trial)); % add poisson noise
        end
    end
end