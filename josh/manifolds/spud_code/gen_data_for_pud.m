%% Generate data in matlab to run through spud in python 
spud_path = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/manifolds/spud_code';

%% Low or high embedding of 1D latent variable

n_neurons = 50; 
phi_highD = linspace(0,10,n_neurons); 
phi_lowD = 10 + zeros(n_neurons,1); 
G = 5;
sigma_low = 20;
sigma_high = 1;
n_trials = 10;
S = repmat((1:.01:10)',[1 n_trials])';
X_lowEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_lowD,G,sigma_low,S); 
X_highEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S); 

figure()
subplot(1,2,1)
imagesc(flipud(mean(cat(3,X_lowEmbedding),3)))
subplot(1,2,2)
imagesc(flipud(mean(cat(3,X_highEmbedding),3)))

synth1d_path = fullfile(spud_path,'/data/1D_synth.mat');
save(synth1d_path,'X_lowEmbedding','X_highEmbedding','S');

%% High-D embedding of a 2D latent variable 

n_neurons = 50; 
phi_var1 = linspace(0,10,n_neurons); 
phi_var2 = zeros(n_neurons,1); 
G = 5;
sigma_low = 20;
sigma_high = 1;
n_trials = 10;
S1 = repmat((1:.1:10)',[1 n_trials])';
S2 = repmat(sin((1:.1:10) * 3)',[1 n_trials])';
X_S1 = gen_manifold_embedding_dataset(n_neurons,phi_highD - 3,G,sigma_high,S1); 
X_S2 = gen_manifold_embedding_dataset(n_neurons,phi_highD - 3,G,sigma_high,S2); 

X_highEmbedding_2D = X_S1 + X_S2; 

figure()
subplot(2,1,1)
imagesc(flipud(mean(cat(3,X_highEmbedding_2D),3)))
subplot(2,1,2);hold on
plot(S1(:),'linewidth',2)
plot(S2(:),'linewidth',2)
legend(["S1","S2"])

synth1d_path = fullfile(spud_path,'/data/2D_synth.mat');
save(synth1d_path,'X_highEmbedding_2D','S1','S2');

%% High vs low d embedding of sin wave
n_neurons = 50; 
phi_highD = linspace(-1,1,n_neurons); 
phi_lowD = zeros(n_neurons,1); 
G = 5;
sigma_low = 1;
sigma_high = 1;
n_trials = 10;
S = repmat(sin((1:.1:10)' / 10 * 2 * pi * 2),[1 n_trials])';
X_lowEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_lowD,G,sigma_low,S); 
X_highEmbedding = gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S); 

figure()
subplot(1,2,1)
imagesc(flipud(mean(cat(3,X_lowEmbedding),3)))
subplot(1,2,2)
imagesc(flipud(mean(cat(3,X_highEmbedding),3)))

synth1d_sin_path = fullfile(spud_path,'/data/1D_synth_sinWave.mat');
save(synth1d_sin_path,'X_lowEmbedding','X_highEmbedding','S');

%% High-D embedding of a 3D latent variable 
n_neurons = 50; 
phi_var1 = linspace(0,10,n_neurons); 
phi_var2 = zeros(n_neurons,1); 
G = 5;
sigma_low = 20;
sigma_high = 1;
n_trials = 10;
S1 = repmat((1:.1:10)',[1 n_trials])';
S2 = repmat(sin((1:.1:10) * 3)',[1 n_trials])';
S3 = repmat(cos((1:.1:10) * 3)',[1 n_trials])';
X_S1 = gen_manifold_embedding_dataset(n_neurons,phi_highD - 3,G,sigma_high,S1); 
X_S2 = gen_manifold_embedding_dataset(n_neurons,phi_highD - 3,G,sigma_high,S2); 
X_S3 = gen_manifold_embedding_dataset(n_neurons,phi_highD - 6,G,sigma_high,S3); 


X_highEmbedding_3D = X_S1 + X_S2 + X_S3; 

figure()
subplot(2,1,1)
imagesc(flipud(mean(cat(3,X_highEmbedding_3D),3)))
subplot(2,1,2);hold on
plot(S1(:),'linewidth',2)
plot(S2(:),'linewidth',2)
plot(S3(:),'linewidth',2)
legend(["S1","S2","S3"])

synth1d_path = fullfile(spud_path,'/data/3D_synth.mat');
save(synth1d_path,'X_highEmbedding_3D','S1','S2','S3');

%% Plot manifold
mice = [1 2 4 5];
mice_names = ["m75","m76","m79","m80"]; 
figure()
for m_ix = 1:numel(mice)
    mIdx = mice(m_ix); 
    subplot(1,numel(mice),m_ix)
    scatter3(pca_full(1,mouseID_full == mIdx),pca_full(2,mouseID_full == mIdx),pca_full(3,mouseID_full == mIdx),.2)
    
end
%% Mouse80 RXNil trials 
% Note that need to run rxnil_umap to get these variables!
m80_zscored_prt = pooled_zscored_prt_full(mouseID_full == 5);
m80_time = time_full(mouseID_full == 5);
m80_tts = trial_types_full(mouseID_full == 5);
m80_pca = pca_full(:,mouseID_full == 5);
m80_fr = nanPadded_fr_mat_full(:,mouseID_full == 5); 
m80_fr(all(isnan(m80_fr),2),:) = []; % get rid of non-m80 FR
% save
m80_path = fullfile(spud_path,'/data/m80.mat');
save(m80_path,'m80_zscored_prt','m80_time','m80_tts','m80_pca','m80_fr');

%% All RXNil data
% save
glmPFC_RXNil_mean_full = roi_pooled_RXNil_mean_full;
rxnil_path = fullfile(spud_path,'/data/rxnil_data.mat');
save(rxnil_path,'tt_starts','glmPFC_RXNil_mean_full');

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