%% Demo of linear decoding theory of high embedding dimensionality computational advantage
% 3 tiling neurons and a ramp

n_neurons = 3; 
phi_highD = [.5 1 1.5]; 
G = 1;
sigma_high = .1;
sigma_low = .5;
phi_lowD = 2;
n_trials = 1;
S = (0:.01:2)';
seq = squeeze(gen_manifold_embedding_dataset(n_neurons,phi_highD,G,sigma_high,S)); 
ramp = squeeze(gen_manifold_embedding_dataset(1,phi_lowD,2 * G,sigma_low,S))'; 

population = [ramp ; seq];

%% Visualize
gmm_colors = [68 119 170; 238 102 119; 34 136 51; 204 187 68; 102 204 238]/255;
offset = 1.1;
figure();hold on
plot(S,ramp,'linewidth',4,'color',gmm_colors(1,:))
plot(S,seq' - offset * [3 2 1],'linewidth',4,'color',gmm_colors(2,:))
yticks([])
ylabel("Firing Rate")
xlabel("Time Since Rew / Decision Variable")
set(gca,'fontsize',18)
% xline(phi_highD(1),'--','linewidth',2);xline(phi_highD(2),'--','linewidth',2);xline(phi_highD(3),'--','linewidth',2)


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
%             R(:,i_timestep,i_trial) = poissrnd(R(:,i_timestep,i_trial)); % add poisson noise
        end
    end
end
