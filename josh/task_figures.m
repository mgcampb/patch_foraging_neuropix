%% Make a figure showing how reward probability decreases over time on patch

tau = 8;  
t = 1:10;  
colors = [.7 .7 .7 ; .4 .4 .4 ; 0 0 0]; 
N0s = [.125 .25 .5]; 
figure(); hold on
for i = 1:numel(N0s) 
    iN0 = N0s(i);
    scatter(1:numel(t),iN0 * exp(-t / tau),100,'o','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',colors(i,:),'linewidth',1.5) 
%     plot(1:numel(t),iN0 * exp(-t / tau)
end  
xlim([0 10.5])
ylim([0 .6]) 
% title("Decaying Probability of Reward Delivery") 
legend(["\rho_{0} = .125","\rho_{0} = .25","\rho_{0} = .50"]) 
xlabel("Time on Patch (sec)") 
ylabel("Probability of Reward Delivery")  
ax = gca; 
ax.XAxis.FontSize = 15;
ax.YAxis.FontSize = 15; 
ax.Legend.FontSize = 16; 
set(gca,'fontsize',16)

%%  
t = 1:.1:10;
N0 = .25;
figure();hold on
for iRewsize = [1 2 4] 
    plot(1:numel(t)+1,gradient(cumsum([iRewsize iRewsize * iN0 * exp(-t*4 / tau)])))
end

%% experiment w/ plotting reward rate over different values of test policy
policies = [1 2 0 4]; 
ITI_penalty = 10; 
timecost = 4; 
nTrials = 100000; 
rewsizes = [1 2 4]; 
test_policies = 1:.5:10;  
test_ITI_penalties = 1:15; 
reward_rate = zeros(numel(test_ITI_penalties),length(test_policies));
% figure(); hold on
for i_ITI_penalty = 1:numel(test_ITI_penalties)
    ITI_penalty = test_ITI_penalties(i_ITI_penalty); 
    for i_test_policy = 1:numel(test_policies)
        policies = test_policies(i_test_policy) * [1 2 0 4];
        total_reward = 0;
        total_timesteps = 0;
        for iTrial = 1:nTrials
            this_rewsize = rewsizes(randi(length(rewsizes),1));
            this_N0 = rewsizes(randi(length(rewsizes),1));
            for iTstep = 1:policies(this_rewsize)
                if iTstep == 1
                    total_reward = total_reward + this_rewsize;
                elseif rand() < this_N0 * exp(-iTstep / tau)
                    total_reward = total_reward + this_rewsize;
                else
                    total_reward = total_reward - timecost;
                end
                total_timesteps = total_timesteps + 1;
            end
            total_reward= total_reward - ITI_penalty;
        end
        reward_rate(i_ITI_penalty,i_test_policy) = total_reward / total_timesteps;
    end
end

%% Visualize 
colors = copper(numel(test_ITI_penalties)); 
figure();hold on
for i_ITI_penalty = 1:numel(test_ITI_penalties) 
    plot(reward_rate(i_ITI_penalty,:),'color',colors(i_ITI_penalty,:),'linewidth',1.5)
end
[max_rewrate,policy_max] = max(reward_rate'); 
plot(policy_max,max_rewrate,'linewidth',1.5,'color','k')