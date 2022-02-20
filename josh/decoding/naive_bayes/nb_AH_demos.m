%% demo the results of nb analyses for alternative hypotheses

% Alternative hypotheses: 
% a/b:
% could be integrate to thresh
% Could be using different mechanisms for trials of different reward sizes --> decoders would not generalize
% Could just be keeping track of variables--> all lines would look the same and curves would not converge at end
% 
% C:
% could be integrator to thresh
% Short PRT trials could just have collapse of information --> short PRT would just go to random 
% Time since reward representation could be independent from behavior --> should collapse (ala cluster 2)
% 
% D: 
% could be anti-correlated (compensatory? / information preserving?)
% could be uncorrelated (acting disjointly)

cool3 = cool(3); 

%% a/b: 

figure()
subplot(2,3,1);hold on
plot([0,2],[1,1] - .025,'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,2],[1,1] + 0.025,'color',cool3(3,:),'linewidth',3)
title(sprintf("Different Representations\n Across Reward Sizes"))

subplot(2,3,2);hold on
plot([0,2]+.05,[0,2],'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,2]-.05,[0,2],'color',cool3(3,:),'linewidth',3)
title("Pure Time Representation")

subplot(2,3,3);hold on
plot([0,2],[0,1.25],'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,1.25],[0,2],'color',cool3(3,:),'linewidth',3)
title("Integrator with Variable Gain")
% 
% subplot(2,3,4);hold on
% plot([0,2],[1,1] - .025,'color',cool3(1,:),'linewidth',3)
% plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
% plot([0,2],[1,1] + 0.025,'color',cool3(3,:),'linewidth',3)
% title(sprintf("Different Representations\n Across Reward Sizes"))

subplot(2,3,4);hold on
plot([0,2],[0,2] - .05,'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,2],[0,2] + .05,'color',cool3(3,:),'linewidth',3)
title("Pure Time Until Leave Representation")

subplot(2,3,5);hold on
plot([0,2],[0,2] + .5,'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,2]-.05,[0,2] - .5,'color',cool3(3,:),'linewidth',3)
title("Pure Time on Patch Representation")

subplot(2,3,6);hold on
plot([0,2],[0,1.25],'color',cool3(1,:),'linewidth',3)
plot([0,2],[0,2],'color',cool3(2,:),'linewidth',3)
plot([0,1.25],[0,2],'color',cool3(3,:),'linewidth',3)
title("Integrator with Constant Threshold")

for i = 1:6
    subplot(2,3,i)
    if i <=3
        xlabel("True Time on Patch (sec)")
    else
        xlabel("True Until Leave (sec)")
        set ( gca, 'xdir', 'reverse' )
    end
    xlim([0,2])
    ylabel(sprintf("2 uL Trials \n Decoded time"))
    ylim([0,2])
    fig = gca;
    set(fig,'fontsize',14)
end

%% C) Time since reward representation could be independent from behavior --> should collapse (ala cluster 2)

rdbu3 = cbrewer('div',"RdBu",10);
rdbu3 = rdbu3([3 7 end],:);

figure() 
subplot(1,3,1);hold on
plot([0,2]+.05,[0,2],'color',rdbu3(3,:),'linewidth',3)
plot([0,2],[0,2],'color',rdbu3(2,:),'linewidth',3)
plot([0,2]-.05,[0,2],'color',rdbu3(1,:),'linewidth',3)
title(sprintf("Representations Unrelated\n to Behavioral Variability"))


subplot(1,3,2);hold on
plot([0,2],[.5,.5],'color',rdbu3(3,:),'linewidth',3)
plot([0,2],[0,2],'color',rdbu3(2,:),'linewidth',3)
plot([0,2],[1.5,1.5],'color',rdbu3(1,:),'linewidth',3)
title(sprintf("Distinct Representations\n Underlie Behavioral Variability"))

subplot(1,3,3);hold on
plot([0,2],[0,1.75],'color',rdbu3(3,:),'linewidth',3)
plot([0,2],[0,2],'color',rdbu3(2,:),'linewidth',3)
plot([0,1.75],[0,2],'color',rdbu3(1,:),'linewidth',3)
title(sprintf("Variable Integrator Gain\n Underlies Behavioral Variability"))

for i = 1:3
    subplot(1,3,i)
    xlabel("True Time on Patch (sec)")
    xlim([0,2])
    ylabel("Decoded Time (sec)")
    ylim([0,2])
    fig = gca;
    set(fig,'fontsize',14)
end

%% D) i. anti-correlated ii. uncorrelated, iii. correlated
n = 50000; 
n_bins = 50;
mu = [0;0];
sigma_corr = [1,.6;.6,1]; 
sigma_uncorr = [1,0;0,1]; 
sigma_anticorr = [1,-.6;-.6,1]; 

figure(); subplot(1,3,1)
dist_uncorr = mvnrnd(mu,sigma_uncorr,n);
b = binscatter(dist_uncorr(:,1),dist_uncorr(:,2),n_bins,'XLimits',[-4 4],'YLimits',[-4 4]); 
% imagesc(b.Values)
ax = gca;
colorbar off
% ax.ColorScale = 'log';
colormap('parula') 

subplot(1,3,2)
dist_anticorr = mvnrnd(mu,sigma_anticorr,n);
b = binscatter(dist_anticorr(:,1),dist_anticorr(:,2),n_bins,'XLimits',[-4 4],'YLimits',[-4 4]); 
% imagesc(flipud(b.Values))
ax = gca;
colorbar off
% ax.ColorScale = 'log';
colormap('parula') 

subplot(1,3,3)
dist_corr = mvnrnd(mu,sigma_corr,n);
b = binscatter(dist_corr(:,1),dist_corr(:,2),n_bins,'XLimits',[-4 4],'YLimits',[-4 4]); 
% imagesc(flipud(b.Values))
ax = gca;
colorbar off
% ax.ColorScale = 'log';
colormap('parula') 
