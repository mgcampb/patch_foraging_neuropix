%% Make a figure showing how reward probability decreases over time on patch

tau = 8;  
t = 1:10;  
colors = [.7 .7 .7 ; .4 .4 .4 ; 0 0 0]; 
N0s = [.125 .25 .5]; 
figure(); hold on
for i = 1:numel(N0s) 
    iN0 = N0s(i);
    scatter(1:numel(t),iN0 * exp(-t / tau),40,'o','MarkerEdgeColor',[0 0 0],'MarkerFaceColor',colors(i,:),'linewidth',1)
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
ax.Legend.FontSize = 15;