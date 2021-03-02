
%% Figure 3: Bars by rew size & frequency
x = [1 2 3 5 6 7 9 10 11];

darkBlack = [0 0 0];
darkCyan = [0 1 1];
darkBlue = [0 0 1];
cmapBlack = ones(4,3);
cmapCyan = ones(4,3);
cmapBlue = ones(4,3);

for i = 1:3
    cmapBlack(:,i) = linspace(1, darkBlack(i), size(cmapBlack,1));
    cmapCyan(:,i) = linspace(1, darkCyan(i), size(cmapCyan,1));
    cmapBlue(:,i) = linspace(1, darkBlue(i), size(cmapBlue,1));
end

cool3 = cool(3);
cool9_light = zeros(9,3);
cool9_dark = zeros(9,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool9_light(1:3,i) = linspace(.9, cool3(1,i), 3);
    cool9_light(4:6,i) = linspace(.9, cool3(2,i), 3);
    cool9_light(7:9,i) = linspace(.9, cool3(3,i), 3);
    
    cool9_dark(1:3,i) = linspace(.4, cool3(1,i), 3);
    cool9_dark(4:6,i) = linspace(.4, cool3(2,i), 3);
    cool9_dark(7:9,i) = linspace(.4, cool3(3,i), 3);
end
cool9_light(2,1) = .7; % adjustment otherwise the 1uL colors don't come out nice

h_bars9 = figure('Position',[100 200 1200 800]);
for iMouse = 1:length(miceGroup)
    
    subplot(3,6,iMouse)
    hold on;
    for i = 1:9
        bar(x(i), mice(iMouse).rFreqSize(i).mean, 'FaceColor', cool9_light(i,:), 'EdgeColor', 'k');
        errorbar(x(i), mice(iMouse).rFreqSize(i).mean, mice(iMouse).rFreqSize(i).sem,'k.');
    end
    
    maxPos = find([mice(iMouse).rFreqSize(:).mean] == max([(mice(iMouse).rFreqSize(:).mean)])); % location of max mean PRT (plus SEM)
    maxMean = mice(iMouse).rFreqSize(maxPos).mean + .1;
    maxPlusSEM = mice(iMouse).rFreqSize(maxPos).mean + mice(iMouse).rFreqSize(maxPos).sem;
    
    maxY = (floor(maxMean / 5) + 1) * 5;
    ylim([0 maxY]);
    
    ax = gca;
    ax.XTick = x;
    ax.XTickLabel = {[]};
    
    if iMouse >= 13
        ax.XTickLabel = {'Lo', 'Md', 'Hi', 'Lo', 'Md', 'Hi', 'Lo', 'Md', 'Hi'};
        ax.XTickLabelRotation = 60;
    else
        ax.XTickLabel = {[]};
    end
    if iMouse == 1 || iMouse == 7 || iMouse == 13
        ylabel('PRT', 'FontSize', 14);
    end
    
    title(mice(iMouse).mouseStr,'FontSize',15);
end
sgtitle('Patch Residence Time (PRT) depends on reward size & frequency', 'FontSize', 24)