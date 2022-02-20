%% Make PPSSM behavior visualizations in matlab for consistency

ppssm_prts_table = readtable('/Users/joshstern/Documents/Undergraduate Thesis/Draft 3/Draft 3 figures/Figure 7/ppssm_n0rewsize_prts.csv');
tt_string_cell = ppssm_prts_table.tt;
prts = ppssm_prts_table.PRT;
% get rid of parentheses
tt_string_cell = cellfun(@(x) regexprep(x,'(',''),tt_string_cell,'un',0);
tt_string_cell = cellfun(@(x) regexprep(x,')',''),tt_string_cell,'un',0);
tt_string_cell = cellfun(@(x) split(x,','),tt_string_cell,'un',0);
rewsize = str2double(cellfun(@(x) x(1),tt_string_cell));
n0 = str2double(cellfun(@(x) x(2),tt_string_cell));

%% Visualize PRT barplot 
x = [1 2 3 5 6 7 9 10 11];

cool3 = cool(3);
cool9_light = zeros(9,3);
cool9_dark = zeros(9,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool9_light(1:3,i) = linspace(.9, cool3(1,i), 3);
    cool9_light(4:6,i) = linspace(.9, cool3(2,i), 3);
    cool9_light(7:9,i) = linspace(.9, cool3(3,i), 3);
end
cool9_light(2,1) = .7; % adjustment otherwise the 1uL colors don't come out nice 
counter = 1; 
first = true;
figure();hold on
for iRewsize = [1 2 4]
    for iN0 = [.125 .25 .5]

        these_trials = (rewsize == iRewsize & n0 == iN0);
        b = bar(x(counter), mean(prts(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k','linewidth',1.5);
        if iN0 ~= .5
            set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        sem = 1.96 * std(prts(these_trials)) / sqrt(numel(find(these_trials)));
        e = errorbar(x(counter), mean(prts(these_trials)),sem,'k.');
        set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        ax = gca;
        ax.XTick = x;
        ax.YAxis.FontSize = 13;
        ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
        ax.XTickLabelRotation = 60;
        
        counter = counter + 1;
    end
    ax.XAxis.FontSize = 13;
    
    [p,tbl,stats] = kruskalwallis(prts(rewsize == iRewsize),n0(rewsize == iRewsize),'off');
    disp(p)
    
    if p < .05
        % add line
        plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [sem + b(1).YData sem + b(1).YData],'k','linewidth',1.5,'HandleVisibility','off')
        % add legs to sides of the line
        if first == true
            leg_length = 1.05 * (sem + b(1).YData) - .95 * 1.05 * (sem + b(1).YData);
            first = false;
        end
        plot([x(counter-3)-.75 x(counter-3)-.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1.5,'HandleVisibility','off')
        plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1.5,'HandleVisibility','off')
        % add significance stars
        if p < .05 && p > .01
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "*",'FontSize',14,'HorizontalAlignment','center');
        elseif p < .01 && p > .001
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "**",'FontSize',14,'HorizontalAlignment','center');
        elseif p < .001
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "***",'FontSize',14,'HorizontalAlignment','center');
        end
    end
    ylim([0 1.1 *  1.1 * (sem + b(1).YData)])
    ylabel("PRT (sec)") 
    xlabel("Initial Reward Probability")
%     title("First Session Pooled Across Mice")
end
set(gca,'fontsize',18)

%% Now do MVT analysis for ppssm 

nTrials = length(prts);
ERew = arrayfun(@(iTrial) calc_E_rew(prts(iTrial),n0(iTrial),rewsize(iTrial)),(1:nTrials)'); 

counter = 1;
first = true;
figure();hold on
for iRewsize = [1 2 4]
    for iN0 = [.125 .25 .5]
        these_trials = (rewsize == iRewsize & n0 == iN0);
        b = bar(x(counter), mean(ERew(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k','linewidth',1.5);
        if iN0 ~= .5
            set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        sem = 1.96 * std(ERew(these_trials)) / sqrt(numel(find(these_trials)));
        e = errorbar(x(counter), mean(ERew(these_trials)),sem,'k.');
        set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        ax = gca;
        ax.XTick = x;
        ax.YAxis.FontSize = 13;
        ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
        ax.XTickLabelRotation = 60;
        
        counter = counter + 1;
    end
    ax.XAxis.FontSize = 13;
    
    [p,tbl,stats] = kruskalwallis(ERew(rewsize == iRewsize),n0(rewsize == iRewsize),'off');
    
    if p < .05
        % add line
        plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [sem + b(1).YData sem + b(1).YData],'k','linewidth',1.5,'HandleVisibility','off')
        % add legs to sides of the line
        if first == true
            leg_length = 1.05 * (sem + b(1).YData) - .95 * 1.05 * (sem + b(1).YData);
            first = false;
        end
        plot([x(counter-3)-.75 x(counter-3)-.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1.5,'HandleVisibility','off')
        plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1.5,'HandleVisibility','off')
        % add significance stars
        if p < .05 && p > .01
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "*",'FontSize',14,'HorizontalAlignment','center');
        elseif p < .01 && p > .001
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "**",'FontSize',14,'HorizontalAlignment','center');
        elseif p < .001
            text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "***",'FontSize',14,'HorizontalAlignment','center');
        end
    end
    ylim([0 1.1 *  1.1 * (sem + b(1).YData)])
    
end

ylabel("E[Reward] at time of leave",'Fontsize',15)

title(sprintf("E[Reward] at time of leave\n Separated by Trial Type"))
set(gca,'fontsize',18)

%% Define N0 pdf 
function E = calc_E_rew(t,N0,rewsize)
    tau = 8; 
    E = rewsize * N0 * exp(-t/tau);  
end

