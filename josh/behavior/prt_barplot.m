paths = struct;
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice'; 
paths = struct;
% paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

paths.neuro_data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
% close all
paths.beh_data = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));

sessions = dir(fullfile(paths.beh_data,'*.mat'));
sessions = {sessions.name};  

mice = ["75","76","78","79","80"];
mouse_grps = cell(length(mice),1); 
for m = 1:numel(mice) 
    mouse_grps{m} = find(cellfun(@(x) strcmp(x(1:2),mice(m)),sessions,'un',1));
end 

mPFC_sessions = [1:8 10:13 15:18 23 25];
% to pare down to just recording sessions
recording_sessions = dir(fullfile(paths.neuro_data,'*.mat'));
recording_sessions = {recording_sessions.name};
% to just use recording sessions
recording_session_bool = cellfun(@(x) ismember(x,recording_sessions(mPFC_sessions)),sessions);

%% Plot PRTs for 5 recorded mice  
mouse_rewsize = cell(numel(mouse_grps),1); 
mouse_N0 = cell(numel(mouse_grps),1); 
mouse_prts = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    mouse_prts{mIdx} = []; 
    mouse_rewsize{mIdx} = []; 
    mouse_N0{mIdx} = [];  
    for i = 1 % :numel(mouse_grps{mIdx})   
        sIdx = mouse_grps{mIdx}(i); 
%         if recording_session_bool(sIdx)
            session = sessions{sIdx}(1:end-4);
            data = load(fullfile(paths.beh_data,session));
            rewsize = mod(data.patches(:,2),10);
            mouse_rewsize{mIdx} = [mouse_rewsize{mIdx} ; rewsize];
            N0 = round(mod(data.patches(:,2),100)/10);
            N0(N0 == 3) = .125; % just reorder in terms of
            N0(N0 == 2) = .25;
            N0(N0 == 1) = .5;
            mouse_N0{mIdx} = [mouse_N0{mIdx} ; N0];
            prts = data.patchCSL(:,3) - data.patchCSL(:,2);
            mouse_prts{mIdx} = [mouse_prts{mIdx} ; prts];
%         end
    end
end

%% PRT barplot 

mouse_names = ["m75","m76","m78","m79","m80"];
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

vis_mice = 1:5; 
vis_quartiles = 1:3;

h2 = figure(); % 'units','inches','position',[0 50 0 50]);
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes

for m = 1:numel(vis_mice)
    mIdx = vis_mice(m);
    counter = 1; 
    first = true;
    for iRewsize = [1 2 4]
        for iN0 = [.125 .25 .5]
            subplot(2,3,m);hold on
            these_trials = (mouse_rewsize{mIdx} == iRewsize & mouse_N0{mIdx} == iN0);
            b = bar(x(counter), mean(mouse_prts{mIdx}(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k');
            if iN0 ~= .5
                set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            end 
            sem = 1.96 * std(mouse_prts{mIdx}(these_trials)) / sqrt(numel(find(these_trials)));
            e = errorbar(x(counter), mean(mouse_prts{mIdx}(these_trials)),sem,'k.');
            set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            
            ax = gca;
            ax.YAxis.FontSize = 13;
            if m >= 4
                ax.XTick = x;
                ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
                ax.XTickLabelRotation = 60;
                xlabel("Initial Reward Probability")
            else
                xticks([])
            end

            counter = counter + 1; 
        end
        ax.XAxis.FontSize = 13; 
        
        [p,tbl,stats] = kruskalwallis(mouse_prts{mIdx}(mouse_rewsize{mIdx} == iRewsize),mouse_N0{mIdx}(mouse_rewsize{mIdx} == iRewsize),'off');
     
        if p < .05 
            % add line
            plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [sem + b(1).YData sem + b(1).YData],'k','linewidth',1,'HandleVisibility','off') 
            % add legs to sides of the line 
            if first == true 
                leg_length = 1.05 * (sem + b(1).YData) - .95 * 1.05 * (sem + b(1).YData); 
                first = false;
            end
            plot([x(counter-3)-.75 x(counter-3)-.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
            plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
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
    
    if m == 1 || m == 4
        ylabel("PRT (sec)",'Fontsize',15)
    end
    
    title(mouse_names(m),'Fontsize',15)
    
    if m == 1
        legend([sprintf("1 μL"),"2 μL","4 μL"],'Fontsize',13)
    end
end 

% suptitle("Behavior during Recorded Sessions")

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)]) 

%% Now pool across mice 
all_rewsize = cat(1,mouse_rewsize{:}); 
all_N0 = cat(1,mouse_N0{:});  
all_prts = cat(1,mouse_prts{:});

counter = 1; 
first = true;
figure();hold on
for iRewsize = [1 2 4]
    for iN0 = [.125 .25 .5]

        these_trials = (all_rewsize == iRewsize & all_N0 == iN0);
        b = bar(x(counter), mean(all_prts(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k','linewidth',1.5);
        if iN0 ~= .5
            set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        sem = 1.96 * std(all_prts(these_trials)) / sqrt(numel(find(these_trials)));
        e = errorbar(x(counter), mean(all_prts(these_trials)),sem,'k.');
        set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        ax = gca;
        ax.XTick = x;
        ax.YAxis.FontSize = 13;
        ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
        ax.XTickLabelRotation = 60;
        
        counter = counter + 1;
    end
    ax.XAxis.FontSize = 13;
    
    [p,tbl,stats] = kruskalwallis(all_prts(all_rewsize == iRewsize),all_N0(all_rewsize == iRewsize),'off');
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
%     title("First Day Pooled Across Mice")
end
set(gca,'fontsize',18)

kruskalwallis(all_prts,all_rewsize,'off')


% savepath = '/Users/joshstern/Documents/Undergraduate Thesis/Draft 1.1/results_figures 1.1/figure2';
% save([savepath '/rewsize_n0_prts.mat'],'all_prts','all_rewsize','all_N0')


