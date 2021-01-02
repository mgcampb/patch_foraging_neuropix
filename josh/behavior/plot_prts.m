%% Plot PRTs for 5 recorded mice 
mouse_rewsize = cell(numel(mouse_grps),1); 
mouse_N0 = cell(numel(mouse_grps),1); 
mouse_prts = cell(numel(mouse_grps),1); 
for mIdx = 1:5
    mouse_prts{mIdx} = []; 
    mouse_rewsize{mIdx} = []; 
    mouse_N0{mIdx} = [];  
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.data,session));  
        rewsize = mod(data.patches(:,2),10);  
        mouse_rewsize{mIdx} = [mouse_rewsize{mIdx} ; rewsize]; 
        N0 = round(mod(data.patches(:,2),100)/10);
        N0(N0 == 3) = .125; % just reorder in terms of 
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5; 
        mouse_N0{mIdx} = [mouse_N0{mIdx} ; N0];  
        prts = data.patchCSL(:,3) - data.patchCSL(:,2);   
        mouse_prts{mIdx} = [mouse_prts{mIdx} ; prts]; 
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

h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes

for m = 1:numel(vis_mice)
    mIdx = vis_mice(m); 
    counter = 1;
    for iRewsize = [1 2 4]
        for iN0 = [.125 .25 .5]
            subplot(2,3,m);hold on
            these_trials = (mouse_rewsize{mIdx} == iRewsize & mouse_N0{mIdx} == iN0);
            b = bar(x(counter), mean(mouse_prts{mIdx}(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k');  
            if iN0 ~= .5
                set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            end
            e = errorbar(x(counter), mean(mouse_prts{mIdx}(these_trials)),std(mouse_prts{mIdx}(these_trials)) / sqrt(numel(find(these_trials))),'k.');
            set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            
            ax = gca;
            ax.XTick = x; 
            ax.YAxis.FontSize = 13;
            if iQuartile == numel(vis_quartiles)
                ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
                ax.XTickLabelRotation = 60;
            else
                ax.XTickLabel = {[]};
            end 
            ax.XAxis.FontSize = 13;
            
            if m == 1 || m == 4
                ylabel("PRT (sec)",'Fontsize',15)
            end  
            
            title(mouse_names(m),'Fontsize',15)
            counter = counter + 1;
        end
    end 
    if m == 1 
        legend(["R = 1 uL","R = 2 uL","R = 4 uL"],'Fontsize',13)
    end 
   
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])