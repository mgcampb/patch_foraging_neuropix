%%  Some more detailed analysis of PRT 
% Need some session inclusion criteria!
% 1. Is the time on patch information in mouse behavior? 
%    a. Test R0 vs RR PRT post last rew to see if there is time on patch
%    b. Test R0R vs RR0 PRT post last reward 
% 2. Is there correlation between cue residence time and PRT? 
%    a. this will need to be among certain subsets of trials or patch
%    effects will outweigh  

% 3. What is distribution of PRT after last reward? 
%    a. Dependent on last reward time or number?
% ---- These are less pressing ----
% 3. Patch history effects? 
% 4. Maybe hazard rate given different reward sequences

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
mouse_names = ["m75","m76","m78","m79","m80"]; 
mouse_grps = cell(length(mice),1); 
for m = 1:numel(mice) 
    mouse_grps{m} = find(cellfun(@(x) strcmp(x(1:2),mice(m)),sessions,'un',1));
end 

% to pare down to just recording sessions
recording_sessions = dir(fullfile(paths.neuro_data,'*.mat'));
recording_sessions = {recording_sessions.name};
% to just use recording sessions
recording_session_bool = cellfun(@(x) ismember(x,recording_sessions),sessions);
savepath = '/Users/joshstern/Documents/Undergraduate Thesis/Draft 1.1/results_figures 1.1/figure2';

%% Get PRTs and RX, RXX barcode for 5 recorded mice  
mouse_rewsize = cell(numel(mouse_grps),1); 
mouse_N0 = cell(numel(mouse_grps),1); 
mouse_prts = cell(numel(mouse_grps),1);   
mouse_qrts = cell(numel(mouse_grps),1);  
mouse_RX = cell(numel(mouse_grps),1);  
mouse_RXX = cell(numel(mouse_grps),1); 
rew_barcodes = cell(numel(mouse_grps),1);  
mouse_post_rew_rts = cell(numel(mouse_grps),1);   
mouse_last_rew_num = cell(numel(mouse_grps),1);   
mouse_last_rew_time = cell(numel(mouse_grps),1);   
mouse_ERew = cell(numel(mouse_grps),1);   
for mIdx = 1:numel(mouse_grps)
    mouse_prts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_rewsize{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_N0{mIdx} = cell(numel(mouse_grps{mIdx}),1);  
    mouse_qrts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_RX{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_RXX{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    rew_barcodes{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_post_rew_rts{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_last_rew_num{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_last_rew_time{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    mouse_ERew{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.beh_data,session));   
        
        % get patch information
        rewsize = mod(data.patches(:,2),10);   
        N0 = round(mod(data.patches(:,2),100)/10);
        N0(N0 == 3) = .125;
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5;   
        
        % get behavior timing information 
        rew_sec = data.rew_ts; 
        patchcue_sec = data.patchCSL(:,1); 
        patchstop_sec = data.patchCSL(:,2); 
        patchleave_sec = data.patchCSL(:,3);  
        cue_rts = patchstop_sec - patchcue_sec;
        prts = patchleave_sec - patchstop_sec;   
        floor_prts = floor(prts);  

        % make barcode matrices
        nTimesteps = 30; 
        nTrials = length(rewsize); 
        rew_barcode = zeros(nTrials , nTimesteps); 
        last_rew_time = nan(nTrials,1); 
        last_rew_num = nan(nTrials,1); 
        post_rew_rts = nan(nTrials,1); 
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
            rew_barcode(iTrial , (max(rew_indices)+1):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
            last_rew_time(iTrial) = rew_indices(end) - 1;  
            post_rew_rts(iTrial) = prts(iTrial) - last_rew_time(iTrial);
            last_rew_num(iTrial) = length(rew_indices); 
        end  
        
        % turn RX and RXX information into strings for easy access 
        % this is not beautiful but dont judgeee
        RX = nan(nTrials,1); 
        RXX = nan(nTrials,1);
        for iRewsize = [1 2 4] 
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0) = double(sprintf("%i0",iRewsize));
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize) = double(sprintf("%i%i",iRewsize,iRewsize));  
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) <= 0 & rew_barcode(:,3) <= 0 & rew_barcode(:,4) < 0) = double(sprintf("%i00",iRewsize)); 
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) <= 0 & rew_barcode(:,4) < 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize & rew_barcode(:,4) < 0) = double(sprintf("%i0%i",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize & rew_barcode(:,4) < 0) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize));
        end
        
        % log data 
        mouse_rewsize{mIdx}{i} = rewsize;  
        mouse_N0{mIdx}{i} = N0;  
        mouse_prts{mIdx}{i} = prts;   
        mouse_qrts{mIdx}{i} = cue_rts;  
        mouse_RX{mIdx}{i} = RX; 
        mouse_RXX{mIdx}{i} = RXX; 
        rew_barcodes{mIdx}{i} = rew_barcode; 
        mouse_post_rew_rts{mIdx}{i} = post_rew_rts;
        mouse_last_rew_num{mIdx}{i} = last_rew_num;
        mouse_last_rew_time{mIdx}{i} = last_rew_time;
        mouse_ERew{mIdx}{i} = arrayfun(@(iTrial) calc_E_rew(prts(iTrial),N0(iTrial),rewsize(iTrial)),(1:nTrials)'); 
    end
end 

%% 0. Session inclusion criteria 
%  - Need to select more sessions than just recording sessions to increase
%    power, but also need to have some kind of behavior to see that we're
%    performing some kind of value-based decision-making 
%  - Ideas: a) Early-mid-late b) Window pooling c) Spectral clustering

close all

cool3 = cool(3);
cool9_light = zeros(9,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool9_light(1:3,i) = linspace(.9, cool3(1,i), 3);
    cool9_light(4:6,i) = linspace(.9, cool3(2,i), 3);
    cool9_light(7:9,i) = linspace(.9, cool3(3,i), 3);
end
cool9_light(2,1) = .7; % adjustment otherwise the 1uL colors don't come out nice 

% start by just plotting mean PRTs acr days to see what this looks like 
tt_mean_prts = cell(numel(mouse_grps),1);  
pvalues = cell(numel(mouse_grps),1); 
for mIdx = 1:5   
    tt_mean_prts{mIdx} = nan(9,numel(mouse_grps{mIdx}));  
    pvalues{mIdx} = nan(3,numel(mouse_grps{mIdx}),1); 
    for iSession = 1:numel(mouse_grps{mIdx})  
        session_tt_mean_prts = nan(9,1); 
        counter = 1; 
        rewsize = mouse_rewsize{mIdx}{iSession}; 
        N0 = mouse_N0{mIdx}{iSession};  
        prts_trialtypes_rewsize = []; 
        for iRewsize = [1 2 4] 
            for iN0 = [.125 .25 .5] 
%                 scatter(iSession,mean(mouse_prts{mIdx}{iSession}(rewsize == iRewsize & N0 == iN0)),[],cool9_light(counter,:)) 
                session_tt_mean_prts(counter) = mean(mouse_prts{mIdx}{iSession}(rewsize == iRewsize & N0 == iN0));
                prts_trialtypes_rewsize = [prts_trialtypes_rewsize ; [mouse_prts{mIdx}{iSession}(rewsize == iRewsize & N0 == iN0) counter+zeros(length(find(rewsize == iRewsize & N0 == iN0)),1) iRewsize+zeros(length(find(rewsize == iRewsize & N0 == iN0)),1)]];
                counter = counter + 1;  
            end  
        end  
        sorted_rewsize = prts_trialtypes_rewsize(:,3); 
        pvalues{mIdx}(1,iSession) = kruskalwallis(prts_trialtypes_rewsize(sorted_rewsize == 1,1),prts_trialtypes_rewsize(sorted_rewsize == 1,2),'off');
        pvalues{mIdx}(2,iSession) = kruskalwallis(prts_trialtypes_rewsize(sorted_rewsize == 2,1),prts_trialtypes_rewsize(sorted_rewsize == 2,2),'off');
        pvalues{mIdx}(3,iSession) = kruskalwallis(prts_trialtypes_rewsize(sorted_rewsize == 4,1),prts_trialtypes_rewsize(sorted_rewsize == 4,2),'off');
        tt_mean_prts{mIdx}(:,iSession) = session_tt_mean_prts;
    end
end

%% More x session visualization 
% h2 = figure();
% h2(1) = subplot(2,3,1);
% h2(2) = subplot(2,3,2);
% h2(3) = subplot(2,3,3);
% h2(4) = subplot(2,3,4);
% h2(5) = subplot(2,3,5); % the last (odd) axes 
% for mIdx = 1:5
%     subplot(2,3,mIdx);hold on
%     for tt = 1:9
%         plot(tt_mean_prts{mIdx}(tt,:),'color',cool9_light(tt,:))
%     end
% end
% 

h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes 
for mIdx = 1:5  
    [score,coeffs,expl] = pca(zscore(tt_mean_prts{mIdx},[],1));
    subplot(2,3,mIdx) ;colormap('cool')
    scatter(score(:,1),score(:,3),[],any(pvalues{mIdx} < .05,1)) % 1:size(score(:,1)))  
    xlabel("PC1")
    ylabel("PC2")
    title(mouse_names(mIdx))
end
pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)]) 

%% -1. Do mice pay attention to reward size? 

cool3 = cool(3); 

% h2 = figure();
% h2(1) = subplot(2,3,1);
% h2(2) = subplot(2,3,2);
% h2(3) = subplot(2,3,3);
% h2(4) = subplot(2,3,4);
% h2(5) = subplot(2,3,5); % the last (odd) axes 

mean_prt_rewsize = nan(numel(mouse_grps),3); 

for mIdx = 1:numel(mouse_grps)   
%     subplot(2,3,mIdx);hold on
    iMouse_rewsize = cat(1,mouse_rewsize{mIdx}{:});
    iMouse_prts = cat(1,mouse_prts{mIdx}{:}); 
    
    for iRewsize = [1 2 4]  
        these_trials = find(iMouse_rewsize == iRewsize); 
%         scatter(.1 * randn(length(these_trials),1) + min(3,iRewsize) + zeros(length(these_trials),1),iMouse_prts(these_trials),2,cool3(min(3,iRewsize),:),'o')
%         bar(min(3,iRewsize),mean(iMouse_prts(these_trials)),'FaceColor',cool3(min(3,iRewsize),:)) 
        mean_prt_rewsize(mIdx,min(3,iRewsize)) = mean(iMouse_prts(these_trials));
    end 
%     ylim([0 mean(iMouse_prts) + 2 * std(iMouse_prts)])
%     xlim([0 4]) 
%     xticks(1:3) 
%     xticklabels(["1 uL","2 uL","4 uL"]) 
%     xlabel("Reward Size") 
%     p = kruskalwallis(iMouse_prts,iMouse_rewsize,'off');  
%     title(sprintf("%s (p = %.3f)",mouse_names(mIdx),p)) 
end 

% pos = get(h2,'Position');
% new = mean(cellfun(@(v)v(1),pos(1:2)));
% set(h2(4),'Position',[new,pos{end}(2:end)])
% new = mean(cellfun(@(v)v(1),pos(2:3)));
% set(h2(5),'Position',[new,pos{end}(2:end)]) 
figure();hold on
for iRewsize = [1 2 4]
    scatter(min(3,iRewsize) + zeros(numel(mouse_grps),1),mean_prt_rewsize(:,min(3,iRewsize)),[],cool3(min(3,iRewsize),:));
    bar(min(3,iRewsize),mean(mean_prt_rewsize(:,min(3,iRewsize))),'FaceColor',cool3(min(3,iRewsize),:),'EdgeColor',cool3(min(3,iRewsize),:),'FaceAlpha',.1) 
end 
xlim([0 4]) 
ylim([0 max(mean_prt_rewsize(:))+2]) 
xticks(1:3) 
xticklabels(["1 uL","2 uL","4 uL"])
xlabel("Reward Size")  
ylabel("PRT (sec)")

%% 1. Is the time on patch information in mouse behavior? 
%  a) Test R0 vs RR PRT post last rew to see if there is time on patch 
%  b) Test R0R vs RR0 PRT post last reward  
%  c) Test R0R vs RRR 
% ohh this R0 vs RR is less interesting 

x = [1 2 4 5]; 

RX_labels = ["20","22","40","44"];

RX_colors = [.75 .75 1 ; .5 .5 1 ; 1 .5 1; 1 0 1];
h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes 

for mIdx = 1:numel(mouse_grps)  
    iMouse_RX = cat(1,mouse_RX{mIdx}{:});
    iMouse_prts = cat(1,mouse_prts{mIdx}{:});
    i_ERew = cat(1,mouse_ERew{mIdx}{:});
    counter = 1; 
    for iRewsize = [2 4]
        for trial_type = [double(sprintf("%i0",iRewsize)) double(sprintf("%i%i",iRewsize,iRewsize))] 
            subplot(2,3,mIdx);hold on
%             these_trials = find(mouse_RX{mIdx} == trial_type); 
            these_trials = find(iMouse_RX == trial_type); 
            
            if trial_type == double(sprintf("%i%i",iRewsize,iRewsize)) 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),iMouse_prts(these_trials)-1,8,RX_colors(counter,:)) 
%                  scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),i_ERew(these_trials),8,RX_colors(counter,:)) 
            else 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),iMouse_prts(these_trials),8,RX_colors(counter,:)) 
%                 scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),i_ERew(these_trials),8,RX_colors(counter,:)) 
            end
 
            counter = counter + 1; 
        end 
        
        % significance?  
        p = ranksum(iMouse_prts(iMouse_RX == double(sprintf("%i%i",iRewsize,iRewsize)))-1, ...
                    iMouse_prts(iMouse_RX == double(sprintf("%i0",iRewsize))));
%         disp(p)
%         p = ranksum(i_ERew(iMouse_RX == double(sprintf("%i%i",iRewsize,iRewsize)))-1, ...
%                     i_ERew(iMouse_RX == double(sprintf("%i0",iRewsize))));        
        if p < .05
            % add line
            %                 plot([x(counter-2)-.75 x(counter-1)+.75],1.05 * [sem + b(1).YData sem + b(1).YData],'k','linewidth',1,'HandleVisibility','off')
            %                 % add legs to sides of the line
            %                 if first == true
            %                     leg_length = 1.05 * (sem + b(1).YData) - .95 * 1.05 * (sem + b(1).YData);
            %                     first = false;
            %                 end
            %                 plot([x(counter-2)-.75 x(counter-2)-.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1,'HandleVisibility','off')
            %                 plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (sem + b(1).YData) 1.05 * (sem + b(1).YData)-leg_length],'k','linewidth',1,'HandleVisibility','off')
            % add significance stars
            %                 mean(mouse_prts{mIdx}(these_trials)-1)
            if p < .05 && p > .01
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "*",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5,"*",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .01 && p > .001
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "**",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5,"**",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .001
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "***",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5, "***",'FontSize',14,'HorizontalAlignment','center');
            end
        end
    end    
    ax = gca;
    ax.XTick = x;
    ax.YAxis.FontSize = 13; 
%     ax.XAxis.FontSize = 13; 
    ax.XTickLabel = RX_labels;  
    xlim([0 max(x)+1]) 
    ylim([0 mean(iMouse_prts(these_trials)) + 3*std(iMouse_prts(these_trials))])
%     ylim([0 mean(i_ERew(these_trials)) + 3*std(i_ERew(these_trials))])
%     ax.XTickLabelRotation = 60;
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])
 

%% 1.b) is there reward memory

x = [1 2 4 5]; 

mouse_names = ["m75","m76","m78","m79","m80"]; 
RX_labels = ["202","222","404","444"];
% RX_labels = ["220","202","440","404"];

RXX_colors = [.75 .75 1 ; .5 .5 1 ; 1 .5 1; 1 0 1];
h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes 
rxx_tt_pool = [];
rxx_prts_pool = [];
mouse_pool = [];
for mIdx = 1:numel(mouse_grps)   
    sig_sessions = any(pvalues{mIdx} < .05,1);  
    sig_RXX = mouse_RXX{mIdx}(sig_sessions);
    sig_RXX = cat(1,sig_RXX{:}); 
    rxx_tt_pool = [rxx_tt_pool ; sig_RXX];
%     sig_RXX = cat(1,mouse_RXX{mIdx}{:});
    sig_prts = mouse_prts{mIdx}(sig_sessions); 
    sig_prts = cat(1,sig_prts{:}); 
    rxx_prts_pool = [rxx_prts_pool ; sig_prts];
%     sig_prts = cat(1,mouse_prts{mIdx}{:}); 
    mouse_pool = [mouse_pool ; mIdx + zeros(length(sig_prts),1)];
    
    counter = 1; 
    for iRewsize = [2 4]
        for trial_type = [double(sprintf("%i0%i",iRewsize,iRewsize)) double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize))] 
            subplot(2,3,mIdx);hold on
            these_trials = find(sig_RXX == trial_type); 
            
            if trial_type == double(sprintf("%i%i%i",iRewsize,iRewsize)) 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),sig_prts(these_trials),8,RXX_colors(counter,:)) 
            else 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),sig_prts(these_trials),8,RXX_colors(counter,:)) 
            end
 
            counter = counter + 1; 
        end 
        
        % significance?  
        p = ranksum(sig_prts(sig_RXX == double(sprintf("%i0%i",iRewsize,iRewsize))), ...
                    sig_prts(sig_RXX == double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize))));
        disp(p)
        if p < .05
            if p < .05 && p > .01
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "*",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5,"*",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .01 && p > .001
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "**",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5,"**",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .001
                %                     text(mean([x(counter-2)-.75 x(counter-1)+.75]),1.1 * (sem + b(1).YData), "***",'FontSize',14,'HorizontalAlignment','center');
                text(mean([x(counter-2)-.75 x(counter-1)+.75]),5, "***",'FontSize',14,'HorizontalAlignment','center');
            end 
        end  
    end    
    ax = gca;
    ax.XTick = x;
    ax.YAxis.FontSize = 13; 
%     ax.XAxis.FontSize = 13; 
    ax.XTickLabel = RX_labels;  
    xlim([0 max(x)+1]) 
    ylim([0 mean(sig_prts(these_trials)) + 3*std(sig_prts(these_trials))])
    ax.XTickLabelRotation = 60; 
    title(mouse_names(mIdx)) 
    if mIdx == 1 || mIdx == 4 
        ylabel("PRT (sec)") 
    end
end  
suptitle("N0 Significant Sessions")

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])

%% 1.b.pool) 

cool4 = zeros(4,3);
cool4(1,:) = [.6,.6,.8];
cool4(2,:) = cool3(2,:);
cool4(3,:) = [.7,.5,.7];
cool4(4,:) = cool3(3,:);


mouse_means = zeros(5,4); 
tts = [202,222,404,444];
for i_mouse = 1:5
    for i_tt = 1:numel(tts)
        these_trials = rxx_tt_pool == tts(i_tt) & mouse_pool == i_mouse;
        mouse_means(i_mouse,i_tt) = mean(rxx_prts_pool(these_trials));
    end
end
[~,tt444_sort] = sort(mouse_means(:,4));
mouse_means = mouse_means(tt444_sort,:);

figure();hold on
these_trials = ismember(rxx_tt_pool,[202,222,404,444]);
b = boxplot(rxx_prts_pool(these_trials),rxx_tt_pool(these_trials),'Notch','on','orientation','horizontal','Color',cool4); 
h = findobj(gcf,'tag','Outliers');
for i = 1:4
    j = i*-1 + 5;
    set(h(j),'MarkerEdgeColor',cool4(i,:));
    set(h(j),'MarkerSize',4);
    scatter(mouse_means(:,i),i + zeros(5,1),20,cool4(i,:),'linewidth',1) %   + linspace(-.05,.05,5)'
end

xlabel("PRT (sec)")
xlim([0,15])
xticks([1:5 10 15])
xticklabels([1:5 10 15])
title("RXX PRT Comparison")
set(gca,'fontsize',14)

md_rxx = ismember(rxx_tt_pool,[202,222]);
lg_rxx = ismember(rxx_tt_pool,[404,444]);
p_4XX = ranksum(rxx_prts_pool(lg_rxx),rxx_tt_pool(lg_rxx));
p_2XX = ranksum(rxx_prts_pool(md_rxx),rxx_tt_pool(md_rxx));

set(b,{'linew'},{2})

save([savepath '/rxx_data.mat'],'rxx_prts_pool','rxx_tt_pool')

% saveas(gcf,[savepath '/rxx_boxplot.png'])

%% 2) Is there correlation between cue residence time and PRT? 
%     - start by doing this within reward size 
%     - then zscore to see if we can compare across reward sizes to incr power
h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes
cool3 = cool(3);
for mIdx = 1:numel(mouse_grps)  
    disp(length(sig_sessions))
    sig_sessions = find(any(pvalues{mIdx} <= .05,1));  
    pooled_zscored_qrts = [];  
    pooled_zscored_prts = [];  
    for iN0 = [.125 .25 .5]
        for iRewsize = [1 2 4]
            subplot(2,3,mIdx);hold on
            
            zscored_qrts = [];
            zscored_prts = [];
            for i = 1:numel(mouse_grps{mIdx})
                zscored_qrts = [zscored_qrts ; zscore(mouse_qrts{mIdx}{i}(mouse_rewsize{mIdx}{i} == iRewsize & mouse_N0{mIdx}{i} == iN0))];
                zscored_prts = [zscored_prts ; zscore(mouse_prts{mIdx}{i}(mouse_rewsize{mIdx}{i} == iRewsize & mouse_N0{mIdx}{i} == iN0))];
            end
            
            % scatter QRTs vs PRTs
            scatter(zscored_qrts,zscored_prts,1,cool3(min(3,iRewsize),:),'o');
            
            % add to pooled data
            pooled_zscored_qrts = [pooled_zscored_qrts ; zscored_qrts];
            pooled_zscored_prts = [pooled_zscored_prts ; zscored_prts];
        end
    end
    xlim([-2 4])
    ylim([-2 4])  
    [r,p] = corrcoef(pooled_zscored_qrts,pooled_zscored_prts);   
    title(sprintf("%s \n r = %.3f, p = %.2e",mouse_names(mIdx),r(2),p(2)),'fontsize',13)   
    ylabel("PRT (z-scored)",'fontsize',13)   
    xlabel("Patch Entrance Latency (z-scored)",'fontsize',13)   
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])


%% 3) Is there time on patch information? 
%   - Does time spent on patch after last reward depend on when last reward
%     was delivered?

h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes

% make a darkening cool color scheme
cool3 = cool(3);
cool6_darkening = zeros(6,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool6_darkening(1:3,i) = fliplr(linspace(.3, cool3(2,i), 3));
    cool6_darkening(4:6,i) = fliplr(linspace(.3, cool3(3,i), 3));
end
% cool6_darkening(2,1) = .7; % adjustment otherwise the 1uL colors don't come out nice 
x = [1 2 3 5 6 7];

pooled_post_rew_rts = []; 
pooled_tt = []; 
pooled_mouse = []; 

for mIdx = 1:numel(mouse_grps)  
    sig_sessions = any(pvalues{mIdx} < .05,1);  
    sig_post_rew_rts = mouse_post_rew_rts{mIdx}(sig_sessions);
    sig_post_rew_rts = cat(1,sig_post_rew_rts{:}); 
    sig_last_rew_num = mouse_last_rew_num{mIdx}(sig_sessions); 
    sig_last_rew_num = cat(1,sig_last_rew_num{:}); 
    sig_last_rew_time = mouse_last_rew_time{mIdx}(sig_sessions); 
    sig_last_rew_time = cat(1,sig_last_rew_time{:}); 
    sig_rewsize = mouse_rewsize{mIdx}(sig_sessions);  
    sig_rewsize = cat(1,sig_rewsize{:}); 
    
    pool_trials = sig_last_rew_num == 2 & ismember(sig_rewsize,[2,4]) & ismember(sig_last_rew_time,[1:3]);
    pooled_post_rew_rts = [pooled_post_rew_rts ; sig_post_rew_rts(pool_trials)]; 
    pooled_tt = [pooled_tt ; sig_rewsize(pool_trials) * 10 + sig_last_rew_time(pool_trials)];
    pooled_mouse = [pooled_mouse ; mIdx + zeros(length(find(pool_trials)),1)]; 
    
    subplot(2,3,mIdx);hold on
    counter = 1; 
    first = true; 
    no_sig = true;
    max_meanPlusSEM = 0; 
    for iRewsize = [2 4]
        for iRewtime = [1 2 3]
            these_trials = find(sig_rewsize == iRewsize & sig_last_rew_time == iRewtime & sig_last_rew_num == 2); 

            b = bar(x(counter),mean(sig_post_rew_rts(these_trials)),'FaceColor', cool6_darkening(counter,:), 'EdgeColor', 'k');
            sem = std(sig_post_rew_rts(these_trials)) / sqrt(numel(these_trials));
            e = errorbar(x(counter),mean(sig_post_rew_rts(these_trials)),sem,'k.');
            set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            
            if iRewtime == 1 
                first_mean = mean(sig_post_rew_rts(these_trials)); 
                first_sem = sem; 
            end
            max_meanPlusSEM = max(max_meanPlusSEM,mean(sig_post_rew_rts(these_trials))+sem); 
            
            counter = counter + 1; 
        end
        
        % significance?
        p = kruskalwallis(sig_post_rew_rts(sig_rewsize == iRewsize & sig_last_rew_time <= 3 & sig_last_rew_num == 2),sig_last_rew_time(sig_rewsize == iRewsize & sig_last_rew_time <= 3 & sig_last_rew_num == 2),'off');
        if p < .05  
            no_sig = false; 
            % add line
            plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [first_sem + first_mean first_sem + first_mean],'k','linewidth',1,'HandleVisibility','off') 
            % add legs to sides of the line 
            if first == true 
                leg_length = 1.05 * (first_sem + first_mean) - .95 * 1.05 * (first_sem + first_mean); 
                first = false;
            end
            plot([x(counter-3)-.75 x(counter-3)-.75],[1.05 * (first_sem + first_mean) 1.05 * (first_sem + first_mean)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
            plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (first_sem + first_mean) 1.05 * (first_sem + first_mean)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
            % add significance stars 
            if p < .05 && p > .01
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "*",'FontSize',14,'HorizontalAlignment','center');   
            elseif p < .01 && p > .001
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "**",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .001
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "***",'FontSize',14,'HorizontalAlignment','center'); 
            end
        end
        ylim([0 1.1 *  1.1 * (first_sem + first_mean)])
    end
    
    if no_sig == true 
       ylim([0 max_meanPlusSEM+1])
    end
    
    xlim([0 max(x + 1)]) 
    ax = gca;
    ax.XTick = x;
    ax.YAxis.FontSize = 12;
    ax.XTickLabel = ["t = 1","t = 2","t = 3","t = 1","t = 2","t = 3"];
    ax.XTickLabelRotation = 60;
    
    if ismember(mIdx,[1 4]) 
        ylabel("PRT After Last Reward (sec)")
    end
    title(mouse_names(mIdx))
    
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])

%% 3pooled) 

mouse_means = zeros(5,4); 
tts = [21,22,23,41,42,43];
for i_mouse = 1:5
    for i_tt = 1:numel(tts)
        these_trials = pooled_tt == tts(i_tt) & pooled_mouse == i_mouse;
        mouse_means(i_mouse,i_tt) = mean(pooled_post_rew_rts(these_trials));
    end
end
[~,tt40_sort] = sort(mouse_means(:,4));
mouse_means = mouse_means(tt40_sort,:);

figure();hold on
b = boxplot(pooled_post_rew_rts,pooled_tt,'Notch','on','orientation','horizontal','Color',cool6_darkening); 
set(b,{'linew'},{2})
xlim([0,15])
h = findobj(gcf,'tag','Outliers');
for i = 1:6
    j = i*-1 + 7;
    set(h(j),'MarkerEdgeColor',cool6_darkening(i,:));
    set(h(j),'MarkerSize',4);
    scatter(mouse_means(:,i),i + zeros(5,1),20,cool6_darkening(i,:),'linewidth',1) %   + linspace(-.05,.05,5)'
end
md_pval = kruskalwallis(pooled_post_rew_rts(floor(pooled_tt / 10) == 2),pooled_tt(floor(pooled_tt / 10) == 2),'off')
lg_pval = kruskalwallis(pooled_post_rew_rts(floor(pooled_tt / 10) == 4),pooled_tt(floor(pooled_tt / 10) == 4),'off')

yticklabels(["22","202","2002","44","404","4004"])

xlabel("PRT After Last Reward (sec)")
xlim([0,15])
xticks([1:5 10 15])
xticklabels([1:5 10 15])
title("PRT by Reward Time Comparison")
set(gca,'fontsize',14)

save([savepath '/r_2_data.mat'],'pooled_post_rew_rts','pooled_tt')
saveas(gcf,[savepath '/r_2_boxplot.png'])

%% 3+) Is there parametric information about number of rewards received? 
% given that last reward is delivered at t = 4, do mice wait diff amount of
% time depending on reward number? parametric comparison of RXX  

% this effect looks real but not significant within mice

h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes

% make a darkening cool color scheme
cool3 = cool(3);
cool6_darkening = zeros(6,3);
for i = 1:3
    %colors4(1:4,i) = linspace(1, cool3(1,i), 4);
    cool6_darkening(1:3,i) = linspace(.3, cool3(2,i), 3);
    cool6_darkening(4:6,i) = linspace(.3, cool3(3,i), 3);
end
% cool6_darkening(2,1) = .7; % adjustment otherwise the 1uL colors don't come out nice 
x = [1 2 3 5 6 7];

for mIdx = 1:numel(mouse_grps)  
    sig_sessions = any(pvalues{mIdx} < .05,1);  
    sig_post_rew_rts = mouse_post_rew_rts{mIdx}(sig_sessions);
    sig_post_rew_rts = cat(1,sig_post_rew_rts{:}); 
    sig_last_rew_num = mouse_last_rew_num{mIdx}(sig_sessions); 
    sig_last_rew_num = cat(1,sig_last_rew_num{:}); 
    sig_last_rew_time = mouse_last_rew_time{mIdx}(sig_sessions); 
    sig_last_rew_time = cat(1,sig_last_rew_time{:}); 
    sig_rewsize = mouse_rewsize{mIdx}(sig_sessions);  
    sig_rewsize = cat(1,sig_rewsize{:}); 
    
    subplot(2,3,mIdx);hold on
    counter = 1; 
    first = true; 
    no_sig = true;
    max_meanPlusSEM = 0; 
    for iRewsize = [2 4]
        for iRewnum = [2 3 4]
            these_trials = find(sig_rewsize == iRewsize & sig_last_rew_num == iRewnum & sig_last_rew_time == 4); 

            b = bar(x(counter),mean(sig_post_rew_rts(these_trials)),'FaceColor', cool6_darkening(counter,:), 'EdgeColor', 'k');
            sem = std(sig_post_rew_rts(these_trials)) / sqrt(numel(these_trials));
            e = errorbar(x(counter),mean(sig_post_rew_rts(these_trials)),sem,'k.');
            set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            
            if iRewnum == 4
                first_mean = mean(sig_post_rew_rts(these_trials)); 
                first_sem = sem; 
            end
            max_meanPlusSEM = max(max_meanPlusSEM,mean(sig_post_rew_rts(these_trials))+sem); 
            
            counter = counter + 1; 
        end
        
        % significance?
        p = kruskalwallis(sig_post_rew_rts(sig_rewsize == iRewsize & sig_last_rew_time == 4),sig_last_rew_num(sig_rewsize == iRewsize & sig_last_rew_time == 4),'off');
        if p < .05  
            no_sig = false; 
            % add line
            plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [first_sem + first_mean first_sem + first_mean],'k','linewidth',1,'HandleVisibility','off') 
            % add legs to sides of the line 
            if first == true 
                leg_length = 1.05 * (first_sem + first_mean) - .95 * 1.05 * (first_sem + first_mean); 
                first = false;
            end
            plot([x(counter-3)-.75 x(counter-3)-.75],[1.05 * (first_sem + first_mean) 1.05 * (first_sem + first_mean)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
            plot([x(counter-1)+.75 x(counter-1)+.75],[1.05 * (first_sem + first_mean) 1.05 * (first_sem + first_mean)-leg_length],'k','linewidth',1,'HandleVisibility','off') 
            % add significance stars 
            if p < .05 && p > .01
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "*",'FontSize',14,'HorizontalAlignment','center');   
            elseif p < .01 && p > .001
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "**",'FontSize',14,'HorizontalAlignment','center');
            elseif p < .001
                text(mean([x(counter-3)-.75 x(counter-1)+.75]),1.1 * (first_sem + first_mean), "***",'FontSize',14,'HorizontalAlignment','center'); 
            end
        end
        ylim([0 1.1 *  1.1 * (first_sem + first_mean)])
    end
    
    if no_sig == true 
       ylim([0 max_meanPlusSEM+1])
    end
    
    xlim([0 max(x + 1)]) 
    ax = gca;
    ax.XTick = x;
    ax.YAxis.FontSize = 12;
    ax.XTickLabel = ["rew 2","rew 3","rew 4","rew 2","rew 3","rew 4"];
    ax.XTickLabelRotation = 60;
    
    if ismember(mIdx,[1 4]) 
        ylabel("PRT After Last Reward (sec)")
    end
    title(mouse_names(mIdx))
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)]) 

%% MVT hypothesis: barplot E_rew between patch types 

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

% pool (yeah it's dumb)
pooled_sigRewsize = [];
pooled_sigN0 = [];
pooled_ERew = []; 

for m = 1:numel(vis_mice)
    mIdx = vis_mice(m);
    sig_sessions = any(pvalues{mIdx} < .05,1);  
    
    sig_rewsize = mouse_rewsize{mIdx}(sig_sessions);  
    sig_rewsize = cat(1,sig_rewsize{:}); 
    pooled_sigRewsize = [pooled_sigRewsize ; sig_rewsize]; 
    sig_N0 = mouse_N0{mIdx}(sig_sessions);  
    sig_N0 = cat(1,sig_N0{:});  
    pooled_sigN0 = [pooled_sigN0 ; sig_N0]; 
    
    i_ERew = mouse_ERew{mIdx}(sig_sessions); 
    i_ERew = cat(1,i_ERew{:}); 
    pooled_ERew = [pooled_ERew ; i_ERew];
    
    counter = 1; 
    first = true;
    for iRewsize = [1 2 4]
        for iN0 = [.125 .25 .5]
            subplot(2,3,m);hold on
            these_trials = (sig_rewsize == iRewsize & sig_N0 == iN0);
            b = bar(x(counter), mean(i_ERew(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k');
            if iN0 ~= .5
                set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            end 
            sem = 1.96 * std(i_ERew(these_trials)) / sqrt(numel(find(these_trials)));
            e = errorbar(x(counter), mean(i_ERew(these_trials)),sem,'k.');
            set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            
            ax = gca;
            ax.XTick = x;
            ax.YAxis.FontSize = 13;
            ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
            ax.XTickLabelRotation = 60;

            counter = counter + 1; 
        end
        ax.XAxis.FontSize = 13; 
        
        [p,tbl,stats] = kruskalwallis(i_ERew(sig_rewsize == iRewsize),sig_N0(sig_rewsize == iRewsize),'off');
     
        if p < .05 
            % add line
            plot([x(counter-3)-.75 x(counter-1)+.75],1.05 * [sem + b(1).YData sem + b(1).YData],'k','linewidth',1,'HandleVisibility','off') 
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
    
    if m == 1 || m == 4
        ylabel("E[Reward] at time of leave",'Fontsize',15)
    end
    
    title(mouse_names(m),'Fontsize',15)
    
    if m == 1
        legend(["R = 1 uL","R = 2 uL","R = 4 uL"],'Fontsize',13)
    end
end 

suptitle(sprintf("E[Reward] at time of leave\n Separated by Trial Type"))

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)]) 

%% Pool MVT hypothesis figure

counter = 1;
first = true;
figure();hold on
for iRewsize = [1 2 4]
    for iN0 = [.125 .25 .5]
        these_trials = (pooled_sigRewsize == iRewsize & pooled_sigN0 == iN0);
        b = bar(x(counter), mean(pooled_ERew(these_trials)),'FaceColor', cool9_light(counter,:), 'EdgeColor', 'k','linewidth',1.5);
        if iN0 ~= .5
            set(get(get(b,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        end
        sem = 1.96 * std(pooled_ERew(these_trials)) / sqrt(numel(find(these_trials)));
        e = errorbar(x(counter), mean(pooled_ERew(these_trials)),sem,'k.');
        set(get(get(e,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
        
        ax = gca;
        ax.XTick = x;
        ax.YAxis.FontSize = 13;
        ax.XTickLabel = {"ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50', "ρ_{0} = .125", 'ρ_{0} = .25', 'ρ_{0} = .50'};
        ax.XTickLabelRotation = 60;
        
        counter = counter + 1;
    end
    ax.XAxis.FontSize = 13;
    
    [p,tbl,stats] = kruskalwallis(pooled_ERew(pooled_sigRewsize == iRewsize),pooled_sigN0(pooled_sigRewsize == iRewsize),'off');
    
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

%% Plot PDFs to check
pdf1 = calc_E_rew(0:.1:30,.125,1); 
pdf2 = calc_E_rew(0:.1:30,.25,1); 
pdf3 = calc_E_rew(0:.1:30,.5,1); 
rewsize_dumb = [1 2 4]; 

figure(); hold on
for i_rewsize = 1:3
    plot(pdf1 * rewsize_dumb(i_rewsize),'linewidth',2,'color',cool9_light(1 + 3 * (i_rewsize - 1),:)); 
    plot(pdf2 * rewsize_dumb(i_rewsize) - .025,'linewidth',2,'color',cool9_light(2 + 3 * (i_rewsize - 1),:)); 
    plot(pdf3 * rewsize_dumb(i_rewsize) - .05,'linewidth',2,'color',cool9_light(3 + 3 * (i_rewsize - 1),:)); 
end
xticks(0:50:300) 
xticklabels(.1*(0:50:300) ) 
xlim([0 200])
xlabel("Time on patch (sec)")
ylabel("E[Reward]")
ylim([0,2])
title("Expected Value Across 9 Patch Types")
legend("1 µL \rho = 0.125","1 µL \rho = 0.25","1 µL \rho = 0.50","2 µL \rho = 0.125","2 µL \rho = 0.25","2 µL \rho = 0.50","4 µL \rho = 0.125","4 µL \rho = 0.25","4 µL \rho = 0.50")
this_fig = gca; 
set(this_fig,'fontsize',16)

%% Where in the second are mice leaving? 

h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes

for mIdx = 1:5 
    sig_sessions = any(pvalues{mIdx} < .05,1);  
%     sig_sessions = any(pvalues{mIdx} < .05,1);  
    last_rew_time_cat = cat(1,mouse_last_rew_time{mIdx}{:}); 
    prts_cat = cat(1,mouse_prts{mIdx}{:}); % (sig_sessions)); 
%     prts_cat = prts_cat(last_rew_time_cat - prts_cat > .5);
    prts_cat = prts_cat((floor(prts_cat) == last_rew_time_cat) & (floor(prts_cat) > 0)); 
%     prts_cat = prts_cat((floor(prts_cat) > 0)); 
%     prts_cat = cat(1,prts_cat{:});
    decimal = prts_cat - floor(prts_cat);
    
    subplot(2,3,mIdx)
    histogram(decimal,0:.1:1)
    [h,p] = kstest(decimal,'CDF',[(0:.01:1)' unifcdf(0:.01:1,0,1)']); % kolmagorov-smirnov test for equality of distn.
    title(sprintf("%s (p = %.3f)",mouse_names(mIdx),p),'fontsize',13) 
    xlabel("Decimal Place of PRT",'fontsize',14) 
    ylabel([])
end

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)]) 
suptitle(sprintf("Decimal Place of PRT Distribution\n Trials Where Leave occurs in rewarded second"))

%% Define N0 pdf 
function E = calc_E_rew(t,N0,rewsize)
    tau = 8; 
    E = rewsize * N0 * exp(-t/tau);  
end
