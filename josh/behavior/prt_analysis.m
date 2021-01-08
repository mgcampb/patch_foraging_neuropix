%%  Some more detailed analysis of PRT
% 1. Is the time on patch information in mouse behavior? 
%    a. Test R0 vs RR PRT post last rew to see if there is time on patch
%    b. Test R0R vs RR0 PRT post last reward 
% 2. Is there correlation between cue residence time and PRT? 
%    a. this will need to be among certain subsets of trials or patch
%    effects will outweigh 
% 3. Patch history effects?

paths = struct;
paths.data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';

addpath(genpath('/Users/joshstern/Documents/UchidaLab_NeuralData'));

sessions = dir(fullfile(paths.data,'*.mat'));
sessions = {sessions.name}; 
mouse_grps = {1:2,3:8,10:13,15:18,[23 25]}; 

%% Get PRTs and RX, RXX barcode for 5 recorded mice  
mouse_rewsize = cell(numel(mouse_grps),1); 
mouse_N0 = cell(numel(mouse_grps),1); 
mouse_prts = cell(numel(mouse_grps),1);   
mouse_qrts = cell(numel(mouse_grps),1);  
mouse_RX = cell(numel(mouse_grps),1);  
mouse_RXX = cell(numel(mouse_grps),1); 
rew_barcodes = cell(numel(mouse_grps),1); 
for mIdx = 1:numel(mouse_grps)
    mouse_prts{mIdx} = []; 
    mouse_rewsize{mIdx} = []; 
    mouse_N0{mIdx} = [];    
    mouse_qrts{mIdx} = []; 
    mouse_RX{mIdx} = []; 
    mouse_RXX{mIdx} = []; 
    rew_barcodes{mIdx} = cell(numel(mouse_grps{mIdx}),1); 
    for i = 1:numel(mouse_grps{mIdx})
        sIdx = mouse_grps{mIdx}(i);
        session = sessions{sIdx}(1:end-4);
        data = load(fullfile(paths.data,session));   
        
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
        rew_barcode = zeros(length(data.patchCSL) , nTimesteps);
        for iTrial = 1:nTrials
            rew_indices = round(rew_sec(rew_sec >= patchstop_sec(iTrial) & rew_sec < patchleave_sec(iTrial)) - patchstop_sec(iTrial)) + 1;
            rew_barcode(iTrial , max(rew_indices):end) = -1; % set part of patch after last rew = -1
            rew_barcode(iTrial , (floor_prts(iTrial) + 1):end) = -2; % set part of patch after leave = -2
            rew_barcode(iTrial , rew_indices) = rewsize(iTrial);
        end  
        
        % turn RX and RXX information into strings for easy access
        RX = nan(nTrials,1); 
        RXX = nan(nTrials,2);
        for iRewsize = [1 2 4]
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0) = double(sprintf("%i0",iRewsize) & rew_barcode(:,3) < 0); 
            RX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize) = double(sprintf("%i%i",iRewsize,iRewsize) & rew_barcode(:,3) < 0);  
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == 0) = double(sprintf("%i00",iRewsize)); 
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == 0) = double(sprintf("%i%i0",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == 0 & rew_barcode(:,3) == iRewsize) = double(sprintf("%i0%i",iRewsize,iRewsize));
            RXX(rew_barcode(:,1) == iRewsize & rew_barcode(:,2) == iRewsize & rew_barcode(:,3) == iRewsize) = double(sprintf("%i%i%i",iRewsize,iRewsize,iRewsize));
        end
        
        % log data 
        mouse_rewsize{mIdx} = [mouse_rewsize{mIdx} ; rewsize];  
        mouse_N0{mIdx} = [mouse_N0{mIdx} ; N0];  
        mouse_prts{mIdx} = [mouse_prts{mIdx} ; prts];   
        mouse_qrts{mIdx} = [mouse_qrts{mIdx} ; cue_rts];  
        mouse_RX{mIdx} = [mouse_RX{mIdx} ; RX]; 
        mouse_RXX{mIdx} = [mouse_RXX{mIdx} ; RXX]; 
        rew_barcodes{mIdx}{i} = rew_barcode; 
    end
end

%% 1. Is the time on patch information in mouse behavior? 
%  a) Test R0 vs RR PRT post last rew to see if there is time on patch 
%  b) Test R0R vs RR0 PRT post last reward 

x = [1 2 4 5 7 8]; 

mouse_names = ["m75","m76","m78","m79","m80"]; 
RX_labels = ["10","11","20","22","40","44"];

RX_colors = [.5 1 1 ; 0 1 1 ; .75 .75 1 ; .5 .5 1 ; 1 .5 1; 1 0 1];
h2 = figure();
h2(1) = subplot(2,3,1);
h2(2) = subplot(2,3,2);
h2(3) = subplot(2,3,3);
h2(4) = subplot(2,3,4);
h2(5) = subplot(2,3,5); % the last (odd) axes 

for mIdx = 1:numel(mouse_grps)  
    counter = 1; 
    for iRewsize = [1 2 4]
        for trial_type = [double(sprintf("%i0",iRewsize)) double(sprintf("%i%i",iRewsize,iRewsize))] 
            subplot(2,3,mIdx);hold on
            these_trials = find(mouse_RX{mIdx} == trial_type); 
            
            if trial_type == double(sprintf("%i%i",iRewsize,iRewsize)) 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),mouse_prts{mIdx}(these_trials)-1,8,RX_colors(counter,:)) 
            else 
                scatter(.1 * randn(length(these_trials),1) + x(counter) + zeros(length(these_trials),1),mouse_prts{mIdx}(these_trials),8,RX_colors(counter,:)) 
            end
 
            counter = counter + 1; 
        end 
        
        % significance?
        p = ranksum(mouse_prts{mIdx}(mouse_RX{mIdx} == double(sprintf("%i%i",iRewsize,iRewsize)))-1, ...
            mouse_prts{mIdx}(mouse_RX{mIdx} == double(sprintf("%i0",iRewsize))));
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
    ylim([0 mean(mouse_prts{mIdx}(these_trials)) + 3*std(mouse_prts{mIdx}(these_trials))])
%     ax.XTickLabelRotation = 60;
end 

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(4),'Position',[new,pos{end}(2:end)])
new = mean(cellfun(@(v)v(1),pos(2:3)));
set(h2(5),'Position',[new,pos{end}(2:end)])

