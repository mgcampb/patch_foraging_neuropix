%% The good, the bad, the behavior 
% what is good behavior? what is love?  
% 1) Plot evolution of PRTs across sessions 
% 2) Explore position / velocity trace changes over sessions 

%% Basics
% close all
datapath = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));

all_data = dir(fullfile(datapath,'*.mat')); 
names = {all_data.name}; % file names

%% 1) PRT drift analysis ... accompany differences in attention to rewards? 

%% Visualize PRT drift over session within mice
colors = cool(3);
mice = [79 80]; 
figcounter = 1;
for mIdx = 1:numel(mice)
    mouse = mice(mIdx);
    this_mouse = contains(names,num2str(mouse)); 
    this_mouse_files = all_data(this_mouse);
    this_mouse_files = {this_mouse_files.name}; 
    for sIdx = 1:numel(this_mouse_files)
        % get the file
        mouse_data = this_mouse_files{sIdx}; 
        session_title = ['m' mouse_data(1:2) ' ' mouse_data(end-6) '/' mouse_data(end-5:end-4)];
        load(mouse_data);
        
        % get session data
        prts = patchCSL(:,3) - patchCSL(:,2);
        patchType = patches(:,2);
        rewsize = mod(patchType,10); 
        
        figure(figcounter) 
        gscatter(1:numel(prts),prts,rewsize,colors,[],10);hold on 
        xlabel("Trial") 
        ylabel("PRT") 
%         title(session_title) 
        figure(figcounter + 1)  
        gscatter((1:numel(prts)) / numel(prts),prts,rewsize,colors,[],10);hold on 
        xlabel("Fraction of session") 
        ylabel("PRT")  
    end  
    figure(figcounter) 
    title(mice(mIdx)) 
    figure(figcounter + 1) 
    title(mice(mIdx))
    figcounter = figcounter + 2;
end  

%% First pass at a regression-based analysis of "good behavior" 

% throw this in python

%% 2) Position / velocity trace analysis ... accompany differences in attention to rewards? 



