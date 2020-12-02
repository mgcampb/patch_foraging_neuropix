%% Plot regression coef to N0 value over days to show development of attention to frequency
close all
datapath = '/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data';
% add behavioral data path
addpath(genpath('/Users/joshstern/Dropbox (Uchida Lab)/patchforaging_behavior/processed_data'));

all_data = dir(fullfile(datapath,'*.mat'));

names = {all_data.name}; % file names
this_mouse = contains(names,num2str(80));

colors = cool(3);
colors2 = jet(5);
more_colors = cool(9); 

mice = [75 76 78 79 80];
nMice = numel(mice);
corr = {nMice};
corrUpper = {nMice};
corrLower = {nMice};
pValues = {nMice};
pValues_prevPatch = {nMice};
corr_prevPatch = {nMice};

for mIdx = 1:nMice
    mouse = mice(mIdx);
    this_mouse = contains(names,num2str(mouse));
    this_mouse_files = all_data(this_mouse);
    this_mouse_files = {this_mouse_files.name};
    
    nSessions = numel(this_mouse_files);
    corr{mIdx} = nan(nSessions,3);
    pValues{mIdx} = nan(nSessions,3);
    corr_prevPatch{mIdx} = nan(nSessions,3);
    pValues_prevPatch{mIdx} = nan(nSessions,3);
    
    for sIdx = 1:nSessions
        % get the file
        mouse_data = this_mouse_files{sIdx};
        load(mouse_data);
        
        % get session data
        prts = patchCSL(:,3) - patchCSL(:,2);
        patchType = patches(:,2);
        rewsize = mod(patchType,10);
        N0 = round(mod(patchType,100)/10);
        N0(N0 == 3) = .125; % just reorder in terms of 
        N0(N0 == 2) = .25;
        N0(N0 == 1) = .5;
        
        % take pearson correlation coefficients for N0
        [r1,p1,ru1,rl1] = corrcoef(N0(rewsize == 1),prts(rewsize == 1));
        [r2,p2,ru2,rl2] = corrcoef(N0(rewsize == 2),prts(rewsize == 2));
        [r4,p4,ru4,rl4] = corrcoef(N0(rewsize == 4),prts(rewsize == 4));
        % some jank code to test previous patch effects 
        prts1 = prts(rewsize == 1); 
        prts2 = prts(rewsize == 2); 
        prts4 = prts(rewsize == 4);
        [r_prev_prt1,p_prev_prt1] = corrcoef(prts1(1:end-1),prts1(2:end));
        [r_prev_prt2,p_prev_prt2] = corrcoef(prts2(1:end-1),prts2(2:end));
        [r_prev_prt4,p_prev_prt4] = corrcoef(prts4(1:end-1),prts4(2:end));
        
        % add to data structure
        corr{mIdx}(sIdx,1) = r1(2);
        corr{mIdx}(sIdx,2) = r2(2);
        corr{mIdx}(sIdx,3) = r4(2);
        pValues{mIdx}(sIdx,1) = p1(2);
        pValues{mIdx}(sIdx,2) = p2(2);
        pValues{mIdx}(sIdx,3) = p4(2);

        corr_prevPatch{mIdx}(sIdx,1) = r_prev_prt1(2);
        corr_prevPatch{mIdx}(sIdx,2) = r_prev_prt2(2);
        corr_prevPatch{mIdx}(sIdx,3) = r_prev_prt4(2);
        pValues_prevPatch{mIdx}(sIdx,1) = p_prev_prt1(2);
        pValues_prevPatch{mIdx}(sIdx,2) = p_prev_prt2(2);
        pValues_prevPatch{mIdx}(sIdx,3) = p_prev_prt4(2);
        
    end
end

%%% First visualizing N0 effect %%% 
% hardcoded 5 mice for now
h = figure();
h(1) = subplot(3,2,1);
h(2) = subplot(3,2,2);
h(3) = subplot(3,2,3);
h(4) = subplot(3,2,4);
h(5) = subplot(3,2,5); % the last (odd) axes

for mIdx = 1:nMice
    subplot(3,2,mIdx)
    set(gca, 'ColorOrder', colors);hold on
    plot(corr{mIdx},'linewidth',1.5);
    sig1Days = find(pValues{mIdx}(:,1) < .05);
    sig2Days = find(pValues{mIdx}(:,2) < .05);
    sig4Days = find(pValues{mIdx}(:,3) < .05);
    scatter(sig1Days,.05 + max(corr{mIdx}(:)) * ones(length(sig1Days),1),[],colors(1,:),'*')
    scatter(sig2Days,.1 + max(corr{mIdx}(:)) * ones(length(sig2Days),1),[],colors(2,:),'*')
    scatter(sig4Days,.15 + max(corr{mIdx}(:)) * ones(length(sig4Days),1),[],colors(3,:),'*')
    xlabel("Session")
    title(sprintf("Mouse %i Correlation between N0 and PRT Over Days of Training",mice(mIdx)))
    plot([0 size(corr{mIdx},1)+1],[0 0],'k--')
end

pos = get(h,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h(5),'Position',[new,pos{end}(2:end)])

%%% Next visualizing previous patch value effect %%% 
h2 = figure();
h2(1) = subplot(3,2,1);
h2(2) = subplot(3,2,2);
h2(3) = subplot(3,2,3);
h2(4) = subplot(3,2,4);
h2(5) = subplot(3,2,5); % the last (odd) axes

for mIdx = 1:nMice
    subplot(3,2,mIdx)
    set(gca, 'ColorOrder', colors);hold on
    plot(corr_prevPatch{mIdx},'linewidth',1.5);
    sig1Days = find(pValues_prevPatch{mIdx}(:,1) < .05);
    sig2Days = find(pValues_prevPatch{mIdx}(:,2) < .05);
    sig4Days = find(pValues_prevPatch{mIdx}(:,3) < .05);
    scatter(sig1Days,.05 + max(corr_prevPatch{mIdx}(:)) * ones(length(sig1Days),1),[],colors(1,:),'*')
    scatter(sig2Days,.1 + max(corr_prevPatch{mIdx}(:)) * ones(length(sig2Days),1),[],colors(2,:),'*')
    scatter(sig4Days,.15 + max(corr_prevPatch{mIdx}(:)) * ones(length(sig4Days),1),[],colors(3,:),'*')
    xlabel("Session")
    title(sprintf("Mouse %i Correlation between PRT and Prev PRT Over Days of Training",mice(mIdx)))
    plot([0 size(corr_prevPatch{mIdx},1)+1],[0 0],'k--')
    if mIdx == 1
        legend("1 uL","2 uL","4 uL")
    end
end

pos = get(h2,'Position');
new = mean(cellfun(@(v)v(1),pos(1:2)));
set(h2(5),'Position',[new,pos{end}(2:end)])

