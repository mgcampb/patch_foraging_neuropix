% makes pie charts of neurons with non-zero coefficients

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_18Sep2020_new_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pie_charts\2uL_and_4uL_18Sep2020';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end

var_category_names = {'Kern','TimeSinceRew','TimeOnPatch','TotalRew'};
N_comb = 2^numel(var_category_names);
legend_labels = {}; 
keep = logical(dec2bin(0:N_comb-1)-'0');
for i = 1:N_comb
    str_this = '';
    for j = 1:numel(var_category_names)
        if keep(i,numel(var_category_names)-j+1)
            str_this = strcat(str_this,sprintf('%s_',var_category_names{j}));
        end
    end
    legend_labels{i} = str_this;
end


for i = 1:numel(session_all)
    load(fullfile(paths.results,session_all{i}));
    
    uL1 = contains(var_name,'1uL');
    uL2 = contains(var_name,'2uL');
    uL4 = contains(var_name,'4uL');
    
    var_category = nan(numel(var_category_names),numel(var_name));
    for j = 1:numel(var_category_names)
        var_category(j,:) = contains(var_name,var_category_names{j});
    end
    
    neuron_category = zeros(size(var_category,1),size(beta_all,2));
    for j = 1:size(var_category,1)
%         neuron_category(j,sum(abs(beta_all(uL1 & var_category(j,:),:)),1)>0 & ...
%             sum(abs(beta_all(uL2 & var_category(j,:),:)),1)>0 & ...
%             sum(abs(beta_all(uL4 & var_category(j,:),:)),1)>0) = 2^(j-1);
        neuron_category(j,sum(abs(beta_all(uL2 & var_category(j,:),:)),1)>0 & ...
            sum(abs(beta_all(uL4 & var_category(j,:),:)),1)>0) = 2^(j-1);
    end
    neuron_category = sum(neuron_category);
    
    category_counts = nan(N_comb,1);
    for j = 1:N_comb
        category_counts(j) = sum(neuron_category==j-1);
    end
    
    labels = repmat({''},size(category_counts));
    hfig = figure('Position',[200 200 1000 600]); 
    hfig.Name = session_all{i};
    pie(category_counts,labels);
    mycmap = [0 0 0; cbrewer('qual','Set1',15)];
    colormap(mycmap);
    title(session_all{i},'Interpreter','none');
    legend(legend_labels,'location','northeastoutside');
    saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
    
end

%% subplot

nrow = 3;
ncol = ceil(numel(session_all)/3);
hfig = figure('Position',[50 50 1200 400*ncol]); 
hfig.Name = 'all sessions tight subplot';

ha = tight_subplot(nrow,ncol);
for i = 1:numel(session_all)
    load(fullfile(paths.results,session_all{i}));
    
    uL1 = contains(var_name,'1uL');
    uL2 = contains(var_name,'2uL');
    uL4 = contains(var_name,'4uL');
    
    var_category = nan(numel(var_category_names),numel(var_name));
    for j = 1:numel(var_category_names)
        var_category(j,:) = contains(var_name,var_category_names{j});
    end
    
    neuron_category = zeros(size(var_category,1),size(beta_all,2));
    for j = 1:size(var_category,1)
%         neuron_category(j,sum(abs(beta_all(uL1 & var_category(j,:),:)),1)>0 & ...
%             sum(abs(beta_all(uL2 & var_category(j,:),:)),1)>0 & ...
%             sum(abs(beta_all(uL4 & var_category(j,:),:)),1)>0) = 2^(j-1);
        neuron_category(j,sum(abs(beta_all(uL2 & var_category(j,:),:)),1)>0 & ...
            sum(abs(beta_all(uL4 & var_category(j,:),:)),1)>0) = 2^(j-1);
    end
    neuron_category = sum(neuron_category);
    
    
    category_counts = nan(N_comb,1);
    for j = 1:N_comb
        category_counts(j) = sum(neuron_category==j-1);
    end
    
    labels = repmat({''},size(category_counts));
    axes(ha(i));
    pie(category_counts,labels);
    mycmap = [0 0 0; cbrewer('qual','Set1',15)];
    colormap(mycmap);
    title(session_all{i},'Interpreter','none');
    %legend(legend_labels,'location','northeastoutside');
    
end
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

figure;
pie(ones(N_comb,1));
colormap(mycmap);
hleg = legend(legend_labels);
