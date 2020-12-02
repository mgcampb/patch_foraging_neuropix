% makes pie charts of neurons with non-zero coefficients

addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions\'));

paths = struct;
paths.results = 'C:\data\patch_foraging_neuropix\GLM_output\run_20201110_all_sessions';
paths.figs = 'C:\figs\patch_foraging_neuropix\glm_pie_charts\20201110_2uL_and_4uL_all_sessions';
if ~isfolder(paths.figs)
    mkdir(paths.figs);
end

opt = struct;
opt.brain_region = 'PFC';
opt.data_set = 'mb';

%%
session_all = dir(fullfile(paths.results,'*.mat'));
session_all = {session_all.name};
for i = 1:numel(session_all)
    session_all{i} = session_all{i}(1:end-4);
end
if strcmp(opt.data_set,'mc')
    session_all = session_all(contains(session_all,'mc'));
elseif strcmp(opt.data_set,'mb')
    session_all = session_all(~contains(session_all,'mc'));
end

var_category_names = {'RewKern','TimeSinceRew','TimeOnPatch','TotalRew'};
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

%% subplot

nrow = 3;
ncol = ceil(numel(session_all)/3);
hfig = figure('Position',[50 50 350*ncol 1000]); 
hfig.Name = sprintf('all sessions tight subplot %s %s cohort',opt.brain_region,opt.data_set);

ha = tight_subplot(nrow,ncol);
counter = 0;
for i = 1:numel(session_all)
    dat = load(fullfile(paths.results,session_all{i}));
    
    if numel(dat.brain_region_rough)==numel(dat.good_cells)
        keep_cell = strcmp(dat.brain_region_rough,opt.brain_region);
    else
        keep_cell = strcmp(dat.brain_region_rough(ismember(dat.good_cells_all,dat.good_cells)),opt.brain_region);
    end
    
    if sum(keep_cell)>0
        
        counter = counter+1;
        
        beta_all = dat.beta_all(:,keep_cell);
           
        uL1 = contains(dat.var_name,'1uL');
        uL2 = contains(dat.var_name,'2uL');
        uL4 = contains(dat.var_name,'4uL');

        var_category = nan(numel(var_category_names),numel(dat.var_name));
        for j = 1:numel(var_category_names)
            var_category(j,:) = contains(dat.var_name,var_category_names{j});
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
        axes(ha(counter));
        pie(category_counts,labels);
        mycmap = [0 0 0; cbrewer('qual','Set1',15)];
        colormap(mycmap);
        title(sprintf('%s\n%d cells',session_all{i},sum(keep_cell)),'Interpreter','none');
        %legend(legend_labels,'location','northeastoutside');
    end
    
end
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');

hfig = figure;
hfig.Name = 'legend';
pie(ones(N_comb,1));
colormap(mycmap);
hleg = legend(legend_labels);
saveas(hfig,fullfile(paths.figs,hfig.Name),'png');
