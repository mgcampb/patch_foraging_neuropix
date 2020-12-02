addpath(genpath('C:\code\patch_foraging_neuropix\malcolm\functions'));

paths = struct;
paths.data = 'Z:\Malcolm\patch_foraging_behavior_data';
paths.figs = 'C:\figs\patch_foraging_neuropix\behavior\PRT_plots_across_training';
if ~isfolder(paths.figs)
    mkdir(paths.figs)
end

% analysis opts
opt = struct;
opt.rew_size = [1 2 4];
opt.mouse = '4';
opt.first_behav_day = '1006';

paths.data = fullfile(paths.data,opt.mouse);

%% sessions to analyze

session_all = dir(paths.data);
session_all = {session_all.name}';
session_all = session_all(3:end);
first_behav_day_idx = find(endsWith(session_all,opt.first_behav_day));
session_all = session_all(first_behav_day_idx:end);

%% create fig of right size

ncol = 7;
nrow = 4;
hfig = figure('Position',[50 50 2000 1200]);
hfig.Name = sprintf('PRTs all training sessions mouse MC%s',opt.mouse);

%%
for session_idx = 1:numel(session_all)
    fprintf('Analyzing session %d/%d: %s\n',session_idx,numel(session_all),session_all{session_idx});

    %% load data 
    data_file = dir(fullfile(paths.data,session_all{session_idx},'Patches*.mat'));
    dat = load(fullfile(data_file.folder,data_file.name));
    
    %%
    rew_size_all = mod(dat.patches(:,2),10);
    N0_all = mod(round(dat.patches(:,2)/10),10);
    
    plot_col = cool(3);
    
    subplot(nrow,ncol,session_idx); hold on;
    for i = 1:3
        counter = -1;
        for j = 1:3
            y_this = dat.patches(rew_size_all==opt.rew_size(i) & N0_all==4-j,5);
            jit = randn(size(y_this))*0.005;
            x_this = i*ones(size(y_this))+counter*0.3;
            my_scatter(x_this+jit,y_this,plot_col(i,:),0.5);
            errorbar(i+counter*0.3+0.1,mean(y_this),std(y_this)/sqrt(numel(y_this)),'o','Color',plot_col(i,:));
            counter = counter+1;
        end
    end
    xticks([0.7 1 1.3 1.7 2 2.3 2.7 3 3.3]);
    xticklabels({'Lo','Md','Hi','Lo','Md','Hi','Lo','Md','Hi'});
    ylim([0 max(ylim)]);
    text(1,-(max(ylim)-min(ylim))/8,'1uL','HorizontalAlignment','center');
    text(2,-(max(ylim)-min(ylim))/8,'2uL','HorizontalAlignment','center');
    text(3,-(max(ylim)-min(ylim))/8,'4uL','HorizontalAlignment','center');
    ylabel('PRT (sec)');
    title(session_all{session_idx},'Interpreter','none');
    xlim([0.5 3.5]);
end

saveas(hfig,fullfile(paths.figs,hfig.Name),'png');