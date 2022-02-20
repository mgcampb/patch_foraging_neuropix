%% Script to look at differences in waveform across GLM clusters 

paths = struct;
paths.waveforms = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/waveforms'; 
paths.waveform_clusters = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/waveform_cluster.mat';
paths.glm_clusters = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/glm_results/gmm/sig_cells_table_gmm_mb_cohort_PFC_waveforms.mat';
paths.transients_table = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/structs/transients_table_gmm.mat';
load(paths.glm_clusters)
load(paths.waveform_clusters) 
load(paths.transients_table)

session_all = unique(sig_cells.Session);

%% Iterate over sessions and get clusters, waveform
waveforms = []; 
for sIdx = 1:numel(session_all) 
    dat = load(fullfile(paths.waveforms,session_all{sIdx}));   
    waveforms = [waveforms ; dat.mean_waveform]; 
end

%% Show mean waveform per waveform type 
colors = cbrewer('div','Spectral',10); 
colors = colors([2],:);  % colors([9 2 4],:); 
%  Across all cells, not just sig cells
waveform_types = ["Regular"]; % ["Narrow","Regular","TriPhasic"]; 
numeric_waveform_types = nan(size(waveform_cluster.WaveformType,1),1);
for i_waveform_type = 1:numel(waveform_types)
    this_waveform_type = waveform_types(i_waveform_type);  
    this_waveform_type = this_waveform_type{:}; 
    these_cells = cellfun(@(x) strcmp(this_waveform_type,x),waveform_cluster.WaveformType);  
    numeric_waveform_types(these_cells) = i_waveform_type;
    n_cells_this_waveform = length(find(these_cells)); 
    type_mean = mean(waveforms(these_cells,:));
    type_sem = 1.96 * std(waveforms(these_cells,:)) / sqrt(n_cells_this_waveform);
    
    shadedErrorBar(1000 * dat.t_spk,type_mean,type_sem,'lineprops',{'linewidth',1.5,'color',colors(i_waveform_type,:)})
end

xlabel("Time (msec)")
ylabel("Evoked Extracellular Potential")
legend(waveform_types)
set(gca,'fontsize',13) 

%% Piechart all waveforms  
figure()  
subplot(1,2,1)
p1 = pie(histcounts(numeric_waveform_types(transients_table.Region == "PFC")));
colormap(colors) 
title("All PFC Cells",'Fontsize',14)
subplot(1,2,2)
p2 = pie(histcounts(numeric_waveform_types(~isnan(transients_table.gmm_cluster))));
colormap(colors) 
title("GLM ID Cells",'Fontsize',14)
legend(waveform_types,'FontSize',14) 

for k = 2:2:(2*3)
    if k < 2*3
        set(p1(k),'FontSize',14)
        set(p2(k),'FontSize',14)
    else
        set(p1(k),'Visible',false)
        set(p2(k),'Visible',false)
    end
end
%% statistical test

[tbl,chi2,p] = crosstab(transients_table.gmm_cluster(~isnan(transients_table.gmm_cluster)),numeric_waveform_types(~isnan(transients_table.gmm_cluster)) );
gmm_idx = transients_table.gmm_cluster(~isnan(transients_table.gmm_cluster));
waveform_gmm = numeric_waveform_types(~isnan(transients_table.gmm_cluster));
mouse = transients_table.Mouse(~isnan(transients_table.gmm_cluster));
anova_test = anovan(waveform_gmm,{gmm_idx,mouse})
%% Percent per cluster 
[tbl,chi2,p_val,labels] = crosstab(sig_cells.GMM_cluster,sig_cells.WaveformType); 
glm_numeric_waveform_types = numeric_waveform_types(~isnan(transients_table.gmm_cluster)); 
figure() 
for i_cluster = 1:4
    subplot(1,4,i_cluster)
    p = pie(histcounts(glm_numeric_waveform_types(sig_cells.GMM_cluster == i_cluster)));
    colormap(colors)
    for k = 2:2:(2*3) 
        if k < 2*3
            set(p(k),'FontSize',14)
        else 
            set(p(k),'Visible',false)
        end
    end 
    title(sprintf("Cluster %i",i_cluster))
    set(gca,'FontSize',14)
end
legend(waveform_types,'FontSize',14) 

%% 
peak_types = {transients_table.Cue_peak_pos transients_table.Rew0_peak_pos transients_table.Rew1plus_peak_pos};
peak_type_names = ["Cue Peak","Rew0 Peak","Rew1+ Peak"];
gmm_cluster = transients_table.gmm_cluster;
x_vals = {0:.05:1 0:.05:2 0:.05:2};
figure();hold on 
for i_peak_type = 1:numel(peak_types)
    for i_gmm_cluster = 1:4
        subplot(numel(peak_types),4,4 * (i_peak_type -1) +   i_gmm_cluster) ;hold on
        for i_waveform_type = 1:2
            these_peaks = peak_types{i_peak_type}(gmm_cluster == i_gmm_cluster & numeric_waveform_types == i_waveform_type);
%             peak_pdf = pdf(fitdist(these_peaks,'Kernel','Kernel','Normal'),x_vals{i_peak_type});
            if i_peak_type == 2 && i_waveform_type == 2 && i_gmm_cluster == 4 % super weird thing here
                peak_pdf = ksdensity(these_peaks(1:2:end),x_vals{i_peak_type}); % ,'support',[min(x_vals{i_peak_type}) max(x_vals{i_peak_type})]);
            else 
                peak_pdf = ksdensity(these_peaks,x_vals{i_peak_type});
            end
            plot(x_vals{i_peak_type},peak_pdf,'linewidth',2) 
%             histogram(these_peaks,x_vals{i_peak_type},'normalization','probability')
        end
        if i_gmm_cluster == 1 && i_peak_type == 1
            legend("Narrow-Spiking","Wide-Spiking")
        end 
        prop_sig_peak = 100 * length(find(~isnan(peak_types{i_peak_type}) & gmm_cluster == i_gmm_cluster)) / length(find(gmm_cluster == i_gmm_cluster));

        if i_peak_type == 1
            n_NS = length(find(gmm_cluster == i_gmm_cluster & numeric_waveform_types == 1));
            n_RS = length(find(gmm_cluster == i_gmm_cluster & numeric_waveform_types == 2));
            title(sprintf("Cluster %i \n %i NS %i RS \n %i%% sig peak",i_gmm_cluster,n_NS,n_RS,round(prop_sig_peak)))
        else 
            title(sprintf("%i%% sig peak",round(prop_sig_peak)))
        end
        if i_gmm_cluster == 1 
            ylabel(sprintf("%s Location",peak_type_names(i_peak_type)))
        end
            set(gca,'fontsize',14)
    end  
end 
%% Just look at cluster 2
i_gmm_cluster = 2;  

for i_peak_type = 2:3
    subplot(2,1,(i_peak_type-1));hold on
    for i_waveform_type = 1:2
        these_peaks = peak_types{i_peak_type}(gmm_cluster == i_gmm_cluster & numeric_waveform_types == i_waveform_type);
        peak_pdf = ksdensity(these_peaks,x_vals{i_peak_type});
        plot(x_vals{i_peak_type},peak_pdf,'linewidth',2) 
%         histogram(these_peaks,x_vals{i_peak_type})
    end
    xlabel(sprintf("%s Location",peak_type_names(i_peak_type)))
    set(gca,'fontsize',14)
end 
legend("Narrow-Spiking","Regular-Spiking")
set(gca,'fontsize',14)
subplot(2,1,1)
title("Cluster 2 Peak Time Separated by Waveform")