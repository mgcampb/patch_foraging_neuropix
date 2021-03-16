function [avgPETH,peak_ix,neuron_order] = avgPETH(FR_decVar,trials_vis,trials_sort,ix_visualize,opt)
% Function to return average of firing rate matrix from FR_decVar from some
% trials with some sorting; just saves a few lines 

%%%% Read in kwargs %%%%
nNeurons = length(FR_decVar.cell_depths);  
if exist('opt', 'var') && isfield(opt,'neurons')
    neurons = opt.neurons;  
    nNeurons = length(neurons); 
else 
    neurons = 1:nNeurons;
end    

dvar = "timesince"; % variable to average over for peaksort
if exist('opt', 'var') && isfield(opt,'dvar')
    dvar = opt.dvar;  
end   

decVar_bins = linspace(0,2,41); % binning for peaksort
if exist('opt', 'var') && isfield(opt,'decVar_bins')
    decVar_bins = opt.decVar_bins;  
end   

sort = true; 
if exist('opt', 'var') && isfield(opt,'sort')
    sort = opt.sort;  
end   

opt.suppressVis = true; 
opt.trials = trials_sort; % only use these trials for sorting

%%%%  

if ~isempty(trials_vis)  
    [~,neuron_order,unsorted_peth] = peakSortPETH(FR_decVar,dvar,decVar_bins,opt);
    tmp_trialsVis_cell = cellfun(@(x) x(neurons,1:ix_visualize),FR_decVar.fr_mat(trials_vis),'UniformOutput',false);
    avgPETH = zscore(mean(cat(3,tmp_trialsVis_cell{:}),3),[],2);  
    if sort == true
        avgPETH = avgPETH(neuron_order,:);  
    end
    [~,peak_ix] = min(unsorted_peth'); % max(unsorted_peth'); 
    peak_ix = peak_ix';
else % just return matrix of zeros
    avgPETH = nan(nNeurons,ix_visualize); 
    neuron_order = nan(nNeurons,1); 
    peak_ix = nan(nNeurons,1);
end

end

