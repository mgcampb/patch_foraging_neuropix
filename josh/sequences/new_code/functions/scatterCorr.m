function p = scatterCorr(r1,p1,r2,p2,varname1,varname2)
%SCATTERCORR scatterplot 2 correlation results

    significance_labels = zeros(numel(r1),1);
    significance_labels(p1 < .05 & p2 > .05) = 1;
    significance_labels(p1 > .05 & p2 < .05) = 2;
    significance_labels(p1 < .05 & p2 < .05) = 3;
    colors = [0 0 0;1 0 0; 0 0 1; .8 .2 .8];
    figure()
    yline(0,'linewidth',1.5)
    xline(0,'linewidth',1.5)
    hold on
    p = gscatter(r1,r2,significance_labels,colors);
    legend(p(1:4),"No significance", sprintf("%s p < .05",varname1), ...
                                     sprintf("%s p < .05",varname2), ... 
                                     sprintf("%s and %s p < .05",varname1,varname2))
    grid()
    xlim([-1,1])
    ylim([-1,1])
    xlabel(sprintf("%s Pearson Corr",varname1))
    ylabel(sprintf("%s Pearson Corr",varname2))
    title("Mid Responsive Neuron Behavioral Correlations Across Sessions")
end

