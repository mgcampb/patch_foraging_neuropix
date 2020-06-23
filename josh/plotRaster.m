function [] = plotRaster (spikeMat, tVec)
% Visualize raster plot
hold all ;
for trialCount = 1: size ( spikeMat ,1)
    spikePos = tVec ( spikeMat ( trialCount ,:) ) ;
    for spikeCount = 1: length ( spikePos )
        plot ([ spikePos ( spikeCount ) spikePos ( spikeCount ) ] , [ trialCount -0.4
            trialCount +0.4]) ;
    end
end
ylim ([0 size(spikeMat , 1) +1]) ;
