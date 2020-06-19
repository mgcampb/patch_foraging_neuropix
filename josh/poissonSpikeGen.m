function [ spikeMat , tVec ] = poissonSpikeGen ( fr , tSim , nTrials )
    dt = 1/1000; % s
    nBins = floor ( tSim / dt ) ;
    spikeMat = rand ( nTrials , nBins ) < fr * dt ;
    tVec = 0: dt : tSim - dt ;
end