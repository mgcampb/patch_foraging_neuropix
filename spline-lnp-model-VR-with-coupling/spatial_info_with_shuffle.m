% Script to compute spatial info in OF recordings and z-score of spatial
% info relative to shuffled spikes
%
% MGC 5/16/2018

%% params

params.dt = 0.02;
params.numshuf = 1000;
params.spatialbinsize = 2.5; % for rate map

% thresholds
thresh.grid = 0.349;
thresh.inter = 10;

%% data
A = readtable('allCells.csv');
A = A(~strcmp(A.SessionTypeVR,'optic_flow_track'),:); % remove OPT sessions
A = A(A.Coverage>90,:); % Don't want to fool around with poor coverage

% keep only sessions after first grid cell was recorded
% keep only sessions on same tetrode as a grid cell was recorded
keep = nan(size(A,1),1);
for i = 1:size(A,1)
    keep(i) = sum(strcmp(A.Mouse,A.Mouse{i}) & A.DateVR<=A.DateVR(i) & ...
        A.Tetrode==A.Tetrode(i) & A.MeanRateOF<thresh.inter & A.GridScore>thresh.grid)>0;
end
A = A(keep==1,:);

% get rid of these mice
% the only grid cell in Daenerys was an axon
% Beyonce and Bjork's grid cells were questionable
A = A(~strcmp(A.Mouse,'Beyonce') & ~strcmp(A.Mouse,'Bjork') & ~strcmp(A.Mouse,'Daenerys'),:);

% find grid cells
grid = A.MeanRateOF < thresh.inter & A.GridScore > thresh.grid;

% get list of all OF sessions to analyze
OF_data_directory = 'C:/Users/malcolmc/Dropbox/Work/Analysis/classifier/data/OF_data/';
%OF_data_directory = 'G:/Dropbox/Work/Analysis/classifier/data/OF_data/';
cell_files = A.SessionOF;
pos_files = A.SessionOF;
boxsize = A.BoxSize;
for i = 1:numel(cell_files)
    mouse = A.Mouse{i};
    session_base = strsplit(A.SessionOF{i},'\');
    session_base = session_base{end};
    cell_files{i} = strcat(OF_data_directory,mouse,'/',session_base,'_T',num2str(A.Tetrode(i)),'C',num2str(A.Cluster(i)),'.mat');
    pos_files{i} = strcat(OF_data_directory,mouse,'/',session_base,'_pos.mat');
end
[cell_files, uniq_idx] = unique(cell_files);
pos_files = pos_files(uniq_idx);
boxsize = boxsize(uniq_idx);
uniqueID = A.UniqueID(uniq_idx);
meanrateOF = A.MeanRateOF(uniq_idx);
grid = grid(uniq_idx);

% still some duplicates
[uniqueID, uniq_idx] = unique(uniqueID);
cell_files = cell_files(uniq_idx);
pos_files = pos_files(uniq_idx);
boxsize = boxsize(uniq_idx);
meanrateOF = meanrateOF(uniq_idx);
grid = grid(uniq_idx);

%% Compute spatial info and shuffle distribution for all cells
numcells = numel(uniqueID);
spatialInfo = nan(numcells,1);
spatialInfoShuffle = nan(numcells,params.numshuf);
for i = 1:numcells
    
    fprintf('Cell %d/%d: %s\n',i,numcells,uniqueID{i});

    % load data
    load(cell_files{i});
    load(pos_files{i});

    % position
    % average the two LEDs
    posx = (posx+posx2)/2;
    posy = (posy+posy2)/2;

    % take out NaN's and replace them with neighboring values
    positions = {posx, posy};
    for k = 1:2
        pos_temp = positions{k};
        nan_ind = find(isnan(pos_temp));
        for m = 1:numel(nan_ind)
            if nan_ind(m) - 1 == 0
                temp = find(~isnan(pos_temp),1,'first');
                pos_temp(nan_ind(m)) = pos_temp(temp);
            else
                pos_temp(nan_ind(m)) = pos_temp(nan_ind(m)-1);
            end
        end
        positions{k} = pos_temp;
    end
    posx = positions{1}; posy = positions{2};

    % scale the coordinates to the size of the box
    minX = nanmin(posx); maxX = nanmax(posx);
    minY = nanmin(posy); maxY = nanmax(posy);
    posx = posx-minX; posy = posy-minY;
    xLength = maxX - minX; yLength = maxY - minY; sLength = max([xLength, yLength]);
    scale = boxsize(i) / sLength;
    posx = posx * scale; posy = posy * scale;

    % compute spatial bin occupancy
    numxbins = ceil((max(posx)-min(posx))/params.spatialbinsize);
    numybins = ceil((max(posy)-min(posy))/params.spatialbinsize);
    offsetx = (min(posx)+numxbins*params.spatialbinsize-max(posx))/2;
    offsety = (min(posy)+numybins*params.spatialbinsize-max(posy))/2;
    xbinedges = min(posx)-offsetx:params.spatialbinsize:min(posx)-offsetx+params.spatialbinsize*numxbins;
    ybinedges = min(posy)-offsety:params.spatialbinsize:min(posy)-offsety+params.spatialbinsize*numybins;
    occupancy = hist3([posx posy],'Edges',{xbinedges,ybinedges});
    occupancy = occupancy(1:numxbins,1:numybins) * params.dt;
    occupancy(occupancy==0) = NaN;

    % compute rate map
    spike_idx = ceil(cellTS/params.dt);
    spikecounts = hist3([posx(spike_idx) posy(spike_idx)],'Edges',{xbinedges,ybinedges});
    spikecounts = spikecounts(1:numxbins,1:numybins);
    ratemap = spikecounts./occupancy;
    meanrate = sum(sum(spikecounts))/nansum(nansum(occupancy));

    % compute spatial info
    logratemap = log2(ratemap/meanrate);
    logratemap(isinf(logratemap)) = 0; % 0*Inf defined as 0 here
    spatialInfo(i) = nansum(nansum((occupancy/nansum(nansum(occupancy))).*(ratemap/meanrate).*logratemap));

    % compute shuffled distribution of spatial info
    for j = 1:params.numshuf
        spike_idx = mod(spike_idx+randsample(100:numel(posx)-100,1),numel(posx))+1;
        spikecounts = hist3([posx(spike_idx) posy(spike_idx)],'Edges',{xbinedges,ybinedges});
        spikecounts = spikecounts(1:numxbins,1:numybins);
        ratemap = spikecounts./occupancy;
        logratemap = log2(ratemap/meanrate);
        logratemap(isinf(logratemap)) = 0; % 0*Inf defined as 0 here
        spatialInfoShuffle(i,j) = nansum(nansum((occupancy/nansum(nansum(occupancy))).*(ratemap/meanrate).*logratemap));
    end
end
spatialInfoZscore = (spatialInfo-nanmean(spatialInfoShuffle,2))./nanstd(spatialInfoShuffle,[],2);