% example_cell.m
% script to run GLM on an example cell
% MGC 4/12/2018

%% load example data
%datafolder = 'C:\Users\malcolmc\Dropbox\Work\Analysis\classifier\data\OF_data\';
datafolder = '/Users/malcolmcampbell/Dropbox/Work/Analysis/classifier/data/OF_data/';
load(strcat(datafolder,'Biggie/0630_rm1_1_T4C5.mat'));
load(strcat(datafolder,'Biggie/0630_rm1_1_pos.mat'));
boxSize = 75;
dt = 0.02;

%% position

% average the two LEDs
x = (posx+posx2)/2;
y = (posy+posy2)/2;

% take out NaN's and replace them with neighboring values
positions = {x, y};
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
x = positions{1}; y = positions{2};

% scale the coordinates to the size of the box
minX = nanmin(x); maxX = nanmax(x);
minY = nanmin(y); maxY = nanmax(y);
x = x-minX; y = y-minY;
xLength = maxX - minX; yLength = maxY - minY; sLength = max([xLength, yLength]);
scale = boxSize / sLength;
x = x * scale; y = y * scale;

%% head_direction
head_direction = atan2(posy2-posy,posx2-posx) + pi;

%% speed
velx = diff([x(1); x]); vely = diff([y(1); y]);
speed = sqrt(velx.^2+vely.^2)/dt;
speed(speed > 100) = NaN;
speed(isnan(speed)) = interp1(find(~isnan(speed)), speed(~isnan(speed)), find(isnan(speed)), 'pchip'); % interpolate NaNs
speed = gauss_smoothing(speed,10); % gaussian kernel smoothing

%% spiketrain
spiketrain = histc(cellTS,post);

%% run GLM
[bestModels,allModelTestFits,tuning_curves,final_pval,fig1] = create_glm(x,y,head_direction,speed,spiketrain,boxSize,dt);