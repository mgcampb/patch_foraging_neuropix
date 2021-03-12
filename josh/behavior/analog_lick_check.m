%% Check that lick ts are reasonable given analog signal  

paths = struct;
paths.daq_data_dir = '/Users/joshstern/Documents/UchidaLab_NeuralData/patch_foraging_neuropix/josh/behavior/m80_200317_g0'; % dataset to process   
paths.neuropix_data = '/Users/joshstern/Documents/UchidaLab_NeuralData/processed_neuropix_data/all_mice';
sessions = dir(fullfile(paths.neuropix_data,'*.mat'));
sessions = {sessions.name}; 
% note that this folder needs to have the same name of the files 
paths.lick_thresh = 2.5; % voltage threshold for detecting licks (what malcolm used)

%% Load daq data 
% file names
[~,main_name]=fileparts(paths.daq_data_dir);
NIDAQ_file = fullfile(paths.daq_data_dir,strcat(main_name,'_t0.nidq.bin'));
NIDAQ_config = fullfile(paths.daq_data_dir,strcat(main_name,'_t0.nidq.meta'));

% load nidaq data
fprintf('loading nidaq data...\n');
fpNIDAQ=fopen(NIDAQ_file);  
daq_data=fread(fpNIDAQ,[9,Inf],'*int16');
fclose(fpNIDAQ);

% get the nidaq sample rate
dat=textscan(fopen(NIDAQ_config),'%s %s','Delimiter','=');
names=dat{1};
vals=dat{2};
loc=contains(names,'niSampRate');
daq_sampling_rate=str2double(vals{loc});
daq_time = (0:size(daq_data,2)-1)/daq_sampling_rate;

ch_licks = 3; % lick sensor
daq_licks = double(daq_data(ch_licks,:));
clear daq_data; % offload the file

%% Load neuropixels data 

data = load(fullfile(paths.neuropix_data,sessions{25}));  
rew_ts = data.rew_ts; 
lick_ts = data.lick_ts;  

%% Now visualize discretized licks over analog lick signal 
%  - maybe around reward times 
close all

preRew_sec = 1; 
postRew_sec = 1;  
below_minLickSignal_vis = 500; 

rew_events = randi(numel(rew_ts),5,1); 
figure() 
for r = 1:numel(rew_events)  
    subplot(1,numel(rew_events),r);hold on
    iRew = rew_events(r);
    this_rew_sec = rew_ts(iRew);
    daq_vis_bool = daq_time > this_rew_sec - preRew_sec & daq_time < this_rew_sec + postRew_sec;  
    lick_ts_vis = lick_ts(lick_ts > this_rew_sec - preRew_sec & lick_ts < this_rew_sec + postRew_sec); 
    y_loc = min(daq_licks(daq_vis_bool)) - below_minLickSignal_vis;
    
    plot(daq_time(daq_vis_bool) - this_rew_sec,daq_licks(daq_vis_bool),'linewidth',1.5,'color',[0 .6 0])  
    text(lick_ts_vis - this_rew_sec,y_loc + zeros(length(lick_ts_vis),1),'x','HorizontalAlignment','center'); 
    for lick = 1:numel(lick_ts_vis)
        xline(lick_ts_vis(lick) - this_rew_sec) 
    end
    
    yl = ylim(); 
    ylim([y_loc-below_minLickSignal_vis yl(2)]) 
    
    if r == 1 
        ylabel("Analog Lick Signal") 
    end 
    xlabel("Session Time (sec)") 
    title(sprintf("Reward #%i",iRew))
end 

