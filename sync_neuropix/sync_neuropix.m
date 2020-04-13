%% SCRIPT FOR SYNCING NEUROPIX SPIKING DATA TO NIDAQ BEHAVIORAL DATA 
% MGC 4/10/2020
% Adapted from Giocomo lab code

%% options
tic
opt = struct;
opt.data_dir = 'D:\neuropix_patchForaging\m80_200317_g0'; % dataset to process
opt.dt = 0.02; % 50 Hz sampling for downsampled velocity
opt.lick_thresh = 2.5; % voltage threshold for detecting licks
opt.save_dir = 'D:\processed_neuropix_data'; % where to save processed data
opt.save_name = '80_20200317.mat'; % file to save processed data to

%% libraries
% need these for loading KS output
addpath(genpath('C:\code\spikes'));
addpath(genpath('C:\code\npy-matlab'));

% need these for processing lick channel
addpath(genpath('C:\code\HGRK_analysis_tools'));

%% location of data
[~,main_name]=fileparts(opt.data_dir);
NIDAQ_file = fullfile(opt.data_dir,strcat(main_name,'_t0.nidq.bin'));
NIDAQ_config = fullfile(opt.data_dir,strcat(main_name,'_t0.nidq.meta'));
spike_dir = fullfile(opt.data_dir,strcat(main_name,'_imec0'));

%% load spike times
fprintf('loading spike data...\n');
sp = loadKSdir(spike_dir);

%% hack for when we entered 30000 into KS2 for NP samp rate instead of true calibrated value (usually something like 30000.27)

% find true NP ap sample rate
np_ap_meta_file = fullfile(spike_dir,strcat(main_name,'_t0.imec0.ap.meta'));
fp_np = fopen(np_ap_meta_file);
dat=textscan(fp_np,'%s %s','Delimiter','=');
names=dat{1};
vals=dat{2};
loc=contains(names,'imSampRate');
true_sampling_rate=str2double(vals{loc});
fclose(fp_np);

% find sample rate that was entered into KS2 (probably 30KHz)
ks_params_file = fullfile(spike_dir,'params.py');
fp_ks = fopen(ks_params_file);
dat=textscan(fp_ks,'%s %s','Delimiter','=');
names=dat{1};
vals=dat{2};
loc=contains(names,'sample_rate');
ks_sampling_rate = str2double(vals{loc});
fclose(fp_ks);

% correct spike times
st=sp.st;
in_samples=st*ks_sampling_rate;
sp.st=in_samples/true_sampling_rate;

%% load nidaq data
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

% load VIRMEN daq data
daq_file_virmen = dir(fullfile(opt.data_dir,'*.daq'));
[daq_data_virmen, daq_time_virmen] = daqread(fullfile(opt.data_dir,daq_file_virmen.name));

%% process nidaq data

fprintf('processing nidaq data...\n');
ch_vel = 2; % signal used to compute velocity from rotary encoder
ch_licks = 3; % lick sensor
ch_rews = 4; % reward valve opening
ch_events = 5; % 'events' signalled from within DAQ for patch cue appearing, etc

% virmen channel numbers
ch_vel_virmen = 1;
ch_licks_virmen = 2;

% VELOCITY 
% load TaskVars*.mat file to obtain baseline rotary encoder voltage to subtract to get actual signal
taskVars_file = dir(fullfile(opt.data_dir,'TaskVars*'));
taskVars = load(fullfile(opt.data_dir,taskVars_file.name));
taskVars = taskVars.taskVars;

% Compute velocity
daq_vel = double(daq_data(ch_vel,:)); % voltage used to compute velocity
daq_vel_virmen = daq_data_virmen(:,ch_vel_virmen);
daq_vel = (daq_vel-min(daq_vel))/(max(daq_vel)-min(daq_vel)); % scale to be 0 to 1, then...
daq_vel = daq_vel*(max(daq_vel_virmen)-min(daq_vel_virmen)) + min(daq_vel_virmen); % scale to be same range as virmen daq data
daq_vel = taskVars.scaling(2) * (daq_vel - taskVars.idle_voltage_offset(1)); % convert voltage to velocity

% downsample velocity
velt = min(daq_time):opt.dt:max(daq_time);
vel = interp1(daq_time,daq_vel,velt,'spline');

% REWARD
rew_voltage = double(daq_data(ch_rews,:));
% binarize signal
rew_voltage(rew_voltage <= max(rew_voltage)/2) = 0;
rew_voltage(rew_voltage > max(rew_voltage)/2) = 1;
rew_idx = find(diff(rew_voltage)>0.5)+1;
rew_ts = daq_time(rew_idx)';
% double check that the number of rewards matches the RewTimes file
RewTimes_file = dir(fullfile(opt.data_dir,'RewTimes*'));
RewTimes = load(fullfile(opt.data_dir,RewTimes_file.name));
num_rew = 0;
for i = 1:numel(RewTimes.RewTimes)
    num_rew = num_rew+numel(RewTimes.RewTimes{i})-2;
end
assert(numel(rew_ts) == num_rew+1);

% LICKS   
% Hyung Goo's lick detection code (modified by Malcolm):
daq_licks = double(daq_data(ch_licks,:));
daq_licks_virmen = daq_data_virmen(:,ch_licks_virmen);
daq_licks = (daq_licks-min(daq_licks))/(max(daq_licks)-min(daq_licks)); % scale to be 0 to 1, then...
daq_licks = daq_licks*(max(daq_licks_virmen)-min(daq_licks_virmen)) + min(daq_licks_virmen); % scale to be same range as virmen daq data
lick_ts = detect_small_lick_by_deflection_malcolm(4-daq_licks,opt.lick_thresh,daq_sampling_rate); 
lick_ts = lick_ts/daq_sampling_rate; % convert to seconds

% PATCH CUE, STOP, and LEAVE
daq_events = double(daq_data(ch_events,:));

% discretize signal
daq_events(daq_events<=2000 & daq_events>=-2000) = 0;
daq_events(daq_events>2000) = 1;
daq_events(daq_events<-2000) = -1;

% patch cue
patchCue_idx = diff(daq_events)>0.25 & daq_events(2:end)>0.25;
patchCue_idx = find(patchCue_idx)+1;
patchCue_ts = daq_time(patchCue_idx)';

% patch leave
patchLeave_idx = diff(daq_events)<-0.25 &  daq_events(2:end)<-0.25;
patchLeave_idx = find(patchLeave_idx)+1;
patchLeave_ts = daq_time(patchLeave_idx)'-0.5; % correct the 0.5 second difference

% make sure there are the same number of patchCue and patchLeave
patchCue_ts = patchCue_ts(1:numel(patchLeave_ts));

% patch stop
patchStop_ts = [];
patchCSL = [];
patchCL = [];
for i = 1:numel(patchCue_ts)
    next_rew_ts = rew_ts(rew_ts>patchCue_ts(i));
    if isempty(next_rew_ts)
        patchCL = [patchCL; patchCue_ts(i) patchLeave_ts(i)];
    else
        if next_rew_ts(1) < patchLeave_ts(i)
            patchStop_ts = [patchStop_ts; next_rew_ts(1)];
            patchCSL = [patchCSL; patchCue_ts(i) next_rew_ts(1) patchLeave_ts(i)];
        else
            patchCL = [patchCL; patchCue_ts(i) patchLeave_ts(i)];
        end
    end
end

% make sure number of patches found here matches the "patches" variable
% load patches file
patches_file = dir(fullfile(opt.data_dir,'Patches*'));
patches = load(fullfile(opt.data_dir,patches_file.name));
patches = patches.patches;
assert(numel(patchStop_ts) == size(patches,1));

%% CORRECT FOR DRIFT BETWEEN IMEC AND NIDAQ BOARDS

% TWO-PART CORRECTION
% 1. Get sync pulse times relative to NIDAQ and Imec boards.  
% 2. Quantify difference between the two sync pulse times and correct in
% spike.st. 
% PART 1: GET SYNC TIMES RELATIVE TO EACH BOARD
% We already loaded most of the NIDAQ data above. Here, we access the sync
% pulses used to sync Imec and NIDAQ boards together. The times a pulse is
% emitted and registered by the NIDAQ board are stored in syncDatNIDAQ below.

fprintf('correcting drift...\n');
syncDatNIDAQ=daq_data(1,:)>1000;

% convert NIDAQ sync data into time data by dividing by the sampling rate
ts_NIDAQ = strfind(syncDatNIDAQ,[0 1])/daq_sampling_rate; 

% ts_NIDAQ: these are the sync pulse times relative to the NIDAQ board
% Now, we do the same, but from the perspective of the Imec board. 
LFP_config = dir(fullfile(spike_dir,'*.lf.meta'));
LFP_config = fullfile(LFP_config.folder,LFP_config.name);
LFP_file = dir(fullfile(spike_dir,'*.lf.bin'));
LFP_file = fullfile(LFP_file.folder,LFP_file.name);
dat=textscan(fopen(LFP_config),'%s %s','Delimiter','=');
names=dat{1};
vals=dat{2};
loc=contains(names,'imSampRate');
lfp_sampling_rate=str2double(vals{loc});

% for loading only a portion of the LFP data
fpLFP = fopen(LFP_file);
fseek(fpLFP, 0, 'eof'); % go to end of file
fpLFP_size = ftell(fpLFP); % report size of file
fpLFP_size = fpLFP_size/(2*384); 
fclose(fpLFP);

% get the sync pulse times relative to the Imec board
fpLFP=fopen(LFP_file);
fseek(fpLFP,384*2,0);
ftell(fpLFP);
datLFP=fread(fpLFP,[1,round(fpLFP_size/4)],'*int16',384*2); % this step used to take forever
fclose(fpLFP);
syncDatLFP=datLFP(1,:)>10; 
ts_LFP = strfind(syncDatLFP,[0 1])/lfp_sampling_rate;

% ts_LFP: these are the sync pulse times relative to the Imec board
% PART 2: TIME CORRECTION
lfpNIDAQdif = ts_LFP - ts_NIDAQ(1:size(ts_LFP, 2)); % calculate the difference between the sync pulse times
fit = polyfit(ts_LFP, lfpNIDAQdif, 1); % linear fit 
correction_slope = fit(1); % this is the amount of drift we get per pulse (that is, per second)

% save the old, uncorrected data as sp.st_uncorrected and save the new,
% corrected data as sp.st (as many of your analyses are using sp.st).
sp.st_uncorrected = sp.st; % save uncorrected spike times (st)
st_corrected = sp.st - sp.st * correction_slope; % in two steps to avoid confusion
sp.st = st_corrected; % overwrite the old sp.st

%% save processed data
fprintf('saving processed data...\n');
if exist(opt.save_dir,'dir')~=7
    mkdir(opt.save_dir);
end
save(fullfile(opt.save_dir,opt.save_name),...
    'sp','velt','vel','rew_ts','lick_ts','patchCue_ts','patchStop_ts','patchLeave_ts','patchCSL','patchCL','patches');
fprintf('done.\n');
toc