function rxnil_dynamics_plot3(X,tts,colors,pre_cue_ix,median_cue_len,post_leave_ix)
    % Plot RXNil dynamics in 3d
    hold on
    for tt = tts
        % pre-cue
        plot3(X{tt}(1,1:pre_cue_ix), ...
              X{tt}(2,1:pre_cue_ix), ...
              X{tt}(3,1:pre_cue_ix),'linewidth',2,'color',[.5 .5 .5]) 
        % cue
        plot3(X{tt}(1,pre_cue_ix:pre_cue_ix + median_cue_len), ...
              X{tt}(2,pre_cue_ix:pre_cue_ix + median_cue_len), ...
              X{tt}(3,pre_cue_ix:pre_cue_ix + median_cue_len),'linewidth',2,'color',[.2 .7 .2]) 
        % trial 
        plot3(X{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix), ...
              X{tt}(2,pre_cue_ix + median_cue_len:end-post_leave_ix), ... 
              X{tt}(3,pre_cue_ix + median_cue_len:end-post_leave_ix),'linewidth',2,'color',colors{tt}) 

        tick_interval = 10;  
        tt_len = length(X{tt}(1,pre_cue_ix + median_cue_len:end-post_leave_ix));

        % add time ticks for cue
        time_ticks = (pre_cue_ix+tick_interval):tick_interval:(pre_cue_ix + median_cue_len - tick_interval);
        plot3(X{tt}(1,time_ticks),X{tt}(2,time_ticks),X{tt}(3,time_ticks), ... 
              'ko', 'markerSize', 8, 'markerFaceColor',[.2 .7 .2]); 
        % add time ticks for on patch
        time_ticks = ((pre_cue_ix + median_cue_len)+tick_interval):tick_interval:(tt_len - tick_interval);
        plot3(X{tt}(1,time_ticks),X{tt}(2,time_ticks),X{tt}(3,time_ticks), ... 
              'ko', 'markerSize', 8, 'markerFaceColor',colors{tt}); 

        % add mark for reward
        if mod(tt,2) == 0
            plot3(X{tt}(1,(pre_cue_ix + median_cue_len)+50),X{tt}(2,(pre_cue_ix + median_cue_len)+50),X{tt}(3,(pre_cue_ix + median_cue_len)+50), ... 
                  'kd', 'markerSize', 15, 'markerFaceColor',colors{tt}); 
        end 

            % add some marks to make the trajectories more interpretable 
        plot3(X{tt}(1,1),X{tt}(2,1),X{tt}(3,1), ...
              'ko', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5]);
        plot3(X{tt}(1,pre_cue_ix),X{tt}(2,pre_cue_ix),X{tt}(3,pre_cue_ix), ...
              'ko', 'markerSize', 15, 'markerFaceColor',[.2 .7 .2]);  

        % add O at start of trial
        plot3(X{tt}(1,pre_cue_ix + median_cue_len),X{tt}(2,pre_cue_ix + median_cue_len),X{tt}(3,pre_cue_ix + median_cue_len), ... 
              'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
        % add X at end of trial
        plot3(X{tt}(1,end-post_leave_ix),X{tt}(2,end-post_leave_ix),X{tt}(3,end-post_leave_ix), ... 
              'ko', 'markerSize', 15, 'markerFaceColor',colors{tt});
        % add X at end of trial
        plot3(X{tt}(1,end-post_leave_ix),X{tt}(2,end-post_leave_ix),X{tt}(3,end-post_leave_ix), ... 
              'kx', 'markerSize', 15, 'markerFaceColor',[.5 .5 .5],'linewidth',2);

    end 
    xticklabels([]);yticklabels([]);zticklabels([]) 
    grid()  
    view(-135,30)
    set(gca,'fontsize',15)
end

