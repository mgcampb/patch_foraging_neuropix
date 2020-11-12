function b = barWithinRewCorr(withinRewsizeCorr,withinRewsizeP,varname1,varname2,mouse_names)
%Barplot summary of within reward size correlations acr mice

    figure(); 
    colors = cool(3); 
    b = bar(withinRewsizeCorr,'FaceColor','flat');
    for k = 1:3
        b(k).CData = colors(k,:);  
        sig05 = find(withinRewsizeP(:,k) < .05 & withinRewsizeP(:,k) > .01);
        sig01 = find(withinRewsizeP(:,k) < .01 & withinRewsizeP(:,k) > .001);
        sig001 = find(withinRewsizeP(:,k) < .001);
        x = b(k).XEndPoints;
        y = b(k).YEndPoints + .1 * sign(withinRewsizeCorr(:,k))';
        text(x(sig05),y(sig05),"*",'HorizontalAlignment','center',...
            'VerticalAlignment','bottom') 
        text(x(sig01),y(sig01),"* *",'HorizontalAlignment','center',...
            'VerticalAlignment','bottom') 
        text(x(sig001),y(sig001),"* * *",'HorizontalAlignment','center',...
            'VerticalAlignment','bottom') 
    end    
    ylim([-1,1]) 
    legend("Pooled 1uL Trials","Pooled 2 uL Trials","Pooled 4 uL Trials") 
    xticklabels(mouse_names) 
    title({"Pooled Within-Reward-Size Correlations Between" sprintf("%s and",varname1) sprintf("%s",varname2)})
end

