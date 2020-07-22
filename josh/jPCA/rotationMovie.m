% This is a hastily written and not well commented function.
% Though fairly self explanatory.

function MV = rotationMovie(Projection, Summary, times, steps2fullRotation, numSteps2show, pixelsToGet,colors,fname)


reusePlot = false;

if ~isempty(fname)
    v = VideoWriter (fname);
    open (v);
end

for step = 1:numSteps2show
    
    
    rotate2jPCA(Projection, Summary, times, steps2fullRotation, step, reusePlot,colors)
    reusePlot = true;
    
    drawnow;
    
    if ~isempty(fname)
%         MV(step) = getframe(gca, pixelsToGet);
        frame = getframe(gca,pixelsToGet);
%         frame = getframe(gca);
        writeVideo (v, frame);
    end
    
end

if ~isempty(fname)
    close(v)
end
end


