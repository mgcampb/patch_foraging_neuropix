function varargout = myplot(varargin)
    % Plot that automatically sets Box to off and TickDir to out
    h=builtin('plot',varargin{:});
    set(get(h,'Parent'),'Box','off');
    set(get(h,'Parent'),'TickDir','out');
    varargout{1}=h;  
end