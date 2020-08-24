function x=gauss_smoothing(x,sigma)
if sigma>0
    % define gaussian filter
    lw=ceil(3*sigma);
    wx=-lw:lw;
    gw=exp(-wx.^2/(2*sigma^2)); 
    gw=gw/sum(gw);
%     gw(1:round(length(gw)/2-1)) = 0; % half gaussian

    x=conv(x,gw,'same');
end

