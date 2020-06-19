function x=gauss_smoothing_half(x,sigma)
if sigma>0
    % define gaussian filter
    lw=ceil(3*sigma);
    wx=0:lw;
    gw=exp(-wx.^2/(2*sigma^2));
    % gw = exp(-wx/sigma);
    gw=gw/sum(gw);

    x2=conv(x,gw,'full');
    x=x2(1:numel(x)); 
end

