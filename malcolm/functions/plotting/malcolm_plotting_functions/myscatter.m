function h = myscatter(x,y,col,alpha,sz)

if exist('sz','var')
    h = scatter(x,y,sz,'MarkerFaceColor',col,'MarkerEdgeColor',col,'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha);
else
    h = scatter(x,y,'MarkerFaceColor',col,'MarkerEdgeColor',col,'MarkerFaceAlpha',alpha,'MarkerEdgeAlpha',alpha);    
end
set(gca,'Box','off');
set(gca,'TickDir','out');

end