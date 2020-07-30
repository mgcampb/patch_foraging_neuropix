function color = get_color(gain,contrast)
    if ~exist('contrast','var')
        contrast = 100;
    end
    gains_all = [1 0.8 0.7 0.6 0.5 0.2]; % all the gain values we use       
    contrasts_all = [100 50 20 10 5 2 0]; % all the contrast values we use
    if sum(gains_all==gain)==0 || sum(contrasts_all==contrast)==0
        color = [1 0 0];
    elseif gain==1
        colors_contrast = gray(numel(contrasts_all)+1);
        colors_contrast = colors_contrast(1:end-1,:);
        color = colors_contrast(contrasts_all==contrast,:);
    else
        colors_gain = [0 0 0; cool(4); 0 0 1];
        color = colors_gain(gains_all==gain,:);
        if contrast==20
            color = color * 0.75;
        elseif contrast==10
            color = color * 0.5;
        elseif contrast==5
            color = color * 0.25;
        end
    end
end