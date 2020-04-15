function [] = save_figs(image_save_dir,h,imtype)
    
    if ~exist('image_save_dir','var')
        image_save_dir = fullfile(pwd,'images');
    end
    
    if exist(image_save_dir,'dir')~=7
        mkdir(image_save_dir);
    end
    
    if ~exist('imtype','var')
        imtype = 'png';
    end
    
    % get all figures
    if ~exist('h','var')
        h = findobj('type','figure');
    end
    
    for i = 1:numel(h)
        if strcmp(imtype,'pdf')
            set(h(i),'PaperOrientation','landscape');
        end
        if isempty(h(i).Name)
            saveas(h(i),fullfile(image_save_dir,num2str(h(i).Number)),imtype);
        else
            saveas(h(i),fullfile(image_save_dir,sprintf('%s.%s',h(i).Name,imtype)),imtype);
        end
    end

end