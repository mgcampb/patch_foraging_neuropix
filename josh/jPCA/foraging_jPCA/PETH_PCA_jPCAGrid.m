function PETH_PCA_jPCAGrid(conditions,jPCA_data,Projection,labels,neuronGroupName)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    for condIdx = conditions
        fr_mat = jPCA_data(condIdx).A';
        % sort PETH
        fr_mat_norm = zscore(fr_mat,[],2);
%         [~,index] = max(fr_mat_norm');
%         if condIdx == min(conditions) % sort by only one condition
%             [~,index_sort] = sort(index);
%         end
        % show sorted PETH
        figure();colormap('jet');
        subplot(1,3,1)
%         imagesc(flipud(fr_mat_norm(index_sort,:)))
        imagesc(flipud(fr_mat_norm))
        xlabel("Time (ms)")
        xticks([0,50,100]) % just for the 2 sec data
        xticklabels([0,1000,2000])
        title(sprintf("%s %s PETH",neuronGroupName,labels{condIdx}))
        subplot(2,3,2)
        plot(Projection(condIdx).tradPCAproj(:,1:2),'linewidth',1.5)
        title("PCA projections over time")
        subplot(2,3,5)
        plot(Projection(condIdx).proj(:,1:2),'linewidth',1.5)
        title("jPCA projections over time")
        subplot(2,3,3)
        plot(Projection(condIdx).tradPCAproj(:,1),Projection(condIdx).tradPCAproj(:,2),'linewidth',1.5)
        title("PCA trajectory")
        subplot(2,3,6)
        plot(Projection(condIdx).proj(:,1),Projection(condIdx).proj(:,2),'linewidth',1.5)
        title("jPCA trajectory")
    end
end

