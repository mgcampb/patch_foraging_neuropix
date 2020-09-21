function Yi = myLinearInterp(X, Y, Xi)   
    % Jan code for fast warping of matrices
    % (https://www.mathworks.com/matlabcentral/answers/64118-use-interp1-to-interpolate-a-matrix-row-wise)
    
    % X and Xi are column vectors, Y a matrix with data along the columns
    [~, Bin] = histc(Xi, X);  
    H            = diff(X);       % Original step size
    
    % Extra treatment if last element is on the boundary:
    sizeY = size(Y);
    if Bin(length(Bin)) >= sizeY(1)
       Bin(length(Bin)) = sizeY(1) - 1;
    end
    
    Xj = Bin + (Xi - X(Bin)) ./ H(Bin);
    % Yi = ScaleTime(Y, Xj);  % FASTER MEX CALL HERE
    % return;
    
    % Interpolation parameters:
    Sj = Xj - floor(Xj);
    Xj = floor(Xj);
    % Shift frames on boundary:
    edge     = (Xj == sizeY(1));
    Xj(edge) = Xj(edge) - 1;
    Sj(edge) = 1;           % Was: Sj(d) + 1;
    % Now interpolate:
    if sizeY(2) > 1
       Sj = Sj(:, ones(1, sizeY(2)));  % Expand Sj
    end
    Yi = Y(Xj, :) .* (1 - Sj) + Y(Xj + 1, :) .* Sj; 
    
     psthStretched = psth(find(tt>event,1)-45:find(tt>event,1)+45)';
end
