function [mu, D] = k_means(X, M, mu_0, max_iter)

%K_MEANS Summary of this function goes here
%   asdf
length = size(X,1);

D = zeros(length,M);
mu = mu_0

  for i = 1:max_iter
    %calc distances for all points to all centers
    for j = 1:M
      D(:,j) = ((X-ones(length,2)*[mu(j,1) 0; 0 mu(j,2)]).^2)*[1 1]';
    end
    D = D.^(1/2);
    
    %get the minimum distances 
    [~, ind] = min(D');
    
    meanvalues = zeros(M,3); %each row: x, y, counter
    for j = 1:length 
       meanvalues(ind(j),:) = meanvalues(ind(j),:) + [X(j,:) 1];
    end
    
    %calc new center
    for j = 1:M
        meanvalues(j,1:2) = meanvalues(j,1:2)./meanvalues(j,3);
    end    
    
    mu = meanvalues(:,1:2);
    
  end


end

