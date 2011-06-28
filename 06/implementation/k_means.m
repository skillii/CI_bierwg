function [mu, D] = k_means(X, M, mu_0, max_iter)

%K_MEANS Summary of this function goes here
%   asdf
length = size(X,1);

min_diff = 1e-4;  % minimum difference for detecting convergence

Dmat = zeros(length,M);
D = zeros(max_iter,1);
mu = mu_0;

last_mu = mu;

  for i = 1:max_iter
    %calc distances for all points to all centers
    for j = 1:M
      Dmat(:,j) = ((X-ones(length,2)*[mu(j,1) 0; 0 mu(j,2)]).^2)*[1 1]';
    end
    Dmat = Dmat.^(1/2);
    
    %get the minimum distances 
    [val, ind] = min(Dmat');
    D(i) = sum(val);
    
    meanvalues = zeros(M,3); %each row: x, y, counter
    for j = 1:length 
       meanvalues(ind(j),:) = meanvalues(ind(j),:) + [X(j,:) 1];
    end
    
    %calc new center
    for j = 1:M
        meanvalues(j,1:2) = meanvalues(j,1:2)./meanvalues(j,3);
    end    
    
    mu = meanvalues(:,1:2);
    
    
    
    if i > 1
        if sum(abs(lastmu-mu)) < min_diff
            D = D(1:i);
            break;
        end
    end
    
    lastmu = mu;
    
  end

end

