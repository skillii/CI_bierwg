function [alpha, mu, Sigma, L] = EM(X, M, alpha_0, mu_0, Sigma_0, max_iter)
%EM calculates expectation maximation
%  size(alpha) = M,1
%  size(mu) = M,size(X,2)
%  size(Sigma) = size(X,2),size(X,2),M

min_diff = 1e-6;  % minimum difference for detecting convergence

alpha = alpha_0;
mu = mu_0;
Sigma = Sigma_0;
L = zeros(1, max_iter);

r_mn = zeros(size(X,1), M);


for i = 1:max_iter    
    
    % === 1 Expectation Step ===
    
    % --- 1.1 Calculate r_mn denominator ---
    r_mn_den = 0;
    
    for m = 1:M
        r_mn_den = r_mn_den + alpha(m) * mvnpdf(X, mu(m,:), Sigma(:,:,m));
    end
    
    % --- 1.2 Calculate r_mn ---
    for m = 1:M
        r_mn(:,m) = alpha(m) * mvnpdf(X, mu(m,:), Sigma(:,:,m)) ./ r_mn_den;
    end
    
    
    % === 2 Maximization Step ===
    
    N_m = sum(r_mn, 1);
    
    for m = 1:M
        mu(m,:) = 1/N_m(m) * sum(repmat(r_mn(:,m), 1, size(X,2)) .* X, 1);
    
        r_mn_temp = repmat(r_mn(:,m), 1, size(X,2));
        mu_temp = repmat(mu(m,:), size(X,1), 1);
        Sigma(:,:,m) = 1/N_m(m) * (r_mn_temp .* (X - mu_temp))' * (X - mu_temp);
        
        alpha(m) = N_m(m) / size(X,1);
    end
    
    
    % === 3 Likelihood Calculation Step ===
    
    L_temp = 0;
    
    for m = 1:M
        L_temp = L_temp + alpha(m) * mvnpdf(X, mu(m,:), Sigma(:,:,m));
    end
    
    L_temp = log(L_temp);
    
    L(i) = sum(L_temp);
    
    
    % --- check convergence ---
    
    if i > 1
        if abs(L(i) - L(i-1)) < min_diff
            L = L(1:i);
            break;
        end
    end
    
end

