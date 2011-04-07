function [weights, error] = gradientDescentAdaptiveHw2(w0, numIter, eta, alpha)

weights = zeros(numIter + 1, 2);
error = zeros(numIter, 1);

weights(1,:) = w0(:)';

delta_w_old = [0 0];

for i = 1:numIter
    grad(1,1) = +40 * exp(-20*(0.5*(weights(i,1) - 1)^2 + (weights(i,2) - 1)^2))*(weights(i,1) - 1) + 0.8*exp(-0.1*(4*(weights(i,1) + 1)^2 + 0.5*weights(i,2)^2))*(weights(i,1)+1);
    grad(1,2) =  80 * exp(-20*(0.5*(weights(i,1) - 1)^2 + (weights(i,2) - 1)^2))*(weights(i,2) - 1) + 0.1*exp(-0.1*(4*(weights(i,1) + 1)^2 + 0.5*weights(i,2)^2))*weights(i,2);

    
    delta_w = -(1-alpha*sign(i-1))*eta * grad + alpha*delta_w_old;
    
    weights(i+1,:) = weights(i,:) + delta_w;
    
    error(i) = -2*exp(-20*(0.5*(weights(i+1,1) - 1)^2 + (weights(i+1,2) - 1)^2)) - exp(-0.1*(4*(weights(i+1,1) + 1)^2 + 0.5*weights(i+1,2)^2));

    
    if(i > 1)
        if(error(i) > error(i-1))
            weights(i+1,:) = weights(i,:);
            error(i) = error(i -1);
            eta = eta*0.7;

            delta_w = [0 0];%delta_w_old;

        else
            eta = eta*1.05;
        end
    end
    
    delta_w_old = delta_w;

end





end

