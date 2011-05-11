function [weights, error] = gradientDescentImpulseHw2(w0, numIter, eta, alpha)

weights = zeros(numIter + 1, 2);
error = zeros(numIter, 1);

weights(1,:) = w0(:)';

lastdiff = [0 0];

for i = 1:numIter
    grad(1,1) = +40 * exp(-20*(0.5*(weights(i,1) - 1)^2 + (weights(i,2) - 1)^2))*(weights(i,1) - 1) + 0.8*exp(-0.1*(4*(weights(i,1) + 1)^2 + 0.5*weights(i,2)^2))*(weights(i,1)+1);
    grad(1,2) =  80 * exp(-20*(0.5*(weights(i,1) - 1)^2 + (weights(i,2) - 1)^2))*(weights(i,2) - 1) + 0.1*exp(-0.1*(4*(weights(i,1) + 1)^2 + 0.5*weights(i,2)^2))*weights(i,2);

    
    
    weights(i+1,:) = weights(i,:) - (1-alpha)*eta * grad + alpha*lastdiff;
    
    error(i) = -2*exp(-20*(0.5*(weights(i,1) - 1)^2 + (weights(i,2) - 1)^2)) - exp(-0.1*(4*(weights(i,1) + 1)^2 + 0.5*weights(i,2)^2));
    lastdiff = weights(i+1,:) - weights(i,:);
end





end