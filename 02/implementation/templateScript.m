clear all;
close all;

load('backprop.mat');

% Surface plot of target function...

figure;
surf(XX,YY, double(reshape(Ytest, 30, 30)));

w = randn(17,1);

eta = 1.0;
numiter = 100;

e = zeros(1,numiter+1);

for i = 1:numiter
    [dw] = backpropNNFull(w, X, Y);
    
    [o] = feedforwardNN(w, X);
    e(i) = (o'-Y)'*(o'-Y);
    
    new_w = eta-dw*eta;
    
    [o] = feedforwardNN(new_w, X);
    e(i+1) = (o'-Y)'*(o'-Y);
    
    if(e(i+1) <= e(i))
        w = new_w;
        eta = eta*0.7;
    else
        eta = eta*1.05;
    end
end

figure
plot(0:numiter, e);


Otest = zeros(length(XX), 1);

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w, Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
