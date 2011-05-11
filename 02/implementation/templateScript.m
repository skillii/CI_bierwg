clear all;
close all;

load('backprop.mat');

% Surface plot of target function...

figure;
surf(XX,YY, double(reshape(Ytest, 30, 30)));

%train your neural network here...

%...
%...
%...
%...

Otest = zeros(length(XX), 1);

% Now make surface plot of learned function
for i = 1:length(XX)
    Otest(i) = feedforwardNN(weights, [XX(i), YY(i)]);
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
