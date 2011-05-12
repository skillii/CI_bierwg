function [output, z, a1, a2] = feedforwardNN(w, x)

 sigma = @(x) 1./(1+exp(-x));

x_size = size(x);

W1 = reshape(w(1:12), 4,3)';
W2 = reshape(w(13:17), 1,5)';


a1 = [x ones(x_size(1),1)]*W1;

z = sigma(a1);

a2 = [z ones(x_size(1),1)]*W2;

output = sigma(a2);
