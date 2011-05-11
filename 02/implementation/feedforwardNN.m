function [output, z, a1, a2] = feedforwardNN(w, x)

x_size = size(x);

W1 = reshape(w(1:12), 4,3)';
W2 = reshape(w(13:17), 1,5)';


a1 = [x ones(x_size(1),1)]*W1;

z = 1./(1+exp(-a1));

a2 = [z ones(x_size(1),1)]*W2;

output = 1/(1+exp(-a2));