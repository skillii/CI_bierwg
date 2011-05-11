function [dw] = backpropNNFull(w, X, Y)

dw = zeros(17,1);

x_size = size(X);

for i = 1:x_size(1)
    dw = dw + backpropNNSingle(w, X(i,:), Y(i));
end
