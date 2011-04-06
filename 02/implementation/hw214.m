close all;
clear all;
clc;


eta = 0.1;





figure(1);

minf = zeros(41,41);


for i = 0:40
    for j = 0:40
        w0 = [-2+0.1*i, -2+0.1*j];
        
        [~, error] = gradientDescentHw2(w0, 100, eta);
        minf(i+1,j+1) = error(end);
      
    end
end



surf(0:40, 0:40, minf);