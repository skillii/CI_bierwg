close all;
clear all;
clc;


eta = 0.1;
minf = zeros(41,41);


figure(1);

for i = 0:40
    for j = 0:40
        w0 = [-2+0.1*i, -2+0.1*j];
        
        [~, error] = gradientDescentHw2(w0, 100, eta);
        minf(i+1,j+1) = error(end);
      
    end
end


lokal = minf < -1;
fraction = 100*sum(sum(lokal)) / (41*41);

disp(['fraction: ', num2str(fraction), '%']);

surf(0:40, 0:40, minf);

title('Effect of local Minima (Gradient Descent) for different w_0 = [-2+0.1i, -2+0.1j]');
xlabel('w0');
ylabel('w1');
zlabel('Error');