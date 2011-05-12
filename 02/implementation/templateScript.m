clear all;
close all;

load('backprop.mat');

% Surface plot of target function...

figure;
surf(XX,YY, double(reshape(Ytest, 30, 30)));
title('learning function');


eta = 1.0;
numiter = 1000;
numtrials = 10;

e = zeros(numiter+1, numtrials);

w0 = zeros(17,numtrials);
w10 = zeros(17,numtrials);
w100 = zeros(17,numtrials);
w300 = zeros(17,numtrials);
w1000 = zeros(17,numtrials);

figure


for trial = 1:numtrials

    w = randn(17,1);
    
    w0(:,trial) = w;

    for i = 1:numiter
        [dw] = backpropNNFull(w, X, Y);

        [o] = feedforwardNN(w, X);
        e(i, trial) = (o-Y)'*(o-Y);

        new_w = w-dw*eta;

        [o] = feedforwardNN(new_w, X);
        e(i+1, trial) = (o-Y)'*(o-Y);

        if(e(i+1, trial) <= e(i, trial))
            w = new_w;
            eta = eta*1.05;
        else
            eta = eta*0.7;
        end
        
        
        if(i == 10), w10(:,trial) = w; end
        if(i == 100), w100(:,trial) = w; end
        if(i == 300), w300(:,trial) = w; end
        if(i == 1000), w1000(:,trial) = w; end
        
    end
    plot(0:numiter, e);
    hold on;
end

title('evolution of the error function');
xlabel('iteration');
ylabel('error');




[~,besttrial] = min(e(numiter, :)');
[~,worsttrial] = max(e(numiter, :)');


%%%%%%%%%%%%%% best trial

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w0(:, besttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 0 iterations, best trial');


% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w10(:, besttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 10 iterations, best trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w100(:, besttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 100 iterations, best trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w300(:, besttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 300 iterations, best trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w1000(:, besttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 1000 iterations, best trial');


%%%%%%%%%%%%%%%%%%%% worst trial

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w0(:, worsttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 0 iterations, worst trial');


% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w10(:, worsttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 10 iterations, worst trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w100(:, worsttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 100 iterations, worst trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w300(:, worsttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 300 iterations, worst trial');

% Now make surface plot of learned function
for i = 1:length(Xtest)
    Otest(i) = feedforwardNN(w1000(:, worsttrial), Xtest(i,:));
end

figure;
surf(XX,YY, reshape(Otest, 30, 30));
title('learned function, after 1000 iterations, worst trial');
