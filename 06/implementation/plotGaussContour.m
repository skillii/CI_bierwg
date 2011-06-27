function plotGaussContour( mu, sigma, colspec )
%Plots three iso-likelihood lines for a bivariate Gaussian distribution. The 
%lines are plotted 1,2 and 3 standard deviations from the mean.
%mu and sigma are mean and covariance matrix, colspec indicates the linestyle.

if nargin < 3; colspec = [0 0 0]; end;
npts = 100;

stdev = sqrtm(sigma);

t = linspace(-pi, pi, npts);
t=t(:);

for i = 1:3
  X = [cos(t) sin(t)] * i * stdev + repmat(mu,npts,1);
  h = line(X(:,1),X(:,2),'color',colspec,'linew',2);
  if(i ~= 1), set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'), end;
end

h = line(mu(1),mu(2),'marker','+','markersize',10,'color',colspec,'linew',2);
set(get(get(h,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');