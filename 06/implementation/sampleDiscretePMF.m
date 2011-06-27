function Y = sampPM(X, PM, N)
%function Y = sampPM(PM, N)
%
%Draw N samples from the given discrete probability mass function PM,
%defined over the support given in X.

Y = zeros(1,N);
PM_cum = cumsum(PM);
offs = rand(1) * 1/N;
comb = (offs) : (1/N) : ((1/N)*(N-1)+offs);

j = 1;
for i=1:N
  while(comb(i) >= PM_cum(j))    %search comb(i) in cumulative sum
    j = j + 1;
  end
  Y(i) = X(j);
end

Y = Y(randperm(N));