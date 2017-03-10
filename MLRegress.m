% MLRegress Multivariate Logistic Regression
% [Bhat Yhat] = MLRegress(X,W,Y,B,alpha)
% Fits model of form: Yhat proportional to  exp(X*Bhat)
% The objective function is multinomial log-likelihood with a 
% quadratic weight penalty term
% This function is optimized using a  Newton-Raphson method.
% There are c dependent variables, n independent variables and m examples.
% X is an m x n  matrix. 
% W is an m x 1 vector specifying the relative importance of each example
% Y is a m x c matrix of response measures  in [0,1] range. Rows add up to 1.
% B is an n x c matrix of initial weight values. B = 0 is recommended. 
% alpha is the weight penalty term. It controls the strenght of a
% Gaussian prior on the B favoring small magnitudes of B. 
% A bias term can be introduced by having a column of  X be all "ones"
% The program automatically detects this fact and does not apply the
% decay term alpha to that column.
% There are many possible solutions to Bhat, we choose the one for which
% the last column is all zeros.

% There is a short tutorial on this topic at the 
% Machine Perception Laboratory's Web site and the Kolmogorov project site.


% Copyright (C) Javier R. Movellan April 2002
% Copyright (C)  Machine Perception Laboratory
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

function [B, Yhat, MyHess]= MLRegress(X,W,Y,B,alpha)
n = size(X,2);
m = size(X,1);
c = size(Y,2);
bias_term = find(((max(X) ==1) + (min(X) == 1)) ==2); % finds whether column of
                                               % ones.
alphaeye  = alpha*eye(n);
alphaeye(bias_term,bias_term) = 0; % gets rid of decay term for the bias

MyHess = eye(n*(c-1));
MyGrad = zeros(n*(c-1),1);
delta =1;
Yhatold = Y;
iteration = 0;
W = abs(W);
W = m*W/sum(W);

while(delta> 0.001 & iteration<100)
   iteration = iteration+1;
   logprobs = X*B;
   logprobs = logprobs - repmat(max(logprobs')',[1 size(logprobs,2)]);
   Yhat = exp(logprobs);
   Yhat = Yhat./repmat(sum(Yhat,2),1,size(Yhat,2)); % normalize rows to add up to 1 
   delta = max(max((abs(Yhatold - Yhat))));   
   % construct the Hessian matrix  and the gradient vector
   % without loss of generality we fix the last column of B to zero.
   for i=1:c-1 
      for j=1:c-1
   		temp = Yhat(:,i).* (kdelta(i,j) - Yhat(:,j)).*W;
			%L = diag(temp);    
                        myn = size(temp,1);
                        L = sparse(1:myn, 1:myn,temp); % creates sparse diagonal
      	MyHess((i-1)*n+1:i*n, (j-1)*n+1: j*n) = - X'*L*X -kdelta(i,j)*alphaeye;    
   	end;
	MyGrad((i-1)*n+1:i*n) = X'*(Y(:,i) - Yhat(:,i))-alphaeye*B(:,i);
end
Yhatold = Yhat;
% The Newton-Raphson algorithm
B(:,1:c-1) = B(:,1:c-1) - reshape(MyHess\MyGrad,n,c-1);
end

function k = kdelta(i,j)
if i == j 
   k = 1;
else
   k=0;
end

