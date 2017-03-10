%%
% Encode a new task using the specified model
%
% inputs -
% model: the ELLA model
% X: the training data (data instances are rows) 
% Y: the training labels
%
% outputs -
% s: the weights over the latent basis vectors to encode the task
% theta: the optimal single task model
% D: the hessian of the loss function evaluated about theta
% taskSpecific: the task specific model component
%
% Copyright (C) Paul Ruvolo and Eric Eaton 2013
%
% This file is part of ELLA.
%
% ELLA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ELLA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ELLA.  If not, see <http://www.gnu.org/licenses/>.
function [s theta D taskSpecific] = encodeTaskELLA(model, X, Y)
    if model.useLogistic
	Yall = zeros(size(Y,1),2);
	Yall(Y==1,1) = 1;
	Yall(Y==0,2) = 1;
	B = zeros(size(X,2),2);
	weighting = 1./size(Y,1)*ones(size(Y,1),1);
	% compute a MAP parameter estimate using multinomial logstic regression
	[bhat yhat myHess] = MLRegress(X,weighting,Yall,B,model.ridgeTerm);
	if any(isnan(bhat(:))) | any(isnan(myHess(:)))
	    bhat = zeros(size(bhat));
	    myHess = zeros(size(myHess));
	end
	theta = bhat(:,1);
	% convert form the format of the Hessian returned by the logisti regression
	% algorithm to the assumptions of our algorithm
	D = -.5*myHess./length(Y);
    else
	% if the last feature is a bias term don't regularize it
	if model.lastFeatureIsABiasTerm
	    theta = inv(X'*X + model.ridgeTerm*diag([ones(1,size(X,2)-1) 0]))*X'*Y;
	    D = ((X'*X) + model.ridgeTerm*diag([ones(1,size(X,2)-1) 0]))./length(Y);
	else
	    theta = inv(X'*X + model.rideTerm*diag(ones(size(X,2),1)))*X'*Y;
	    D = ((X'*X) + model.ridgeTerm*eye(size(X,2)))./length(Y);
	end
    end
    % use the sparse additive modeling toolbox to encode the task
    [s taskSpecific] = sparseEncode(model,theta,D);
end
