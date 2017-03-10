%%
% Create a new ELLA model
%
% inputs -
% options: a struct consisting of key value pairs
% The valid options are:
%	d (required): the dimensionality of the input data points 
%	k (required): the number of latent basis functions
%	useLogistic (default false): true if you want to do a logistic model
%		with log-loss and false for a linear model with squared loss.
%	mu (default 1): the L1 regularization constant
%	lambda (default 1): the L2 regularization constant
%	ridgeTerm (default 1): the L2 regularization constant for learning theta
%	muRatio (default Inf): the L2 regularization constant penalty for the
%			       task specific model component
%	initializeWithFirstKTasks (default false): true if you want to use the
%						   first k single task models to
%						   initialize L, false to
%						   initialize L randomly
%	lastFeatureIsABiasTerm (default false):	true if the last feature is a
%						bias term.  Bias features are
%						not regularized.
%	basisInitializationSeed: used to control the random initialization of
%				 the basis L.  This is useful for peforming
%				 controlled experiments.
%
% outputs -
% model: the created model
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
function model = initModelELLA(options)
    addpath('./externalLibs/spams-matlab/build');

    model = [];
    if ~exist('options','var')
	disp('Must specify model options');
	return; 
    end
    d = options.d;
    k = options.k;
    if ~ismember('useLogistic',fields(options))
	useLogistic = false;
    else
	useLogistic = options.useLogistic;
    end
    if ~ismember('mu',fields(options))
	mu = 1;
    else
	mu = options.mu;
    end
    if ~ismember('lambda',fields(options))
	lambda = 1;
    else
	lambda = options.lambda;
    end
    if ~ismember('ridgeTerm',fields(options))
	ridgeTerm = 1;
    else
	ridgeTerm = options.ridgeTerm;
    end
    if ~ismember('muRatio',fields(options))
	muRatio = Inf;
    else
	muRatio = options.muRatio;
    end
    if ~ismember('initializeWithFirstKTasks',fields(options))
	initializeWithFirstKTasks = false;
    else
	initializeWithFirstKTasks = options.initializeWithFirstKTasks;
    end
    if ~ismember('lastFeatureIsABiasTerm',fields(options))
	lastFeatureIsABiasTerm = false;
    else
	lastFeatureIsABiasTerm = options.lastFeatureIsABiasTerm;
    end

    T = 0;
    D = cell(T,1);
    S = zeros(k,T);
    A = zeros(d*k);
    b = zeros(d*k,1);
    theta = cell(T,1);
    taskSpecific = cell(T,1);
    if ismember('basisInitializationSeed',fields(options))
	rng(options.basisInitializationSeed);
    end
    L = randn(d,k);
    model = struct('d',d,...
		   'k',k,...
		   'useLogistic',useLogistic,...
		   'mu',mu,...
		   'lambda',lambda,...
		   'ridgeTerm',ridgeTerm,...
		   'muRatio',muRatio,...
		   'initializeWithFirstKTasks',initializeWithFirstKTasks,...
		   'lastFeatureIsABiasTerm',lastFeatureIsABiasTerm,...
		   'T',T,...
		   'A',A,...
		   'b',b,...
		   'D',{D},...
		   'S',S,...
		   'theta',{theta},...
		   'taskSpecific',{taskSpecific},...
		   'L',L);
end
