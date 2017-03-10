%%
% Use ELLA on the landmine data with a linear-logistic model and log-loss
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
function y = runExperimentLogisticRegression(useLogistic)
    if nargin < 1
	useLogistic = true;
    end
    load('Datasets/landminedata','feature','label');

    % shuffle the tasks
    T = length(feature);
    r = randperm(T);

    feature = {feature{r}};
    label = {label{r}};

    for t = 1 : T
	feature{t}(:,end+1) = 1;
    end
    d = size(feature{1},2);

    X = cell(T,1);
    Xtest = cell(T,1);
    Y = cell(T,1);
    Ytest = cell(T,1);
    for t = 1 : T
	r = randperm(size(feature{t},1));
	traininds = r(1:floor(length(r)/2));
	testinds = r(floor(length(r)/2)+1:end);
	X{t} = feature{t}(traininds,:);
	Xtest{t} = feature{t}(testinds,:);
	Y{t} = label{t}(traininds);
	Ytest{t} = label{t}(testinds);
    end

    model = initModelELLA(struct('k',2,...
	'd',d,...
	'mu',exp(-12),...
	'muRatio',Inf,...
	'lambda',exp(-10),...
	'ridgeTerm',exp(-5),...
	'initializeWithFirstKTasks',true,...
	'useLogistic',useLogistic,...
	'lastFeatureIsABiasTerm',true));
    for t = 1 : T
	model = addTaskELLA(model,X{t},Y{t},t);
    end
    perf = zeros(T,1);
    for t = 1 : T
	preds = predictELLA(model,Xtest{t},t);
	perf(t) = roc(preds,Ytest{t});
    end
    y = mean(perf);
end
