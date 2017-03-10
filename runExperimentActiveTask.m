%%
% Do an active task selection experiment on the landmine data
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
function y = runExperimentActiveTask()
	useLogistic = true;

	load('Datasets/landminedata','feature','label');
	T = length(feature);
	learningCurves = zeros(T);
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
	    			     'lambda',exp(-10),...
	    			     'ridgeTerm',exp(-5),...
	    			     'initializeWithFirstKTasks',true,...
	    			     'useLogistic',useLogistic,...
	    			     'lastFeatureIsABiasTerm',true));
	learned = logical(zeros(length(Y),1));
	unlearned = find(~learned);
	for t = 1 : T
	    % change the last input to 1 for random, 2 for InfoMax, 3 for Diversity, 4 for Diversity++
	    idx = selectTaskELLA(model,{X{unlearned}},{Y{unlearned}},2);
	    model = addTaskELLA(model,X{unlearned(idx)},Y{unlearned(idx)},unlearned(idx));
	    learned(unlearned(idx)) = true;
	    unlearned = find(~learned);
	    % encode the unlearned tasks
	    for tprime = 1 : length(unlearned)
		model = addTaskELLA(model,X{unlearned(tprime)},Y{unlearned(tprime)},unlearned(tprime),true);
	    end
	    for tprime = 1 : T
	    	preds = predictELLA(model,Xtest{tprime},tprime);
		learningCurves(t,tprime) = roc(preds,Ytest{tprime});
	    end
	end
	y = mean(learningCurves,2);
end
