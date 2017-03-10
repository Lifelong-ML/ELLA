%%
% Select the next task to learn using a specified particular criterion
%
% inputs -
% model: the ELLA model
% Xs: a cell array containing some seed data to use for selecting the next task
%     to learn
% Ys: a cell array containing the labels of the seed data
% selectionCriterion: 1 (random)
% 		      2 (InfoMax)
% 		      3 (diversity)
% 		      4 (diversity++)
% Xtarget (optional): the data for the target task
% Ytarget (optional): the labels for the target task
%
% outputs -
% taskid: the next task to learn (as selected by the speciied selection
% 	  criterion)
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
function taskid = selectTaskELLA(model,Xs,Ys,selectionCriterion,Xtarget,Ytarget)
    doTargeted = nargin == 6;
    if ~ismember(selectionCriterion,[1:5])
	error('Invalid Task Selection Criterion Specified');
    end
    if doTargeted & selectionCriterion ~= 2
	error('Targeted selection only works with InfoMax');
    end
    if selectionCriterion == 1		% random selection
	taskid = randint(1,1,[1 length(Ys)]);
	return;
    end
    taskGoodness = zeros(length(Ys),1);
    if doTargeted
	[sTarget thetaTarget DTarget taskSpecificTarget] = ...
	    encodeTaskELLA(model,Xtarget,Ytarget);
    end
    for t = 1 : length(Ys)
	[sCurr wCurr DCurr taskSpecificCurr] = ...
	    encodeTaskELLA(model,Xs{t},Ys{t});
	if selectionCriterion == 2
	    ACurr = (model.A + kron(sCurr*sCurr',DCurr))./(model.T+1) + model.lambda*eye(model.d*model.k);
	    if ~doTargeted
		% compute using the d-optimality criterion
		taskGoodness(t) = logdet(ACurr);
	    else
		% compute using the d-optimality for the parameter vector for the task
		Psi = kron(sTarget, eye(model.d));
		taskGoodness(t) = logdet(Psi'*ACurr*Psi);
	    end
	end
	if selectionCriterion == 3 | selectionCriterion == 4
	    if sum(sCurr) ~= 0
		% encode using the difference in loss between single task model and encoded model (Diversity Heuristic)
		taskGoodness(t) = (wCurr - model.L*sCurr - taskSpecificCurr)'*DCurr*(wCurr - model.L*sCurr - taskSpecificCurr);
	    end
	end
    end
    if selectionCriterion ~= 4
	% always pick the best task
	[dc taskid] = max(taskGoodness);
    else
	% select task probabilistically
	if sum(taskGoodness) == 0
	    taskid = 1;
	else
	    probs = taskGoodness.^2./sum(taskGoodness.^2);
	    taskid = min(find(cumsum(probs)>rand));
	end
    end
end
