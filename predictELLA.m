%%
% Compute the predictions for the specified model on the specified data
%
% inputs -
% model: the ELLA model
% X: the input data to output predictions for
% taskid: the identity of the task to output predictions for
%
% outputs -
% preds: the predictions organized as a column vector
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
function preds = predictELLA(model,X,taskid)
    % compute the model parameter vector for the task
    thetac = model.taskSpecific{taskid}+model.L*model.S(:,taskid);
    if model.useLogistic
	preds = 1./(1+exp(-X*thetac));
    else
	preds = X*thetac;
    end
end
