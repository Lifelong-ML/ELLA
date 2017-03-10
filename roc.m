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
function [a one_minus_specificity sensitivity] = roc(y,s)
	if nargin == 2
		y = [y s];
	end
	if length(unique(y(:,1)))==size(y,1)
		% use a faster version of ROC that doesn't handle tied outputs properly
		a = ROC_efficient(y(:,1),y(:,2));
		one_minus_specficity = [];
		sensitivity = [];
		return;
	end
	% make sure the labels are in {-1,1}
	y(:,1) = -1*y(:,1); 		% lame should not be doing this, but need to reverse sign
	y(y(:,2)==0,2) = -1;
        y = sortrows(y,1);
	totalPos = length(find(y(:,2) ==1));
	totalNeg = length(find(y(:,2) ==-1));
	currPos = 0;
	currNeg = totalNeg;
	vals = unique(y(:,1));
	sensitivity = zeros(length(vals) + 1,1);
	one_minus_specificity = zeros(length(vals) + 1,1);
	for j = 1 : length(vals)
		indices = find(y(:, 1) == vals(j));
		currPos = currPos + length(find(y(indices,2) == 1));
		currNeg = currNeg - length(find(y(indices,2) == -1));
		sensitivity(j + 1) = currPos / totalPos;
		one_minus_specificity(j + 1) = 1 - currNeg / totalNeg;
	end
	a = trapz(one_minus_specificity, sensitivity);
end
function A=ROC_efficient(Y,X)
   n=size(Y,1);
   for i=1:size(Y,2)
     [s t]=sort(Y(:,i));
     f=find(X(t,i)==1);
     s=size(f,1);
     A(i)=(mean(f)-(s+1)/2)/(n-s);
   end
end
