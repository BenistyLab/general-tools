function R = project_to_so_from_svd(O1, O2)
%PROJECT_TO_SO_FROM_SVD Construct orientation-preserving relative rotation.
%
% Inputs
%   O1, O2   r-by-r orthogonal matrices obtained from
%
%                 [O1, S, O2] = svd(V1' * V2);
%
% Output
%   R        r-by-r matrix in SO(r), so det(R) = +1.
%
% The usual orthogonal polar factor is
%
%       O1 * O2'
%
% which can have determinant -1.
%
% This function applies the standard orientation correction
%
%       R = O1 * D * O2'
%
% where
%
%       D = diag(1,...,1,det(O1*O2'))
%
% so that det(R) = +1.

r = size(O1, 1);

D = eye(r, 'like', O1);

relativeDeterminant = det(O1 * O2');

if relativeDeterminant < 0
    D(end,end) = -1;
end

R = O1 * D * O2';

% Remove tiny numerical asymmetries / imaginary roundoff.
R = real(R);

end