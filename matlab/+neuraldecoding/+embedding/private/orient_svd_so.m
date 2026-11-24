function [U, S, V] = orient_svd_so(U, S, V)
%ORIENT_SVD_SO Orient square SVD factors to have determinant +1.
%
% Given
%
%       Z = U*S*V'
%
% this function changes signs of columns of U and/or V and compensates
% by changing the corresponding diagonal entries of S, so that the
% reconstruction U*S*V' is unchanged.
%
% If U is square:
%       det(U) = +1
%
% If V is square:
%       det(V) = +1
%
% S is allowed to have negative diagonal entries.
%
% For rectangular U or V, determinant is not defined, so no orientation
% constraint is applied to that factor.

r = size(S, 1);

% -------------------------------------------------------------
% First impose a deterministic sign convention on each pair.
%
% Choose the sign of U(:,j) so that its largest-magnitude entry
% is positive. Apply the same sign change to V(:,j), so S does
% not need to change.
% -------------------------------------------------------------

for j = 1:r

    [~, idx] = max(abs(U(:, j)));

    if U(idx, j) < 0
        U(:, j) = -U(:, j);
        V(:, j) = -V(:, j);
    end

end


% -------------------------------------------------------------
% Force U to have determinant +1 if U is square.
%
% Flipping one column of U changes det(U).
% To preserve U*S*V', flip the corresponding entry of S.
% -------------------------------------------------------------

if size(U, 1) == size(U, 2)

    if det(U) < 0
        U(:, end) = -U(:, end);
        S(end, end) = -S(end, end);
    end

end


% -------------------------------------------------------------
% Force V to have determinant +1 if V is square.
%
% Again, compensate by flipping the corresponding singular value.
% -------------------------------------------------------------

if size(V, 1) == size(V, 2)

    if det(V) < 0
        V(:, end) = -V(:, end);
        S(end, end) = -S(end, end);
    end

end

end