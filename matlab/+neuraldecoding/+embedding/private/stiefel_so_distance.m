function [d, dGrassmann, dSO, principalAngles, R] = ...
    stiefel_so_distance(V1, V2, k, tolerance)
%STIEFEL_SO_DISTANCE Distance between two ordered orthonormal frames.
%
% V1 and V2 must be T-by-r matrices satisfying:
%
%       V1' * V1 = I
%       V2' * V2 = I
%
% The distance combines:
%
%   1. Grassmann distance between the temporal subspaces.
%   2. SO(r) distance between the ordered bases within those subspaces.
%
%       d^2 = dGrassmann^2 + k*dSO^2
%
% Usage:
%
%   d = stiefel_so_distance(V1, V2);
%
%   [d, dGrassmann, dSO] = ...
%       stiefel_so_distance(V1, V2);
%
%   [d, dGrassmann, dSO, theta, R] = ...
%       stiefel_so_distance(V1, V2, k, tolerance);
%
% Inputs:
%   V1, V2    T-by-r orthonormal matrices
%   k         Nonnegative weight for SO(r) component; default is 1
%   tolerance Numerical tolerance; optional
%
% Outputs:
%   d               Combined distance
%   dGrassmann      Grassmann geodesic distance
%   dSO             Canonical SO(r) distance
%   principalAngles Principal angles between the subspaces
%   R               Relative in-subspace rotation
%
% Important:
%   For r < T, this is a product distance and not the exact canonical
%   Stiefel geodesic distance. The SO(r) term assumes consistent ordering
%   and sign alignment of the singular vectors.

if nargin < 3 || isempty(k)
    k = 1;
end

if nargin < 4 || isempty(tolerance)
    tolerance = 1e-10 * max(size(V1, 1), 1);
end

% Validate input types.
if ~isnumeric(V1) || ~isnumeric(V2) || ...
        ~isreal(V1) || ~isreal(V2) || ...
        ndims(V1) > 2 || ndims(V2) > 2
    error('stiefel_so_distance:InvalidInput', ...
        'V1 and V2 must be real numeric matrices.');
end

if ~isequal(size(V1), size(V2))
    error('stiefel_so_distance:SizeMismatch', ...
        'V1 and V2 must have the same size T-by-r.');
end

[T, r] = size(V1);

if r < 1 || T < r
    error('stiefel_so_distance:InvalidFrameSize', ...
        'Inputs must have size T-by-r with 1 <= r <= T.');
end

if any(~isfinite(V1(:))) || any(~isfinite(V2(:)))
    error('stiefel_so_distance:NonfiniteInput', ...
        'V1 and V2 must contain only finite values.');
end

if ~isscalar(k) || ~isreal(k) || ~isfinite(k) || k < 0
    error('stiefel_so_distance:InvalidWeight', ...
        'k must be a finite, real, nonnegative scalar.');
end

if ~isscalar(tolerance) || ~isreal(tolerance) || ...
        ~isfinite(tolerance) || tolerance <= 0
    error('stiefel_so_distance:InvalidTolerance', ...
        'tolerance must be a finite, real, positive scalar.');
end

% Verify that both matrices are on St(T,r).
I = eye(r, 'like', V1);

orthogonalityError1 = norm(V1' * V1 - I, 'fro');
orthogonalityError2 = norm(V2' * V2 - I, 'fro');

if orthogonalityError1 > tolerance || ...
        orthogonalityError2 > tolerance
    error('stiefel_so_distance:NotStiefel', ...
        ['Columns must be orthonormal. Errors are %.3e and %.3e; ' ...
        'tolerance is %.3e.'], ...
        orthogonalityError1, ...
        orthogonalityError2, ...
        tolerance);
end

% Cross-Gram matrix between the two frames.
crossGram = V1' * V2;

% Joint SVD:
%
%   V1' * V2 = O1*cos(Theta)*O2'
%
% The diagonal values are the cosines of the principal angles.
[O1, cosineMatrix, O2] = svd(crossGram);

cosines = diag(cosineMatrix);

% Protect acos against numerical roundoff.
cosines = min(max(cosines, 0), 1);

principalAngles = acos(cosines);

% Grassmann geodesic distance.
dGrassmann = norm(principalAngles, 2);

% Orthogonal polar factor describing the relative orientation of the
% ordered frames within their identifiable subspaces.
% R = O1 * O2';
% R = real(R);

% determinantR = det(R);
% if determinantR < 0 && ...
%             abs(determinantR + 1) <= 10 * tolerance
%         error('stiefel_so_distance:OppositeFrameOrientations', ...
%             ['The relative polar factor has determinant -1 and is not ' ...
%              'in SO(r). Align one singular-vector sign consistently, ' ...
%              'or use only the Grassmann component.']);
%     elseif abs(determinantR - 1) > 10 * tolerance
%         error('stiefel_so_distance:InvalidPolarFactor', ...
%             'The computed polar factor is not numerically in SO(r).');
%     end
R = project_to_so_from_svd(O1, O2);

determinantR = det(R);

if abs(determinantR - 1) > 10 * tolerance
    error('stiefel_so_distance:InvalidPolarFactor', ...
        'The corrected relative rotation is not numerically in SO(r).');
end



% Canonical SO(r) distance using eigenvalue phases.
phases = angle(eig(R));
dSO = norm(phases, 2) / sqrt(2);

% Combined product distance.
d = sqrt(dGrassmann^2 + k * dSO^2);
end