function d = so_distance(Q1, Q2)
%SO_DISTANCE Geodesic distance between two matrices in SO(r).
%
%   d = so_distance(Q1, Q2)
%
% Assumptions:
%   - Q1 and Q2 are square matrices of equal size.
%   - Q1 and Q2 are orthogonal.
%   - det(Q1) = det(Q2) = +1.
%   - SVD sign ambiguity has already been resolved externally.
%
% Distance:
%
%   d(Q1,Q2) = ||log(Q1' * Q2)||_F / sqrt(2).
%
% The implementation uses eigenvalue phases instead of LOGM to avoid
% small, spurious complex components caused by numerical roundoff.

    narginchk(2, 2);

    if ~isnumeric(Q1) || ~isnumeric(Q2) || ...
            ~isreal(Q1) || ~isreal(Q2)
        error('so_distance:InvalidInput', ...
            'Q1 and Q2 must be real numeric matrices.');
    end

    if ~ismatrix(Q1) || size(Q1,1) ~= size(Q1,2) || ...
            ~isequal(size(Q1), size(Q2))
        error('so_distance:InvalidDimensions', ...
            'Q1 and Q2 must be square matrices of equal size.');
    end

    if any(~isfinite(Q1(:))) || any(~isfinite(Q2(:)))
        error('so_distance:NonfiniteInput', ...
            'Q1 and Q2 must contain only finite values.');
    end

    r = size(Q1, 1);
    I = eye(r, 'like', Q1);

    tolerance = 100 * r * eps(class(Q1));

    orthogonalityError1 = norm(Q1.' * Q1 - I, 'fro');
    orthogonalityError2 = norm(Q2.' * Q2 - I, 'fro');

    if orthogonalityError1 > tolerance || ...
            orthogonalityError2 > tolerance
        error('so_distance:NotOrthogonal', ...
            'Q1 and Q2 must be numerically orthogonal.');
    end

    if abs(det(Q1) - 1) > 10*tolerance || ...
            abs(det(Q2) - 1) > 10*tolerance
        error('so_distance:NotSO', ...
            'Q1 and Q2 must have determinant +1.');
    end

    % Relative rotation.
    R = Q1.' * Q2;

    % Eigenvalues of R have the form exp(1i*theta_j).
    theta = angle(eig(R));

    % Equivalent to norm(log(R),'fro') / sqrt(2).
    dSquared = 0.5 * sum(theta.^2);
    d = sqrt(max(real(dSquared), 0));
end