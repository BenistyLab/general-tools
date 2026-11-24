function [d, dU, dR] = SpsdDist(A, B, r, k, VA, VB)

    Symm = @(M) (M + M') / 2;
    
    if nargin < 4
        k = 1;
    end

    A = Symm(A);
    B = Symm(B);

    if nargin < 6
        % Callers can reuse one eigendecomposition per trial across pairs.
        [VA, LA] = eig(A, 'vector');
        [VB, LB] = eig(B, 'vector');
        [~, ia] = sort(LA, 'descend'); VA = VA(:, ia(1:r));
        [~, ib] = sort(LB, 'descend'); VB = VB(:, ib(1:r));
    end
    
    [OA, S, OB] = svd(VA' * VB);
    % Principal-angle cosines can exceed one slightly through roundoff.
    vTheta      = acos(max(-1, min(1, diag(S))));
    
    UA = VA * OA;
    UB = VB * OB;
    
%     UA = VA;
%     UB = VB * OB * OA';
    
    RA = Symm(UA' * A * UA);
    RB = Symm(UB' * B * UB);
    
    dU = norm(vTheta);
    dR = SpdDist(RA, RB);
    d  = sqrt(dU^2 + k * dR^2);
end
