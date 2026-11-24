function [matrices, info] = extractNeuronMatrices(X, representationNames)
%EXTRACTNEURONMATRICES Per-trial covXt/corrXt without the centered-SVD gate.
% X is neurons-by-time-by-trials. Matrices are neurons-by-neurons-by-trials.
% Numerical ranks refer to the square matrices, not the activity matrices.
% Constant neurons have zero rows/columns in corrXt (including the diagonal).
% Pearson correlations between nonconstant neurons are unchanged. No neurons
% or trials are removed, no diagonal loading is applied, and data use double.
if nargin < 2, representationNames = ["covXt","corrXt"]; end
names = string(representationNames);
assert(isvector(names) && ~isempty(names) && ~any(ismissing(names)) && ...
    all(ismember(names,["covXt","corrXt"])) && numel(unique(names))==numel(names), ...
    'extractNeuronMatrices:Names','Choose distinct covXt/corrXt names.');
validateattributes(X, {'single','double'}, {'real','finite','nonempty'});
assert(ndims(X)<=3 && size(X,2)>=2, 'extractNeuronMatrices:Shape', ...
    'X must be neurons-by-time-by-trials with at least two time samples.');
[n,t,k] = size(X);
matrices = struct;
info = struct;
for name = reshape(names,1,[])
    matrices.(name) = zeros(n,n,k);
    info.(name) = struct('Rank',zeros(k,1),'IsFullRank',false(k,1), ...
        'ZeroVarianceNeurons',false(n,k),'MatrixDimension',n, ...
        'NumTimeSamples',t,'ConstantNeuronPolicy',"zeroRowsAndColumns");
end
for trial = 1:k
    A = double(X(:,:,trial));
    % Test constancy before centering: mean(0.1*ones(...)) can round away
    % from 0.1 and would otherwise create a spurious nonzero variance.
    constant = all(A==A(:,1),2);
    A = A-mean(A,2);
    A(constant,:) = 0;
    C = (A*A.')/(t-1);
    C = (C+C.')/2;
    variance = diag(C);
    constant = constant | variance==0;
    for name = reshape(names,1,[])
        M = C;
        if name=="corrXt"
            scale = sqrt(variance);
            scale(constant) = 1;
            M = (C./scale)./scale.';
            M(constant,:) = 0; M(:,constant) = 0;
            M = (M+M.')/2;
        end
        matrices.(name)(:,:,trial) = M;
        info.(name).Rank(trial) = rank(M);
        info.(name).IsFullRank(trial) = info.(name).Rank(trial)==n;
        info.(name).ZeroVarianceNeurons(:,trial) = constant;
    end
end
end
