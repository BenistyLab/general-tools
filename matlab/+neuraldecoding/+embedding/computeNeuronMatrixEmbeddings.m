function embedding = computeNeuronMatrixEmbeddings( ...
    X, numCoordinates, representationNames, affinityParameters)
%COMPUTENEURONMATRIXEMBEDDINGS Independent covXt/corrXt diffusion embeddings.
% Uses every supplied trial, with no SVD rank requirement or regularization.
% SPD for full/full pairs; SPSD at min(rank_i,rank_j) for all other pairs.
% numCoordinates counts NONTRIVIAL coordinates (unlike the legacy API).
% psi is coordinates-by-trials. info contains each trial's rank and a pairwise
% SPSD mask/rank. Zero-rank matrices cannot use this distance and raise errors.
if nargin<2 || isempty(numCoordinates), numCoordinates=20; end
if nargin<3, representationNames=["covXt","corrXt"]; end
if nargin<4, affinityParameters=struct('knn',30,'eps',2); end
validateattributes(numCoordinates,{'numeric'},{'scalar','integer','positive','finite'});
assert(size(X,3)>=2,'computeNeuronMatrixEmbeddings:Trials', ...
    'At least two trials are required.');
[matrices,info] = neuraldecoding.embedding.extractNeuronMatrices(X,representationNames);
names = cellstr(reshape(string(representationNames),1,[]));
params = struct('maxInd',min(numCoordinates+1,size(X,3)));
embedding = struct('representationNames',{names},'info',info, ...
    'diffusionParameters',params,'affinityParameters',affinityParameters);
for i=1:numel(names)
    name=names{i};
    [D,distanceInfo] = neuraldecoding.embedding.pairwise_manifold_distances( ...
        matrices.(name),'SPD');
    K = neuraldecoding.embedding.dist2affinity(D,affinityParameters);
    assert(all(isfinite(K(:))),'computeNeuronMatrixEmbeddings:Affinity', ...
        'Degenerate affinity for %s; inspect trial distances.',name);
    psi = neuraldecoding.embedding.calcDiffusionMap(K,params);
    assert(isreal(psi) && all(isfinite(psi(:))), ...
        'computeNeuronMatrixEmbeddings:Coordinates','Invalid coordinates for %s.',name);
    embedding.distance.(name)=D;
    embedding.affinity.(name)=K;
    embedding.psi.(name)=psi;
    embedding.info.(name).Distance=distanceInfo;
end
end
