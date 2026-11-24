function [D, info] = pairwise_manifold_distances(matrices, geometry)
%PAIRWISE_MANIFOLD_DISTANCES Euclidean, SPD/SPSD, SO/frame trial distances.
% Numeric arrays are rows-by-columns-by-trials; cell vectors are also accepted.
% 'SPD' checks ALL matrices. Each full/full pair uses SpdDist; any pair with
% a deficient matrix uses SpsdDist at the smaller of its two numerical ranks.
% 'SPSD' forces the latter rule even for full/full pairs. No regularization,
% trial dropping, or common-rank truncation is applied. Mixed-rank distances
% use different retained dimensions per pair; info records those dimensions.
% Zero-rank and materially indefinite matrices are rejected explicitly.
narginchk(2,2);
geometry = upper(string(geometry));
assert(isscalar(geometry) && any(geometry==["SPD","SPSD","EUC","SO"]), ...
    'pairwise_manifold_distances:InvalidGeometry','Choose SPD, SPSD, EUC, or SO.');
if iscell(matrices)
    assert(isvector(matrices) && ~isempty(matrices), ...
        'pairwise_manifold_distances:Shape','Expected a nonempty cell vector.');
    series=matrices(:);
else
    validateattributes(matrices,{'single','double'},{'real','finite','nonempty'});
    assert(ndims(matrices)<=3,'pairwise_manifold_distances:Shape', ...
        'Expected rows-by-columns-by-trials.');
    series=cell(size(matrices,3),1);
    for i=1:numel(series), series{i}=matrices(:,:,i); end
end
n=numel(series);
shape=size(series{1});
bases=cell(n,1);
info=struct('Rank',nan(n,1),'IsFullRank',false(n,1), ...
    'UsesSPSD',false(n,n),'PairRank',nan(n,n),'Geometry',geometry);
for i=1:n
    validateattributes(series{i},{'single','double'},{'2d','real','finite','nonempty'});
    assert(isequal(size(series{i}),shape),'pairwise_manifold_distances:Shape', ...
        'All matrices must have the same dimensions.');
    if any(geometry==["SPD","SPSD"])
        A=double(series{i});
        assert(size(A,1)==size(A,2),'pairwise_manifold_distances:Shape', ...
            'SPD/SPSD matrices must be square.');
        tolerance=100*size(A,1)*eps*max(norm(A,'fro'),realmin);
        assert(norm(A-A.','fro')<=tolerance, ...
            'pairwise_manifold_distances:Symmetry','Matrix %d is not symmetric.',i);
        A=(A+A.')/2;
        [basis,eigenvalues]=eig(A,'vector');
        assert(min(eigenvalues)>=-tolerance,'pairwise_manifold_distances:Indefinite', ...
            'Matrix %d has a negative eigenvalue beyond numerical tolerance.',i);
        series{i}=A;
        [~,order]=sort(eigenvalues,'descend');
        bases{i}=basis(:,order);
        info.Rank(i)=rank(A);
        info.IsFullRank(i)=info.Rank(i)==size(A,1);
        assert(info.Rank(i)>0,'pairwise_manifold_distances:ZeroRank', ...
            'Matrix %d has rank zero; the SPSD distance requires positive rank.',i);
    end
end
D=zeros(n,n);
for i=2:n
    for j=1:i-1
        A=series{i}; B=series{j};
        switch geometry
            case {"SPD","SPSD"}
                r=min(info.Rank([i j]));
                useSPSD=geometry=="SPSD" || ~all(info.IsFullRank([i j]));
                if useSPSD, distance=SpsdDist(A,B,r,1,bases{i}(:,1:r),bases{j}(:,1:r));
                else, distance=SpdDist(A,B); end
                info.UsesSPSD(i,j)=useSPSD; info.UsesSPSD(j,i)=useSPSD;
                info.PairRank(i,j)=r; info.PairRank(j,i)=r;
            case "EUC"
                distance=norm(A-B,'fro');
            case "SO"
                if size(A,1)==size(A,2), distance=so_distance(A,B);
                else, distance=stiefel_so_distance(A,B); end
        end
        assert(isreal(distance) && isfinite(distance) && distance>=0, ...
            'pairwise_manifold_distances:Distance','Invalid distance for matrices %d and %d.',i,j);
        D(i,j)=distance; D(j,i)=distance;
    end
end
end
