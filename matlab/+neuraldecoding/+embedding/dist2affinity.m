%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% converts a distance matrix into an affinity marix by taking 
% exp(-d/2/sigma^2) where sigma is the median of the distnaces
% to the knn (median of all radii
%
% Written by: Gal Mishne & Hadas Benisty
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function aff_mat = dist2affinity(d, params)

if nargin == 1
    params.knn = 30;
    params.eps = 2;
end
nn_dist = sort(d.').';
params.knn = min(params.knn, size(nn_dist, 2));
sigma = params.eps * nanmedian(reshape(nn_dist(:, 1:params.knn), size(nn_dist,1)*params.knn,1));
if sigma==0
    sigma = params.eps * mean(reshape(nn_dist(:, 1:params.knn), size(nn_dist,1)*params.knn,1));
end
aff_mat = exp(-d.^2/(2*sigma^2));