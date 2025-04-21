function Z_trunc = truncated_SVD(Z, r)
    % calculate truncated SVD
    [U,S,V] = svds(Z,r);
    if any(isnan(S(:)))  % for stable
        [U,S,V] = svd(Z, 'econ');
        r_trunc = min(r, length(diag(S)));
        S = S(1:r_trunc, 1:r_trunc);
        U = U(:, 1:r_trunc);
        V = V(:, 1:r_trunc);
        Z_trunc = U * S * V';    
    else
        Z_trunc = U*S*V';
    end
end