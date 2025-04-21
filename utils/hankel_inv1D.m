function [x] = hankel_inv1D(Z,D)
    x = get_antidiagonal(Z) ./ D;
end

function [diag_sums] = get_antidiagonal(A)
    % calculate all sum of anti-diagonal
    % e.g. [a11,a12; a21,a22] -> [a11,a12+a21,a22]
    [m, n] = size(A);
    indices = (1:m)' + (1:n) - 1;
    max_index = m + n - 1;
    diag_sums = accumarray(indices(:), A(:), [max_index, 1]);
end

