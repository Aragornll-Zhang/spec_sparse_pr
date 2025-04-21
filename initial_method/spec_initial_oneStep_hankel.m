function [x] = spec_initial_oneStep_hankel(A, ymag, r)
    x = initialize_pr( A , ymag.^2);
    [T,N] = size(A);
    [n1, D] = get_split_num(N);
    Z = hankel(x(1:n1),x(n1:end));
    Z_r = truncated_SVD(Z,r);
    x = hankel_inv1D(Z_r,D);
    modular = norm(ymag)/sqrt(T);
    x = x * modular / norm(x);  % normalization
end


function [x_init] = initialize_pr(A , y)
    [T,N] = size(A);
    Y = A' * diag(y) * A / T; 
    [eig_vec , ~ ] = eigs(Y,1);
    modular = sqrt( N * sum(y)/ norm(A,'fro')^2 );
    x_init = modular*eig_vec;
end