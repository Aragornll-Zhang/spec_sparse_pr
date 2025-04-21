function [x_init] = spec_initial_onGrid(A, ymag, r)
    T = size(A,1);
    N = size(A, 2);
    N_grid = N;
    f_grid = (0:N_grid-1)/N_grid;   % linspace(0, 1, Nf);
    t = (0:N-1)'; 
    F_grid = exp(2*pi*1i * t * f_grid);
    AFmat   = A*F_grid / sqrt(N);
    ymag_scale = ymag / sqrt(N);
    c_init = spec_initial(AFmat,ymag_scale,r);
    x_init = F_grid * c_init;
    modular = norm(ymag)/sqrt(T);   % re-normalization
    x_init = x_init * modular / norm(x_init); 
end