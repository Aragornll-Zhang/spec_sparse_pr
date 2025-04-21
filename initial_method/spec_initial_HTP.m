function [x_init] = spec_initial_HTP(A, ymag, r)
    N = size(A,2);
    N_grid = N;
    f_grid = (0:N_grid-1)/N_grid; % linspace(0, 1, Nf);
    t = (0:N-1)'; 
    F_grid = exp(2*pi*1i * t * f_grid);
    AFmat   = A*F_grid / sqrt(N);
    ymag_scale = ymag / sqrt(N);
    c_init = spec_initial(AFmat,ymag_scale,r);
    c_gt = randn(N,1);            

    % default params
    [max_iter, tol, mu] = deal(100, 1e-8, 0.00075);

    [c_HTP,~] = myHTP1D(AFmat,ymag_scale,c_init,r,max_iter,c_gt,tol,mu);
    x_init = F_grid * c_HTP; 
end

function [x,err] = myHTP1D(A, ymag,x_init,r,max_iter,x_gt,tol,mu)
    % 复现 HTP
    N = size(A,2);
    xk = x_init;
    err = norm(x_gt - exp(-1i * angle(trace(x_gt' * xk))) * xk, 'fro') / norm(x_gt, 'fro');

    for k = 1:max_iter
        zk = A * xk;
        yk = sign(zk) .* ymag;

        x_temp = xk + mu*(A' * (yk - zk));
        [~, idx] = sort(abs(x_temp), 'descend');
        Sk = idx(1:r);
        
        A_Sk = A(:, Sk);
        b_Sk = yk;
        x_temp = zeros(N, 1);
        x_temp(Sk) = A_Sk \ b_Sk;
        
        xk1 = x_temp;
        if norm(xk1 - xk) < tol
            break;
        end
        xk = xk1;
        err  = [err; norm(x_gt - exp(-1i * angle(trace(x_gt' * xk))) * xk, 'fro') / norm(x_gt, 'fro')];
    end
    x = xk;
end

