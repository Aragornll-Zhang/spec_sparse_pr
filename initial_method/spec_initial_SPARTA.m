function [x_init] = spec_initial_SPARTA(A, ymag, r)
    
    % On grid - Sparse-PR
    T = size(A,1);
    N = size(A, 2);
    N_grid = N;
    f_grid = (0:N_grid-1)/N_grid;   % linspace(0, 1, Nf);
    t = (0:N-1)'; 
    F_grid = exp(2*pi*1i * t * f_grid);
    Amatrix   = A*F_grid / sqrt(N);
    ymag_scale = ymag / sqrt(N);


    % SPARTA params
    if exist('Params_sparta', 'var')           == 0,  Params_sparta.n2            = 1;    end
    if isfield(Params_sparta, 'n1')            == 0,  Params_sparta.n1            = N; end             % signal dimension
    if isfield(Params_sparta, 'nonK')          == 0,  Params_sparta.nonK          = N-r;   end		% nonK = n1 - K, where x is a K-sparse signal
    if isfield(Params_sparta, 'm')             == 0,  Params_sparta.m             = floor(T) ;  end     % number of measurements
    if isfield(Params_sparta, 'cplx_flag')     == 0,  Params_sparta.cplx_flag     = 1;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
    if isfield(Params_sparta, 'T')             == 0,  Params_sparta.T             = 200;  end    	% number of iterations
    if isfield(Params_sparta, 'mu')            == 0,  Params_sparta.mu            = 1 * (1 - Params_sparta.cplx_flag) + .9 * Params_sparta.cplx_flag;  end		% step size / learning parameter
    if isfield(Params_sparta, 'gamma_lb')      == 0,  Params_sparta.gamma_lb      = 1;   end	% thresholding of throwing small entries of Az 0.3 is a good one
    if isfield(Params_sparta, 'npower_iter')   == 0,  Params_sparta.npower_iter   = 100;   end		% number of power iterations
    if isfield(Params_sparta, 'power_trunc')   == 0,  Params_sparta.power_trunc   = 5;   end		% 1 means truncated power iterations per iteration
    if isfield(Params_sparta, 'tol')           == 0,  Params_sparta.tol           = 1e-9;  end     % number of MC trials
    if isfield(Params_sparta, 'support_opi')   == 0,  Params_sparta.support_opi   = 2;   end      % 1 means truncated power iterations per iteration
    
    y = ymag_scale.^2;
    x_gt_rand = randn(N,1)+1j*randn(N,1);
    [~, z] = SPARTA1D(y , x_gt_rand  , Params_sparta, Amatrix);
    x_init = F_grid*z;
end


%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Sparse Phase Retrieval via Truncated Amplitude Flow’’ by G. Wang, L. Zhang, 
% G. B. Giannakis, M. Akcakaya, and J. Chen
% from `https://gangwg.github.io/SPARTA/codes.html`


function [Relerrs, z] = SPARTA1D(y, x, Params, Amatrix) %#ok<INUSD>
    Arnorm  = sqrt(sum(abs(Amatrix).^2, 2)); % norm of rows of Amatrix
    ymag    = sqrt(y);
    normest = Params.m * Params.n1 / sum(sum(abs(Amatrix))) * (1 / Params.m) * sum(ymag);
    ynorm   = ymag ./ (Arnorm .* normest);
    
    %% finding largest normalized inner products
    Anorm      = bsxfun(@rdivide, Amatrix, Arnorm);
    [ysort, ~] = sort(ynorm, 'ascend');
    ythresh    = ysort(round(Params.m / (1.2))); % 6/5 the orthogonality-promoting initialization parameter
    ind        = (abs(ynorm) >= ythresh);
    
    %% estimate the support of x
    Aselect  = Anorm(ind, :);
    if Params.support_opi == 1
        % based on orthogonality-promoting initialization
        rdata_opi= sum(abs(Aselect).^2, 1);
        [~, sind_opi] = sort(rdata_opi, 'descend');
        Supp_opi = sind_opi(1 : round(Params.n1 - Params.nonK));
    else
        % based on squared quantities
        Ya       = bsxfun(@times, abs(Amatrix).^2, y);
        rdata    = sum(Ya, 1);
        [~,sind] = sort(rdata, 'descend');
        Supp_opi = sind(1 : (Params.n1 - Params.nonK));
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% power iterations for sparse orthogonality-promoting initialization
    if Params.power_trunc == 0
        
        z0       = randn(Params.n1, Params.n2);
        z0       = z0 / norm(z0, 'fro');    % Initial guess
        for t    = 1:Params.npower_iter                   % Truncated power iterations
            z0   = Aselect' * (Aselect * z0);
            z0   = z0 / norm(z0, 'fro');
        end
        
        z0(setdiff((1:Params.n1), Supp_opi)) = 0;
        z0       = z0 / norm(z0, 'fro');
        z0        = normest * z0;                   % Apply scaling
        
    elseif Params.power_trunc == 1
        
        z0       = randn(Params.n1, Params.n2);
        z0       = z0 / norm(z0, 'fro');    % Initial guess
        for t    = 1:Params.npower_iter                   % Truncated power iterations
            z0   = Aselect' * (Aselect * z0);
            sz0  = sort(abs(z0), 'descend');
            z0(abs(z0) < sz0(Params.n1 - Params.nonK) - eps) = 0;
            z0   = z0 / norm(z0, 'fro');
        end
        
    elseif Params.power_trunc == 2
        
        Aselect  = Anorm(ind, Supp_opi);
        zk0      = randn(Params.n1 - Params.nonK, Params.n2);
        zk0      = zk0 / norm(zk0, 'fro');   
        for t    = 1:Params.npower_iter                  
            zk0  = Aselect' * (Aselect * zk0);
            zk0  = zk0 / norm(zk0, 'fro');
        end
        z0       = zeros(Params.n1, 1);
        z0(Supp_opi) = zk0;
        
    else
        
        Asample  = Amatrix(:, Supp_opi);
        Arnormx  = sqrt(sum(abs(Asample).^2, 2)); % norm of rows of Amatrix
        
        % finding largest normalized inner products
        Anormx   = bsxfun(@rdivide, Amatrix, Arnormx);
        ynormx   = ymag ./ (Arnormx .* normest);
        ysortx   = sort(ynormx, 'ascend');
        
        ythreshx = ysortx(round(Params.m / (1.2))); % 6/5 the orthogonality-promoting initialization parameter
        indx     = (abs(ynormx) >= ythreshx);
        Aselectx = Anormx(indx, Supp_opi);
        
        zk0      = randn(Params.n1 - Params.nonK, Params.n2);
        zk0      = zk0 / norm(zk0, 'fro');    % Initial guess
        for t    = 1:Params.npower_iter                   % Truncated power iterations
            zk0  = Aselectx' * (Aselectx * zk0);
            zk0  = zk0 / norm(zk0, 'fro');
        end
        
        z0      = zeros(Params.n1, 1);
        z0(Supp_opi) = zk0;
        
    end
    
    z       = normest * z0;                   % Apply scaling
    Relerrs = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error
    
    for t = 1: Params.T
        
        Az       = Amatrix * z; %A(z);
        ratio    = abs(Az) ./ ymag;
        yz       = ratio > 1 / (1 + Params.gamma_lb);
        ang      = Params.cplx_flag * exp(1i * angle(Az)) + (1 - Params.cplx_flag) * sign(Az);
        
        grad     = Amatrix' * (yz .* ymag .* ang - yz .* Az) / Params.m;
        z        = z + Params.mu * grad;
        [sz, ~]  = sort(abs(z), 'descend');
        z(abs(z) < sz(Params.n1 - Params.nonK) - eps) = 0;
        Relerrs  = [Relerrs; norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x,'fro')]; %#ok<AGROW>
        if Relerrs(end) < Params.tol - eps
            break;
        end
        
    end

end