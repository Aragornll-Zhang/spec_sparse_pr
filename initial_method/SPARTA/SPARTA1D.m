%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Sparse Phase Retrieval via Truncated Amplitude Flow’’ by G. Wang, L. Zhang, 
% G. B. Giannakis, M. Akcakaya, and J. Chen


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

