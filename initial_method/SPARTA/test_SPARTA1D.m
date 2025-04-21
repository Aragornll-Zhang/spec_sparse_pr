%% Implementation of the Sparse Truncated Amplitude Flow algorithm proposed in the paper
%  ``SPARTA: Sparse phase retrieval by truncated amplitude flow'' 
% by G. Wang, L. Zhang, G. B. Giannakis, J. Chen, and M. Ackacaya.

clear;
% clc;
close all;
% rng(1)
n = 200;
m = 150;
r = 5;

if exist('Params', 'var')           == 0,  Params.n2            = 1;    end
if isfield(Params, 'n1')            == 0,  Params.n1            = n; end             % signal dimension
if isfield(Params, 'nonK')          == 0,  Params.nonK          = n-r;   end		% nonK = n1 - K, where x is a K-sparse signal

if isfield(Params, 'm')             == 0,  Params.m             = floor(m) ;  end     % number of measurements
if isfield(Params, 'cplx_flag')     == 0,  Params.cplx_flag     = 1;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'T')             == 0,  Params.T             = 200;  end    	% number of iterations
if isfield(Params, 'mu')            == 0,  Params.mu            = 1 * (1 - Params.cplx_flag) + .9 * Params.cplx_flag;  end		% step size / learning parameter
if isfield(Params, 'gamma_lb')      == 0,  Params.gamma_lb      = 1;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'npower_iter')   == 0,  Params.npower_iter   = 100;   end		% number of power iterations
if isfield(Params, 'power_trunc')   == 0,  Params.power_trunc   = 5;   end		% 1 means truncated power iterations per iteration
if isfield(Params, 'tol')           == 0,  Params.tol           = 1e-18;  end     % number of MC trials
if isfield(Params, 'support_opi')   == 0,  Params.support_opi   = 2;   end      % 1 means truncated power iterations per iteration

display(Params);

%% Make signal and data (noiseless)
Amatrix    = (1 * randn(Params.m, Params.n1) + Params.cplx_flag * 1i * randn(Params.m, Params.n1)) / (sqrt(2)^Params.cplx_flag);
% Amatrix    = dftmtx(Params.n1) * diag(randsrc(Params.n1, 1, [1i -1i 1 -1])) / Params.n1;
x          = 1 * randn(Params.n1, 1)  + Params.cplx_flag * 1i * randn(Params.n1, 1);
nonSupp    = (1:Params.nonK); % randperm(Params.n1, Params.nonK); % support complementary
x(nonSupp) = 0; % (n1-K)-sparse signal x
% x          = x / norm(x);

Supp       = setdiff((1:Params.n1), nonSupp);

A       = @(I) Amatrix  * I;
At      = @(Y) Amatrix' * Y;
var     = 0.0;
noise   = var * randn(Params.m, 1);
y       = abs(A(x) + noise).^2;
disp(norm(y));

%% run TAF algorithm
timetaf = tic;
[Relerrs_SPARTAF, z] = SPARTA1D(y, x, Params, Amatrix); 
times = toc(timetaf);
disp('----------SPARTA done!----------');

%% plot the relative error of TAF
figure,
semilogy(Relerrs_SPARTAF)
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('SPARTA: Relerr vs. itercount')
grid
