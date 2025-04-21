%% Implementation of Phase Retrieval of Spectrally Sparse Signals

addpath("utils\","algorithms\","initial_method\");

%% hyper params
N = 3000;                           % Dim of signal
T = 2000;                           % # of measurements 
r = 15;                             % spectral sparsity
N_grid = N;                         % Grid Num
init_method = 'on-grid';            % Str 'spec', 'one-step-hankel', 'on-grid', 'sparsePR-SPARTA','sparsePR-HTP'
use_Fast_GDAP = true;               % Bool, use GDAP or Fast-GDAP

%% generate spectrally sparse 1D signal and measurements
A = 1/sqrt(2)*randn(T,N)+ 1i/sqrt(2)*randn(T,N); % Guassian - matrix
c_true = randn(r,1)+1i*randn(r,1);
f_true = rand(r,1);
x0 = zeros(N,1);
for i = 1 : N
    for k = 1 : r
        x0(i) = x0(i) + c_true(k) * exp(2*pi*1i*f_true(k)*(i-1));
    end
end
ymag = abs(A*x0);

%% initialization
tic;
if init_method == "spec"     %  original spectral method is useless in our experiments.
    x_init = spec_initial(A,ymag,0);  % without extra structure
elseif init_method == "one-step-hankel"
    x_init = spec_initial_oneStep_hankel(A,ymag,r);
elseif init_method == "on-grid"
    x_init = spec_initial_onGrid(A,ymag,r);
elseif init_method == "sparsePR-SPARTA"
    x_init = spec_initial_SPARTA(A,ymag,r);
elseif init_method == "sparsePR-HTP"
    x_init = spec_initial_HTP(A,ymag,r);
end

stage1_time = toc;
fprintf('initial time %.4f s\n',stage1_time);

%% GDAP / Fast-GDAP
% params
if exist('Params', 'var')           == 0,  Params.max_iter      = 200;    end               % Max Iteration time. 
if isfield(Params, 'beta')          == 0,  Params.beta          = 1;     end		        % params for line search. 
if isfield(Params, 'alpha')         == 0,  Params.alpha         = 0.5;   end	            % params for line search.  
if isfield(Params, 'tolerr_x')      == 0,  Params.tolerr_x      = 1e-9;  end                % stop criterion. diff(x_{k-1} , x_k} <= tolerr_x 
if isfield(Params, 'tolerr_y')      == 0,  Params.tolerr_y      = 1e-8;  end                % stop criterion. diff(abs(Ax_{k}) , ymag} <= tolerr_y 
if isfield(Params, 'Subspace_START_TIME')  == 0,  Params.Subspace_START_TIME   = 1;  end    % Int, determine when to calculate Subspace for FIHT 

if use_Fast_GDAP
    [x,relerr] = Fast_GDAP(ymag,x_init,A,r,x0,Params);
else
    [x,relerr] = GDAP(ymag,x_init,A,r,x0,Params);
end
all_process_time = toc;
fprintf('all time %.4f s\n', all_process_time);


%% results
figure,
semilogy(relerr)
xlabel('Iteration');
ylabel('Relative error with ground truth (log10)');
title('Relerr vs. itercount')
grid








