function [x_init] = spec_initial(A, ymag, r)
    % Conventional Spectral Method for Initialization in Sparse Phase Retrieval
    % used by CoPRAM / HPT
    % when r == 0 -> spectral initialization for Phase Retrieval
    [T,N] = size(A);
    
    if r == 0
        A_S_hat = A;
        S_hat = 1:N;
    else
        % estimate S_hat
        contributions = zeros(N, 1);
        for j = 1:N
            contributions(j) = (1/T) * sum((ymag.^2) .* (abs(A(:, j)).^2));
        end
        [~, idx] = sort(contributions, 'descend');
        S_hat = idx(1:r);  
        A_S_hat = A(:, S_hat);
    end


    % apply spectral method on S_hat 
    M = A_S_hat' * diag(ymag.^2) * A_S_hat / T; 
    [V, ~] = eigs(M, 1);
    x_init = zeros(N, 1);
    modular = norm(ymag) / sqrt(T);
    x_init(S_hat) = V * modular;
end
