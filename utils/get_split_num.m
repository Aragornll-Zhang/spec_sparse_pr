function [n1,D_vec] = get_split_num(N)
    % From length-N vector -> n1 * n2 Hankel Mat with (n1 , n2=N+1-n1)
    % hankel(x())
    if mod(N,2) == 0
        n1 = N/2;
        D_vec = [1:n1 n1 n1-1:-1:1].';
    else
        n1 = (N + 1)/2;
        D_vec = [1:n1 n1-1:-1:1].';
    end
end