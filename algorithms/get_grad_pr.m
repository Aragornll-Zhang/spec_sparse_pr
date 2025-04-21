function [obj,grad_x] = get_grad_pr(A,x,ymag)
    % gradient of f(x) = (1/2m) * || |Ax| - ymag ||^2
    [m, ~] = size(A);
    Ax = A*x;
    angle = sign(Ax);
    grad_x = A' * ( Ax - ymag.*angle ) / m;
    obj = sum( (abs(Ax) - ymag).^2 ) / (2*m);
end