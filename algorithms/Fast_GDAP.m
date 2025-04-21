function [x,relerr] = Fast_GDAP(ymag,x_init,A,r,x_gt,Params)
    % find spec sparse signal x , s.t.  |A*x| = ymag 
    [n1, D] = get_split_num(length(x_init));
    x = x_init;
    % Initialize Ur, Vr. used when Params.Subspace_START_TIME
    L0 = hankel(x(1:n1),x(n1:end));
    [Ur,~,Vr] = svds(L0,r);
    
    relerr = [];
    for kk = 1:Params.max_iter
        x_lastIter = x;
        % GD step
        [curr_func, grad] = get_grad_pr(A, x, ymag); 
        beta = Params.beta;
        alpha = Params.alpha;
        [curr_new, ~] = get_grad_pr(A, x-beta*grad , ymag);  
        while curr_new > curr_func && beta > 1e-7 
            beta = alpha * beta;   
            [curr_new, ~] = get_grad_pr(A, x-beta*grad ,ymag);  
        end
        x = x -beta*grad;  
  
        % Fast-IHT step   
        Z = hankel(x(1:n1),x(n1:end));
        if kk <= Params.Subspace_START_TIME
            [Ur,Sr,Vr] = svds(Z,r);
            Z_r = Ur*Sr*Vr';
        else
            UtHx = Ur'*Z;
            C = UtHx*Vr;
            Xt = UtHx-C*(Vr');
            X = Xt';
            Y = Z*Vr-Ur*C;   
            [Q1,R1] = qr(X,0);
            [Q2,R2] = qr(Y,0);
            M = [C R1';R2 zeros(r,r)];

            [Uc,Snew,Vc]=svds(M,r);
            % next iteration
            Ur=[Ur Q2] * Uc;
            Vr=[Vr Q1] * Vc;
            Z_r = Ur*Snew*Vr';       
        end
        x = hankel_inv1D(Z_r,D);
        % record relative error with ground truth. (For drawing pictures only)
        relerr = [relerr,cal_relerr(x,x_gt)];

        % Early stop
        if cal_relerr(x_lastIter,x) < Params.tolerr_x / norm(x,'fro') || cal_relerr(abs(A*x),ymag) < Params.tolerr_y
            break
        end
    end
end


function relerr = cal_relerr(x,x0)
    % relerr(z,x) = dist(z,x) / ||x||;  dist(z,x) := min_{\phi \in [0,2\pi)} ||x - ze^{j\phi}||
    relerr = norm(x0 - exp(-1i * angle(trace(x0' * x))) * x, 'fro') / norm(x0,'fro'); 
end

