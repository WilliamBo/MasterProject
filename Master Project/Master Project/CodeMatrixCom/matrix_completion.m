function [ L_hat ] = mat_completion( rk,Rn,delta,K )
    
    % We create the matrix P_K
    
	Pk = zeros(K);
	q = ceil(K*delta);
	ind = [];
	for i = 1:(K-q-1)
		ind = [ind,((i-1)*K + i+q+1):(i*K)];
	end
	Pk(ind) = 1;
	Pk = Pk + Pk';
    
    % We set the option for the optimization using the function fminunc
	options = optimset;
	options = optimset(options,'Display', 'off');
	options = optimset(options,'GradObj', 'on');
	options = optimset(options,'Hessian', 'off');

	% We perform the minimization of the function contained in the file a_min_l2_grad.m (this function has to be saved in the same folder as this function)

	StartV = Rn;
	[U,S,V] = svd(StartV);
	C_init = U(:,1:rk)*sqrt(S(1:rk,1:rk));
	[x,fval,exitflag,output,grad,hessian] = fminunc(@(C)min_l2_norm(C,Rn,Pk),C_init,options);       
	
	L_hat = x*x';

end

