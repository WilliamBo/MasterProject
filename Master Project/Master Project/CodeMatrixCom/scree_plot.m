% Performs a low rank matrix completion of the empirical covariance matrix R_n^K to which 
% a band of width delta has been removed, for different ranks. 
%
%--------------
% Input :
%--------------
% seq_rk : vector of ranks for which we perform the low-rank matrix completion
% Rn : the empirical covariance matrix of the data
% delta : bandwidth of the banded covariance matrix B (delta in (0,1/4))
% K : dimension of Rn
%
%--------------
% Output :
%--------------
% min_fun : the minimum of the optimisation problem 7.1 of the paper for
% different values of the rank


function min_fun= scree_plot(seq_rk, Rn, delta, K)

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

	% We perform the minimization of the function contained in the file min_l2_norm.m (this function has to be saved in the same folder as this function) for a range of different rank rk

	StartV = Rn;
	[U,S,V] = svd(StartV);
    min_fun = zeros(1,length(seq_rk));
    
	for i = 1:length(seq_rk)

		C_init = U(:,1:seq_rk(i))*sqrt(S(1:seq_rk(i),1:seq_rk(i)));
		[x,fval,exitflag,output,grad,hessian] = fminunc(@(C)min_l2_norm(C,Rn,Pk),C_init,options);  
		min_fun(1,i) = fval;
        
    end
    
    plot(seq_rk,min_fun,'b--o')

end

