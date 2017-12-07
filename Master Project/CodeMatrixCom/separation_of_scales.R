# Download of library


library(MASS)
library(pracma)

 
 
#### Function Pred used to predict the smooth and rough part of the data, and to obtain the estimation of the plug-in estimator of B #### 

#------------------------
# Input
#------------------------
# X = nxK matrix containing the data
# Rn = empirical covariance matrix of X 
# n = sample size
# K = number of grid points
# rankL = rank selected by the scree-plot approach
# delta = bandwidth of the banded matrix B (between 0 and 1/4)
# hat_L = estimation of the matrix L obtained with the Matlab function matrix.completion.r
# grid = vector of length K containing the grid of points on which every function X_i is evaluated

#------------------------
# Output
#------------------------
# hatB = plug-in estimator of B
# hatY = prediction of the smooth part of the data
# hatW = predictions of the rough part of the data

Pred <- function(X,Rn,n,K,rankL,delta,hat_L,grid){
	
################# Calculation of the plug-in estimator of B #################		
	# Creation of a mask matrix : 1 on a band of width delta and 0 everywhere else
	ind <- c()
	q <- ceiling(K*delta)
		
	for(i in 1:(K-q-1)){
		ind <- c(ind,((i-1)*K + i+q+1):(i*K))
	}
	mask <- matrix(1,ncol=K,nrow=K)
	mask[ind] <- 0
	mask <- mask * t(mask)
	
	Bnew <- Rn-hat_L
	Bold <- Rn
	while(normF(Bold-Bnew)>0.0001){
		Bold <- Bnew
		temp <- mask*Bnew
		eigen_dec_B <- eigen(temp)
		beta <- eigen_dec_B$values
		rk_B <- length(which(beta>0.00001))
		Bnew <- eigen_dec_B$vectors[,1:rk_B]%*%diag(beta[1:rk_B])%*%t(eigen_dec_B$vectors[,1:rk_B])
	}
	hat_B <- Bnew
		
		
################# Calculation of estimated prediction of Y and W #################		
		
	# Calculation of the eigendecomposition of hat_L
	eigen_dec_L <- eigen(hat_L)
	eta <- (t(eigen_dec_L$vectors[,1:rankL]))*sqrt(K) #dim rankL x K
	lambda <- (eigen_dec_L$values[1:rankL])/K
	
	# Calculation of the xi
		
	inv_hat_r <- ginv(hat_L+hat_B)
	mat_xi <- matrix(0,nrow=n,ncol=rankL)
	for(i in 1:rankL){
		mat_xi[,i] <- lambda[i]*eta[i,]%*%inv_hat_r%*%t(X)
	}
	hat_Y <- mat_xi%*%eta # line i contains the estimation of Y_i 
	hat_W <- X - hat_Y # line i contains the estimation of W_i 
			
	return(list('hatY'= hat_Y,'hatW'= hat_W,'hatB'=hat_B))
	

}