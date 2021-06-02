library(pbivnorm)
library(rootSolve)
library(pcalg)
library(Matrix)

### Generate random causal graph. Half data are continuous, the other half is binary
### It is only possible for X_i to influence X_j, where i<j
### In the pcalg::pc function, the adjacency matrix in the summary should be an upper diagonal matrix.
### The as(pc.fit0, "amat") gives the transpose of the adjacency matrix, which is lower diagonal.

generate_DAG_CBO = function(n, p, s, noise = 1)
{
  A = matrix(0, p, p)
  for(i in 1:p)
  {
    for(j in 1:p)
    {
      if(i>j) 
      { tmp = rbinom(1,1,s)
      if(tmp) {A[i,j] = runif(1, min = 0.1, max = 0.5)}
      }
    }
  }
  X = matrix(0, n, p)
  X[,1] = rnorm(n)
  for(i in 2:p)
  {
    X[,i] = X%*%A[i,]+noise*rnorm(n)
  }
  sigmahat = cor(X)
  X[,1:(p/3)] = apply(X[,1:(p/3)], 2, function(s){s^3})
  X[,(p/3+1):(p*2/3)] = apply(X[,(p/3+1):(p*2/3)], 2, function(s){s>runif(1, min = -1, max = 1)})
  X[,(2*p/3+1):p] = apply(X[,(2*p/3+1):p], 2, function(s){(s>runif(1, min = -1, max = -0.3)) + (s>runif(1, min = 0.3, max = 1)) })
  
  X <- data.frame(X)
  colnames(X) <- paste("V", 1:dim(X)[2], sep = "")
  list(X=X, A=A, sigmahat=sigmahat)
}

   
### Generate random causal graph. All data are continuous
DAG_random = function(n, p, s, noise = 1)
{
  A = matrix(0, p, p)
  for(i in 1:p)
  {
    for(j in 1:p)
    {
      if(i>j) 
      { tmp = rbinom(1,1,s)
      if(tmp) {A[i,j] = runif(1, min = 0.1, max = 1)}
      }
    }
  }
  X = matrix(0, n, p)
  X[,1] = rnorm(n)
  for(i in 2:p)
  {
    X[,i] = X%*%A[i,]+noise*rnorm(n)
  }
  X <- data.frame(X)
  colnames(X) <- paste("V", 1:dim(X)[2], sep = "")
  
  list(X=X,A=A)
}

### Tunning parameter selection with the StARS method.
StARS = function(X, a1 = seq(0.01, 0.1, by = 0.01), TunModel = "LGC")
{
  l1 = length(a1)
  n = dim(X)[1]
  b = round(n*0.75)
  D = rep(0, l1)
  N = 100
  for(i in 1:l1)
  {
    cat(i, "\r")
    all.edges = matrix(0, N, p*p)
    for(j in 1:N)
    {
      #cat(j, "\r")
      ind1 = sample(1:n, size = b, replace = F)
      X1 = X[ind1,]
      
      if(TunModel == "LGC") sig1 = getCOV_LGC(X1) 
      if(TunModel == "Gaussian") sig1 = cor(X1)
      if(TunModel == "Rank") sig1 = cor(X1, method = "kendall")
      
      pc.fit1 <- pc(suffStat = list(C = sig1, n = b), indepTest = gaussCItest, alpha=a1[i], 
                    labels = colnames(X), skel.method = "stable",verbose = F)
      t1 = as.numeric(as(pc.fit1, "amat"))
      all.edges[j,] = t1
    }
    thetahat = apply(all.edges, 2, function(s){2*mean(s)*(1-mean(s))})
    D[i] = mean(thetahat, na.rm = T)
  }
  
  list(alpha = a1, D = D, cumD = cummax(D), est = approx(a1,cummax(D), xout=0.05))
  
}




