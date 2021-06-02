#### Simulations for latent PC algorithm.
rm(list = ls())
library(Rcpp)
library(RcppArmadillo)
library(RcppNumerical)
sourceCpp("LGC_functions.cpp")
source("data_generate.R")
set.seed(1)

#n = 100
#p = 12

### The significance level alpha
aa = 0.01
### The expected degrees
exp.deg = 5
### The simulation times. Set as 1 only for one fast realization of the algorithm.
simutime = 1

nn = c(50, 100, 150, 200)
pp = c(9, 27, 81, 243)

for(ii in 1:length(nn)){
  n = nn[ii]
  p = pp[ii]
  cat(n, "\r")

  s = exp.deg/(p-1)
  la = c(rep("continuous", p/3), rep("binary", p/3), rep("ordinal", p/3))
  
  r0 <- r1 <- r2 <- r3 <- r4 <- c()
  s1 <- s2 <- s3 <- s4 <- c()
  TPR_oracle <- FPR_oracle <- TPR1 <- FPR1 <- TPR2 <- FPR2 <- TPR3 <- FPR3 <- c()
  
  
  for(i in 1:simutime)
  {
    #cat(i, "\r")
    d1 = generate_DAG_CBO(n, p, s)
    X = d1$X
    A = d1$A
    t0 = as.numeric(A>0)
    
    sig_oracle = d1$sigmahat
    sig3 = cor(X)
    sig2 = cor(X, method = "kendall")
    sig1 = LGC_sigma_AllType(as.matrix(X), la)
    sig1[which(is.na(sig1), arr.ind = T)] = sig3[which(is.na(sig1), arr.ind = T)]
    sig1 = nearPD(sig1, corr = TRUE)$mat
    
    s1 = c(s1, norm(sig_oracle-sig1, "2"))
    s2 = c(s2, norm(sig_oracle-sig2, "2"))
    s3 = c(s3, norm(sig_oracle-sig3, "2"))
    
    
    pc.fit0 <- pc(suffStat = list(C = sig_oracle, n = n),
                  indepTest = gaussCItest, ## indep.test: partial correlations
                  alpha=aa, labels = colnames(X), skel.method = "stable",verbose = F)
    tsig_oracle = as.numeric(as(pc.fit0, "amat"))
    
    
    pc.fit1 <- pc(suffStat = list(C = sig1, n = n),
                  indepTest = gaussCItest, ## indep.test: partial correlations
                  alpha=aa, labels = colnames(X), skel.method = "stable",verbose = F)
    t1 = as.numeric(as(pc.fit1, "amat"))
    
    pc.fit2 <- pc(suffStat = list(C = sig2, n = n),
                  indepTest = gaussCItest, ## indep.test: partial correlations
                  alpha=aa, labels = colnames(X), skel.method = "stable",verbose = F)
    t2 = as.numeric(as(pc.fit2, "amat"))
    
    pc.fit3 <- pc(suffStat = list(C = sig3, n = n),
                  indepTest = gaussCItest, ## indep.test: partial correlations
                  alpha=aa, labels = colnames(X), skel.method = "stable",verbose = F)
    t3 = as.numeric(as(pc.fit3, "amat"))
    
    
    tp = sum(t0[which(tsig_oracle==1)]==1)
    tn = sum(t0[which(tsig_oracle==0)]==0)
    fp = sum(tsig_oracle[which(t0==0)]==1)
    fn = sum(tsig_oracle[which(t0==1)]==0)
    TPR_oracle = c(tp/(tp+fn), TPR_oracle)
    FPR_oracle = c(fp/(fp+tn), FPR_oracle)
    
    tp = sum(t0[which(t1==1)]==1)
    tn = sum(t0[which(t1==0)]==0)
    fp = sum(t1[which(t0==0)]==1)
    fn = sum(t1[which(t0==1)]==0)
    TPR1 = c(tp/(tp+fn), TPR1)
    FPR1 = c(fp/(fp+tn), FPR1)
    
    tp = sum(t0[which(t2==1)]==1)
    tn = sum(t0[which(t2==0)]==0)
    fp = sum(t2[which(t0==0)]==1)
    fn = sum(t2[which(t0==1)]==0)
    TPR2 = c(tp/(tp+fn), TPR2)
    FPR2 = c(fp/(fp+tn), FPR2)
    
    tp = sum(t0[which(t3==1)]==1)
    tn = sum(t0[which(t3==0)]==0)
    fp = sum(t3[which(t0==0)]==1)
    fn = sum(t3[which(t0==1)]==0)
    TPR3 = c(tp/(tp+fn), TPR3)
    FPR3 = c(fp/(fp+tn), FPR3)
    
    r0 = c(r0, sum(abs(tsig_oracle-t0)))
    r1 = c(r1, sum(abs(t1-t0)))
    r2 = c(r2, sum(abs(t2-t0)))
    r3 = c(r3, sum(abs(t3-t0)))
    
  }
  
  
  CovError = list(s1, s2, s3)
  SHDEst = list(r0, r1, r2, r3)
  TruePR = list(TPR_oracle, TPR1, TPR2, TPR3)
  FalsPR = list(FPR_oracle, FPR1, FPR2, FPR3)
  result = list(CovError = CovError, SHDEst = SHDEst, TruePR = TruePR, FalsPR = FalsPR)
  print(lapply(SHDEst, mean))
  print(lapply(TruePR, mean))
  print(lapply(FalsPR, mean))
  filename = paste("SIMU0", "-", n, "-", p, "-", exp.deg, aa, ".rda", sep = "")
  save(result, file = filename)
  
}



