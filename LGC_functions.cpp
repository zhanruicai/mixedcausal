// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppNumerical.h>

using namespace Numer;
using namespace arma;

// P(a1 < X1 < b1, a2 < X2 < b2), (X1, X2) ~ N([0], [1   rho])
//                                            ([0], [rho   1])
class BiNormal: public MFunc
{
private:
  const double rho;
  double const1;  // 2 * (1 - rho^2)
  double const2;  // 1 / (2 * PI) / sqrt(1 - rho^2)
public:
  BiNormal(const double& rho_) : rho(rho_)
  {
    const1 = 2.0 * (1.0 - rho * rho);
    const2 = 1.0 / (2 * M_PI) / std::sqrt(1.0 - rho * rho);
  }
  
  // PDF of bivariate normal
  double operator()(Constvec& x)
  {
    double z = x[0] * x[0] - 2 * rho * x[0] * x[1] + x[1] * x[1];
    return const2 * std::exp(-z / const1);
  }
};


double bivaNormal(double x, double y, double rho)
{
  BiNormal f(rho);  // rho is the correlation
  Eigen::VectorXd lower(2);
  lower << -20, -20;  // set the lower bound to be very small
  Eigen::VectorXd upper(2);
  upper << x, y;
  double err_est;
  int err_code;
  const double res = integrate(f, lower, upper, err_est, err_code);
  return res;
}


class BiNormalDeri: public MFunc
{
private:
  const double rho;
  double const1;  // 2 * (1 - rho^2)
  double const2;  // 1 / (2 * PI) / sqrt(1 - rho^2)
  double const3;  // rho / (2 * PI) / sqrt(1 - rho^2) / (1 - rho^2)
public:
  BiNormalDeri(const double& rho_) : rho(rho_)
  {
    const1 = (1.0 - rho * rho);
    const2 = 1.0 / (2 * M_PI) / std::sqrt(1.0 - rho * rho);
    const3 = rho / (2 * M_PI) / std::sqrt(1.0 - rho * rho) / (1.0 - rho * rho);
  }
  
  // PDF of derivative of the bivariate normal
  double operator()(Constvec& x)
  {
    double z1 = x[0] * x[0] - 2 * rho * x[0] * x[1] + x[1] * x[1];
    double z2 = x[0] * x[1] * const1 - rho * z1;
    return const3 * std::exp(-z1 / const1 /2) + const2 * std::exp( z2 /const1/const1);
  }
};

double bivaNormalDeri(double x, double y, double rho)
{
  BiNormalDeri f1(rho);  // rho is the correlation
  Eigen::VectorXd lower(2);
  lower << -20, -20;  // set the lower bound to be very small
  Eigen::VectorXd upper(2);
  upper << x, y;
  double err_est;
  int err_code;
  const double res = integrate(f1, lower, upper, err_est, err_code);
  return res;
}


class NormPDF: public Func
{
private:
  double a;
  double b;
public:
  NormPDF(double a_, double b_) : a(a_), b(b_) {}
  
  double operator()(const double& x) const
  {
    return std::exp(-0.5*(x-a)*(x-a)/b/b)/b/std::sqrt(2*M_PI);
  }
};


double CDFnorm(double x)
{
  const double a = 0, b = 1;
  const double lower = -20, upper = x;
  NormPDF f(a, b);
  double err_est;
  int err_code;
  const double res = integrate(f, lower, upper, err_est, err_code);
  return res;
}


// [[Rcpp::export]]
double kendall_tau (arma::vec x, arma::vec y)
{
  int n=x.n_elem, i, j;
  double na=0, nb=0, nc=0, nd=0, ne=0, nf=0;
  for(i = 0; i<n; i++)
  {
    for(j = 0; j<=i; j++)
    {
      if(x[i]>x[j] && y[i]>y[j]) {na = na+1;}
      if(x[i]>x[j] && y[i]<y[j]) {nb = nb+1;}
      if(x[i]<x[j] && y[i]>y[j]) {nc = nc+1;}
      if(x[i]<x[j] && y[i]<y[j]) {nd = nd+1;}
      if(x[i]==x[j] && y[i]!=y[j]) {ne = ne+1;}
      if(x[i]!=x[j] && y[i]==y[j]) {nf = nf+1;}
    }
  }
  return (na+nd-nb-nc)/std::sqrt((na+nd+nb+nc+ne)*(na+nd+nb+nc+nf));
}

double ContinuousInC(vec x, vec y)
{
  double tau = kendall_tau(x, y);
  double sighat = std::sin(tau*M_PI/2);
  return sighat;
}

double binaryContiFunc(double x, double deltak, double tau)
{
  return 4*bivaNormal(deltak, 0, x/std::sqrt(2)) - 2*R::pnorm(deltak, 0.0, 1.0, 1, 0) - tau;
}

double easyDeriBinaryContiFunc(double x, double deltak, double tau)
{
  double ep1 = 0.000001;
  double d1 = (binaryContiFunc(x, deltak, tau) - binaryContiFunc(x-ep1, deltak, tau))/ep1;
  return d1;
}


double BinaryContiInC(vec x, vec y)
{
  double tau = kendall_tau(x, y);
  double xbar = mean(x);
  double deltak = R::qnorm(1 - xbar, 0.0, 1.0, 1, 0);
  double diff = 100.0, xnew = 10, xold = 0.5, f0, fd;
  while (diff>0.000001){
    f0 = binaryContiFunc(xold, deltak, tau);
    fd = easyDeriBinaryContiFunc(xold, deltak, tau);
    xnew = xold - f0/fd;
    diff = xnew - xold;
    xold = xnew;
    //cout << diff << "\n";
  }
  return xnew;
}


double binaryFunc(double x, double deltak, double deltal, double tau)
{
  double p1 = R::pnorm(deltak, 0.0, 1.0, 1, 0);
  double p2 = R::pnorm(deltal, 0.0, 1.0, 1, 0);
  return 2*bivaNormal(deltak, deltal, x) - 2*p1*p2 - tau;
}

double easyDeribinaryFunc(double x, double deltak, double deltal, double tau)
{
  double ep1 = 0.000001;
  double d1 = (binaryFunc(x, deltak, deltal, tau) - binaryFunc(x-ep1, deltak, deltal, tau))/ep1;
  return d1;
}

double BinaryInC(vec x, vec y)
{
  double tau = kendall_tau(x, y);
  if(tau==0) {return 0;}
  double xbar = mean(x), ybar = mean(y);
  double deltak = R::qnorm(1 - xbar, 0.0, 1.0, 1, 0);
  double deltal = R::qnorm(1 - ybar, 0.0, 1.0, 1, 0);
  double diff = 100.0, xnew = 10, xold = 0.5, f0, fd;
  while (diff>0.000001){
    f0 = binaryFunc(xold, deltak, deltal, tau);
    fd = easyDeribinaryFunc(xold, deltak, deltal, tau);
    xnew = xold - f0/fd;
    diff = xnew - xold;
    xold = xnew;
    //cout << diff << "\n";
  }
  return xnew;
}

mat generate_dummy_Categorical(vec x){
  vec a = unique(x);
  int n = x.size(), numCat = a.size(), i, j, tmp = 999999;
  mat dummy = zeros(n, numCat-1);
  for(i=0; i<n; i++){
    for(j=0; j<numCat-1; j++){
      if(x(i)==a(j)) {dummy(i,j) = 1;}
    }
  }
  return dummy;
}

mat generate_dummy_Ordinal(vec x){
  vec a = unique(x);
  std::sort(a.begin(), a.end());
  int n = x.size(), numCat = a.size(), i, j, tmp = 999999;
  mat dummy = zeros(n, numCat-1);
  for(i=0; i<n; i++){
    for(j=0; j<numCat-1; j++){
      if(x(i)>=a(j+1)) {dummy(i,j) = 1;}
    }
  }
  return dummy;
}


double OrdinalOrdinal(vec x, vec y)
{
  mat m1 = generate_dummy_Ordinal(x);
  mat m2 = generate_dummy_Ordinal(y);
  int n1 = m1.n_cols, n2 = m2.n_cols,i, j;
  double xnew=0.0;
  for(i=0; i<n1; i++){
    for(j=0; j<n2; j++){
      double a = BinaryInC(m1.col(i),m2.col(j));
      if(::isnan(a)) {a = 0;}
      xnew = xnew + a;
    }
  }
  return (xnew/(n1*n2));
}

double CategoricalCategorical(vec x, vec y)
{
  mat m1 = generate_dummy_Categorical(x);
  mat m2 = generate_dummy_Categorical(y);
  int n1 = m1.n_cols, n2 = m2.n_cols,i, j;
  double xnew=0.0;
  for(i=0; i<n1; i++){
    for(j=0; j<n2; j++){
      xnew = xnew + BinaryInC(m1.col(i),m2.col(j));
    }
  }
  return (xnew/(n1*n2));
}

double CategoricalOrdinal(vec x, vec y)
{
  mat m1 = generate_dummy_Categorical(x);
  mat m2 = generate_dummy_Ordinal(y);
  int n1 = m1.n_cols, n2 = m2.n_cols,i, j;
  double xnew=0.0;
  for(i=0; i<n1; i++){
    for(j=0; j<n2; j++){
      xnew = xnew + BinaryInC(m1.col(i),m2.col(j));
    }
  }
  return (xnew/(n1*n2));
}

double OrdinalContiuous(vec x, vec y)
{
  mat m1 = generate_dummy_Ordinal(x);
  int n1 = m1.n_cols,i;
  double xnew=0.0;
  for(i=0; i<n1; i++){
      xnew = xnew + BinaryContiInC(m1.col(i), y);
  }
  return (xnew/n1);
}

double CategoricalContiuous(vec x, vec y)
{
  mat m1 = generate_dummy_Categorical(x);
  int n1 = m1.n_cols,i;
  double xnew=0.0;
  for(i=0; i<n1; i++){
    xnew = xnew + BinaryContiInC(m1.col(i), y);
  }
  return (xnew/n1);
}

double CategoricalBinary(vec x, vec y)
{
  mat m1 = generate_dummy_Categorical(x);
  int n1 = m1.n_cols,i;
  double xnew=0.0;
  for(i=0; i<n1; i++){
    xnew = xnew + BinaryInC(m1.col(i), y);
  }
  return (xnew/n1);
}


double OrdinallBinary(vec x, vec y)
{
  mat m1 = generate_dummy_Ordinal(x);
  int n1 = m1.n_cols,i;
  double xnew=0.0, a;
  for(i=0; i<n1; i++){
    a = BinaryInC(m1.col(i), y);
    if(::isnan(a)) {a = 0;}
    xnew = xnew + a;
  }
  return (xnew/n1);
}


// [[Rcpp::export]]
mat LGC_sigma_AllType(mat x, std::vector<std::string> label){
  int p = x.n_cols, i, j;
  arma::mat sig = arma::ones(p,p);
  //cout<<label[1]<<endl;
  for(int i=0; i<p-1; i++)
  {
    for(int j=i+1; j<p; j++)
    {
      if(label[i]=="continuous" && label[j] == "continuous") {sig(i,j) = ContinuousInC(x.col(i), x.col(j));}
      if(label[i]=="binary" && label[j] == "continuous") {sig(i,j) = BinaryContiInC(x.col(i), x.col(j));}
      if(label[i]=="continuous" && label[j] == "binary") {sig(i,j) = BinaryContiInC(x.col(j), x.col(i));}
      if(label[i]=="binary" && label[j] == "binary") {sig(i,j) = BinaryInC(x.col(i), x.col(j));}
      if(label[i]=="continuous" && label[j] == "ordinal") {sig(i,j) = OrdinalContiuous(x.col(j), x.col(i));}
      if(label[i]=="ordinal" && label[j] == "continuous") {sig(i,j) = OrdinalContiuous(x.col(i), x.col(j));}
      if(label[i]=="continuous" && label[j] == "categorical") {sig(i,j) = CategoricalContiuous(x.col(j), x.col(i));}
      if(label[i]=="categorical" && label[j] == "continuous") {sig(i,j) = CategoricalContiuous(x.col(i), x.col(j));}
      if(label[i]=="ordinal" && label[j] == "binary") {sig(i,j) = OrdinallBinary(x.col(i), x.col(j));}
      if(label[i]=="binary" && label[j] == "ordinal") {sig(i,j) = OrdinallBinary(x.col(j), x.col(i));}
      if(label[i]=="categorical" && label[j] == "binary") {sig(i,j) = CategoricalBinary(x.col(i), x.col(j));}
      if(label[i]=="binary" && label[j] == "categorical") {sig(i,j) = CategoricalBinary(x.col(j), x.col(i));}
      if(label[i]=="ordinal" && label[j] == "categorical") {sig(i,j) = CategoricalOrdinal(x.col(j), x.col(i));}
      if(label[i]=="categorical" && label[j] == "ordinal") {sig(i,j) = CategoricalOrdinal(x.col(i), x.col(j));}
      if(label[i]=="ordinal" && label[j] == "ordinal") {sig(i,j) = OrdinalOrdinal(x.col(i), x.col(j));}
      if(label[i]=="categorical" && label[j] == "categorical") {sig(i,j) = CategoricalCategorical(x.col(i), x.col(j));}
      
      
      sig(j,i) = sig(i,j);
    }
  }
  return (sig);
}



// [[Rcpp::export]]
mat LGC_sigma_BC(mat x, std::vector<std::string> label){
  int p = x.n_cols, i, j;
  arma::mat sig = arma::ones(p,p);
  //cout<<label[1]<<endl;
  for(i=0; i<p-1; i++)
  {
    for(j=i+1; j<p; j++)
    {
      if(label[i]=="continuous" && label[j] == "continuous") {sig(i,j) = ContinuousInC(x.col(i), x.col(j));}
      if(label[i]=="binary" && label[j] == "continuous") {sig(i,j) = BinaryContiInC(x.col(i), x.col(j));}
      if(label[i]=="continuous" && label[j] == "binary") {sig(i,j) = BinaryContiInC(x.col(j), x.col(i));}
      if(label[i]=="binary" && label[j] == "binary") {sig(i,j) = BinaryInC(x.col(i), x.col(j));}
      
      sig(j,i) = sig(i,j);
    }
  }
  return (sig);
}




