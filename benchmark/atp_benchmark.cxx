#include <nlopt.hpp>

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <array>
#include <cassert>
#include <iostream>

using real_type = double;
using vec_type = std::vector<real_type>;

extern "C"
{
void sgemm_(char* transa, char* transb, int* m, int* n, int* k, float* alpha, float* a, int* lda, float* b, int* ldb, float* beta, float* c, int* ldc);
void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);
void sgemv_(char* trans, int* m, int* n, float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);
void dgemv_(char* trans, int* m, int* n, double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);
}

namespace lapack_wrapper
{
template<typename T>
auto gemm
    (   char transa_
    ,   char transb_
    ,   int m_
    ,   int n_
    ,   int k_
    ,   T alpha_
    ,   const T* a_
    ,   int lda_
    ,   const T* b_
    ,   int ldb_
    ,   T beta_
    ,   T* c_
    ,   int ldc_
    )
{
    if constexpr ( std::is_same_v<T, float> )
    {
        return sgemm_(&transa_, &transb_, &m_, &n_, &k_, &alpha_, const_cast<T*>(a_), &lda_, const_cast<T*>(b_), &ldb_, &beta_, c_, &ldc_);
    }
    else if constexpr ( std::is_same_v<T, double> )
    {
        return dgemm_(&transa_, &transb_, &m_, &n_, &k_, &alpha_, const_cast<T*>(a_), &lda_, const_cast<T*>(b_), &ldb_, &beta_, c_, &ldc_);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>, "Only implemented for floating-point types.");
    }
}
template<typename T>
auto gemv
    (   char trans_
    ,   int m_
    ,   int n_
    ,   T alpha_
    ,   const T* a_
    ,   int lda_
    ,   const T* x_
    ,   int incx_
    ,   T beta_
    ,   T* y_
    ,   int incy_
    )
{
    if constexpr ( std::is_same_v<T, float> )
    {
        return sgemv_(&trans_, &m_, &n_, &alpha_, const_cast<T*>(a_), &lda_, const_cast<T*>(x_), &incx_, &beta_, y_, &incy_);
    }
    else if constexpr ( std::is_same_v<T, double> )
    {
        return dgemv_(&trans_, &m_, &n_, &alpha_, const_cast<T*>(a_), &lda_, const_cast<T*>(x_), &incx_, &beta_, y_, &incy_);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>, "Only implemented for floating-point types.");
    }
}
} /* namespace lapack_wrapper */


struct func_data
{
    vec_type mu;
    vec_type w0;
    vec_type wb;
    real_type mb;
    real_type alpha = 0.1;
    real_type t = 0.1;
    vec_type beta;
    vec_type gamma;
    std::array<vec_type, 2> covS;
};

real_type atp_objective
    (   const vec_type& x_
    ,   vec_type& grad_
    ,   void* data_
    )
{
    const func_data& fdat = *reinterpret_cast<func_data*>(data_);
    const auto& mu = fdat.mu;
    const auto& beta = fdat.beta;
    const auto& gamma = fdat.gamma;
    const auto& w0 = fdat.w0;
    assert(x_.size() == mu.size());
    assert(x_.size() == beta.size());
    assert(x_.size() == gamma.size());
    assert(x_.size() == w0.size());

    if ( !grad_.empty() )
    {
        grad_ = mu;
        for(int i=0; i<grad_.size(); ++i)
        {
            real_type sign = x_[i] >= w0[i] ? 1.0 : -1.0;
            grad_[i] -= sign*(beta[i] + static_cast<real_type>(3./2.)*std::sqrt(std::abs(x_[i] - w0[i]))*gamma[i]);
        }
    }

    real_type res = std::transform_reduce(mu.begin(), mu.end(), x_.begin(), static_cast<real_type>(0));
    vec_type abs_dw(x_.size());
    std::transform(x_.begin(), x_.end(), w0.begin(), abs_dw.begin(), [](real_type w_, real_type w0_) -> real_type { return std::abs(w_ - w0_); });
    res += std::transform_reduce(abs_dw.begin(), abs_dw.end(), beta.begin(), static_cast<real_type>(0));
    res += std::transform_reduce(abs_dw.begin(), abs_dw.end(), gamma.begin(), static_cast<real_type>(0), std::plus<real_type>(), [](real_type w_, real_type g_) { return std::pow(w_, 3./2.) * g_; });
    return res;
}

template<int SIdx>
real_type cov_constraint
    (   const vec_type& x_
    ,   vec_type& grad_
    ,   void* data_
    )
{
    const func_data& fdat = *reinterpret_cast<func_data*>(data_);
    const auto& mu = fdat.mu;
    const auto& alpha = fdat.alpha;
    const auto& t = fdat.t;
    const auto& wb = fdat.wb;
    const auto mb = fdat.mb;
    const real_type rhs = SIdx == 0 ? t / 2.66 : alpha*t;
    const auto& S = std::get<SIdx>(fdat.covS);
    const int len = x_.size();
    assert(len == mu.size());
    assert(len == wb.size());
    assert(S.size() == len*len);
    vec_type xtmp;
    if constexpr ( SIdx == 1 )
    {
        xtmp.resize(len);
        std::transform(x_.begin(), x_.end(), wb.begin(), xtmp.begin(), [mb](real_type w_, real_type wb_) { return w_ - mb*wb_; });
    }
    const auto& wvec = SIdx == 1 ? xtmp : x_;

    // Save S*w in gradient
    vec_type gtemp;
    if ( grad_.empty() )
    {
        gtemp.resize(len);
    }
    auto& grad = grad_.empty() ? gtemp : grad_;
    lapack_wrapper::gemv('N', len, len, 1.0, S.data(), std::max(1, len), wvec.data(), 1, 0.0, grad.data(), 1);

    // Compute function value.
    const real_type norm = std::sqrt(std::transform_reduce(wvec.begin(), wvec.end(), grad.begin(), static_cast<real_type>(0)));

    // Scale gradient (if not empty)
    std::for_each(grad_.begin(), grad_.end(), [n=-1.0/norm](real_type& x_) { x_ *= n; });

    // Return result
    return rhs - norm;
}


int main(int argc, char* argv[])
{
    std::mt19937 rng(777);
    std::normal_distribution<real_type> ndist(0.0, 1.0);
    std::uniform_real_distribution<real_type> udist(0.0, 10.0);

    const int dim = argc > 1 ? std::atoi(argv[1]) : 20000;
    vec_type x(dim, 1.0/dim);
    nlopt::opt opt("LD_SLSQP", dim);
    vec_type lb(dim, 0.0);
    opt.set_lower_bounds(lb);
    func_data data;
    data.mu.resize(dim);
    std::generate(data.mu.begin(), data.mu.end(), [&] { return udist(rng); });
    data.beta.resize(dim, 0.01);
    data.gamma.resize(dim, 0.001);
    data.alpha = 0.1;
    data.t = 10.0;
    data.w0 = x;
    data.wb.resize(dim, 1.0/dim);

    vec_type sqrt_covs(dim*dim);
    std::generate(sqrt_covs.begin(), sqrt_covs.end(), [&] { return ndist(rng); });
    data.covS[0].resize(dim*dim);
    lapack_wrapper::gemm('T', 'N', dim, dim, dim, 0.5, sqrt_covs.data(), dim, sqrt_covs.data(), dim, 0.0, data.covS[0].data(), dim);
    data.covS[1].resize(dim*dim);
    std::generate(sqrt_covs.begin(), sqrt_covs.end(), [&] { return ndist(rng); });
    lapack_wrapper::gemm('T', 'N', dim, dim, dim, 0.5, sqrt_covs.data(), dim, sqrt_covs.data(), dim, 0.0, data.covS[1].data(), dim);

    vec_type dummy_grad;
    const auto es3m99_ref = cov_constraint<0>(data.wb, dummy_grad, &data);
    data.mb = data.t / es3m99_ref;

    opt.set_max_objective(atp_objective, &data);
    opt.add_equality_constraint(cov_constraint<0>, &data);
    opt.add_inequality_constraint(cov_constraint<1>, &data);
    opt.set_xtol_rel(1.e-6);

    real_type max_val = 0.0;

    try
    {
        opt.optimize(x, max_val);
        std::cout   <<  " Opt value:        " << max_val << "\n"
                    <<  " Sum of weights:   " << std::reduce(x.begin(), x.end()) << "\n"
                    <<  std::flush;
    }
    catch(const std::exception& e)
    {
        std::cerr << "NlOpt failed: " << e.what() << std::endl;
    }
    
    return 0;
}