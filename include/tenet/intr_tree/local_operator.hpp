#pragma once
// include/tenet/intr_tree/local_operator.hpp
//
// AbstractLocalOperator<B> and concrete LocalOperator<B>, IdentityOperator<B>.
// These are the building blocks of SparseMPOTensor.
// See docs/C++重构设计方案.md §4.2.

#include "tenet/core/backend.hpp"

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <string>

namespace tenet {

// ── Abstract base (runtime polymorphism for container storage) ────────────────
template<TensorBackend B = DenseBackend>
class AbstractLocalOperator {
public:
    using Tensor = typename B::Tensor;

    virtual ~AbstractLocalOperator() = default;

    virtual int                    site()        const = 0;
    virtual const std::string&     name()        const = 0;
    virtual const Eigen::MatrixXcd& matrix()     const = 0;
    virtual bool                   is_identity() const { return false; }
    virtual std::unique_ptr<AbstractLocalOperator<B>> clone() const = 0;
};

// ── Concrete local operator ───────────────────────────────────────────────────
template<TensorBackend B = DenseBackend>
class LocalOperator : public AbstractLocalOperator<B> {
public:
    LocalOperator(Eigen::MatrixXcd mat, std::string name, int site,
                  std::optional<std::complex<double>> strength = std::nullopt);

    int                    site()    const override { return site_; }
    const std::string&     name()    const override { return name_; }
    const Eigen::MatrixXcd& matrix() const override { return mat_; }

    std::unique_ptr<AbstractLocalOperator<B>> clone() const override;

private:
    Eigen::MatrixXcd mat_;
    std::string      name_;
    int              site_;
    std::optional<std::complex<double>> strength_;
};

// ── Identity operator ─────────────────────────────────────────────────────────
template<TensorBackend B = DenseBackend>
class IdentityOperator : public AbstractLocalOperator<B> {
public:
    explicit IdentityOperator(int site, int dim = 1);

    int                    site()        const override { return site_; }
    const std::string&     name()        const override { return name_; }
    const Eigen::MatrixXcd& matrix()     const override { return mat_; }
    bool                   is_identity() const override { return true; }

    std::unique_ptr<AbstractLocalOperator<B>> clone() const override;

private:
    Eigen::MatrixXcd mat_;
    std::string      name_ = "I";
    int              site_;
};

} // namespace tenet
