# TenetCpp C++ 重构设计方案

> 版本：v2.0
> 日期：2026-03-01
> 策略：**第一阶段不处理对称性**，使用稠密张量实现全部算法；通过 Backend 模板参数预留对称张量接入点，未来可无痛扩展。

---

## 目录

1. [重构目标与原则](#1-重构目标与原则)
2. [技术选型](#2-技术选型)
3. [C++ 项目结构](#3-c-项目结构)
4. [Julia → C++ 核心语言映射](#4-julia--c-核心语言映射)
5. [可扩展 Backend 架构（核心设计）](#5-可扩展-backend-架构核心设计)
6. [稠密张量系统（DenseTensor）](#6-稠密张量系统densetensor)
7. [MPS / MPO 模块设计](#7-mps--mpo-模块设计)
8. [环境张量模块设计](#8-环境张量模块设计)
9. [哈密顿量与 Krylov 求解器设计](#9-哈密顿量与-krylov-求解器设计)
10. [相互作用树与稀疏 MPO 构建](#10-相互作用树与稀疏-mpo-构建)
11. [可观测量系统设计](#11-可观测量系统设计)
12. [局域空间（LocalSpace）设计](#12-局域空间localspace设计)
13. [算法层设计（DMRG / TDVP / CBE / SETTN）](#13-算法层设计dmrg--tdvp--cbe--settn)
14. [并发与线程安全设计](#14-并发与线程安全设计)
15. [内存管理策略](#15-内存管理策略)
16. [构建系统与依赖管理](#16-构建系统与依赖管理)
17. [测试策略](#17-测试策略)
18. [迁移路线图](#18-迁移路线图)
19. [未来对称性扩展路径](#19-未来对称性扩展路径)

---

## 1. 重构目标与原则

### 1.1 重构动机

| 维度 | Julia 现状 | C++ 目标 |
|------|-----------|---------|
| 启动时间 | JIT 编译，首次运行慢 | 提前编译，启动即用 |
| 内存控制 | GC 不可预测 | RAII，确定性析构 |
| 跨语言集成 | 较繁琐 | 原生 C 接口，易于绑定 |
| 部署 | 需要 Julia 运行时 | 单一可执行文件 |
| 对称性处理 | TensorKit 深度集成 | **第一阶段跳过**，预留扩展点 |

### 1.2 两阶段策略

```
第一阶段（当前）：稠密张量 + 完整算法
  DenseTensor（Eigen 矩阵）→ MPS/MPO/DMRG/TDVP 全部可运行

第二阶段（未来）：替换 Backend，接入对称张量
  SymmetricTensor（块稀疏）→ 算法层代码不变，仅换 Backend 参数
```

### 1.3 设计原则

1. **Backend 参数化**：所有核心数据结构模板化于 `Backend`，当前默认 `DenseBackend`
2. **Space 接口抽象**：`TrivialSpace`（当前，纯整数维度）和未来的 `VectorSpace`（带量子数）满足同一 concept
3. **零开销抽象**：编译期多态优先，模板特化替代运行时分发
4. **RAII 优先**：禁止裸指针持有资源，值语义为主
5. **接口稳定**：算法层代码对 Backend 透明，换 Backend 不改算法
6. **错误处理分层**：编程错误（维度不匹配、腿索引越界）用 `assert`；运行时错误（I/O 失败、内存不足）抛出异常；数值问题（Krylov 不收敛、SVD 退化）通过返回值标志（`converged`、`truncation_err`）告知调用方而非抛出异常，由算法层决策如何响应

---

## 2. 技术选型

### 2.1 C++ 标准

**选用 C++20**：

| 特性 | 用途 |
|------|------|
| `concepts` | 约束 Backend、Space、Tensor 模板参数 |
| `std::span` | 非拥有内存视图，避免拷贝 |
| `std::optional<T>` | 替代 Julia 的 `Union{Nothing, T}` |
| `std::variant<Ts...>` | 替代 Julia 的多分支 Union |
| `std::ranges` | 惰性序列操作 |
| `std::jthread` | 可自动 join 的线程 |
| `[[nodiscard]]` | 强制检查关键返回值 |

### 2.2 核心依赖库

| 库 | 版本 | 用途 | 替代 Julia 包 |
|----|------|------|--------------|
| **Eigen3** | ≥ 3.4 | 矩阵/向量运算（DenseTensor 的存储后端） | BLAS/LAPACK 封装 |
| **Intel MKL** | 最新 | BLAS/LAPACK 高性能后端（可选 OpenBLAS） | MKL.jl |
| **HDF5 C++** | ≥ 1.12 | 数据序列化与存储 | JLD2.jl |
| **OpenMP** | ≥ 4.5 | 粗粒度并行（环境更新） | Julia Threads |
| **spdlog** | ≥ 1.11 | 结构化日志输出 | println |
| **Google Test** | ≥ 1.13 | 单元测试框架 | — |
| **CMake** | ≥ 3.25 | 构建系统 | — |
| **{fmt}** | ≥ 10 | 格式化输出 | — |

### 2.3 可选加速库

| 库 | 用途 |
|----|------|
| **CUDA + cuBLAS** | GPU 加速稠密张量缩并 |
| **Spectra** | 大规模稀疏特征值（Lanczos），或自行实现 |
| **oneTBB** | Intel 任务并行库（细粒度并行） |

---

## 3. C++ 项目结构

```
TenetCpp/
├── CMakeLists.txt
├── cmake/
│   ├── Dependencies.cmake
│   ├── CompilerFlags.cmake
│   └── FindMKL.cmake
│
├── include/tenet/
│   │
│   ├── core/                         # 张量核心（与 Backend 无关的工具）
│   │   ├── backend.hpp               # Backend concept + DenseBackend 定义
│   │   ├── space.hpp                 # TrivialSpace（当前）+ Space concept
│   │   ├── dense_tensor.hpp          # DenseTensor：N 阶稠密张量
│   │   ├── tensor_ops.hpp            # 缩并、置换、matricize 等操作
│   │   └── factorization.hpp         # SVD、QR、特征分解
│   │
│   ├── mps/
│   │   ├── mps_tensor.hpp            # MPSTensor<B>、AdjointMPSTensor<B>
│   │   ├── mps.hpp                   # DenseMPS<B>、AdjointMPS<B>
│   │   ├── mpo_tensor.hpp            # DenseMPOTensor<B>、SparseMPOTensor<B>
│   │   ├── mpo.hpp                   # DenseMPO<B>、SparseMPO<B>
│   │   └── canonical.hpp             # 左/右正则化，中心移动
│   │
│   ├── environment/
│   │   ├── env_tensor.hpp            # LeftEnvTensor<B>、RightEnvTensor<B>
│   │   │                             # SparseLeftEnvTensor<B>、SparseRightEnvTensor<B>
│   │   ├── environment.hpp           # Environment<B>（主类）
│   │   ├── env_push.hpp              # push_right<B>、push_left<B>
│   │   └── cbe_environment.hpp       # CBEEnvironment<B>
│   │
│   ├── hamiltonian/
│   │   ├── projective_ham.hpp        # SparseProjectiveHamiltonian<B>
│   │   ├── ham_action.hpp            # apply()：H_eff|ψ⟩
│   │   └── ham_builders.hpp          # proj0/proj1/proj2 工厂函数
│   │
│   ├── intr_tree/
│   │   ├── local_operator.hpp        # LocalOperator<B>、IdentityOperator<B>
│   │   ├── interaction_leave.hpp     # InteractionTreeLeave
│   │   ├── interaction_node.hpp      # InteractionTreeNode
│   │   └── add_interaction.hpp       # addIntr1~N、addIntrStr 接口
│   │
│   ├── observables/
│   │   ├── obs_node.hpp              # ObservableTreeNode<B>
│   │   ├── add_observable.hpp        # addObs 接口
│   │   └── cal_observable.hpp        # calObs 接口
│   │
│   ├── local_space/
│   │   ├── spin.hpp                  # 自旋算符（无对称性，纯矩阵）
│   │   └── fermion.hpp               # 费米子算符（无对称性）
│   │
│   ├── algorithm/
│   │   ├── dmrg.hpp                  # DMRG1<B>、DMRG2<B>
│   │   ├── tdvp.hpp                  # TDVP1<B>、TDVP2<B>、tanTRG<B>
│   │   ├── cbe.hpp                   # CBE<B>
│   │   └── settn.hpp                 # SETTN<B>
│   │
│   ├── krylov/
│   │   ├── lanczos.hpp               # Lanczos 特征值 + 线性方程求解
│   │   └── arnoldi.hpp               # Arnoldi（TDVP 指数传播）
│   │
│   ├── process_control/
│   │   ├── config.hpp                # DMRGConfig、TDVPConfig、CBEConfig
│   │   ├── scheme.hpp                # SingleSite、DoubleSite、SVD 方案标签
│   │   └── sweep_info.hpp            # DMRGSweepInfo、TDVPSweepInfo 等
│   │
│   └── utils/
│       ├── timer.hpp                 # 性能计时器
│       ├── logger.hpp                # spdlog 封装
│       └── io.hpp                    # HDF5 读写工具
│
├── src/
│   ├── core/
│   │   ├── dense_tensor.cpp
│   │   ├── tensor_ops.cpp
│   │   └── factorization.cpp
│   ├── mps/
│   │   ├── mps.cpp
│   │   ├── mpo.cpp
│   │   └── canonical.cpp
│   ├── environment/
│   │   ├── environment.cpp
│   │   └── env_push.cpp
│   ├── hamiltonian/
│   │   ├── projective_ham.cpp
│   │   └── ham_action.cpp
│   ├── intr_tree/
│   │   └── interaction_tree.cpp
│   ├── observables/
│   │   └── cal_observable.cpp
│   ├── algorithm/
│   │   ├── dmrg.cpp
│   │   ├── tdvp.cpp
│   │   ├── cbe.cpp
│   │   └── settn.cpp
│   └── krylov/
│       ├── lanczos.cpp
│       └── arnoldi.cpp
│
├── tests/
│   ├── test_dense_tensor.cpp
│   ├── test_factorization.cpp
│   ├── test_mps.cpp
│   ├── test_mpo.cpp
│   ├── test_environment.cpp
│   ├── test_lanczos.cpp
│   ├── test_dmrg.cpp
│   └── test_tdvp.cpp
│
└── examples/
    ├── heisenberg_dmrg.cpp
    ├── hubbard_tdvp.cpp
    └── finite_temperature_settn.cpp
```

---

## 4. Julia → C++ 核心语言映射

### 4.1 参数化类型 → C++ 模板

```julia
# Julia
struct DenseMPS{L, T <: Union{Float64, ComplexF64}}
    ts     :: Vector{MPSTensor}
    center :: Vector{Int64}
end
```

```cpp
// C++ — Backend 参数控制张量类型；系统大小 L 改为运行时参数
template<typename B = DenseBackend>
class DenseMPS {
public:
    using Tensor = typename B::Tensor;
    using Space  = typename B::Space;

    explicit DenseMPS(int L);

    int length()  const { return L_; }
    Tensor&       site(int i)       { return sites_[i]; }
    const Tensor& site(int i) const { return sites_[i]; }
    std::pair<int,int> center() const { return {cleft_, cright_}; }

private:
    int L_;
    std::vector<Tensor> sites_;
    int cleft_{0}, cright_{0};
};

// 当前默认实例化
using MPS = DenseMPS<DenseBackend>;
```

### 4.2 抽象类型 → Concepts（编译期）+ 虚基类（运行时）

**编译期多态（性能关键路径）：**

```cpp
// 替代 abstract type AbstractMPSTensor
template<typename T>
concept MPSTensorLike = requires(T t) {
    { t.left_dim()  } -> std::convertible_to<int>;
    { t.phys_dim()  } -> std::convertible_to<int>;
    { t.right_dim() } -> std::convertible_to<int>;
    { t.as_matrix() } -> std::convertible_to<Eigen::MatrixXcd>;
    { t.adjoint()   };
};
```

**运行时多态（容器存储，如 LocalOperator）：**

```cpp
class AbstractLocalOperator {
public:
    virtual ~AbstractLocalOperator() = default;
    virtual int site() const = 0;
    virtual const std::string& name() const = 0;
    virtual const Eigen::MatrixXcd& matrix() const = 0;
    virtual bool is_identity() const { return false; }
    virtual std::unique_ptr<AbstractLocalOperator> clone() const = 0;
};
```

### 4.3 Union 类型 → std::optional / std::variant

```julia
# Julia
H    :: Union{Nothing, SparseMPO}
Env  :: Union{Nothing, LeftEnvTensor, String}
```

```cpp
// C++
std::optional<SparseMPO>       H;
std::variant<std::monostate,
             LeftEnvTensor,
             std::string>       env_cache;

// 访问 variant
std::visit(overloaded{
    [](std::monostate)        { /* 未初始化 */ },
    [](LeftEnvTensor& e)      { /* 使用缓存 */ },
    [](const std::string& s)  { /* 磁盘路径 */ }
}, env_cache);
```

### 4.4 多重分发 → 标签类型 + 模板特化

```julia
# Julia
function DMRG!(env, alg::DMRGalgo{SingleSite}, info::DMRGsweepinfo{L2R}) ... end
function DMRG!(env, alg::DMRGalgo{DoubleSite}, info::DMRGsweepinfo{L2R}) ... end
```

```cpp
// C++ — 标签类型（空结构体，零开销）
struct SingleSite {};
struct DoubleSite {};
struct L2R {};
struct R2L {};

// 主模板声明（不定义 → 非法组合触发编译错误）
template<typename Scheme, typename Dir, typename B>
void dmrg_sweep(Environment<B>&, DMRGConfig&, DMRGSweepInfo&) = delete;

// 有效特化
template<typename B>
void dmrg_sweep<SingleSite, L2R, B>(Environment<B>&, DMRGConfig&, DMRGSweepInfo&);

template<typename B>
void dmrg_sweep<DoubleSite, L2R, B>(Environment<B>&, DMRGConfig&, DMRGSweepInfo&);
```

### 4.5 伴随类型 → CRTP 零开销包装

```cpp
template<typename Base>
class Adjoint {
public:
    explicit Adjoint(Base& b) : base_(b) {}
    Base& unadjoint() { return base_; }
    int length() const { return base_.length(); }
    auto site(int i) const { return base_.site(i).adjoint(); }

private:
    Base& base_;   // 非拥有引用，零拷贝
};

// 使用
DenseMPS<> psi(L);
auto psi_dag = Adjoint(psi);                   // ⟨ψ|，零开销
auto env     = Environment(psi, H, psi_dag);
```

---

## 5. 可扩展 Backend 架构（核心设计）

这是整个重构的关键设计决策：通过 `Backend` 模板参数将**张量存储实现**与**算法逻辑**解耦。

### 5.1 Backend Concept

```cpp
// include/tenet/core/backend.hpp

// Space concept：当前由 TrivialSpace 满足，未来由 VectorSpace（带量子数）满足
template<typename S>
concept SpaceLike = requires(S s) {
    { s.dim()  } -> std::convertible_to<int>;
    { s.dual() } -> std::same_as<S>;
    { S::trivial(int{}) } -> std::same_as<S>;  // 工厂函数
};

// Tensor concept：当前由 DenseTensor 满足，未来由 SymmetricTensor 满足
template<typename T, typename S>
concept TensorLike = requires(T t, S s) {
    // 形状查询
    { t.rank()        } -> std::convertible_to<int>;
    { t.dim(int{})    } -> std::convertible_to<int>;
    { t.space(int{})  } -> std::same_as<S>;
    // 基本操作
    { t.permute(std::vector<int>{}) } -> std::same_as<T>;
    { t.adjoint()                   } -> std::same_as<T>;
    // 线性代数接口
    { t.as_matrix(std::vector<int>{}, std::vector<int>{}) }
      -> std::convertible_to<Eigen::MatrixXcd>;
};

// Backend concept：将 Tensor 和 Space 绑定，提供工厂方法
template<typename B>
concept TensorBackend = requires {
    typename B::Tensor;
    typename B::Space;
    typename B::Scalar;
    requires SpaceLike<typename B::Space>;
    requires TensorLike<typename B::Tensor, typename B::Space>;
} && requires(typename B::Space s, std::vector<typename B::Space> legs) {
    { B::zeros(legs) } -> std::same_as<typename B::Tensor>;  // 零张量
    { B::random(legs) } -> std::same_as<typename B::Tensor>; // 随机张量
};
```

### 5.2 DenseBackend（第一阶段实现）

```cpp
// ===== TrivialSpace：仅含维度信息 =====
class TrivialSpace {
public:
    explicit TrivialSpace(int dim) : dim_(dim), dual_(false) {}

    int  dim()  const { return dim_; }
    bool is_dual() const { return dual_; }

    TrivialSpace dual() const { return TrivialSpace(dim_, !dual_); }
    static TrivialSpace trivial(int dim) { return TrivialSpace(dim); }

    bool operator==(const TrivialSpace&) const = default;

    // 预留接口（未来 SymmetricSpace 会实现）
    // virtual std::vector<Sector> sectors() const;

private:
    TrivialSpace(int dim, bool dual) : dim_(dim), dual_(dual) {}
    int  dim_;
    bool dual_;
};

// ===== DenseBackend =====
struct DenseBackend {
    using Scalar = std::complex<double>;
    using Space  = TrivialSpace;
    using Tensor = DenseTensor;   // 见第 6 章

    static DenseTensor zeros(const std::vector<TrivialSpace>& legs);
    // 随机张量：使用 std::mt19937_64 引擎，支持显式种子以保证可重现性
    // 多线程环境下每个线程持有独立引擎（thread_local）
    static DenseTensor random(const std::vector<TrivialSpace>& legs,
                              std::optional<uint64_t> seed = std::nullopt);
    static DenseTensor identity(const TrivialSpace& space);
};

// 默认别名（用户代码无需写 Backend 参数）
using DefaultBackend = DenseBackend;
```

### 5.3 为什么这样设计

```
当前调用方式：
  dmrg(psi, H)          →  自动推导 B = DenseBackend

未来接入对称张量：
  using SymB = SymmetricBackend<U1Charge>;
  DenseMPS<SymB> psi = ...;
  SparseMPO<SymB> H  = ...;
  dmrg(psi, H)          →  自动推导 B = SymB
  // 算法层代码完全不变！
```

所有依赖对称性的逻辑只需在 `SymmetricBackend` 和 `SymmetricTensor` 中实现，不触碰算法层。

---

## 6. 稠密张量系统（DenseTensor）

### 6.1 DenseTensor 数据结构

```cpp
// include/tenet/core/dense_tensor.hpp

class DenseTensor {
public:
    // ===== 构造 =====
    DenseTensor() = default;

    // 从形状构造（全零）
    explicit DenseTensor(std::vector<TrivialSpace> legs);

    // 从形状 + 数据构造
    DenseTensor(std::vector<TrivialSpace> legs,
                std::vector<std::complex<double>> data);

    // ===== 形状查询 =====
    int rank()     const { return legs_.size(); }
    int dim(int i) const { return legs_[i].dim(); }
    int64_t numel()const;   // 总元素数
    const TrivialSpace& space(int i) const { return legs_[i]; }
    const std::vector<TrivialSpace>& spaces() const { return legs_; }

    // ===== 数据访问 =====
    // 按多维下标访问（调试用，性能敏感路径用 data()）
    std::complex<double>& operator()(std::initializer_list<int> idx);
    const std::complex<double>& operator()(std::initializer_list<int> idx) const;

    // 底层数据（行优先展平）
    std::complex<double>*       data()       { return data_.data(); }
    const std::complex<double>* data() const { return data_.data(); }

    // ===== 核心操作 =====

    // 置换腿的顺序（会重排数据）
    [[nodiscard]] DenseTensor permute(const std::vector<int>& perm) const;

    // 重塑（仅改变 shape 元信息，不移动数据）
    // 要求总元素数不变
    [[nodiscard]] DenseTensor reshape(std::vector<TrivialSpace> new_legs) const;

    // 矩阵化：将指定腿合并为行/列，返回 Eigen 矩阵（零拷贝视图）
    // row_legs: 构成行的腿的下标（从左到右）
    // col_legs: 构成列的腿的下标
    // 若腿顺序已满足连续要求，返回 Eigen::Map（零拷贝），否则先置换再 Map
    Eigen::MatrixXcd matricize(const std::vector<int>& row_legs,
                               const std::vector<int>& col_legs) const;

    // 逆矩阵化：从 Eigen 矩阵恢复张量（指定各维度大小）
    static DenseTensor from_matrix(const Eigen::MatrixXcd& mat,
                                   std::vector<TrivialSpace> legs,
                                   bool row_is_left);

    // 伴随（共轭转置：翻转所有腿的 dual 标记，取共轭）
    [[nodiscard]] DenseTensor adjoint() const;

    // 原地共轭
    DenseTensor& conj();

    // 线性组合：this = α * this + β * other
    DenseTensor& axpby(std::complex<double> alpha,
                       std::complex<double> beta,
                       const DenseTensor& other);

    // 范数
    double norm() const;
    DenseTensor& normalize();

    // ===== 腿信息 =====
    // 融合连续腿（返回新张量，被融合腿合并为一腿）
    [[nodiscard]] DenseTensor fuse(int from, int to) const;

    // 劈开一腿（逆操作）
    [[nodiscard]] DenseTensor split(int leg,
                                    const std::vector<TrivialSpace>& sub_spaces) const;

private:
    std::vector<TrivialSpace>      legs_;
    std::vector<std::complex<double>> data_;   // 行优先存储

    // 从多维下标计算线性位移
    int64_t linear_idx(const std::vector<int>& idx) const;
};
```

### 6.2 张量操作工具（tensor_ops.hpp）

```cpp
// include/tenet/core/tensor_ops.hpp

// ===== 两张量缩并 =====
// 类比 Julia 的 @tensor C[i,k] := A[i,j] * B[j,k]
//
// contracted: {(腿A的下标, 腿B的下标)} 对，指定被收缩的维度
// 剩余腿顺序：A 的开放腿（按原顺序） ++ B 的开放腿（按原顺序）
DenseTensor contract(
    const DenseTensor& A, const std::vector<int>& A_legs,  // A 参与缩并的腿
    const DenseTensor& B, const std::vector<int>& B_legs,  // B 参与缩并的腿
    const std::vector<std::pair<int,int>>& contracted      // 收缩的腿对
);

// 底层实现：将两张量矩阵化后调用 BLAS ZGEMM
// A(rows, contract) × B(contract, cols) = C(rows, cols)
Eigen::MatrixXcd matmul(const DenseTensor& A,
                        const std::vector<int>& A_row_legs,
                        const std::vector<int>& A_con_legs,
                        const DenseTensor& B,
                        const std::vector<int>& B_con_legs,
                        const std::vector<int>& B_col_legs);

// ===== 内积 =====
// ⟨A|B⟩ = Tr(A† B)，要求形状相同
std::complex<double> inner(const DenseTensor& A, const DenseTensor& B);

// ===== 直积 =====
DenseTensor outer(const DenseTensor& A, const DenseTensor& B);

// ===== 工具 =====
DenseTensor zeros_like(const DenseTensor& t);
DenseTensor random_like(const DenseTensor& t);
```

### 6.3 矩阵分解（factorization.hpp）

```cpp
// include/tenet/core/factorization.hpp

// ===== 截断参数 =====
struct TruncParams {
    int    maxD     = 0;       // 最大保留奇异值数（0 = 不限）
    double cutoff   = 1e-12;   // 相对截断阈值（截去 σ/σ_max < cutoff 的部分）
    bool normalize  = false;   // 截断后是否重归一化
};

// ===== QR 分解 =====
// T[split_at 腿以左 | split_at 腿以右] = Q * R
// Q: 左正交（Q†Q = I），R: 上三角
struct QRResult {
    DenseTensor Q;
    DenseTensor R;
    int         bond_dim;
};
QRResult qr(const DenseTensor& T, int split_at);

// 右 QR（R * Q）：Q 右正交
QRResult rq(const DenseTensor& T, int split_at);

// ===== SVD 分解 =====
// T = U * S * Vt（可截断）
struct SVDResult {
    DenseTensor          U;                 // 左奇异向量
    Eigen::VectorXd      S;                 // 奇异值（降序）
    DenseTensor          Vt;                // 右奇异向量（已转置/共轭）
    int                  bond_dim;          // 截断后的键维度
    double               truncation_err;    // 截断误差（丢弃的奇异值平方和）
};
SVDResult svd(const DenseTensor& T, int split_at,
              const TruncParams& trunc = {});

// 随机化 SVD（低秩近似，用于 CBE 的 randSVD 方案）
// target_rank: 目标秩，oversample: 过采样比（Halko 算法）
SVDResult rand_svd(const DenseTensor& T, int split_at,
                   int target_rank, double oversample = 1.5,
                   const TruncParams& trunc = {});

// ===== 特征分解（厄米矩阵）=====
struct EigenResult {
    Eigen::VectorXd eigenvalues;     // 升序
    DenseTensor     eigenvectors;
};
EigenResult eigh(const DenseTensor& T);

// ===== 矩阵指数（用于 TDVP）=====
// exp(α * H) * v，H 为厄米矩阵（通过 Krylov 近似）
DenseTensor matrix_exp_times_vec(const DenseTensor& H,
                                  const DenseTensor& v,
                                  std::complex<double> alpha,
                                  double tol = 1e-10);
```

---

## 7. MPS / MPO 模块设计

### 7.1 MPS 张量（三腿：左虚 × 物理 × 右虚）

```cpp
// include/tenet/mps/mps_tensor.hpp

template<typename B = DenseBackend>
class MPSTensor {
public:
    using Tensor = typename B::Tensor;
    using Space  = typename B::Space;

    // 构造：指定左/物理/右空间
    MPSTensor(Space vl, Space phys, Space vr);

    // 形状查询
    int left_dim()  const { return data_.dim(0); }
    int phys_dim()  const { return data_.dim(1); }
    int right_dim() const { return data_.dim(2); }
    const Space& left_space()  const { return data_.space(0); }
    const Space& phys_space()  const { return data_.space(1); }
    const Space& right_space() const { return data_.space(2); }

    // 底层张量访问
    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    // 矩阵化视图（用于 QR/SVD）
    // left_ortho: 将 (vl×phys) 作为行，vr 作为列
    // right_ortho: 将 vl 作为行，(phys×vr) 作为列
    Eigen::MatrixXcd as_matrix_left()  const;  // (D_l * d) × D_r
    Eigen::MatrixXcd as_matrix_right() const;  // D_l × (d * D_r)

    // 伴随
    MPSTensor adjoint() const;

    // 正则化：就地 QR，返回剩余矩阵（中心矩阵）
    Eigen::MatrixXcd left_canonicalize();    // 本格点变为左正交
    Eigen::MatrixXcd right_canonicalize();   // 本格点变为右正交

    // 从矩阵重建（SVD/QR 后更新格点）
    static MPSTensor from_left_matrix(const Eigen::MatrixXcd& mat,
                                       Space vl, Space phys, Space vr);
    static MPSTensor from_right_matrix(const Eigen::MatrixXcd& mat,
                                        Space vl, Space phys, Space vr);

private:
    Tensor data_;   // rank-3
};
```

### 7.2 DenseMPS

```cpp
// include/tenet/mps/mps.hpp

template<typename B = DenseBackend>
class DenseMPS {
public:
    using Tensor = typename B::Tensor;
    using Space  = typename B::Space;
    using SiteTensor = MPSTensor<B>;

    explicit DenseMPS(int L) : L_(L), sites_(L) {}

    // 格点访问
    SiteTensor&       operator[](int i)       { return sites_[i]; }
    const SiteTensor& operator[](int i) const { return sites_[i]; }

    int length() const { return L_; }

    // 正交中心管理
    int center_left()  const { return cleft_; }
    int center_right() const { return cright_; }
    void set_center(int l, int r) { cleft_ = l; cright_ = r; }
    bool is_canonical() const { return cleft_ == cright_; }

    // 正交化到指定格点
    void move_center_to(int target);    // 就地，最优路径
    void left_canonicalize(int from, int to);
    void right_canonicalize(int from, int to);

    // 内积 ⟨this|other⟩（要求相同结构）
    std::complex<double> inner(const DenseMPS& other) const;

    double norm() const;
    void normalize();

    // 键维度查询
    int bond_dim(int bond) const;           // bond ∈ [0, L]（0=左边界，L=右边界）
    int max_bond_dim() const;

    // 伴随（零拷贝引用包装）
    Adjoint<DenseMPS> adjoint() { return Adjoint<DenseMPS>(*this); }

    // I/O（HDF5）
    void save(const std::string& path) const;
    static DenseMPS load(const std::string& path);

    // 随机初始化
    static DenseMPS random(int L, int D,
                            const std::vector<Space>& phys_spaces);

private:
    int L_;
    std::vector<SiteTensor> sites_;
    int cleft_{0}, cright_{0};
};
```

### 7.3 稀疏 MPO

```cpp
// include/tenet/mps/mpo_tensor.hpp

// SparseMPOTensor：(D_in × D_out) 矩阵，元素为 LocalOperator 或 nullptr
template<typename B = DenseBackend>
class SparseMPOTensor {
public:
    SparseMPOTensor(int d_in, int d_out)
        : d_in_(d_in), d_out_(d_out), ops_(d_in * d_out) {}

    // 元素访问
    AbstractLocalOperator<B>*       operator()(int i, int j);
    const AbstractLocalOperator<B>* operator()(int i, int j) const;
    void set(int i, int j, std::unique_ptr<AbstractLocalOperator<B>> op);
    bool has(int i, int j) const { return ops_[i * d_out_ + j] != nullptr; }

    int d_in()  const { return d_in_; }
    int d_out() const { return d_out_; }

    // 迭代非零元素：fn(row, col, op)
    void for_each_nonzero(
        std::function<void(int, int, const AbstractLocalOperator<B>&)> fn) const;

private:
    int d_in_, d_out_;
    std::vector<std::unique_ptr<AbstractLocalOperator<B>>> ops_;
};

// SparseMPO
template<typename B = DenseBackend>
class SparseMPO {
public:
    explicit SparseMPO(int L) : L_(L), sites_(L) {}

    SparseMPOTensor<B>&       operator[](int i)       { return sites_[i]; }
    const SparseMPOTensor<B>& operator[](int i) const { return sites_[i]; }

    int length() const { return L_; }
    std::pair<int,int> bond_dim(int i) const;  // (D_in, D_out) at site i

private:
    int L_;
    std::vector<SparseMPOTensor<B>> sites_;
};
```

### 7.4 正则化（canonical.hpp）

```cpp
// include/tenet/mps/canonical.hpp

// 将 MPS 左正交化至 [from, to)，把中心矩阵推到 to
template<typename B>
void left_canonicalize(DenseMPS<B>& psi, int from, int to);

// 将 MPS 右正交化至 (from, to]，把中心矩阵推到 from
template<typename B>
void right_canonicalize(DenseMPS<B>& psi, int from, int to);

// 将正交中心从当前位置移动到 target（自动选择方向）
template<typename B>
void move_center(DenseMPS<B>& psi, int target);

// 计算冯诺依曼纠缠熵（需要先做 SVD）
double von_neumann_entropy(const Eigen::VectorXd& singular_values);
```

---

## 8. 环境张量模块设计

### 8.1 环境张量类型

```cpp
// include/tenet/environment/env_tensor.hpp

// 左环境张量：⟨ψ|H|ψ⟩ 从左侧部分缩并的结果
// 形状：(D_mps* × D_mpo × D_mps)，矩阵化为 (D_mps* · D_mpo) × D_mps
template<typename B = DenseBackend>
class LeftEnvTensor {
public:
    using Tensor = typename B::Tensor;

    explicit LeftEnvTensor(Tensor data) : data_(std::move(data)) {}

    Tensor&       data()       { return data_; }
    const Tensor& data() const { return data_; }

    // 便捷访问：矩阵形式 (D_bra · D_mpo) × D_ket
    Eigen::MatrixXcd as_matrix() const;

    // 边界条件：最左侧空环境（1×1×1 张量，值为 1）
    static LeftEnvTensor boundary();

private:
    Tensor data_;
};

template<typename B = DenseBackend>
class RightEnvTensor { /* 对称结构 */ };

// 稀疏左环境张量：对应 SparseMPO 的 D_mpo 个 LeftEnvTensor
// envs_[i] 对应 MPO 键维度的第 i 行（nullptr 表示零贡献）
template<typename B = DenseBackend>
class SparseLeftEnvTensor {
public:
    explicit SparseLeftEnvTensor(int D) : D_(D), envs_(D) {}

    int dim() const { return D_; }
    LeftEnvTensor<B>*       operator[](int i)       { return envs_[i].get(); }
    const LeftEnvTensor<B>* operator[](int i) const { return envs_[i].get(); }
    void set(int i, std::unique_ptr<LeftEnvTensor<B>> e) { envs_[i] = std::move(e); }
    bool has(int i) const { return envs_[i] != nullptr; }

    // 边界：D=1，envs_[0] 为单位边界张量
    static SparseLeftEnvTensor boundary(int D);

private:
    int D_;
    std::vector<std::unique_ptr<LeftEnvTensor<B>>> envs_;
};

template<typename B = DenseBackend>
class SparseRightEnvTensor { /* 对称结构 */ };
```

### 8.2 主环境类

```cpp
// include/tenet/environment/environment.hpp

template<typename B = DenseBackend>
class Environment {
public:
    // 构造：传入 MPS 和 SparseMPO，自动创建三层结构
    Environment(DenseMPS<B>& psi, SparseMPO<B>& H);

    // 层引用
    DenseMPS<B>&   psi() { return *psi_; }
    SparseMPO<B>&  H()   { return *H_; }
    int length()   const { return L_; }

    // 环境缓存访问
    // left_env(0)  = 最左边界（初始化为单位环境）
    // left_env(L)  = 整个系统的左环境（不常用）
    SparseLeftEnvTensor<B>&  left_env(int site);
    SparseRightEnvTensor<B>& right_env(int site);

    // 当前正交中心（从 MPS 同步）
    int center() const { return psi_->center_left(); }

    // ===== 初始化 =====
    // 从当前 MPS 正则形式出发，建立全套左/右环境
    // 调用前：psi 需已正则化到某个格点
    void build_all();    // 重建全部（第一次调用）
    void rebuild();      // 在 psi 修改后重建

    // ===== 扫描推进（核心接口）=====
    // 将 left_env[site] → left_env[site+1]
    // 前提：psi[site] 已被更新，且已左正交化
    void push_right(int site);

    // 将 right_env[site] → right_env[site-1]
    void push_left(int site);

    // ===== 缓存管理 =====
    void release_left_cache(int site);    // 释放 left_env[site]
    void release_right_cache(int site);
    void release_distant_cache(int current_site, int window = 2);

private:
    int            L_;
    DenseMPS<B>*   psi_;    // 非拥有
    SparseMPO<B>*  H_;      // 非拥有

    // 大小均为 L+1：索引 i 表示格点 i 左侧/右侧的边界
    std::vector<std::optional<SparseLeftEnvTensor<B>>>  left_envs_;
    std::vector<std::optional<SparseRightEnvTensor<B>>> right_envs_;
};
```

### 8.3 环境推进实现

```cpp
// src/environment/env_push.cpp

template<typename B>
void Environment<B>::push_right(int site) {
    // left_envs_[site] 必须已存在
    assert(left_envs_[site].has_value());

    auto& L_env   = *left_envs_[site];
    auto& psi_t   = psi_->operator[](site);        // MPSTensor
    auto& H_t     = H_->operator[](site);          // SparseMPOTensor
    int   d_out   = H_t.d_out();

    SparseLeftEnvTensor<B> new_L(d_out);

    // 稀疏循环：只计算 MPO 非零元素的贡献
    H_t.for_each_nonzero([&](int row, int col, const AbstractLocalOperator<B>& op) {
        if (!L_env.has(row)) return;   // 此行环境为零，跳过

        // 核心缩并：new_L[col] += L_env[row] ⊗ ψ[site] ⊗ op ⊗ ψ*[site]
        // 张量网络图：
        //
        //   [L_env[row]]---ψ*[site]---
        //         |           |
        //        op[row,col]  |  （物理指标缩并）
        //         |           |
        //   [new_L[col]]---ψ[site]---
        //
        auto contrib = contract_env_step_right(
            *L_env[row], psi_t, op, psi_t);

        if (!new_L.has(col))
            new_L.set(col, std::make_unique<LeftEnvTensor<B>>(contrib));
        else
            new_L[col]->data().axpby(1.0, 1.0, contrib);
    });

    left_envs_[site + 1] = std::move(new_L);
}
```

---

## 9. 哈密顿量与 Krylov 求解器设计

### 9.1 稀疏投影哈密顿量

```cpp
// include/tenet/hamiltonian/projective_ham.hpp

// 单格点有效哈密顿量 H_eff：作用于 MPSTensor，供 Lanczos 使用
template<typename B = DenseBackend>
class SparseProjectiveHamiltonian {
public:
    SparseProjectiveHamiltonian(
        const SparseLeftEnvTensor<B>&  env_l,
        const SparseRightEnvTensor<B>& env_r,
        const SparseMPO<B>&            H,
        int                            site,
        double                         E0 = 0.0);

    // 核心：H_eff * x（Krylov 求解器调用此接口）
    // 输入输出均为展平的复向量（Eigen 格式）
    void apply(const Eigen::VectorXcd& x, Eigen::VectorXcd& y) const;

    // 向量 ↔ MPSTensor 转换（用于 apply 内部）
    MPSTensor<B>         vec_to_tensor(const Eigen::VectorXcd& x) const;
    Eigen::VectorXcd     tensor_to_vec(const MPSTensor<B>& T) const;

    int dim() const;     // 向量空间维数（D_l * d * D_r）

private:
    const SparseLeftEnvTensor<B>&  env_l_;
    const SparseRightEnvTensor<B>& env_r_;
    const SparseMPO<B>&            H_;
    int                            site_;
    double                         E0_;
    // 预计算有效非零 (row, col) 对，加速 apply
    std::vector<std::tuple<int,int>> valid_inds_;
};

// ===== 工厂函数 =====

// 单格点投影哈密顿量
template<typename B>
SparseProjectiveHamiltonian<B>
make_proj1(const Environment<B>& env, int site, double E0 = 0.0);

// 双格点投影哈密顿量（用于 DMRG2/TDVP2）
// 返回的算子作用于 CompositeMPSTensor（两格点合并）
template<typename B>
SparseProjectiveHamiltonian<B>
make_proj2(const Environment<B>& env, int site, double E0 = 0.0);
```

### 9.2 Lanczos 特征值求解器

```cpp
// include/tenet/krylov/lanczos.hpp

struct LanczosConfig {
    int    krylov_dim = 8;      // Krylov 子空间维数（DMRG 默认）
    int    max_iter   = 1;      // 最大重启次数
    double tol        = 1e-6;   // 残差收敛容差
    bool   eager      = true;   // 提前终止（残差低于 tol 即停）
    int    verbosity  = 0;      // 日志等级
};

struct LanczosResult {
    double           eigenvalue;
    Eigen::VectorXcd eigenvector;
    int              num_iter;
    double           residual;
    bool             converged;
};

// 模板化求解器：接受任何提供 apply() 和 dim() 的线性算子
// LinearOp 需满足：
//   void op.apply(const Eigen::VectorXcd& x, Eigen::VectorXcd& y)
//   int  op.dim()
template<typename LinearOp>
LanczosResult lanczos_eigsolve(
    const LinearOp&         op,
    const Eigen::VectorXcd& init,    // 初始向量
    const LanczosConfig&    cfg = {});

// 线性方程求解 (op - shift*I) x = rhs（用于 TDVP 中心格点更新）
template<typename LinearOp>
Eigen::VectorXcd lanczos_linsolve(
    const LinearOp&         op,
    const Eigen::VectorXcd& rhs,
    std::complex<double>    shift,
    const LanczosConfig&    cfg = {});

// 内部：修正 Gram-Schmidt 正交化（数值稳定性优于标准 GS）
namespace detail {
void modified_gram_schmidt(Eigen::VectorXcd& v,
                            const std::vector<Eigen::VectorXcd>& basis);
}
```

### 9.3 Arnoldi 指数传播（TDVP 用）

```cpp
// include/tenet/krylov/arnoldi.hpp

// 计算 exp(α * op) * v（矩阵指数作用于向量），用于 TDVP 时间步
// 通过 Krylov 子空间近似，避免显式构造矩阵指数
template<typename LinearOp>
Eigen::VectorXcd krylov_expm_times_vec(
    const LinearOp&         op,
    const Eigen::VectorXcd& v,
    std::complex<double>    alpha,   // 通常为 -i*τ（实时）或 -τ（虚时）
    int    krylov_dim = 32,
    double tol        = 1e-8);
```

---

## 10. 相互作用树与稀疏 MPO 构建

### 10.1 局域算符类型

```cpp
// include/tenet/intr_tree/local_operator.hpp

// 抽象基类（无对称性：张量就是普通 Eigen 矩阵）
template<typename B = DenseBackend>
class AbstractLocalOperator {
public:
    virtual ~AbstractLocalOperator() = default;
    virtual int site() const = 0;
    virtual const std::string& name() const = 0;
    // 当前：纯矩阵（phys_dim × phys_dim）
    // 未来 SymmetricBackend：SymmetricTensor（带量子数）
    virtual const typename B::Tensor& tensor() const = 0;
    virtual bool is_identity() const { return false; }
    virtual std::complex<double> strength() const { return 1.0; }
    virtual std::unique_ptr<AbstractLocalOperator> clone() const = 0;
};

// 普通局域算符
template<typename B = DenseBackend>
class LocalOperator : public AbstractLocalOperator<B> {
public:
    LocalOperator(typename B::Tensor tensor,
                  std::string name,
                  int site,
                  std::complex<double> strength = 1.0);
    // 实现纯虚接口
};

// 恒等算符（不存储矩阵，按需生成单位矩阵）
template<typename B = DenseBackend>
class IdentityOperator : public AbstractLocalOperator<B> {
public:
    IdentityOperator(int site, int phys_dim);
    bool is_identity() const override { return true; }
    const typename B::Tensor& tensor() const override;  // 返回缓存的单位张量
};
```

### 10.2 相互作用树

```cpp
// include/tenet/intr_tree/interaction_node.hpp

// 树叶数据
struct InteractionLeaf {
    std::vector<Eigen::MatrixXcd> operators;  // 各格点的算符矩阵
    std::vector<int>              sites;       // 格点索引（有序）
    std::vector<std::string>      names;       // 算符名称
    std::vector<bool>             fermionic;   // 是否需要 JW 弦
    std::complex<double>          strength;    // 耦合系数
    std::optional<Eigen::MatrixXcd> jw_op;    // JW 算符（费米子系统）
};

// 树节点
class InteractionTreeNode {
public:
    InteractionTreeNode() = default;

    // 添加子节点（返回子节点引用，支持链式调用）
    InteractionTreeNode& add_child(Eigen::MatrixXcd op,
                                    std::string name, int site);
    void set_leaf(InteractionLeaf leaf);

    bool is_leaf()              const { return leaf_.has_value(); }
    const InteractionLeaf& leaf()const { return *leaf_; }
    auto children()                   { return std::span(children_); }
    void print(std::ostream& = std::cout, int depth = 0) const;

private:
    std::optional<Eigen::MatrixXcd>    op_;        // 本节点算符
    std::string                         op_name_;
    int                                 site_{-1};
    InteractionTreeNode*                parent_{};
    std::vector<InteractionTreeNode>    children_;
    std::optional<InteractionLeaf>      leaf_;
};

// 相互作用树（面向用户的接口）
class InteractionTree {
public:
    explicit InteractionTree(int L) : L_(L) {}

    // 添加 1~N 体相互作用
    void add1(const Eigen::MatrixXcd& op, int site,
               std::string name = "", std::complex<double> strength = 1.0);

    void add2(const Eigen::MatrixXcd& op1, int s1, std::string n1,
               const Eigen::MatrixXcd& op2, int s2, std::string n2,
               std::complex<double> strength = 1.0,
               bool fermionic = false);

    // 通用 N 体接口
    void addN(std::vector<Eigen::MatrixXcd> ops,
               std::vector<int> sites,
               std::vector<std::string> names,
               std::vector<bool> fermionic,
               std::complex<double> strength);

    // 弦算符（Jordan-Wigner）
    void add_string(std::vector<Eigen::MatrixXcd> ops,
                     std::vector<int> sites,
                     const Eigen::MatrixXcd& jw_op,
                     std::complex<double> strength);

    // 编译为 SparseMPO（有限状态自动机算法）
    template<typename B = DenseBackend>
    SparseMPO<B> to_sparse_mpo() const;

    void print(std::ostream& = std::cout) const;

private:
    int L_;
    InteractionTreeNode root_;
    SparseMPO<DenseBackend> compile_automata() const;
};
```

---

## 11. 可观测量系统设计

```cpp
// include/tenet/observables/obs_node.hpp

struct ObservableLeaf {
    std::vector<int>         sites;
    std::vector<std::string> op_names;
    mutable std::complex<double> value{std::numeric_limits<double>::quiet_NaN(), 0};
};

template<typename B = DenseBackend>
class ObservableTreeNode {
public:
    void add_child(ObservableTreeNode child);
    void set_leaf(ObservableLeaf leaf);

    bool is_leaf() const { return leaf_.has_value(); }
    ObservableLeaf& leaf() { return *leaf_; }

    // 缓存的部分缩并环境（避免重复计算共同前缀）
    mutable std::optional<LeftEnvTensor<B>> cached_env;
    std::optional<Eigen::MatrixXcd>         op_matrix;
    int                                      site{-1};

    auto children() { return std::span(children_); }

private:
    std::vector<ObservableTreeNode> children_;
    std::optional<ObservableLeaf>   leaf_;
};

// 可观测量计算接口
template<typename B = DenseBackend>
class ObservableForest {
public:
    // 注册单格点可观测量：⟨op[site]⟩
    void add(const Eigen::MatrixXcd& op, int site,
              const std::string& name = "");

    // 注册两格点关联：⟨op1[s1] op2[s2]⟩
    void add_correlation(const Eigen::MatrixXcd& op1, int s1, std::string n1,
                          const Eigen::MatrixXcd& op2, int s2, std::string n2);

    // 计算所有期望值，返回 {名称 → 值}
    std::map<std::string, std::complex<double>>
    compute(const Environment<B>& env) const;

    // 并行版本（多线程遍历独立分支）
    std::map<std::string, std::complex<double>>
    compute_parallel(const Environment<B>& env, int nthreads = 0) const;

private:
    std::vector<ObservableTreeNode<B>> trees_;
};
```

---

## 12. 局域空间（LocalSpace）设计

第一阶段不使用量子数，所有算符均为普通复矩阵（`Eigen::MatrixXcd`）。

```cpp
// include/tenet/local_space/spin.hpp
namespace local_space {

// 每个命名空间提供：
//   - phys_dim(): 物理维度
//   - 各算符的 Eigen::MatrixXcd

namespace SpinHalf {   // 自旋-1/2（2×2 矩阵）
    inline int phys_dim() { return 2; }

    Eigen::MatrixXcd Sz();            // 对角 {1/2, -1/2}
    Eigen::MatrixXcd Sp();            // 升算符
    Eigen::MatrixXcd Sm();            // 降算符
    Eigen::MatrixXcd Id();            // 单位矩阵

    // 返回相互作用项的算符对，直接传入 add2()
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> SpSm();  // (S+, S-)
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> SmSp();
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> SzSz();
}

namespace Spin1 {      // 自旋-1（3×3 矩阵）
    inline int phys_dim() { return 3; }
    Eigen::MatrixXcd Sz();
    Eigen::MatrixXcd Sp();
    Eigen::MatrixXcd Sm();
    // ...
}

} // namespace local_space

// include/tenet/local_space/fermion.hpp
namespace local_space {

namespace Spinless {   // 无自旋费米子（2×2：|0⟩, |1⟩）
    inline int phys_dim() { return 2; }
    Eigen::MatrixXcd Z();    // Jordan-Wigner: diag(1, -1)
    Eigen::MatrixXcd n();    // 粒子数
    Eigen::MatrixXcd Cdag(); // 产生算符
    Eigen::MatrixXcd C();    // 湮灭算符
    // FdagF = (Cdag*Z, C)，fermionic = true
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> CdagC();
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> CCdag();
}

namespace SpinfulHalf { // 自旋-1/2 费米子（4×4：|0⟩, |↑⟩, |↓⟩, |↑↓⟩）
    inline int phys_dim() { return 4; }
    Eigen::MatrixXcd Z();
    Eigen::MatrixXcd n();
    Eigen::MatrixXcd n_up(), n_dn();
    Eigen::MatrixXcd nd();    // 双占据
    Eigen::MatrixXcd Sz();
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> Cup_dag_Cup();
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> Cdn_dag_Cdn();
    std::pair<Eigen::MatrixXcd, Eigen::MatrixXcd> SpSm(), SmSp(), SzSz();
}

} // namespace local_space
```

---

## 13. 算法层设计（DMRG / TDVP / CBE / SETTN）

### 13.1 配置结构体

```cpp
// include/tenet/process_control/config.hpp

// ===== DMRG =====
struct DMRGConfig {
    enum class Scheme { SingleSite, DoubleSite } scheme = Scheme::SingleSite;

    TruncParams    trunc{.maxD = 256, .cutoff = 1e-10};
    LanczosConfig  solver{.krylov_dim = 8, .tol = 1e-6};

    int    max_sweeps  = 20;
    double E_tol       = 1e-8;   // 能量收敛阈值
    double S_tol       = 1e-5;   // 纠缠熵收敛阈值

    // CBE 选项
    struct CBEOptions {
        bool   enabled = false;
        double lambda  = 1.2;
        int    n_full  = 4;      // 前几轮用 fullSVD
    } cbe{};

    int gc_every_sweep = 2;      // 每 N 次扫描释放远端缓存
    int gc_every_site  = -1;     // -1 = 禁用

    int verbosity = 1;           // 0=静默，1=每轮扫描，2=每格点
};

// ===== TDVP =====
struct TDVPConfig {
    enum class Scheme { SingleSite, DoubleSite } scheme = Scheme::SingleSite;
    TruncParams   trunc;
    LanczosConfig solver{.krylov_dim = 32, .tol = 1e-8};
    bool          use_cbe    = false;
    double        cbe_lambda = 1.2;
    int           cbe_n_full = 4;
    int           verbosity  = 1;
};

// ===== CBE =====
struct CBEConfig {
    enum class SVDMode { Full, Rand, Dynamic } mode = SVDMode::Dynamic;
    double lambda     = 1.2;   // 随机过采样比（randSVD）
    int    n_boundary = 4;     // Dynamic 模式：边界范围内用 fullSVD
    int    target_D   = 0;     // 目标键维度（0 = 由 trunc 决定）
};
```

### 13.2 运行信息追踪

```cpp
// include/tenet/process_control/sweep_info.hpp

struct SiteInfo {
    int    site;
    double energy;
    double entropy;           // 冯诺依曼纠缠熵
    double trunc_err;         // SVD 截断误差
    int    bond_dim;          // 截断后键维度
    double wall_ms;           // 本格点耗时（毫秒）
};

struct SweepInfo {
    int                    sweep_idx;
    double                 energy;
    double                 delta_energy;
    double                 max_entropy;
    double                 delta_entropy;
    std::vector<SiteInfo>  site_infos;
    bool                   converged;
    double                 wall_s;        // 本轮扫描耗时（秒）
};

struct DMRGResult {
    double                 ground_energy;
    std::vector<SweepInfo> sweeps;
    bool                   converged;
    int                    total_sweeps;

    void print_summary(std::ostream& os = std::cout) const;
    void save_hdf5(const std::string& path) const;
};
```

### 13.3 DMRG 主接口与内循环

```cpp
// include/tenet/algorithm/dmrg.hpp

// 主接口（单/双格点由 cfg.scheme 决定）
template<typename B = DenseBackend>
DMRGResult dmrg(DenseMPS<B>& psi, SparseMPO<B>& H,
                const DMRGConfig& cfg = {});

// 底层：一次完整扫描（L2R + R2L）
namespace detail {

template<typename B>
SweepInfo dmrg_sweep(Environment<B>& env,
                      const DMRGConfig& cfg,
                      int sweep_idx,
                      double prev_energy = 0.0,
                      double prev_entropy = 0.0);

// 单格点，一次格点优化
// 返回：能量、熵、键维度
template<typename B>
SiteInfo optimize_site_single(Environment<B>& env, int site,
                               const DMRGConfig& cfg, bool l2r);

// 双格点，一次两格点优化
template<typename B>
SiteInfo optimize_site_double(Environment<B>& env, int site,
                               const DMRGConfig& cfg, bool l2r);

} // namespace detail
```

**DMRG 主循环（概要）：**

```cpp
template<typename B>
DMRGResult dmrg(DenseMPS<B>& psi, SparseMPO<B>& H, const DMRGConfig& cfg) {
    // 1. 初始化环境
    Environment<B> env(psi, H);
    env.build_all();

    DMRGResult result;
    double prev_E = 0.0, prev_S = 0.0;

    for (int sweep = 0; sweep < cfg.max_sweeps; ++sweep) {
        // 2. 完整扫描（内含 L2R + R2L）
        auto info = detail::dmrg_sweep(env, cfg, sweep, prev_E, prev_S);
        result.sweeps.push_back(info);

        // 3. 打印进度
        if (cfg.verbosity >= 1)
            spdlog::info("Sweep {:3d}: E = {:.12f}, ΔE = {:.3e}, S = {:.6f}",
                          sweep, info.energy, info.delta_energy, info.max_entropy);

        // 4. 收敛判断
        if (sweep > 0 && info.converged) {
            result.converged = true;
            break;
        }

        // 5. 定期释放缓存
        if (cfg.gc_every_sweep > 0 && (sweep + 1) % cfg.gc_every_sweep == 0)
            env.release_distant_cache(env.center(), 2);

        prev_E = info.energy;
        prev_S = info.max_entropy;
    }

    result.ground_energy = result.sweeps.back().energy;
    result.total_sweeps  = result.sweeps.size();
    return result;
}
```

**单格点优化核心：**

```cpp
template<typename B>
SiteInfo detail::optimize_site_single(
    Environment<B>& env, int site, const DMRGConfig& cfg, bool l2r)
{
    auto t0 = Clock::now();

    // [可选] CBE 扩展键维度
    if (cfg.cbe.enabled)
        cbe(env, site, CBEConfig{.lambda = cfg.cbe.lambda});

    // 提取有效哈密顿量
    auto H_eff = make_proj1(env, site);

    // Lanczos 求基态
    auto init   = env.psi()[site].as_vector();
    auto result = lanczos_eigsolve(H_eff, init, cfg.solver);
    double E    = result.eigenvalue;

    // 更新格点张量
    auto new_tensor = H_eff.vec_to_tensor(result.eigenvector);
    env.psi()[site] = new_tensor;

    // SVD 截断 + 正交化
    SVDResult svd_res;
    if (l2r) {
        svd_res = svd(new_tensor.data(), 2, cfg.trunc); // split after phys leg
        env.psi()[site]     = MPSTensor<B>::from_left_matrix(svd_res.U * svd_res.S.asDiagonal(), ...);
        env.psi()[site + 1] = absorb_left(svd_res.Vt, env.psi()[site + 1]);
        env.push_right(site);
    } else {
        svd_res = svd(new_tensor.data(), 1, cfg.trunc);
        env.psi()[site]     = MPSTensor<B>::from_right_matrix(svd_res.S.asDiagonal() * svd_res.Vt, ...);
        env.psi()[site - 1] = absorb_right(svd_res.U, env.psi()[site - 1]);
        env.push_left(site);
    }

    double S  = von_neumann_entropy(svd_res.S);
    auto   t1 = Clock::now();

    return SiteInfo{
        .site      = site,
        .energy    = E,
        .entropy   = S,
        .trunc_err = svd_res.truncation_err,
        .bond_dim  = svd_res.bond_dim,
        .wall_ms   = duration_ms(t0, t1)
    };
}
```

### 13.4 TDVP 接口

```cpp
// include/tenet/algorithm/tdvp.hpp

struct TDVPResult {
    std::vector<double>    times;
    std::vector<SweepInfo> sweep_infos;
    std::vector<double>    norms;
};

// 实时演化：对每个时间点 times[i] 返回状态
template<typename B = DenseBackend>
TDVPResult tdvp(Environment<B>& env,
                const std::vector<double>& times,
                const TDVPConfig& cfg = {});

// 虚时热化（tanTRG）
struct TanTRGResult {
    std::vector<double> betas;
    std::vector<double> free_energies;
    std::vector<double> energies;
};

template<typename B = DenseBackend>
TanTRGResult tan_trg(DenseMPS<B>& rho, SparseMPO<B>& H,
                      const std::vector<double>& betas,
                      const TDVPConfig& cfg = {});
```

### 13.5 CBE 接口

```cpp
// include/tenet/algorithm/cbe.hpp

struct CBEResult {
    int    old_D;
    int    new_D;
    double trunc_err;
};

template<typename B = DenseBackend>
CBEResult cbe(Environment<B>& env, int site, const CBEConfig& cfg = {});

namespace detail {
template<typename B>
CBEResult cbe_full_svd(Environment<B>& env, int site, const CBEConfig& cfg);

template<typename B>
CBEResult cbe_rand_svd(Environment<B>& env, int site, const CBEConfig& cfg);

template<typename B>
CBEResult cbe_dynamic(Environment<B>& env, int site, const CBEConfig& cfg) {
    int L = env.length();
    // 距边界较近用 fullSVD，内部用 randSVD
    if (std::min(site, L - 1 - site) <= cfg.n_boundary)
        return cbe_full_svd(env, site, cfg);
    else
        return cbe_rand_svd(env, site, cfg);
}
} // namespace detail
```

---

## 14. 并发与线程安全设计

### 14.1 并行层次

| 层次 | 策略 | 工具 |
|------|------|------|
| 扫描格点（外层） | 串行（顺序依赖） | — |
| 稀疏 MPO 列（中层） | 并行（列间独立） | OpenMP |
| BLAS 操作（最内层） | 自动多线程 | MKL/OpenBLAS |

### 14.2 环境更新并行化

```cpp
template<typename B>
void Environment<B>::push_right(int site) {
    auto& L_env = *left_envs_[site];
    int   d_out = H_->operator[](site).d_out();

    // 收集非零项（串行，快速）
    std::vector<std::tuple<int,int,const AbstractLocalOperator<B>*>> nonzeros;
    H_->operator[](site).for_each_nonzero([&](int row, int col, const auto& op) {
        if (L_env.has(row)) nonzeros.emplace_back(row, col, &op);
    });

    // 并行计算各列贡献
    std::vector<std::optional<typename B::Tensor>> accum(d_out);

    #pragma omp parallel for schedule(dynamic) if(nonzeros.size() > 4)
    for (int k = 0; k < (int)nonzeros.size(); ++k) {
        auto [row, col, op] = nonzeros[k];
        auto contrib = contract_env_step_right(*L_env[row],
                                               psi_->operator[](site), *op);
        #pragma omp critical(env_accum)
        {
            if (!accum[col].has_value()) accum[col] = contrib;
            else                         accum[col]->axpby(1.0, 1.0, contrib);
        }
    }

    // 写入新环境
    SparseLeftEnvTensor<B> new_L(d_out);
    for (int col = 0; col < d_out; ++col)
        if (accum[col].has_value())
            new_L.set(col, std::make_unique<LeftEnvTensor<B>>(std::move(*accum[col])));

    left_envs_[site + 1] = std::move(new_L);
}
```

### 14.3 可观测量并行计算

```cpp
// 每棵独立的树并行计算（树间无数据依赖）
template<typename B>
std::map<std::string, std::complex<double>>
ObservableForest<B>::compute_parallel(const Environment<B>& env, int nthreads) const {
    if (nthreads <= 0) nthreads = omp_get_max_threads();

    std::map<std::string, std::complex<double>> result;
    std::mutex result_mutex;

    #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
    for (int i = 0; i < (int)trees_.size(); ++i) {
        auto local = compute_tree(trees_[i], env);
        std::lock_guard lock(result_mutex);
        result.merge(local);
    }
    return result;
}
```

---

## 15. 内存管理策略

### 15.1 所有权规则

| 对象 | 所有权 | 理由 |
|------|--------|------|
| MPS/MPO 格点张量 | `std::vector<MPSTensor<B>>` 值语义 | 连续存储，缓存友好 |
| 环境缓存 | `std::optional<SparseLeftEnvTensor<B>>` | 可按需释放 |
| 树节点 | 值语义（子节点嵌入父节点） | 父析构自动级联 |
| MPO 元素（LocalOperator） | `std::unique_ptr<AbstractLocalOperator>` | 多态，明确所有权 |
| 投影哈密顿量 | 非拥有引用（`const&`） | 生命周期 < Environment |

### 15.2 张量数据存储选择

```
DenseTensor 内部使用 std::vector<complex<double>>（行优先展平）
优点：
  - 标准库管理生命周期，无需手动 malloc/free
  - 连续内存，对 BLAS 友好
  - 移动语义（std::vector::operator=）零拷贝转移所有权
缺点（可接受）：
  - 小张量有 vector 元数据开销（3个指针，24字节）
  - 无法直接 in-place reshape（需要新建 vector 视图）

替代（性能优化阶段考虑）：
  - Eigen::Tensor<cx,N>（支持 Map，可与 BLAS 直接集成）
  - 自定义内存池（减少碎片）
```

### 15.3 扫描过程中的缓存管理

```cpp
// 扫描时只保留当前格点附近的环境缓存
// 节省内存：O(window × D² × D_mpo) 而非 O(L × D² × D_mpo)
void Environment<B>::release_distant_cache(int current, int window) {
    for (int i = 0; i <= L_; ++i) {
        if (std::abs(i - current) > window) {
            left_envs_[i].reset();
            right_envs_[i].reset();
        }
    }
}
```

---

## 16. 构建系统与依赖管理

### 16.1 根 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.25)
project(TenetCpp VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CompilerFlags.cmake)
include(cmake/Dependencies.cmake)

# 核心库
add_library(tenet STATIC
    src/core/dense_tensor.cpp
    src/core/tensor_ops.cpp
    src/core/factorization.cpp
    src/mps/mps.cpp
    src/mps/mpo.cpp
    src/mps/canonical.cpp
    src/environment/environment.cpp
    src/environment/env_push.cpp
    src/hamiltonian/projective_ham.cpp
    src/hamiltonian/ham_action.cpp
    src/intr_tree/interaction_tree.cpp
    src/observables/cal_observable.cpp
    src/algorithm/dmrg.cpp
    src/algorithm/tdvp.cpp
    src/algorithm/cbe.cpp
    src/krylov/lanczos.cpp
    src/krylov/arnoldi.cpp
)

target_include_directories(tenet PUBLIC include/)
target_link_libraries(tenet PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    HDF5::HDF5
    spdlog::spdlog
    $<$<BOOL:${USE_MKL}>:MKL::MKL>
)

# Release 优化
target_compile_options(tenet PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -DNDEBUG -ffast-math>
    $<$<CONFIG:Debug>:-O0 -g -fsanitize=address,undefined>
)

enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
```

### 16.2 编译器选项

```cmake
# cmake/CompilerFlags.cmake
add_compile_options(-Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)  # LTO
```

### 16.3 依赖查找

```cmake
# cmake/Dependencies.cmake
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(spdlog REQUIRED)
find_package(GTest REQUIRED)

option(USE_MKL "Use Intel MKL as BLAS backend" ON)
if(USE_MKL)
    include(cmake/FindMKL.cmake)
    find_package(MKL REQUIRED)
    # 让 Eigen 使用 MKL 后端
    add_compile_definitions(EIGEN_USE_MKL_ALL)
endif()
```

---

## 17. 测试策略

### 17.1 测试层次

| 层次 | 内容 | 工具 |
|------|------|------|
| 单元测试 | 每个类/函数独立正确性 | Google Test |
| 数值回归 | 与 Julia 版结果对比（无对称性版本） | Python + numpy + h5py |
| 集成测试 | 完整算法运行已知解 | Google Test |
| 性能基准 | 关键路径耗时，与 Julia 对比 | Google Benchmark |

### 17.2 关键测试用例

```cpp
// test_dense_tensor.cpp
TEST(DenseTensor, ContractCorrectness) {
    // C[i,k] = Σ_j A[i,j] * B[j,k]
    // 对比 Eigen GEMM 结果
}

TEST(DenseTensor, SVD_Roundtrip) {
    // A ≈ U * diag(S) * Vt，验证 ||A - U*S*Vt|| < 1e-12
}

TEST(DenseTensor, Permute_Consistency) {
    // permute 后缩并结果与未 permute 一致
}

// test_mps.cpp
TEST(DenseMPS, Canonicalization) {
    // 左正则化后 Q†Q = I（验证各格点）
    // 右正则化后 QQ† = I
}

TEST(DenseMPS, InnerProduct) {
    // 随机 MPS，⟨ψ|ψ⟩ == norm²
}

// test_dmrg.cpp
TEST(DMRG, Heisenberg_L10) {
    // L=10 Heisenberg 链，J=1
    // 精确对角化基态能量（参考值由 Lanczos 独立计算）
    // 要求 |E_DMRG - E_exact| < 1e-6
    auto H = build_heisenberg(10, 1.0);
    auto psi = DenseMPS<>::random(10, 32, 2);  // D=32, d=2
    auto result = dmrg(psi, H, {.max_sweeps = 20, .E_tol = 1e-10});
    EXPECT_NEAR(result.ground_energy, -4.2580, 1e-4);  // 已知精确值
}

TEST(DMRG, Convergence_Criterion) {
    // 验证收敛标志位正确设置
}

TEST(TDVP, Norm_Conservation) {
    // 实时演化保持 ||ψ||² = 1（误差 < 1e-8）
}
```

### 17.3 与 Julia 结果的数值对比

```python
# scripts/regression_test.py
# 1. 运行 Julia 版（无对称性模式或稠密模式）保存到 HDF5
# 2. 运行 C++ 版，保存到 HDF5
# 3. 逐项对比

import h5py, numpy as np

def check_dmrg(julia_f, cpp_f, tol=1e-6):
    with h5py.File(julia_f) as jf, h5py.File(cpp_f) as cf:
        np.testing.assert_allclose(
            jf['ground_energy'][()], cf['ground_energy'][()], atol=tol)
        np.testing.assert_allclose(
            jf['entanglement_entropy'][:], cf['entanglement_entropy'][:], atol=tol)
```

---

## 18. 迁移路线图

### 阶段一：核心张量层（约 4 周）

**目标**：`DenseTensor` 可用，分解正确。

- [ ] `DenseTensor`：形状、数据、permute、reshape、fuse/split
- [ ] `tensor_ops.cpp`：contract（基于 BLAS ZGEMM）、inner、axpby
- [ ] `factorization.cpp`：QR、SVD（含截断）、rand_svd
- [ ] 单元测试：contract 正确性、SVD 回路、permute 一致性

**验收**：L=10 Heisenberg 链单格点张量的 SVD 与 Julia 稠密版误差 < 1e-12

---

### 阶段二：MPS / MPO（约 3 周）

- [ ] `MPSTensor<B>`：三腿张量，矩阵化视图，左/右正交化
- [ ] `DenseMPS<B>`：格点数组，中心追踪，内积，正则化
- [ ] `SparseMPOTensor<B>`、`SparseMPO<B>`：稀疏矩阵结构
- [ ] `canonical.cpp`：left_canonicalize、right_canonicalize、move_center
- [ ] 单元测试：正则化正交性验证，MPS 内积

**验收**：L=10 随机 MPS 正则化后误差 < 1e-14

---

### 阶段三：局域空间 + 相互作用树（约 3 周）

- [ ] `local_space/spin.hpp`：SpinHalf（Sz, Sp, Sm），Spin1
- [ ] `local_space/fermion.hpp`：Spinless，SpinfulHalf（含 JW）
- [ ] `LocalOperator<B>`、`IdentityOperator<B>`
- [ ] `InteractionTree`：树构建，automata 编译，SparseMPO 输出
- [ ] 单元测试：Heisenberg MPO 键维度、Hubbard MPO 矩阵元

**验收**：Heisenberg 和 Hubbard 模型的 SparseMPO 与 Julia 版吻合

---

### 阶段四：环境张量（约 2 周）

- [ ] `LeftEnvTensor<B>`、`SparseLeftEnvTensor<B>` 及其右侧镜像
- [ ] `Environment<B>`：初始化、push_right、push_left、缓存管理
- [ ] 单元测试：L=10 Heisenberg 链，proj1 矩阵元验证

**验收**：单格点有效哈密顿量矩阵元与 Julia 版误差 < 1e-12

---

### 阶段五：Krylov 求解器（约 2 周）

- [ ] `lanczos_eigsolve`：修正 Gram-Schmidt，收敛判断，重启
- [ ] `lanczos_linsolve`：线性方程（TDVP 中心格点）
- [ ] `krylov_expm_times_vec`：Arnoldi 指数传播（TDVP 前向步）
- [ ] 单元测试：随机厄米矩阵特征值，与 Eigen 精确解对比

---

### 阶段六：DMRG 算法（约 3 周）

- [ ] 单格点 DMRG（L2R + R2L 扫描）
- [ ] 双格点 DMRG
- [ ] CBE 集成
- [ ] 收敛判断、SweepInfo 追踪、日志输出

**验收**：
- L=20 Heisenberg 链，|E_DMRG - E_exact| < 1e-6
- 性能不低于 Julia 稠密版

---

### 阶段七：TDVP / SETTN / 可观测量（约 4 周）

- [ ] 单/双格点 TDVP（实时 + 虚时）
- [ ] tanTRG 热化
- [ ] SETTN
- [ ] ObservableForest：单体、两体关联函数
- [ ] 并行可观测量计算

---

### 阶段八：完善与优化（持续）

- [ ] OpenMP 多线程环境更新（目前先单线程正确性优先）
- [ ] MKL 后端验证与调优
- [ ] 内存池（减少频繁小块分配）
- [ ] HDF5 I/O（MPS 保存/载入）
- [ ] pybind11 Python 绑定（可选）
- [ ] 文档与示例

---

## 19. 未来对称性扩展路径

当第一阶段算法稳定后，接入对称张量只需以下步骤，**算法层代码完全不改**：

### 19.1 新增 SymmetricBackend

```cpp
// 未来：include/tenet/core/symmetric_backend.hpp

// Step 1：实现 VectorSpace（带量子数的 Space）
template<typename Charge>   // 例如 Charge = U1Charge
class VectorSpace {
public:
    struct Sector { Charge charge; int multiplicity; };
    explicit VectorSpace(std::vector<Sector> sectors, bool dual = false);
    int dim() const;          // Σ multiplicity
    VectorSpace dual() const;
    static VectorSpace trivial(int dim);  // 无量子数，与 TrivialSpace 兼容
    // 满足 SpaceLike concept
};

// Step 2：实现 SymmetricTensor（块稀疏）
template<typename Charge>
class SymmetricTensor {
    // 每个 Sector 组合对应一个 Eigen::MatrixXcd 块
    std::map<BlockKey<Charge>, Eigen::MatrixXcd> blocks_;
    std::vector<VectorSpace<Charge>> legs_;
public:
    // 满足 TensorLike concept
    int rank() const;
    int dim(int i) const;
    VectorSpace<Charge> space(int i) const;
    SymmetricTensor permute(const std::vector<int>& perm) const;
    SymmetricTensor adjoint() const;
    Eigen::MatrixXcd as_matrix(std::vector<int> rows, std::vector<int> cols) const;
};

// Step 3：定义 SymmetricBackend
template<typename Charge>
struct SymmetricBackend {
    using Scalar = std::complex<double>;
    using Space  = VectorSpace<Charge>;
    using Tensor = SymmetricTensor<Charge>;

    static Tensor zeros(const std::vector<Space>& legs);
    static Tensor random(const std::vector<Space>& legs);
};

// 常用别名
using U1Backend  = SymmetricBackend<U1Charge>;
using SU2Backend = SymmetricBackend<SU2Charge>;
```

### 19.2 用法示例（未来）

```cpp
// 现在（稠密）
auto H1  = build_heisenberg<DenseBackend>(L, 1.0);
auto psi1 = DenseMPS<DenseBackend>::random(L, D, 2);
auto res1 = dmrg(psi1, H1);   // 编译 DenseBackend 路径

// 未来（U(1) 对称）：同一份算法代码，不同 Backend
auto H2  = build_heisenberg<U1Backend>(L, 1.0);
auto psi2 = DenseMPS<U1Backend>::random(L, D, SpinHalf_U1::phys_space());
auto res2 = dmrg(psi2, H2);   // 编译 U1Backend 路径，块稀疏加速
```

### 19.2 Krylov 求解器与块稀疏的兼容性

`lanczos_eigsolve` 和 `krylov_expm_times_vec` 当前模板化于 `LinearOp` concept，只要求：
```cpp
void op.apply(const Eigen::VectorXcd& x, Eigen::VectorXcd& y);
int  op.dim();
```

接入 `SymmetricBackend` 时，`SparseProjectiveHamiltonian<SymB>::apply` 的实现应**直接在块结构上迭代**，而非先将 `SymmetricTensor` 展平为稠密向量再计算。具体而言：向量 `x` 应理解为按量子数扇区分块的稀疏向量，`apply` 只对守恒量子数匹配的块进行 GEMM，计算量与块数目成正比而非与总维度平方成正比。Krylov 求解器本身的接口（`apply` + `dim`）无需修改，差异完全封装在 `apply` 的实现中。

### 19.3 验证策略

对于每个新支持的对称性，验证：
1. 无量子数（`DenseBackend`）与有量子数版本的能量误差 < 1e-8
2. 有量子数版本的内存占用 < 稠密版本 × (1/d_sector)²（理论上界）
3. 有量子数版本的速度提升 ≥ 对称性带来的理论加速比

---

*本文档（v2.0）描述第一阶段（无对称性）的完整 C++ 重构方案，Backend 接口预留对称张量的扩展点。*
