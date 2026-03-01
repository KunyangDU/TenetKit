# TENET C++ 重构执行指南 (Claude Code 专用)

## 1. 上下文对齐 (Context Mapping)
在执行任何操作前，必须优先查阅以下文档：
- **核心架构：** `docs/C++重构设计方案.md`
- **逻辑背景：** `docs/架构总结.md`
- **质量标准：** `docs/建议.md`

## 2. 核心技术约束 (Hard Constraints)
- **标准：** 严格遵循 C++20。
- **内存：** - 严禁在 `src` 或 `include` 中出现裸 `new`/`delete`。
    - 必须实现 `TensorMetadata` 与 `Storage` 的解耦。
- **并发：** 为后续多线程/CUDA 留出 `stream` 或 `context` 接口。
- **模板：** 优先使用 `std::concepts` 约束模板参数，避免晦涩的 `SFINAE`。

## 3. Claude Code 工作流 (Step-by-Step)
1. **测试先行：** 每当重构一个子模块（如 `TensorCore`），同步在 `tests/` 下创建对应的 GTest 单元测试。
2. **构建验证：** 每次修改代码后，主动尝试运行 `cmake --build build` 以确保静态检查通过。
3. **性能审计：** 涉及大规模收缩（Contraction）的代码，需保持循环的连续性以保证 Cache Locality。

## 4. 迁移指令参考
- 如果需要参考旧逻辑：读取 `legacy/` 目录下的相关源文件，但**禁止**直接复制，必须按新设计重写。
- 如果设计方案有冲突：优先遵循 `docs/建议.md` 中的现代化改进建议。