# 数值检查

运行示例、修改 DSL 或改进编译 pass 时，使用用例登记的数值合同。

| 数据或操作 | 检查内容 |
|---|---|
| Encoding 与 prepared packed storage | 精确字段／字节映射、padding 和布局身份 |
| 整数算术 | 声明的 signedness、位宽、回绕和累加行为 |
| 浮点输出 | 用例的绝对／相对容差与有限值规则 |
| 可变 state 和 alias | 完整更新与共享存储关系 |
| 模型接入 | 登记的 logits 检查和测量阶段的替换计数 |

## 使用 Bench 现有入口

```sh
weft-bench op q4_k_dot --target k1 --check-only \
  --output output/current/k1/q4_k_dot-check
weft-bench verify --target k1 --list
```

检查语言或 runtime 功能时，从列表选择 verification ID：

```sh
weft-bench verify --target k1 --case 'operators/q4_k_dot' \
  --output output/current/k1/verification
```

配置套件规定具体检查范围。Leaf 与 decode 检查登记的完整输出；
大型 GEMM／prefill 可以结合 scalar 输出抽样与全输出有限值检查，结果保留这一范围。

## 聚焦修复

保留失败输入、source/configuration 与生成程序，修复对应层，
然后重跑该用例和直接受影响的共享路径。
[复现指南](../experiments/reproduce.md)说明固定参考、结果查看和模型阶段。

[详细数值与测量合同](https://github.com/Tianchen-ai/triton-weft/blob/main/docs/specification/development/validation.md)。
