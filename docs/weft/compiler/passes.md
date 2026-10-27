# RISC-V 编译 Pass

所有 pass 都改写同一份 [RISC-V IR](riscv-ir.md)。`implementation`、`schedule`、
`lane_operand` 等中间属性附着在具体 operation 上，并在其消费者 pass 中删除；它们不是
module 外的 assignment 表，也不构成新的 IR 层。

整数范围、转换与乘积和溢出的证明由 `Weft/Analysis/IntegerFacts` 与 x86 共用。
RISC-V 适配层只描述物理操作的数值含义；layout conversion 不改变数值范围，
窄化或潜在回绕不能继承未经证明的窄区间。查询只在单次有界遍历内保存 memo，
不会跨 pass 保留随 SSA 改写失效的事实。边界使用任意精度整数，覆盖完整 signed/unsigned
机器类型，包括 i64/u64；乘积、累加和 magnitude 查询先扩展分析位宽，避免分析自身回绕。
普通 MLIR 操作复用 `InferIntRangeInterface` 的 signed/unsigned 与模运算合同，目标适配层
只解释其局部操作。未知 producer 保留完整类型范围，不提供不存在的窄区间证明。
物理 splat 保留标量范围；masked negate 交由共同分析求原值与取负值的并集。
分析在扩展位宽内计算取负，若结果可能跨过机器类型边界则退回完整类型域，
不能用对 signed-min 无效的绝对值证明允许更多窄累加。

partial source-lane 候选按正式 repack leaf 的临时资源评估输入／输出交接：
操作前计入仍活跃的 source 与 leaf scratch，操作后计入 result。完整寄存器 split
是原 payload 的视图，不将已消费的 source 与 result 当成两套同时分配的寄存器；
需要 slice/copy 的 realization 仍计入其真实 scratch，之后重新验证完整程序的峰值。

`Analysis/PhysicalDependence` 与 x86 共用 SCF 的精确静态迭代数，包括空域、非单位步长和
有符号/无符号比较。RISC-V 的循环不变量读取据此证明循环确实执行；未知边界不获得默认次数。
驻留成本查询可以保留动态循环次数为符号因子，不给条件分支编造执行概率；这些查询在改写后重建。
共享的 block-relative frequency 查询使用目标控制展开后的实际 SCF 边界；RISC-V Level 的
同轴子域不会用名义 extent 代替父窗口的 active extent。未知计数按循环身份保留，不能把不同
动态循环仅按嵌套深度合并。

`Analysis/StorageBytes` 与 x86 共用 Encoding 字段到 bit span / byte fragment 的解释。
RISC-V 的 `fieldFacts`、record-local bit projection 与 x86 的字段传输选择消费同一份
natural、grouped/layered、joined 几何事实；byte/bit order 和 persistent interleave
仍来自作者声明。查询不包含 ISA、量化公式或新布局选择，也不在 IR 改写之间缓存结果。

`PlanRISCVMemory` 的字段子程序将不能由已选连续传输覆盖的完整 record 访问，
展开为显式的记录循环、byte load、数值位组合与 masked byte store。跨 byte 的字段
消费同一份 fragment 映射；写入只覆盖该字段的 bits，保留共享 byte 的其它字段。
编码 Value 的数值副本在原 load 边界完成，后续 field 使用不能穿过别名写入重新解引用。
同一 stage 的多个字段表示保留同一个作者 birth；跨 block 的 stage 在自己的作用域
初始化其存储。局部 byte leaf 只描述一个已投影 record，不拥有外层遍历或 Encoding 算法。
这一通用路径目前要求 owner 的 encoded element axis 覆盖一个完整 record；任意前缀
记录轴由实际 descriptor extent 遍历。这不覆盖任意整记录 reshape 传输或字段窗口；
这些访问仍须具有自己的完整物理程序，不能留给 emitter 临时解释。

已加载记录的 whole-byte natural 字段在 local storage projection 状态下保留 storage edge；
record 对齐弱于元素对齐不要求提前解码整个字段。indexed-entry pass 根据实际消费者窗口和共享 alignment
证明选择 byte-load/reinterpret，完整记录的别名快照仍由统一 read-snapshot pass
在原 load 时刻闭合。实际数值 field 与直接读取字段 View 的完整 load 仍保存其完整读取值。

已选择矩阵 contraction 的字段供给先保留 Encoding 几何，交给同一 fragment materializer
验证并建立精确字段传输。若该消费者需要普通数值 packing，后续 memory pass 再完成字段的
字节快照；不能在消费者决定前把已可用的 packed 字段展开成完整 RVV 数组。字段写入枚举
LMUL 时同时验证 SEW、LMUL 和完整 layout，不能只按 VLEN 能容纳的 bit 数判断合法性。
自然整字字段的 scalar/replica 传输沿用已有 alignment-safe byte copy，不因 alignment
小于字段位宽而提前建立逐字节临时数组。原 encoded read 的快照仍由共同存储分析闭合。
未对齐的单个 base-record F16 字段在目标具有 Zfhmin 时选择整数 byte load 与 `fmv.h.x`，
直接交付相同的半精度 bits，避免原生工具链将未对齐 half load 转成栈上的 store/load。
该局部 transfer 保留读取时点与 NaN payload；能力和 Encoding 条件由选择及 verifier 共同检查，
emitter 只拼写已选的位传输，不改变累加类型、对齐声明或持久布局。
普通 scalar F16/16-bit integer `bitcast` 复用同一 Zfhmin 能力判断，分别选择 `fmv.h.x`
或 `fmv.x.h`；后者按结果的 16-bit 类型保留低位。操作选择和晚期 leaf 收尾调用共同的
instruction 查询，不能在布局已经闭合后把该位传输重新解释成数值转换或栈中转。
同一 natural scalar field 沿已声明 interleave cohort 的 RVV 读取也复用已有 unit-byte 或
indexed-byte transfer；它保持字节对齐，不把未对齐地址强制当作自然对齐的元素指针。

`MaterializeRISCVEncodedValues` 在布局选择前处理有界 base-record 的
load → reshape → field/store 子图。原 load 处生成私有记录字节和逐记录有效性，
reshape 的输入轴次序生成显式逆坐标运算；整记录 store 只传输有效记录，字段数值
沿用共享 byte-fragment 关系，并进入共同的数值有效性传播。读取时刻不因后续
重排或可能别名的写入而改变，padding 随整记录逐字节保存。该物理程序目前不接管
derived layout、跨控制结果或 stage 的 encoded reshape 子图。

descriptor 的请求 extent 与 backing storage 的有效范围分别验证：`MemoryBounds`
生成穿过 subview、slice 和轴改名的显式 SSA 条件。嵌套 domain point 替换同轴的
绝对基坐标，普通 index/subview offset 仍按相对坐标组合，避免重复加上父层 base。
共享 origin 查询先检查最近的 slice/extract 坐标，再回看 backing；内层子窗口不能
继承成外层 record 的起点。IME 将一个 contraction 分成多个 K atom 时沿该起点
生成子窗口，并合成已有的 domain projection，保留其它轴的选择和原读取快照。
已经拥有实际 allocation 的 `LocalBind` 保持 local carrier；布局传播不能隐式将
存储身份变成寄存器数值，消费者必须经过真实的 load 或 conversion。
分组的局部 Value 向 RVV 投影时，group 坐标、保留轴的 stride、slice 和 load 在
composite lowering 中显式形成；它与标量副本投影共用同一分组坐标计算，后续 memory
pass 选择真实传输，不把尚未选择的局部数组访问留给 emitter。

## 1. 固定顺序

```text
ConvertWeftToRISCV
→ SpecializeRISCVPointerOffsets
→ MaterializeRISCVEncodedValues
→ MaterializeRISCVReductions
→ MaterializeRISCVGroupedArithmetic
→ MaterializeRISCVAtomics
→ SelectRISCVOperations
→ PropagateRISCVLayouts
→ MaterializeRISCVLocalWindows
→ SelectRISCVOperations
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ SelectRISCVOperations
→ SelectRISCVSharedSupply
→ LowerRISCVComposites
→ RetainRISCVFragmentAccumulators
→ PropagateRISCVConsumerLayouts
→ MaterializeRISCVValuePrograms
→ MaterializeRISCVPrograms
→ VectorizeRISCVRecordLoops
→ CanonicalizeRISCVLayouts
→ PlanRISCVNestedMemory
→ FuseRISCVBitplanes
→ HoistRISCVLoopInvariants
→ SeparateRISCVFullTiles
→ ScheduleRISCVLevels
→ PipelineRISCVLevels
→ SCCP
→ RetainRISCVFragmentAccumulators
→ ShareRISCVLayeredWindows
→ PlanRISCVPartialTopologies
→ MaterializeRISCVPartialAccumulators
→ UnrollRISCVLevels
→ ShareRISCVLayeredWindows
→ CoalesceRISCVPartialExtractions
→ MaterializeRISCVLocalProjections
→ EliminateDeadRISCVLayouts
→ PropagateRISCVConsumerLayouts
→ CanonicalizeRISCVLayouts
→ PlanRISCVMemory
→ SelectRISCVMemoryByteProjections
→ MaterializeRISCVReplicaStorageLoads
→ HoistRISCVLoopInvariants
→ SpecializeRISCVMemory
→ MaterializeRISCVNumericConversions
→ SelectRISCVOperations
→ SelectRISCVIntegerMACs
→ FinalizeRISCVLeaves
→ SelectRISCVScalarLoadPrimes
→ MaterializeRISCVReadSnapshots
→ MaterializeRISCVReductionBounds
→ CloseRISCVLeafResources
→ RISCV physical CSE
→ FuseRISCVPhysicalIssueLoops
→ EliminateDeadRISCVLayouts
→ MaterializeRISCVResources
→ EliminateDeadRISCVLayouts
→ FinalizeRISCVHelpers
→ VerifyFinalRISCV
```

该顺序是依赖关系，不是阶段标签。`LowerRISCVComposites` 必须先产生带 typed
operation/effect 的显式 physical loop，`schedule` 才能从真实 use-def 分配 stage/order，
`pipeline` 再只负责版本化 SSA 值并生成 prologue/steady-state/epilogue。partial planner 必须在
generic unroll 之前读取未复制的 contraction/reduction use-def，冻结 carrier、combine topology
与 issue unroll；materializer 生成 issue loop后，`UnrollRISCVLevels`才机械复制迭代。展开后才完整
出现的 grouped/layered issue relation 再交给同一个幂等 sharing owner，物化为共享
raw-window load 与显式 layer decode。exact leaf
完成后，resource pass 才能把 leaf temporary 与 SSA live interval 一起计入峰值。
`UnrollRISCVLevels` 形成的剩余循环和尾循环禁止系统再次展开，保留所选访存分组及
operation-major 交错的 issue 窗口。流水与 partial planner 已声明的限制继续保留。
Level-scaled partial 的专门展开在每轮内部收集并归约 partial，外围只携带原标量累加值；
它不自动增加禁止 native 展开的限制。机器循环优化、调度与寄存器分配仍由 native
工具链完成，不能把一次局部 partial 收集当成需要固定的外围机器循环。

整块 stage 的多轴投影在 ordered loop 闭合后、contraction 物化前确定供给，和后续
Value materializer 复用同一实现。可证明没有干扰写入时，staged descriptor 保留原 birth，
失去消费者的 numeric marker/read 随之退出，避免再次生成无人读取的完整拷贝。
必须保存的 oversized Value 使用已有 bounded copy program，在同一读取 epoch 和作者
作用域内形成私有快照；静态 local storage 仍受目标预算约束。完整 pointwise Value 的
资源判断复用候选 issue 的成本查询，计入完整输入／输出活跃集，避免后续 FMA broadcast
把先前仅靠 last-use reuse 合法的表示推到寄存器上限之外。
同形状、同物理几何的普通逐 lane 二元指令可复用最后一次使用的输入寄存器；成本查询
同时计算操作前峰值与复用后的结果峰值，不将其计成额外完整 Value。广播、转换、select
与数学展开不使用该证明，实际 leaf 临时量和最终 spill 仍由正式资源管线收口。

最终 layout 改写前先清理失去消费者的纯 index/conversion 链，避免死路径阻止基于真实
use 数量的 rematerialization；资源闭合后的清理不能替代这个位置的清理。

`MaterializeRISCVReductionBounds` 在 carrier 映射固定后，沿 register reshape 的
`part_order` 将归约各 part 对应回保存的数值域，把真实活动长度写成 index SSA operands。
长度包含 lane offset 和其它轴的活动条件；空 part 使用零长度并保留归约 seed。
逐元素算术、转换和广播沿逻辑坐标投影这些长度；轴改名后的算术结果不能单凭布局的
`full` 标签扩大原保存数值的有效域。显式长度阻止不合法的整宽 stream 合并，并更新
对应的 temporary register 需求。
这些 operands 随 spill/reload 保留，emitter 只读取它们，不按改名后的轴猜测尾块长度。
当前前缀长度表示要求一个 lane axis；有孔的 Cartesian 域必须有显式 mask，不能伪装成前缀。

派生节点继承已有 `source_origin`；只有没有来源标签时才从 location 建立初始来源。
解析 Physical IR 的文件位置不能覆盖作者来源，否则同一程序重放会改变 CSE 的匹配集合。

完整 fragment 与其 consumer layout 在 Value program 的局部存储决策之前先闭合一次，
避免将本可驻留的 accumulator 与 pointwise 链提前拆成数组。后续 materialization 新产生的
fragment 仍经过相同 passes；整数 MAC 保留在访存关系确定之后选择。

内部 RISC-V pipeline 可在这次清理、consumer propagation 与 `CanonicalizeRISCVLayouts`
之前观察同一层 Physical IR。module 的 `weft.riscv.layout_input` 保存尚未消费的
`scalar_load_prime` 单值绑定；其余 target/binding 已体现在 typed program 中。
内部 replay 从该边界运行与普通编译完全共用的
layout、memory、leaf、read-snapshot、resource 和 final verification 后缀，再机械生成 C。
缺少标记、已经资源闭合或试图更换 target/binding 均拒绝；不清除 final resource
marker 来强行重放。该边界不增加 IR 层，也不是备用编译主链。

## 2. 各 pass 合同

### `SpecializeRISCVPointerOffsets`

在 layout 选择前，从一维 pointer offsets 的 typed use-def 证明完整 logical domain 是
`base + coordinate * element_stride`，将地址供应物化为 scalar base 和显式 element stride。
分析有 64 个节点预算，接受 integer iota、标量广播、add/sub、常量乘法及保值转换。
穿过符号转换或改变位宽时，必须由已知整数区间或基点整除关系证明整个 domain 不跨越
wrap/sign 边界；不能把窄整数回绕后的索引误当作宽整数中的连续区间。
没有证明的表达式保留 indexed offsets。mask、fill、访问顺序、来源和 alias 均不改变。
失去消费者的纯地址链在 layout 前删除，避免将可由标量提供的地址广播物化成 local 数组。
该 pass 不选择 RVV 指令；后续 memory pass 根据已证明的关系与 carrier 选择 unit/strided
或 scalar leaf，value-program 分片显式加入当前 logical coordinate 对基点的偏移。

### `MaterializeRISCVReductions`

将联合 `reduce_region` 的纯标量 combiner 物化为 primitive 内的相邻配对平衡树。
每层显式产生偶数/奇数索引和 typed gather，再将同一个 combiner 提升为 pointwise
操作；所有 free axes 保留，最后去掉归约轴。内部 pair axis 是闭合 primitive 的
临时身份，不成为 Canonical 作者 Value，也不改变 Level 或内存 effects。
随后共用 layout、访存、leaf 与资源 passes，emitter 不选择归约算法。
当前支持正二次幂 extent；不支持的 extent 和与全部 tuple 输入无关的结果明确报错。
该 pass 不改写已有单值 add/min/max `reduce`。

同一 pass 将 `scan` 物化为距离逐次倍增的 inclusive 前缀树。每一步的 safe gather
保留非活动位置原值，避免构造越界索引；tuple 各成员共同使用同一顺序和有效域。
reverse 从较大坐标向较小坐标组合，仍以此前前缀为左参数。

### `MaterializeRISCVGroupedArithmetic`

独立 `mac_groups` 及非融合消费者通过共享 `Analysis/LogicalAxes` 验证分组轴、
free-axis 对齐和完整分组范围，再展开为闭合 primitive 内的 typed gather、转换、
乘加与 SSA carry。输入先转换到声明的结果位宽，整数乘加保留该位宽的模运算边界。
物理分组坐标使用私有轴；输出先完成坐标置换，再恢复原 grouped axis identity，
不修改作者的 Level、state 或 workspace。已有 single-use 混合 byte
`mac_groups → widen → reduce(add)` 仍交给 widening-dot 专门物化路径。
普通数值 `sum_pairs` 复用相同的分组坐标、类型转换与有效域传播，只累加每组的两个输入；
已闭合、自然对齐的 signed i16 编码字段仍可使用现有的直接 pair-fold leaf。数值快照
不被伪装成 encoded field，也不要求作者为不同物理载体改写 pair-sum 算法。

分组有效性是各原始有效项的逻辑 OR，不能把物理补零后的完整寄存器当作全部有效。
整数 contraction 的无效输入在同一物理数值域内显式补零；原 contraction、累加类型、
缩并轴和 engine 约束保持不变，不展开成可能在窄输入位宽回绕的乘法。
结果有效域来自双方谓词在共同逻辑轴上的交集，再对缩并轴取 OR，保留所有自由轴；
全无效的输出不因补零而成为有效数值。浮点的零乘无穷大不具有这个中性项合同，
没有独立实现的显式谓词浮点 contraction 继续明确拒绝。
独立 grouped program 显式携带该谓词，并在每个乘积进入累加前补 identity；
`NumericalValidity` 沿提取、转换、pointwise、普通 `for/while/if`、Level 和 helper 返回传递对应的
物理 SSA 谓词，最终作用于 memory effect。谓词随实际执行的控制路径传递，不重新读取
原输入来重建快照。尚无闭合传播规则的消费者明确拒绝，不能默认为完整有效。
整数分组输入的有效范围沿 lookup、字段 gather 和 reshape 保留。索引已证明落在容量内、
而原字段只可能完整或为空时，gather 仍携带原字段的非空条件。reshape 保留未改变的
前缀维度及其尾部；只对完整或空的后缀重排矩形范围，不把任意部分矩形当作完整数据。
谓词的逻辑域覆盖整个容量，非活动位置明确为 false。谓词合取与选择使用私有坐标
别名完成，再投影回原数值轴，避免外层数值 tail 的 VL 截短谓词本身；同域的单一
谓词直接复用，不为窄化后再扩宽额外生成向量合取。
标量 Extract 的所有用途完成上述 SSA 谓词传播后，记录 `weft.riscv.validity_lowered`；
后续 numerical-value pass 不再从旧矩形 shape 重建另一份谓词。重建不同数值投影时不复制
这一完成标记。私有 scalar store 同样消费有效谓词，全真谓词不额外生成分支；由显式
identity seed 初始化的 stream reduction 保留其 surviving free axes 的已定义坐标。

分组结果的自由轴也在完整私有谓词域内计算，不能依赖后来恰好选到哪条 SIMD 轴。
归约先按相同数值合同给无效项补 identity，再对原谓词做 OR；完全无效的输出仍无效。
shaped store 物化为有界坐标遍历和受谓词控制的真实写入，不先读取或写入无效目标。
`while` 的入口和退出 tuple 分别追踪；Level 的物理有效性 companion 用
`weft.riscv.validity_slots` 关联原 handoff，不增加作者 state/stage birth 或 handoff 计数。

### `MaterializeRISCVNumericConversions`

闭合物理数值转换的机器宽度步骤。composite lowering 与晚期 leaf selection 共用同一
实现：后续 Value program 新产生的 RVV 转换也必须显式形成合法 narrowing 链。
F16→F64 的 RVV 扩宽显式形成 F16→F32→F64 两步，保留相同坐标和有效域；两步均为精确
扩宽，中间寄存器布局及其活跃区间进入正常指令选择和资源分析。terminal 不合成转换链。
signless Boolean predicate 转为数值时物化同坐标的 0/1 select。Encoding `ui1` 的普通
数值表示已是规范的 unsigned byte 0/1，沿用 zero extension，避免 byte→mask→数值的往返。
已规范符号扩展到 byte carrier 的有限有符号整数扩大到 i8 时保留原 carrier；转成 u8
只改变机器类型解释，不发明不存在的 4-bit RVV 扩宽指令。逻辑宽度与机器 SEW 分开判断。
真正的 packed predicate register 由专门 op 与 ingress/egress 明确表达，不能按 byte
向量 reinterpret。新值进入正常 leaf、生命周期与资源分析，terminal 不补造转换程序。
后续 bitplane、signed-mask reduction 和 masked-negate 选择共用该 0/1 关系，接受转换
物化后的 `select(predicate, 1, 0)`，不要求前一 pass 保留已经退出的 cast 形式。
额外数值消费者、非布尔输入及改变坐标的转换继续受各局部实现的 use-def 和类型条件约束。

`Analysis/ReductionIdentity` 由 x86 与 RISC-V 共用。整数极值使用声明 signedness 的边界；
浮点 max/min 使用 NaN identity，保证全部有效元素为 NaN 时仍返回 NaN。标量尾段的
f32/f64 极值选择明确的 `fmax.s/d`、`fmin.s/d` leaf，保留 NaN 与 signed-zero 合同。
terminal 以 Clang 23 的 `fmaximum_num` / `fminimum_num` builtin 拼写该已选操作，
让原生编译器继续看见相同的 IEEE 数值语义并调度依赖；不用不透明 inline asm 隔断标量递推。

### `MaterializeRISCVAtomics`

将 shaped pointer atomic 物化为显式的有序 lane 循环与私有旧值结果存储。
各 lane 的 mask、地址、更新值和返回值坐标保持对应；同址冲突由原生原子操作解决。
标量操作选择明确的 atomic leaf；浮点加法与整数极值用保持指定 ordering 的 CAS 实现。
后续局部结果与 pointer store 共用 bounded value program，不让 emitter 创建外层遍历。

### `FinalizeRISCVHelpers`

在布局、转换及资源闭合后，收集 helper 真正使用的外部 SSA 和 descriptor 依赖，
闭合 native helper ABI。返回值转换先由 layout pass 写入 IR；terminal 只按
已选类型输出函数参数、返回槽和调用。noinline 保留为原生函数属性；内层返回只退出该函数。
所有结果使用统一的 `selected-parts` ABI；源级 noinline 和控制流返回语义保持不变。
参数保留 capture 的已有物理部分、完整 descriptor、坐标及 shape 参数；原生工具链继续
优化私有函数的传参，不因结果恰好是标量而选择另一套调用约定。
encoded Record capture 明确传递原读取的指针、byte strides、origins 与 interleave 坐标，
不能在 helper 中重新读取源 View 代替快照。返回的存储 alias 由共同来源分析追踪。
可写 capture 只有在独立 captures 和全部真实访存都被证明不重叠时才标记 `restrict`。
同一 allocation 上的不同指针参数不按同源关系合并；未知地址或 effects 阻止该限定。
final verifier 重算同一证明，terminal 只拼写已冻结的限定。

同一收尾 pass 为原生 kernel 入口写入 `native_entry_alignment=32`。这是有界的代码放置
策略，最多增加 30 bytes 对齐填充，不是 VLEN 或 cache-line 大小声明。它避免入口仅按
压缩指令的两字节要求放置、使相同指令随链接器布局产生不稳定的取指表现。
final verifier 要求显式的合法二次幂对齐；C emitter 只输出已选 `aligned` 属性。
包含 RVV/IME 操作且没有保留子循环的有序循环，另行保存 `weft.riscv.code_alignment=32`，
通过 Clang 的 `code_align` 属性固定回边目标的最小对齐。未知次数、超过四次或显式禁止展开的
子循环视为保留；其外层遍历不重复添加 padding。静态小子循环可能在原生代码生成时完全展开，
外层仍保留自己的对齐决定。四次是有限放置策略的边界，不是 ISA 事实或展开正确性的前提。
该策略不假定更大边界总是更快，也不根据 kernel 名称选择代码偏移。
原生工具链显式配置 `-falign-loops` 或 `-fno-align-loops` 时，backend 将循环放置交还
该配置，Physical kernel 保存 `native_loop_alignment=0`，不再用 statement attribute
覆盖用户的选择。其它情况保留 32-byte 策略；checkpoint 重放继承同一放置决定。

布局选择同时核对一个 control edge 上全部 carry/result slots 的常驻寄存器占用，
并保留更新 issue 的空间；不能用各 slot 单独放得下来证明它们可以同时存活。
超出容量的 slot 集合使用现有 local 数值表示、显式 control transfer 与分段
value program。没有扩大寄存器预算，也没有更改作者的 SSA carry 或归约公式。

### `ConvertWeftToRISCV`

输入是一个已经实例化的 Canonical Kernel IR candidate、一份 target profile 和一组单值
physical parameter binding。它建立 target-aware `ValueType`、`MemDescType`、Level/control、
完整 runtime `memory_view`、artifact builder 与 numerical operations，然后删除 canonical
operations。初始 layout、access 和 leaf 可以是 `unassigned`，但 source axis、Value、Level、
birth、handoff、effect、Encoding 与 ABI identity 必须一一保留。

一个 physical module 只含一组 meta/unroll/pipeline binding。候选枚举和实测发生在 driver
之外；pass pipeline 不接收列表，也不在 module 内保存备用值。

转换时可用入口 View 的完整 record 与 interleave cohort 合同证明 domain 无尾部。
证明只接受已绑定的正 partition，以及可整除的静态 extent 或同一 shape symbol 上的
Encoding 整除约束；随后将 Physical domain 和 point 标为 `exact`。不能仅凭轴名、常见
模型维度或样本 shape 去掉边界，slice 的局部 extent 也不继承整个入口 View 的整除性质。
Canonical domain、遍历顺序与 runtime 维度不变；无法证明时保留原 tail 合同。

### `SelectRISCVOperations`

读取 operation semantics、typed operands、axes 和 target capability，按 target 固定优先级
写入一个 `ImplementationAttr`：scalar、RVV 或 IME 的局部结构 family。它不选择 memory
form。没有合法局部结构时当前 module 失败。

对已经物化、尚未资源冻结的局部 target op，也可选择合同等价的基本操作：逐轴同形、
有效域一致的 byte 乘积，其 i16 编译期整数零初值可由一个 widening multiply 替代。
独立乘积及累加链的首项使用同一规则，后续 accumulate 和最终 reduction 不变。
signedness 来自 typed operands，继承完整 carrier、坐标及尾部，不重选 partial plan；
显式 active-length 的 undisturbed partial 和更宽 partial 不适用。
新 leaf 与资源仍由原 verifier/materializer 检查，不能从代数简化直接推断性能收益。

完整 f32 stream accumulator 的最终 add 归约可选择 `rvv.stream-finalize.m1`：
按已选 m2/m4/m8 carrier 的完整寄存器分段构造平衡加法，再对一个 m1 归约。
单对合并只增加一次 vector add；更深的树仅在每寄存器至少容纳 16 个 f32 lane 时选择，
避免短向量上的多条合并指令抵消归约收益。这是有界 issue-work 策略，不是 CPU 周期模型。
仅在所有分段都属于同一归约轴且完全初始化时合法；没有跨 source reduction 或改变
累加精度。selector 写入确定 leaf，verifier 核对 VLEN、分段和临时驻留；terminal
按所选 leaf 拼写寄存器提取和加法，资源 pass 继续核验完整程序。

单轴 full-valid i16 loop carry 若仅沿同 block 的 byte widening product + add 链流向其
对应 yield，可将整条链选择为已有 widening accumulate。每个 product 必须单 use、
逐坐标同形；保留原 i16 加法边界和迭代顺序，不合并不同 carry，也不只改写链的前缀。

独立 pointwise 乘法的两个 widening 来源位宽不同时，若一侧已经是结果宽度的一半，
另一侧更窄且 signedness 相同，可先显式扩展较窄侧到共同精度，再选择原有 widening
multiply。原结果精度、全部 axes 和 carrier factors 必须闭合，不移动或放宽任何窄化。
逐元素整数乘加的最终融合由访存决策之后的 `SelectRISCVIntegerMACs` 执行；这里保留
地址算术的乘法与加法关系，供连续窗口、stride 和窄整数 wrap 分析使用。

两个单 use 的整数 `widen` 若直接供给同一个 add，且窄 operands 的 signedness/type 与完整
物理坐标相同、宽 result 保持 signedness 并具有双 SEW/LMUL，可以选择局部
`rvv_widen_add`。op 只保留原 active VL 上的逐元素加法，分别拼写 `vwadd` 或 `vwaddu`；
不接受 mixed signedness，不穿过其它计算或改变 reduction。选择器与 verifier 共用类型及
target 合法性判据，memory reads 和外层 traversal 不变；其 window projection、资源重闭合
和 terminal part 映射仍必须显式成立。

已计算的一维 RVV value 被静态连续 iota 切片时，selector 证明索引加法无 wrap、窗口在
原值域内且完整落入已选 carrier，然后物化 `rvv_issue_slice`。索引证明与布局传播共用；
子域的较小 extent 不反向缩窄父值。该规则不重读 encoded field，也不把任意 gather 当作切片。

### `PropagateRISCVLayouts`

跨 pointwise broadcast 传播的是实际保留坐标的约束。source 的 scalar tuple、memory anchor
或完整 lane 范围不能约束 consumer 新增的轴；同名 singleton 被扩展时也不是相同坐标表示。
完整同域 control/cast 仍传递完整约束，目标端必要的表示转换保持显式。

静态容量的 RVV 值在行主序 reshape 后可保留由 time、replica 与 lane 共同形成的 register tuple。
pass 将已有 physical radices 分配到结果各轴，并用 `RegisterCoordinates` 逐项核对
每个 `(part, lane)` 的线性元素序号；只相等的总元素数不足以通过。reshape 本身保持
SEW、LMUL、VL、原有效域标签与驻留数，consumer 需要其它分布时插入显式 `convert_layout`。
尾部不妨碍相同 carrier 的坐标重命名；有效范围继续由数值有效域传播处理，不能因 reshape 变成 full。
该规则支持标准 `tl.sum` 的展平路径，不将 replica 数据误当成一个更宽 vector 的已有 lanes。
完整轴置换在 lane radices 和顺序相同时，以代数证明 lane 映射，只枚举 register parts
的置换；不因完整 tile 超过小型逐元素分析上限而拒绝可表示的转置。其余 lane 重排仍
逐坐标验证，实际寄存器驻留继续由资源 pass 处理。

control carry 的最终 type 必须先确定，再闭合其下游 reshape，避免后续循环结果调整
使已经验证的 reshape 映射失效。
被消除的轴若只占 issue time、而其它 SIMD 轴仍存活，composite pass 将其完整 time 域
转换为现有 register replicas，再由相同的 add/max/min 归约实现消费；不能因为初始
replica factor 为 1 而遗漏该数值域。转换后的完整驻留重新进入资源分析。

完整 numeric lookup table 可以由纯计算产生。若其完整坐标域能放入消费者的 register
group，而两端 LMUL 不同，使用该 lookup edge 上的显式 conversion 适配寄存器容量；
不更改表的逻辑 extent，不改变其它消费者，也不回溯重读构造表时的输入。

完整 codebook entry 的 indexed supply 可沿同 shape/axis 的 pointwise 与有序 carry
查询下游归约；标准全元素 sum 的静态线性展平也提供原轴全部被消费的证明。
已证明被归约的 entry axes 可以与 payload 共同进入 lanes，由原有 packed-entry read
一次供应多个条目。该分析只选择表示，不移动 lookup、scale 或归约的数值位置；未被
证明消费的自由轴仍保留独立表示，资源与转换继续由后续 passes 验证。

从 selected operation anchors、logical axes、producer/consumer、control carry 和 target VLEN
推导每个 shaped SSA value 的 time/lane/register-replica/local factors、SEW、LMUL、`vl`、
validity 与 register groups。普通无特殊 anchor 的 shaped value使用 target 的固定规则：
最内逻辑轴进入 lane；一元素表示保持 scalar。LMUL 是实现该 lane extent 的最小合法值，
不是 emitter 默认值。
primary lane axis 即使也由其它 use 传播到 coalesced 集合中，其 lane span 只计一次。
同位宽 integer Cast 的输入与结果共享完整物理 mapping；所需表示变化在 Cast 外显式
插入 `convert_layout`，op verifier 与 final verifier 均检查这个关系。
bounded register reduction 可以保留一致的 free-axis memory supply，以 register replica
表示被消除轴；结果必须逐轴保留其余 mapping，不强制将归约轴搬入 SIMD lane。
当多个轴已共用 lanes、但只归约其中一轴时，先用纯 `lane_repack` 将该轴移入 register
replicas，再执行 register reduction，保留其它 lane 坐标。单个返回 scalar 的 RVV lane
reduction 不能实现仍有独立 lane 轴的结果。
拆分后的 VL 保留准确坐标数，LMUL 从目标合法集合选择能容纳该 VL 的最小载体；小于
最小 fractional LMUL 容量的切片仍然合法，不要求原 LMUL 能被切片数整除。每个副本
按实际寄存器组计费，新增转换和联合驻留继续由资源 pass 核验。
已实例化的 LMUL binding 是基础表示宽度，不是所有 SSA 值的统一上限。
保持同一 logical lane span 的 cast/widen/narrow 链会按 SEW 比例唯一派生每个值的
LMUL；同整数元素类型、同 shape/axis/lane axis 的普通逐元素 consumer 继续保留这个派生宽度，
不能在下一次 add/mul 等运算处重置为基础 binding，立即把同一数值链重新拆成较小 issue。
传播仅前向增加 result 的合法宽度，不改变计算图、storage layer 上限或最终资源判据。
`weft-pointwise-width` 输出实际派生宽度的 value 数，不将这项计数当成最终指令减少数。
widening contraction 的两个 operand 再共享该 lane span。一维同轴的宽度转换还须
按 operand SEW 与目标最大合法 LMUL 限制 result lane capacity，其余元素保留为 issue time；
不能让窄 result 要求无法表示的宽 operand。没有合法派生 LMUL 或最终联合 live set
超资源时，当前 candidate 失败，不回头改作者树。
整除的 proper domain Extract 是投影，不是父子 lane-span 等价关系：子域可以使用更窄的
carrier，但不能反向压缩父值。父值保持独立合法表示，后续 typed extract 或 partitioned
reduction 消费它；逻辑分组、数值宽度与 reduction 边界不改变。
storage layer 宽度约束 raw field 和保持表示的 cast/conversion 链；lookup 产生新的
数值结果，不继承其 indices 来源的 storage lane 上限。读取与查询的不同 lane span
由显式表示转换连接，不能把超过 grouped/layered 宽度的逻辑 lane 伪装成一次读取。
完整 register table 与查询使用相同 SEW/LMUL，但有效 table 元素数可以少于查询 lanes。
table 完整元素能放入所需 register capacity 时，shared materialize 或同 SEW 的单消费者
table load 可匹配查询的 LMUL；原读取点、逻辑 extent、VL 和 lane/time mapping 均不改变。
不同 SEW 的 memory lookup 不属于该寄存器匹配候选，仍由 typed indexed-memory 关系选择。
完整自然 record field 可以保持数值 load 和显式 `field_view`，无需先复制整张 encoded 表。
`field_view` 仍保留原字段的存储对齐证明；indexed-entry 选择据此读取完整条目，不能因
descriptor 的元素对齐较弱而遗失已经成立的 record 对齐。读取快照与别名写入继续由共同
存储分析约束；表容量先决定寄存器候选是否合法，再核对寄存器 gather 的同 SEW 关系。

dense stage 的 addressable supply 不可只凭投影形式或大小选成 reload。候选必须证明从原
load 到每个 consumer 都处于同一 read epoch，包括重复子区域中 consumer 之后的写入。
无法证明时保留数值快照，不能把存储 descriptor 当作原 stage 的内容。

Encoded stage 同样遵守原读取快照。连续完整 record 区域可以具有多个轴和动态行长；
`LocalSize`、allocation 和原 load 的 snapshot edge 明确保存实际字节数及所有权。
对完整 ABI 内存区间的 `storage_disjoint` 检查可以证明当前调用没有干扰写入：此时
同一个 physical load 借用原 descriptor；否则在原读取点复制完整 record 区域。
这个选择不专门化调用地址、不改变参数的 alias 合同，也不生成另一份 kernel。
borrow 条件必须覆盖该读取在原作用域内的全部可能写入，后置 verifier 重新检查。
动态快照受现有聚合 local capacity 限制；无干扰路径不为整个逻辑 Value 分配或复制字节。
已经具有精确静态容量、且符合私有存储预算的完整 record 快照保留固定 allocation 和已选 copy。
不能仅将省掉的 payload copy 计为借用收益，却忽略每次迭代新增的动态 allocation、容量检查
和条件地址；动态范围继续使用完整 ABI disjoint 证明及原读取点的容量保护。
返回数值 SSA 的读取/转换结束原存储依赖，后续纯算术不因输入地址被改写而再次快照。
通用字段字节解码也消费从保存的 Encoding Value 提取出的 record，不绕回原 View 重读。
展开前的 point 有效长度查询在 Level 展开后折叠到对应 `PhysicalPoint.active`，因此字段
读取、尾部归约和流水 SSA 使用同一真实有效域。归约选择只检查被归约轴是否完整，
不能因 free/output 轴有尾块就引入不必要的清零和 tail-undisturbed accumulator。

同一个 widening contraction 的两个 operand 必须共享一套 reduction carrier。两侧
storage proposal 一致时保留该 mapping；不一致但拥有相同 reduction axes 与 extents 时，
按作者声明的 reduction-axis 顺序建立共同 carrier：最后一轴作为 primary lane，其余轴在其外
coalesce。storage order 只决定 operand 怎样供应这个 carrier，不能分别替两个 operand 决定
互不相容的 contraction layout；所需差异必须由后续显式 memory form 或 conversion 承载。

pointwise、state 与 control handoff 的要求不一致时，pass 插入真实 `convert_layout`；
`for/if/while` 的 carried/result types 同时被改写。无法保持 logical axes 或无合法 LMUL 时
失败，不通过静默 scalarization 继续。

若一个单轴 encoded metadata value 经保持既有 axis 的 pointwise 链扩展成 lookup index，
原 metadata axis 的 storage-contiguous carrier 仍沿 use-def 保留；新加的 entry/payload axis
由 lookup relation 决定。该规则只接受保持原 axis extent 的 typed 链，不能把任意 affine
storage proposal 当成 contraction carrier。

### `PlanRISCVMemory`

完整的一维 byte entry 若只被纯 scalar-replica conversion 消费，memory pass 可把
原位置的 vector load 与逐 lane extract 改为既有 regular-repeat scalar load（repeat=1）。
byte offset 与元素 offset 在这里相同；字段必须自然、byte-aligned，输入不得有其它 consumer。
选择只去掉向量中转与 slide，不移动读取、不改变坐标、元素位宽或后续数值结合。
其它位宽、tail、跨轴或仍有向量消费者的 entry 保持原物理供应。

indexed-entry 的 source 必须是实际可寻址的字段或 local payload。由纯计算产生的
scalar tuple 保持数值提取；不能仅因索引具有连续 payload 形状，就将其当作内存地址。
后续 typed extract/layout conversion 负责它的寄存器供应，不改变读取快照或作者的计算位置。

读取 `MemDescType`、Encoding field mapping 和 value layout，为每条 memory edge写入
`AccessAttr` 与 transfer leaf。dense access 得到 unit/strided/indexed；encoded field 得到
natural/grouped-layered/joined 的完整 storage geometry。lookup table被保留为 descriptor edge，
不是先整表 load。contract 的 lane operand与source storage relation在这里确定，并在
composite lowering中消费。partial materialization后新出现的replica memory edge会再次经过
该 pass写入typed storage relation；它最终采用unit、strided还是保留indexed，唯一由后置的
`MaterializeRISCVReplicaStorageLoads`根据最终consumer layout决定。

普通 dense load/store 的单条 unit/strided leaf 只覆盖一个 lane 轴。多轴计算 carrier 在
memory use 上先拆成 innermost-lane carrier 与 replicas，读取后再按需合并；不能把最外轴的
stride 用于整个合并后的 VL。转换保持原访问点、完整 shape/axes 与 memory effects。

对 stride 为 ABI/runtime operand 的 dense lane axis，unit-stride 探测是局部物理选择。
选择沿 Slice/Subview 回看底层 descriptor：只为最内层非单例维度加入该探测；被切片隐藏的
后续交错维度不能误判为不存在。其它动态 stride 保留 strided leaf，访问语义不变。

从 admitted table 改成 consumer-local memory lookup，必须证明原读取点到 consumer
没有可能别名写入，包括嵌套循环的较早迭代。无法保持原读取且没有合法的已物化 table
表示时，明确拒绝该候选。memory-descriptor lookup 声明 Read effect；只有寄存器 table
lookup 是 pure，CSE 不得跨写入合并前者。
同一 block 内，lookup 的 index 若已支配原 load，或是可以原样复制的常量，则可以在
原 load 点保存所需的数值。此时跨写入传递的是已读取的结果，不能在写入后重读 View；
未被观察的整张表不需要占据局部存储。该规则不跨控制边界，不移动作者 stage birth，
其余消费者继续使用原 Value。动态 table 不能仅凭一个 register layout 就被当作完整寄存器表。
同轴的一维full寄存器table若以`iota + 常量`读取完整、对齐且不越界的连续窗口，
可直接选已有`rvv_issue_slice`，不再生成向量gather索引。常量求值最多访问32个节点，
只处理无溢出的index加乘与值域可精确保留的cast；iota及加法也必须在索引位宽内。
这只投影已经读取的numeric SSA，不新增、延迟或扩大memory read。
不能合法延迟的静态连续 dense table 在原 load 点形成 `local_alloc → dense_snapshot`，
后续 lookup 读取该私有 descriptor。显式 stage 的 owner/birth/lifetime 保存在 allocation；
没有 stage 时采用原读取的 Level owner。超出目标 local 容量或非连续、非静态表明确拒绝。
由算术得到的完整静态 table 若无法由一份寄存器 gather 表示，则复用 numeric local-value
分配、初始化与 view：在原数值定义的作用域保存快照，后续 lookup 消费同一逻辑值。
已有 local carrier 直接提供其 descriptor，不再复制一次。该物理表示不重新执行原读取，
也不改变作者的 Level、stage 或数值支路；容量仍受同一目标预算约束。
尾部 numeric Value 被保存为容量固定的 local object 时，容量不等于有效元素数。标量投影
在表示改写前捕获实际 active extents，局部读取受该谓词保护；共同 validity propagation
把同一谓词传给后续 store、atomic、分支和 carry。无效位置既不读取未初始化的快照，也不
产生可观察写入，不能用填零代替这项 effect 合同。
非连续 dense table 的 lookup 在访存 pass 中把 row-major 逻辑 index 分解成坐标，再按
descriptor 的实际 stride 合成为目标宽度的 element offset；动态 stride、转置与广播使用同一
SSA 地址程序。`weft.riscv.lookup_storage_offsets` 标记这项已经完成的物理映射，后续窗口
物化与布局转换必须保留它，不能再次把 offset 当作逻辑 table index。终端 gather 只执行
这些显式偏移，不能假定所有 dense Encoding 都是连续存储。

indexed-entry 的地址索引若仅由同形、同轴的 unsigned 8/16-bit value 扩展至 32-bit，
且完整 byte offset 范围可由 16-bit 表示，可以直接选择 16-bit extension/identity 与
对应 EEW。必须验证 byte stride、范围、两侧 layout projection 和目标能力；不在一般
32-bit 算术链末尾追加窄化，也不改变 table/data 的读取宽度或位置。

partial 的 issue-window 重建 indexed-entry 时，标量 entry 地址仍对应独立的 payload
replica。若消费端窗口把这些 entry 合并为 lane，先保持 unit-load 原有的地址与 replica
映射，再插入显式 layout conversion；不能让已选 unit-load leaf 隐式承担合并。

物理 word read 后只观察一个完整字节时，可以选择 `rvv_byte_gather`。当前匹配 unsigned
u16/u32 read 的 byte-aligned shift 加 `&255`，或非饱和 `rtz` 的低字节窄化链；shift 的
byte selector 必须经有界 use-def 与范围证明位于原 word 内。encoded field 必须是
little-endian、完整的一维 natural storage；dense descriptor 必须连续。word 到最终投影
的中间链只能有唯一消费者；读取与投影之间最多检查 256 个同 block 操作，可跨普通
只读 effect，但拒绝写入、分配/释放、未知 effect 和 region。不能把已物化 register table
改为重新读内存。返回字节以显式 widening 恢复原消费者的类型与 layout，不改变数值图。
该 widening 也适用上述 indexed-entry 16-bit 地址规则，包括较早 nested-memory pass
已选中的 entry edge；不为其它 32-bit 解码链追加窄化。
字节地址中的 unsigned quotient/remainder 可按
`(base + (x >> k)) * 2^k + (x & (2^k - 1)) = base * 2^k + x`
合成，当前只处理 2/4-byte word、至多四层 pure axis/layout projection，以及同一 SSA
或完整 type/start/end 一致的 iota。该等式保持原整数宽度的 wrap 语义，不跨读取边界，
不使用无界结构比较。`SelectRISCVMemoryByteProjections` 只在最终 layout canonicalization
后的 memory 收尾选择该 RVV leaf，不在较早的 memory planning 提前冻结通用 lookup；scalar consumer 的反向
rematerialization 必须先完成。

已选 byte gather 的地址若可证明为同宽 unsigned 仿射表达式，连续的 lane 后缀可形成
多字节窗口。分析最多访问 64 个节点、深度 24；只穿过 pure layout/broadcast 与同宽
add/sub/常数乘法，不穿过 shaped narrowing 或数据查表。匹配只形成一至四个窗口，
每个窗口至少两个字节，result 必须是一个完整 RVV carrier；不枚举其它 layout。
相对 storage 的 `byte_base` 数值与窗口偏移必须是 power-of-two 窗口宽度的倍数，从而
证明窗口内部不会跨过原 unsigned index 的 wrap 边界；这不提高 storage 地址的对齐要求。
各窗口起点仍按原位宽计算并物化为显式 scalar SSA bases，原 gather 的字节集合、顺序、
读取点和 snapshot 关系不变。选择结果及窗口数/宽度/bases 均可直接从 Physical IR 观察；
它只替换字节读取和地址供应，不改变后续索引值、contraction 或 reduction。

同一 leaf 也可承载连续 byte descriptor 上的双窗口 pointwise lookup：
`(a*(1-h)+b*h)*w+p`，其中 `h=lane/w`、`p=lane%w`，lane 是同一一维逻辑域内
`0..2*w` 的 unsigned iota，w 是至少 2 的 power-of-two。关系最多比较两个乘法顺序，
每条坐标链最多穿过 16 个 unsigned value-preserving cast/widen 或 pure layout conversion。
穿过的每个 value 与 iota 必须具有一致的 full-valid time/lane/replica 等 pointwise mapping；
允许不同 SSA 的等价 iota，不把 SSA 身份当成逻辑坐标相等的必要条件。
`h` 的取值只有 0/1，两个 scalar `a*w`、`b*w` 在原 unsigned 位宽内计算，其低位为零，
所以窗口内部不会 wrap。两个 unit load 和一个 slide pack 替代向量插值地址与 indexed load；
原 source、读取点、byte signedness、输出 lane 顺序和后续数值运算保持不变。
此关系也接受多维 contiguous dense table：selector 与 verifier 共用 stride 证明，
最内层 stride 必须为 1，向外依次等于全部已知内层 extent 的乘积，且乘积不溢出。
只有最外层 extent 可以动态；带 padding、转置或未知内层 stride 的 descriptor 不匹配。
source descriptor 与 slice origin 原样保留，不把逻辑 rank 当作物理连续性的替代条件。

同一block内，若一个byte-aligned natural encoded scalar沿纯一元链形成一个多消费者
supply，pass插入`register_materialize(realization=physical-share)`。该op没有作者
birth；它只冻结一次已选load/decode结果，使后续不同layout consumer共享同一SSA值。

若typed regular-repeat relation证明少量encoded scalar各自覆盖完整lane window，pass先物化
typed regular-repeat gather，再沿pure pointwise use-def追到第一个vector consumer。只有source
count、repeat、lane/time/replica mapping与每个physical part的source base全部闭合时，才把该链
改写成`rvv_regular_repeat_scalar_load`与scalar-carrier pointwise链；vector boundary保留并消费
scalar-vector leaf。该改写不依赖format名字，也不把任意storage proposal当作contraction
carrier。无法闭合或不会到达vector boundary时保留原typed gather。

joined field 的最后一个 role header 与 low-bit payload 物理相邻时，完整零基 source
window 可选择 `rvv.regular-repeat-joined-unit`：一次连续 byte load 后用已物化的 repeat
indices 做 register gather。group 为不超过128的二次幂，完整两组byte均落在field storage
内，所有结果同type、full、SEW8且carrier容纳整个窗口。shift后不再需要的mask由planner
证明并冻结为leaf参数，verifier重算；scratch固定4倍result register groups。该关系不把
不同逻辑field当成同一field，也不移动读取越过write或unknown effect。

单轴、full-valid、unit-repeat 的 joined 窗口若被拆成多个 fractional carrier，可在目标
合法的至多 m1 carrier 内容纳同一完整逻辑窗口，再用显式 conversion 恢复 consumer
的 lane/time mapping。原 shape 必须由 lane×time 精确覆盖，不扩大读取或改变字段轴。

对一个完整 encoded field 同时有外层 RVV consumer 与内层 `group_index` scalar
consumer 的情形，pass 只在 field owner/name、logical shape/axes、parent/sub-Level point
几何、分组步长和重复 decode 全部一致时，在两类 use 的共同支配 Level 物化
`local_alloc → local_bind → rvv_local_materialize`，子 Level 以真实 `local_load` 读取。
这种 `physical-share` local lifetime 完全属于物理 IR，不增加作者 staged birth。
其他跨block/跨Level placement 仍需更一般的 dominance/LCA、effect 与 cost-independent
reuse 证明，不由该窄合同猜测。

### `PropagateRISCVConsumerLayouts`

在 fragment accumulator retention 与 issue grouping 之后，从已选 `ime_unpack` 提取保序
RVV 候选；沿普通 cast、pointwise、new 与 scf.for init/iter_arg/yield 的同域 use-def 传播。
值按 SSA identity 去重，一个 component 至多 128 个值、1024 条 use edge，越界明确报告。
每次 pass 对一个 kernel 至多比较两个完整物理候选；不激活 generic ValueType 的 IME carrier，也不换作者树。
供应侧可以穿过只省略单位物理轴的 rank-subdomain cast/pointwise，保持其原有 shape 与 axis，
避免先分段转换 dtype 再拼回完整 carrier。numeric read 仍是边界，不改读取点。

`local_tile_strategy` 是显式 backend config：`atom-partitioned` 保留独立 atom 的正常候选，
`register-grouped` 要求可成组的完整 atom handoff。`target-default` 对已注册的至多四 atom、
完整 M4 行与相邻 N atom 选择 register grouping，其余几何使用 atom-partitioned 的布局规则。
M1 不强行消费 M4 的无用行。显式请求无适用 contraction、几何不支持、坐标或资源不合法时失败；
final verifier 核对实际 unpack 实体，不允许仅记录请求却继续另一种实现。
单行 accumulator 还可在合法 LMUL 与单 atom 寄存器容量上界内，显式压紧至多四个 N atom
形成完整行 carrier。该候选按内层工作量与真实资源合法性比较；动态尾部仍使用原有效范围。

原有 numeric read 是输入边界，不能追溯 storage 重读；尚未读取的 field projection 可在原
consumer 读取点通过 `field_read` 直接供应已选单 lane-axis 布局。多 lane-axis 内存读取仍先
使用已有 memory layout，再显式 repack。rank-subdomain 的局部 binary broadcast 由 exact
leaf 与共享坐标/临时量合同闭合；其它 consumer 需要的旧布局在该 use-site 显式转换。

成组方案仍由普通 cast/mul/add 和 control carry 组成；行 scale 可以采用 scalar replica 读取，
列 scale 经显式 repack 供应单个 row-scale multiply。`IME/FragmentSupply` 把同 block、同一 numeric
row SSA 的广播单独物化并共享，后续 fragment pointwise 消费该已展开值。跨 operation 的供应
因此存在于 Physical SSA，不依靠 terminal 缓存或 system compiler 偶然 CSE。局部初始化可选择
明确的 local leaf，以保持其物理生成位置，不改变 canonical state birth。
同一供应共享关系也适用于单行完整 carrier。资源闭合前，纯 `lane_repack` 与返回完全相同
type 的逆转换可以消除；这不授权穿过已选 coordinate exchange 任意重建 producer。

单行 fragment 的有效 atom 小于一个硬件寄存器、完整 row 大于一个寄存器时，handoff
可先以独立的小范围 slide 组成完整寄存器，再通过已验证的 `vset` 映射合并。自然 atom、
中间完整寄存器及最终 row 都有显式 type/conversion；局部工作估计不将无搬运的
`vget/vset` 子寄存器映射计为 slide，资源统计仍包含实际驻留的 parts。

候选先检查 IR、坐标与共享 region-aware 驻留预算。成组采用明确的 target 规则；独立 atom
布局按更内层动态 loop 的局部工作优先；已知循环次数乘入工作量，不把仅有两次迭代的
内层循环视为无限优先，忽略外层转换代价。平局保留现有布局。诊断分别输出请求、规则、峰值实体、
算术与 broadcast/conversion/initializer 工作量。该估计描述 system compiler 优化前的局部
展开，不是整个 tile 的周期预测，不包含未变化的 MMA、decode、地址计算与 memory 延迟；
cycle estimate 明确 unavailable。未知局部工作不填零，后续 memory/leaf/resource 失败也不
触发另一套隐藏结构的重试。
当原程序的已知峰值已经超出寄存器容量，而候选不增加该峰值时，前面的布局 pass 可将
该候选交给后续真实资源闭合，不提前宣称可驻留。后续必须通过缩短活跃区间或显式
spill/reload 使完整程序满足预算，否则拒绝。未知峰值不适用此规则；它不增加目标预算。

已注册的 N16 accumulator 可按 row atom 合并，或在 VLEN 可容纳两行时保留连续行对。
当原布局的完整活跃集合超出寄存器预算，而候选无需 spill 即可驻留时，优先合法驻留候选；
其它情况继续比较显式局部工作。标量子域到行对的广播可选有序 scalar runs，重复完整向量
tile 可选 slide 拼接；共享坐标证明约束每个源 part 与目的 lane，不改读取点或数值类型。

### `ScheduleRISCVLocalSupplies`

leaf 已选定后，在同一 block 内沿 unpack/cast 的 use-def 找到 contraction 后消费的独立
scale supply。仅移动既有 field/read/widen/cast/mul/纯 conversion closure，至多 24 个 op；
不改变读取次数、Level、iteration 或浮点运算树。跨越的操作必须只有读取 effect 或无 memory
clobber 的已选 fragment MMA，随后用完整 region live set 检查 vector/fragment 容量。
容量不足保留原顺序。调度结果写回实际 SSA 顺序，terminal emitter 不重新安排该供应。
unpack 的 fragment 来自内层 `scf.for` 结果时，沿对应 yield 追踪 MMA 结果，以该循环作为
同 block 调度锚点；不因循环中存在无关 MMA 就选择它。被跨越循环的所有嵌套 effect
必须可读或纯计算；无 memory clobber 的 MMA 只更新 SSA fragment。独立供应仍只执行
一次，不移入循环，不跨越自身的外层 Level。容量分析前按真实最后 use 更新 fragment
accumulator reuse，避免把可以原位更新的 accumulator 误算成同时存在的新旧两份。
循环中的单 use fragment zero 若只占硬件 fragment 的前缀，选择既有 local-prefix leaf，
以能容纳有效元素且满足 SEW/LMUL 合法性的最小载体初始化。有效 VL 不变；不用完整
硬件 fragment 的常量复制来初始化未被作者结果观察的行。
沿同 block 的 lane-local add/sub/mul 可继续追踪到唯一 unpack，因此独立整数修正不会
遮蔽 contraction 依赖。分析遇到多个不同 unpack 或 lifecycle birth 时停止；只移动独立
供应，不移动这些算术节点，也不重排其运算树。

### `CanonicalizeRISCVLayouts`

该 pass 只做已经实现的真实 rewrite：相邻 pure 逆 conversion 消除、单 use 纯 pointwise producer
的 backward rematerialization，以及同一 block 内相同 conversion 的 SSA CSE。它不声称已经
实现任意跨 block hoist/sink、全局 conversion algebra 或任意 producer rematerialization。
conversion CSE 与 backward rematerialization 处于同一个固定点：合并conversion后新出现的
single-use producer须在本次pass内继续处理，不留给下一次解析/重放才改变其表示。
重物化前先核对全部 shaped operands 的 layout 投影；例如窄结果的 lane 数会使宽整数
index 超过合法 LMUL 时，保留原 producer 与显式 conversion，不插入半套转换再报错。
这是同一物理程序的可选 rewrite，不改变作者计算、目标引擎或访存边界。

普通 `scf.for` 的某个完整 RVV carry，若 block argument 与循环输出的全部用途都是
相同的 pure scalar extract，可将该 carry 改为消费者的 scalar 表示。init 与 yield 显式转换，
argument/result 同步改型；dtype、shape、axes、源 birth/handoff 和数值计算不变。
仅接受至多 16 个完整静态 scalar parts，并排除 deferred storage projection、混合消费者、
tail 与已冻结资源。没有读取或数学运算被移出空循环；后续物理 pass 继续选择真实 supply，
最终资源重新核算。该机制对应 Triton `RemoveLayoutConversions.cpp:552-567` 的循环
representation 传播，CPU 使用 scalar/RVV 载体，不引入 GPU ownership。

partial materialization 后，`physical-share` 的 input/result 都变成单 use 时，若到唯一
conversion 的同 block 区间没有写入或未知 effect，允许移除这条已无共享用途的边界，
继续按原有 typed index/layout 合同 backward rematerialization。作者 materialization、
多 use supply、跨 block、跨写入和 final resource closure 后的情形不适用。
同位宽 integer Cast 可随 conversion 重物化，但其 input/result 必须采用完全相同的
目标 layout，不能在 reinterpret intrinsic 内隐式改变 LMUL 或 logical mapping。
indexed Extract 的 index 投影可以沿单 use 纯 pointwise 链和 iota 重物化；预检查至多访问
32 个索引节点，并逐输入核对可执行的 layout projection 和与 verifier 共用的 index EMUL 合同。
投影去掉部分 lane 轴时不能只比较元素数而忽略 index EEW/data SEW 所要求的寄存器比例。
不能直接拼接旧 lane parts，
不等于该纯 index 程序不能在目标 layout 中重新形成；memory producer 不由这条规则复制。
内存规划先识别连续 field/entry 窗口，只有仍需 indexed 访存的 Extract 才使用这条 EMUL
合同。连续窗口会消除地址索引，不能先按索引 EEW 拆窄数据载体，再把小读取拼回。
真正的索引在自由轴上广播时，先以索引所承载的 lane 域完成
读取，再由显式 conversion 形成消费者的合并 lane 布局；不能只扩大数据 LMUL 而沿用窄索引。
若 Extract 仍投影 grouped/layered storage，不能把超过对应 storage layer 的 SIMD lane
span 向读取侧重物化；保留合法窗口读取后的显式 numeric pack，而不是制造未闭合的跨层读取。
若最小合法 LMUL 使 source part 留有空余 capacity，part-to-lane pack 仍按 pieces 倍率
使用闭合的中间 carrier，再显式缩到 consumer LMUL；pure conversion 合成不得消除这个
必需中间表示。两条边保持完整逻辑坐标、VL 与读取点，资源按实际 carrier 重新闭合。
局部多轴读取展平时，仍在活动 Level 中的尾部轴不能复用为合并后的线性轴，否则行数会被
误当成元素数。局部 gather 的寄存器坐标相对于已建立的私有 Value，不再加外层 Level 起点。

单 use memory lookup 接 scalar conversion 时，在同 block 无跨写入的条件下，可以将
indices 投影为 consumer 的 scalar time/replica mapping 并直接 scalar lookup。新的 memory
form 和 leaf 由后续 `PlanRISCVMemory` 重选；该规则不把寄存器 table 退回内存，也不 hoist
潜在空循环中的读取。
完整的 regular-repeat gather 仅被 scalar conversion 使用时，可在原读取位置直接形成
scalar supply，复用重复的 source byte。该规则保留归约轴及单元素 free-axis mapping，
要求同 block、无跨写入、完整窗口与精确 part-base；不需要重物化整段坐标/算术程序。
`repeat=1` 是每个 source 对应一个 scalar part 的合法退化情形，仍检查完整窗口与 part-base。

一个单 use、无 snapshot 的一维 dense unit load，经 pure conversion 只供给一个 pointwise
consumer 时，可以在原读取点直接采用该 consumer 要求的分片。仅接受已物化的 domain-point
slice、整除的 full lane/time 分片、同 element/axis/validity、无 replica/local/fragment 扩张，
并检查目标 layout 与同 block 无跨写入。读取字节集合和读取位置均不变；memory form、leaf
与资源须重新闭合，不把宽 load 本身当作收益，也不复制多 use 或 indexed 读取。

同一读取点的二维 dense tile 也可在 lane 与 replica 轴精确互换时重物化 load：两个轴均由
PhysicalPoint 选择，time/local/fragment 均为 1，前后 element、domain 与 validity 相同，且
每侧只有一个 lane 轴。仍要求 single-use、无 snapshot、同 block 无跨写入；后续 memory
planner 根据新的 lane 轴选择 unit/strided 读取。该关系不依赖 consumer 是 RVV 还是 IME。

### `SelectRISCVSharedSupply`

在已有 layout 与 memory access 闭合之后、shared materialization 展开之前，对 dense
数值 stage 比较保留完整寄存器值和按 consumer 的窗口重读。只穿过完整 domain 不变的
representation alias；同形状的数值 projection 也不能被当作原 load 的坐标原点。
所有用途必须有可实现的 Extract 读取，并证明原 load 到每个 consumer 没有别名写入；
作者 snapshot、birth、owner、lifetime 和数值类型保持不变。

每个候选复制当前 module，通过同一 registry 运行 shared-supply 之后的完整正式管线：
`PrepareRISCVComposites`、局部 packing/指令选择、流水、snapshot、显式 spill/reload、资源
收口和 final verifier。LMUL 与 scalar-load-prime 使用正式编译的剩余绑定，其他参数从当前
Physical IR 读取。候选不会递归进入 supply selector，也没有简化的候选专用 lowering。
owning Level 与 stage 一同展开，不能验证半展开的 birth 计数。
未改变的独立活跃值、原 read 的其它用途都留在候选中，不用减去一个名义 buffer 的方式
推定资源收益。早期必须 addressable 的表示仍由原合法性规划决定，这一 pass 不把其改回
未实现的 register 路线。

读取和写入比较要求完整、静态窗口的字节数与实际 carrier transfer 数，unit、strided、indexed
等访存类别分别比较，不用较少字节替代新增 gather 的成本；静态循环次数使用共享
分析精确展开，动态次数以真实 bounds/step 表达式为符号，不能用克隆后的 loop 序号代替。
最终候选还记录 allocation、地址计算、数值 carrier、控制与转换；未定价的 operation 保留
类型、参数、常量与频次，不能按零消除。SCF if 保留真实条件与分支极性；其外围循环也作为
域符号保留，避免把依赖 induction 的条件按固定概率展开。未知 active domain 和不可描述的
频次不能得到虚构成本。每个候选只评估一次；同一 kernel 后续选择复用当前方案的已有评估，
每个新替代方案仅执行一次完整后缀。

只有各符号项的读写字节和 transfer 均不增加、至少一项下降，已计价结构工作不增加、
未定价项没有新增，且最终寄存器峰值与局部存储不增加，才依据成本选择重读。若原方案
被明确标记为资源不足或不支持，而替代方案通过最终验证，则选择可执行方案；普通 verifier
和实现错误仍然使编译失败，不作为可忽略的候选拒绝。分支、局部存储及其它消费者全部
保留在评估范围。当前成本是结构工作模型，不把未知 encoded/fragment transfer 当零成本，
也不声称预测具体 CPU 周期。

`weft.riscv.residency_choices` 保存候选的读写多项式、完整结构工作、未定价项、最终
峰值、局部字节、spill/reload site、剩余绑定和评估次数，并从 Triton compilation metadata
的 `supply_choices` 提供给调用方。它不是后续
分析输入，所有分析在改写后重新建立。参考 Triton
`ReorderInstructions.cpp:104-123` 将会增加寄存器压力的转换推近 consumer，以及
`:164-170` 在移动读取前检查写入边界；这里增加了完整候选、复用频次与 SSA 驻留比较，
不引入 GPU 的 shared-memory 或 thread ownership。

### `MaterializeRISCVLocalWindows`

该 pass 在 local carrier 布局确定后、memory planning 前，将具有完整连续索引关系的私有窗口
显式改写为 `local_view`、`subview/slice` 和带 Read/Write effects 的普通 load/store。
它核对静态窗口、每个 axis 的 scalar index 或 unit-step iota、shape symbol 与 row-major
stride，并在消除 functional update result 前核对旧 state version 的独占更新条件。
未缩小的 `all` axis 可保留完整动态 extent，使用 descriptor extent SSA 绑定；仍需缩小的
动态窗口不得被当作完整 axis。静态私有值供给 contraction/reduction 时，所需 carrier
转换同样形成显式 local load，不把 storage carrier 当作计算输入布局。
访存形式、RVV 宽度和资源仍由后续既有 passes 选择；terminal 只绑定已经存在的 local
allocation 和 descriptor。未闭合的动态窗口或非独占更新明确拒绝，不由 emitter 生成循环。

### `LowerRISCVComposites`

把 source-level physical composites 改写为终端前的真实程序结构：

- Level loop变成带 Level identity 的 `scf.for`；
- state/local materialization变成 local alloc/bind/load/store 或 register/staged SSA；
- grouped/encoded reduction变成显式 reduction loop与 typed window load/step；
- ordinary contract变成显式 reduction loop与 RVV step；
- IME contract变成 typed fragment 初始化、operand supply、带 accumulator 的 MMA 与 unpack；
- layout conversion获得唯一 terminal conversion leaf。

整数 widening 后紧接单轴 sum 时，`rvv_widen_reduce` 直接消费窄向量，保持双宽结果
的模整数语义。结果可以是裸标量，也可以保留所有未归约的单位轴；后者必须具有一个
scalar carrier，axis identity 恰好等于输入删除归约轴后的序列。非单位 free axis 仍需
逐行/分片归约，不能被这条优化合并为一个标量。pass 与 verifier 共享该坐标检查，
terminal 继续只拼写已选的 widening reduction，不先生成整向量扩宽。
有界 value program 将原先的数组/scalar traversal 转为 RVV issue 后，再闭合一次数值
conversion；例如 i64 截到 i8 仍显式形成 64→32→16→8 的 narrowing 链，不能把尚未
选定的多宽度转换交给 terminal 猜测实现。

小型、单 reduction 轴的 memory-backed integer `vx` operand 可以采用 scalar tuple，
直接供应各次 issue 的 GPR；相应 projection 保留完整坐标，不先向量加载再逐 lane 提取。
闭合整数 contract 在 full-valid、每个 accumulator 至多两组寄存器、偶数 unroll 且静态
extent 至少覆盖一次展开时，物化两条交替更新的 partial chain，末尾以同宽整数加法合并。
原有外部整数 seed 只进入第一条 chain；第二条从零开始。此重排遵守模整数加法，不跨越
浮点或显式转换边界；新增驻留仍由完整 resource pass 核验。

`DenseContractions` 在闭合浮点 operation 内枚举有限 free-axis tile 和 LMUL。选择同时计入
实际 issue 指令、向量宽度工作、重复输入和尾块的无效工作；输入加载、转换、FMA、最终归约
分别计算驻留峰值，末尾 scalar reduction 的 seed 不与已经死亡的输入重复计数。最终 resource
pass 仍重算完整程序，代价排序不代替合法性。共享 batch 轴保持逐点对应，多 reduction axes
以显式 loop carry 连接，不把每个物理窗口提前收敛为作者可观察的 scalar。

结果只沿同域 pointwise/纯表示链流向一个 disjoint store，且移动到该 store 不跨写入时，
materializer 可直接写每个完成的输出微块。bias 等外部 shaped operand 必须有稳定 dense
read 和完整坐标投影；不复制任意已计算的外部表达式。结果仍有其它观察者时保留完整 numeric
representation。动态 free extents 来自当前 subview，不回退到同轴 ABI 全长。

两个输入均以内层 K 供给的闭合 contraction，可枚举双方转成输出 lane 的有界表示，比较加载、
逐行地址、搬运与 partial-output 流量；不能遇到第一个合法表示就停止。私有 partial 的
最内轴服务于计算 lane；K 完成后，最终 pointwise/store 可以采用输出 descriptor 的轴，
由显式读写转换两种表示。计算期与最终 epilogue 分别核算驻留，不把已死亡的 FMA operand
计入最终输出期，也不将 bias/activation 重复施加到每个 K panel。单行、单列及小 K 的
分支使用实际 View extents，保持其原有完整 reduction 边界。
静态 K 窗口小于一次完整 issue 时，只物化尾部路径，不在零次循环里保留越界 subview。
流水绑定由所选 contraction issue 消费；完整 issue 不存在时，该绑定也不转移到外层作者循环。
表示代价按实际的私有 panel 计算 tile 计数，而不是用最终 epilogue tile 代替。已知 L1
geometry 时，packed operand、partial output 和当前 scalar window 的合计工作集超出 L1，
重复扫描额外计入 cache-line 流量；这是代价排序，不限制合法私有存储容量，也不保证命中率。
已有自然输出 lane 时保留该方向；把另一侧变成非连续 scalar K 的跨 cache line/page 代价不由
这个有限的单侧 panel 候选域代替建模。

连续 K 的 scalar supply packet 由有限 `contraction_k_steps` 绑定、cache line 和标量驻留
共同约束。它物化为有界 `scalar_tile_load` 与逐 K `extract`，FMA 顺序不变；非完整 packet
仍由单步尾循环覆盖。packet 减少 loop control，不消除逐行地址及加载成本。

只流向独立外部 store 的输出遍历可以拆成完整微块与尾块：非 lane 轴的余项按有限的
二次幂微块分解，只保留真实 replicas；lane 轴余项保留已选载体并使用实际 VL。每个输出仍完成原来的全部
K 累加后才运行 epilogue；共享完整 numeric value 与私有快照不能借此省略未写坐标。
Epilogue 的稳定读取与输出不别名证明在展开前闭合，其 descriptor 在各微块中显式投影；
不在已插入部分输出 store 的图上反复重新推断原读取的合法性。
私有 panel 搬运同样显式区分完整向量与余项；完整 segmented store 必须重新证明
descriptor 各轴范围和目标 VLMAX，不能只凭 tuple 的字段数省略尾部。

普通整数和浮点 contraction 共同使用显式数值 operand supply。K 坐标先由 dense slice/load、
encoded `field_read` 或 numeric projection 消费，widen 是独立 operation，之后才进入纯
scalar/vector FMA leaf。不能把解码、读取或数据类型转换藏在 FMA 的 terminal binding 中。

浮点 `rsqrt` 在该 pass 内展开为同 type/layout 的 `sqrt`、同 element type 的常量 1
和 `div`；RVV 的常量通过显式 `rvv_splat` 形成，临时 SSA 均进入原有资源与存活区间分析。
该 rewrite 保持 scalar、logical axes、tail、Level 和数值舍入边界，不产生外围遍历或
workspace。RISC-V 实现接受 f32/f64，其它 dtype 明确 unsupported。后续 selector/finalizer
选择 `scalar.sqrt` 或 `rvv.vfsqrt.v` 和既有除法 leaf；final verifier 拒绝未展开的 `rsqrt`，
terminal emitter 不重新构造倒平方根算法。

传播 NaN 的 `maximum/minimum` 在该 pass 内展开为 `max/min`、逐输入 NaN 比较与
`select`，保留 signed-zero 规则；predicate、分支和结果保持相同逻辑坐标映射。
`select` 的终端 leaf 明确选择 scalar 条件表达式或 RVV mask/merge，核验 predicate 与
数据的 lane/part 映射、SEW/LMUL mask 比例及 scalar splat/mask 临时资源。emitter 不选择
NaN 策略，不把选择改写成乘加，也不隐式重排 layout。

未知 conversion kind、缺失 reduction extent、缺失 access 或不闭合 fragment capability均明确
失败。该 pass 不创建 source outer traversal、Level、workspace 或 persistent Encoding。

IME operand 的 contiguous field-load 关系由 primitive 的行/内轴、packing tile、source field
storage chunk、cohort、physical point 的 partition/alignment 和 read/snapshot 条件共同证明。
一个完整 chunk 跨相邻行组成连续局部 operand 时，选择 `ime_fragment_storage_load`；
sub-byte shift/mask 独立为 pure fragment decode，整数 cast 及同域 add/sub 进入显式
fragment pointwise，row 参数的广播与临时寄存器同样闭合。
不满足该关系的合法 operand 以显式 RVV layout conversion 和 `ime_pack` 表达，不改变作者树。

selector 将单 atom 与 M/N repetition 固定在 implementation parameters；materializer 形成
fragment/replica 分解明确的类型，并共享不同 N atom 的 lhs。最多四个 atom、K 不重复；
terminal 只枚举已选重复数和明确的输出坐标，不决定 microtile 或扩大 kernel 遍历。

### `MaterializeRISCVValuePrograms`

同一有界 dense stage 被多个 consumer 同时按两个或更多 domain 轴切分时，在 producer graph
改写前先闭合整组供应。可安全延迟的读取使用原 descriptor；其它情况在原数值 producer 后
保存一次私有快照，所有投影复用该快照。原 dtype、完整 shape 和轴身份必须相同，源窗口的
physical origins 与活动范围必须可证明。子 point 的全局 base 减去该源窗口 base，得到私有
存储的相对坐标；活动长度同时受原快照边界与当前 child 限制。空窗口使用安全的零 offset。
结果是显式 physical point、slice 和 load，不由 emitter 重建多轴提取算法。已有单轴
寄存器投影仍保留原路线，作者的 stage birth 与原读取位置不变。

严格子域上的数学 producer 可以先在自身已有轴上完成一个 bounded value program，再供应
较大的 broadcast consumer。不得把本可向量执行的子域 exp/log 强制嵌入另一个 lane 轴、
因此重复为每个 replica 做标量数学。已存在的 register boundary 在其定义后形成必要私有副本；
循环外不变值的副本不随内层 consumer 重复建立，loop carry 的副本仍属于该次 iteration。

动态 scalar index 跨寄存器 replicas、无法选择直接寄存器提取时，使用同一数值快照的
`local_view → slice → load`。有定义点的 SSA 在其 producer 后建立一次副本，后续提取
复用支配的私有 descriptor，不在每个 lane 重存完整 Value，也不重新读取源 View。

scalar carrier 若保留非单位 lane window，表示该窗口内已证明一致的标量供应；
time/replica parts 是有限标量 tuple，不能仅因 time factor 大于一就建立局部数组。
普通逐元素 scalar traversal、动态数值域和超寄存器预算的程序仍使用原来的物化规则。
scalar reduction 的被消去轴若含多个 time parts，必须物化实际遍历；标量 tuple
即使能放入寄存器，也不能交给只覆盖 register replica 的终端归约指令。
I16/I32/I64 向 2～7 位整数的普通截断先按合法 RVV 宽度缩到 byte carrier，再执行
已有有限位宽转换；这些中间步骤显式进入 Physical IR，不用不存在的 sub-byte SEW。

对于完整静态范围、相同 VL 且至多四个 issue 的寄存器输入，pass 可直接写出有限 pieces：
`rvv_issue_slice` 供应原值，已完成的 export pieces 由 `rvv_assemble_value` 交还原数值载体。
选择时同时计入已完成 pieces 与当前 issue 的驻留；包含 stream reduction 时，所属循环
显式禁止 system compiler 再展开。范围、尾部或坐标不闭合时采用普通显式私有表示。

完整 Value 的表示与单次 issue 的表示分开。共同 producer graph 查询只消费同 block 的
typed dense read、保持坐标的 conversion 和逐元素 use-def；跨越 region 时必须递归证明
只有读 effect，未知 effect 或无法闭合的投影停止选择。查询有界，不从算子名称恢复算法。普通 store 的便宜图若已能
留在目标寄存器预算内，保留该表示；跨标量数学调用的长驻留及超预算图可采用有界 issue。
归约前的逐元素链在同一窗口内求值，完整初始化 lane accumulator，在原归约边界收敛一次。
独占的 f32 multiply/add-reduce 沿既有 stream-dot 合同保持允许的 FMA contraction；product
还有其它观察者时必须保留其数值结果。

整图选择和逐值物化共用单 operation 的 operands/results/scratch 驻留上界；若某个 operation
超限，先闭合其 producer/consumer 窗口，不先把每个中间值分别快照到本地数组。
标量 tuple 的全部 parts 同样占用整数或浮点寄存器；布局控制边界、数值物化和流水 frontier
共用该计数，不能以 RVV groups 为零推定整数组可驻留。
fresh local allocation 不改变已存在的输入，允许纯读取供应跨越该分配；原 alias/effect 检查仍适用。
普通 dense read 跨越写入时查询实际 storage origins；只写不相关私有对象不构成输入依赖。
尾部轴经完整轴置换重命名后，从原读取及 pointwise use-def 追溯其支配的 active extent；
所有尾部供应必须给出同一个范围，不以目标容量或另一个 Level 代替该范围。

选择后立即形成真实 `scf.for`、physical point、load/compute/store 与必要的私有对象。
同一窗口内部共享 SSA；窗口外的 consumer 通过完整快照共享数值，不能任意重算。
可能与输出重叠的输入在原读取点完成快照；只读的首个归约 consumer 若无跨越写入、改变
执行路径或读取次数，可以在读取窗口时同时填共享快照。跨越写入、未知 effect 或较早 consumer 时不允许
这种延迟。该查询在 composite lowering 的直接 stream matcher 之前使用，并在新 composite
program 产生后由本 pass 再次消费；已经形成的单 issue 不再扩大改写。

有界尾部读取同样可以物化快照：私有容量等于已声明 partition，复制遍历由原 descriptor
的实际 extent 限制，数值边界仍保留 tail。不能因为 active 小于 capacity 而延迟读取跨过
可能别名的写入，也不能将未复制的容量尾部作为有效数值。整数 add 的流式 carry 支持
有符号或无符号 8/16/32/64 位；逐 lane 更新和最后归约均按原位宽取模，显式 widening
边界仍保留，不把窄整数累加自动改成无限精度求和。

issue 资源包含数学 leaf 的临时数据载体和 mask；数据载体随 LMUL 增长，mask 按实际
寄存器计数。不能只计算输入和输出，或用固定的 m1 临时开销为更宽的指数 issue 选型。

完整数值状态按其精确 Value shape 物化；私有对象可用完整向量窗口及标量余项遍历。
Boolean byte storage 与 RVV mask 不混同。分配、descriptor、axis projection 和原 stage 的
owner/birth/lifetime 使用共同 builder；另一个 materializer 生成对象后必须刷新 identity 游标。
同一 numeric SSA 的私有快照可在支配的使用点复用。覆盖旧 buffer 前须证明旧 numerical
version 及其地址别名已无后续观察者；尚由表示 bridge 承载的 snapshot 也参与证明。
普通有序控制的 ownership transfer 与互斥分支使用共同的有界 lifetime 查询。

dense read、完整私有数值、普通 pointwise、store 和 add/max/min reduction 使用同一个
producer graph 与 issue 资源查询。来自外层作用域的 local-read bridge 只有在无干扰写入时
才能继续供应同一私有数值；不能把任意动态 numeric SSA 当成可重新读取的地址。
作者已经声明的 free-axis point 可以形成 1/2/4 个寄存器副本，同一次 issue 共享广播输入；
成本包含实际 part 数、数学临时量、归约 carry 与无效尾行。没有 point 的 ABI 轴保持现有
逐坐标遍历，不新增逻辑 cohort。每个归约输出仍在原闭合边界收敛一次。

只有 store 且没有数值导出的图，可按目标 View 的活动域限制遍历。完整局部 Value 的
容量不能扩大目标尾块的写入范围。控制边界先从外向内闭合词法逃逸，使每条边都指向真实
私有 slot，再从内向外按并行赋值的读写依赖安排复制；有环或未证明的别名关系先保存一个来源，重复来源
共享该快照。互换、扇出以及边界之后仍需观察的旧值不能因 SSA 名称不同而被覆盖。

私有对象间的完整复制可在覆盖范围为无重叠矩形、source 独占且死亡、所有 destination
读取晚于复制、size/alignment/owner/lifetime 相容时，转为 `local_reuse`。删除已证明多余的
copy traversal，destination 的 stage identity 保留。部分初始化、gather、跨干扰写入或
未证明的旧版本观察者不能用该关系消除。
一个动态轴可通过精确的完整向量主循环与标量余项证明覆盖；动态 extents 和分配容量须
保持等价，不能给未知长度编造复制成本。完整 `View[:]` 与原 View 共用坐标关系，
只在映射可证明单射时允许同坐标原地读写。快照、真实 payload 观察者及其顺序仍分别检查。

共同寄存器 liveness 按 storage identity 合并表示别名，沿全部 use 计算区间，以事件扫描求
驻留量。选择与最终 resource closure 消费同一事实；子 region 的峰值在父 live set 下计算，
互斥 region 不相加。估计不能覆盖最后的 leaf 临时量、完整资源或数值合法性检查。

在 `scf.if` 处终止的数值按各 arm 的真实最后 use 借入子 block。若该 arm 只把它交给一个
loop init，可将其驻留移交给 loop carry；另一 arm 的语法 use 不会使原值永久与 carry 共存。
条件之后仍有观察者的值继续保留，循环中反复捕获的 invariant 也不能按单次最后 use 回收。

### `RetainRISCVFragmentAccumulators`

SCCP 之后、流水切分之前识别 `unpack(MMA) + carried i32 → yield` 的单 block、single-use 关系。仅在 register
`new(0)`、相同 domain/type、正向静态 loop trip count 与 i32 累加上界均可证明时，把该 loop
carry 改为 FragmentType，并将 unpack 放到 loop 出口。初值仍属于同一个 New 的 birth/owner/
lifetime，不把初始化吞成无身份的 leaf。signed-i8 的保守单步界为 `128*128*K_atom`，总步数
不得使该界超过 i32；转换不越过外层 FP scale、rounding 或其它数值节点。该 pass 不创造外层
traversal，也不凭 kernel/格式名挑选循环。
已验证的 pure `lane_repack` 可以位于 unpack 与累加之间。每一段必须单 use，并保留相同
element type、shape 和 axis identity；完整转换链随 unpack 放到循环出口。非纯转换、
可观察的中间结果和 lifecycle birth 保留原有位置。
分组布局把标量零初值展开为显式 RVV splat 时，同一精确零值仍可初始化 fragment；只替换
已有 New 的物理初值，不增加或合并 birth。分组数值的 spill/reload 保留既有逐 part 存储
合同，由各 op 检查 element/domain、完整 slot 大小与选定 leaf，最终 verifier 接受这些
已验证的本地存储操作。

资源闭合前的 `weft-riscv-cse` 使用 MLIR 的公共子表达式消除实现，然后重新根据当前 use-def
选择 fragment accumulator/pointwise 的 copy 或 reuse 策略及资源费用。共享初始化值后不能
继续沿用原 single-use 的覆盖许可；最终资源 pass 仍重新核算完整活跃集合。

`MaterializeRISCVPrograms` 紧随其后，把仍以 transient macro op 承载的已选
local-pack 与 grouped-MAC program 展开成真实 `scf` control、point、load/step 和 SSA
use-def，然后删除 macro op。该 pass 只展开已冻结的 plan，不重选 layout、memory
form、tail 或 leaf；final verifier 拒绝任何尚待 terminal emitter 解释的 macro program。

`VectorizeRISCVRecordLoops` 消费 target profile 的 record-axis 固定结构优先级。选择
`across-records` 时，它只接受 ascending、zero-based、unit-step、无 carry 的 exact Level，且
source 与 destination 都必须证明为同一一维 unit-stride record axis；point 必须有静态且相等的
active/partition，store value不能依赖被替换 loop 的 induction variable。loop body只允许一个
domain-point store、同一 encoded record 的 field load、闭合 pure pointwise use-def，以及不被
可观察闭包使用的 dead pure op；额外 read/write 或未知 effect 都拒绝。pass 将
动态 trip count显式拆成整 cohort 的主 `scf.for` 与原语义的 scalar tail，并物化
`record_cohort`、record-strided storage load/decode/store。cohort width、每条 field 的 storage
unit、静态 byte stride、logical ownership 与 output offset全部进入 typed op；terminal emitter
不能重新推导 ramp、stride 或 tail。任何 proof 不闭合时保留原 within-record physical program，
而不是猜测跨 record 映射。

contraction 的 unroll 只能属于它真实的多 issue reduction loop。单 issue 已经覆盖
完整 reduction carrier 时，不存在可展开的 local issue loop；该 unroll 不得改挂到
最近的外层 Level，否则会复制整个 storage-block program。

### `FuseRISCVBitplanes`

连续 packed-u1 的 bitmask decode 根据 field geometry、record origin、子 Level point 与
目标 time/lane 范围选择。pure `time_to_lane` conversion 与已经直接承载 RVV layout 的
`extract` 共用同一关系；直接 extract 限于每 byte 八个连续 logical bit 的 layer=1 表示。
选择前核对 byte alignment、lo-first 位序、完整 group、point partition 与唯一 streamed axis，
不以 conversion 节点是否恰好仍存在作为能力边界。

点式 signed integer `data * (1 - 2 * widen(bit))` 可以在同一 block 的单 use 链上选择
`rvv_masked_negate`：bit 必须来自已物化的 logical-u1 window，sign 与 data 的完整 type
一致，mask 与 data 的全部 physical coordinates 和 mask ratio 一致。匹配最多剥离 16 个
同 type、单 use 的 pure copy；不穿过 read conversion，不复制或移动 mask 读取。
data 上已有的 widening/cast 原位保留，因而不能用窄整数取负替代宽整数取负；也不改变
scale/reduction 的位置。`1-2*bit` 与 reduction 使用的 `2*bit-1` 分别匹配，不能混用符号方向。

尚未被product/reduction关系消费的两种符号表达，也可直接选择 `splat(±1)` 加既有
`rvv_masked_negate`，复用 window-load 或 point-anchored bitmask-decode 的 mask-only leaf。
mask读取点、time/lane/replica、tail、下游原宽乘法均保持；取负对象只有常量±1，不将
任意i8数据取负后再widen。一个选择展开成多个register replica时，所在物理loop保留
已选unroll绑定，禁止系统编译器根据缩短的intrinsic表面再次展开该register cohort。

### `SeparateRISCVFullTiles`

RISC-V 与 x86 共用 `separateFullTiles`，在现有 SCF traversal 内分离完整 tile 与原始
masked remainder。它只消除由该循环边界证明恒真的比较和 `min(remaining, tile)`；
尾部沿原顺序消费完整循环的 carries，保留读取快照、作者 Level 和 state/stage 的归属。
独立轴的边界在入口已可用时，可共享一个完整域分支，其条件同时检查所有参与轴的整除性。
已选择完整域版本的循环树保留一份原始 masked 分支，不再对子循环重复建立完整／尾部版本；
其它独立循环树继续使用剩余预算。该规则限制同一计算体的复制，不删除尾部有效域或改变遍历。
候选必须实际减少结构化条件、向量谓词工作或存储操作，且不增加另外两类；仅折叠 scalar
extent/min 算术不能证明复制整份数值循环值得。没有这项收益证明时保留原有带界遍历。
RISC-V 的 encoded field 可能在较晚的 transfer lowering 才产生 guard；它从真实 point
的 active/partition 关系计入这些受限访问，在最小所属循环上建立完整版本。
祖先不会再次复制已经版本化的子树。父域 `base < end` 与子域上界不超过 `end-base`
共同证明 backing remainder 覆盖该子块，不能只从加法形式假设坐标不溢出。
完整版本的有限 native 展开量在资源闭合后从 1/2/4 中确定：循环外驻留只计一次，
重复 issue 的本地 SSA 与 leaf 临时量按副本计入。已有系统展开限制优先；最终 LLVM
仍负责机器寄存器分配与指令调度。emitter 不重新选择展开量。
嵌套循环从内向外收口；外层预算递归计入内层已选的 native 展开量，不能将两个各自
合法的局部展开相乘后仍沿用单层峰值。循环外存活值仍只按其真实驻留计一次。

候选只来自原循环，新增副本不再次进入本 pass 的选择；按简化后的新增 operation 数扣除
每个 kernel 的有限预算，并设置绝对代码增长上限。调度、buffer 版本和最终资源分析在
变换后重新处理实际程序。RISC-V 物理 index 保留无符号范围，不用有符号负数裁剪它；
无符号循环关系也不能被用来证明一个有符号比较。x86 保留原 signed SCF 空域规则。
完整块与尾块仍使用已有 SCF 简化；RISC-V carrier／fragment 的分支结果保留结构化控制，
不应用将任意 dialect type 改成 `arith.select` 的通用折叠。最终 verifier 也检查 scalar
`arith.select` 的输入类型，不能让缺少物理 transfer 合同的选择直到 C emission 才失败。

### `ScheduleRISCVLevels` 与 `PipelineRISCVLevels`

`ScheduleRISCVLevels` 是 scheduler。它读取已 lowering 的 physical `scf.for`、SSA use-def、
loop carry 和 operation effect，先建立 pure/read producer 与依赖 carry 的消费边界。
共用的有界调度按读取及其数值依赖划分中间 stage，把更新 carry 的 consumer 留在
stage `pipeline_depth - 1`，并在具体 operation 上写入
stage/order。已经数值化或具有直接 field supply 的 widening multiply/dot 与 fragment MMA
也建立 consumer 边界，避免整段 contraction 都落入 producer、只重叠最终 scalar add。
直接 field 在该边界前物化为 `field_read`，跨阶段传递真实 numeric value；带 shaped index 的
storage projection 不使用这条前移规则。无 memory clobber、copy accumulator 的 MMA
只写入新的私有 SSA fragment，其依赖链保持在 consumer stage；最终物理 reuse 仍归资源闭合。
它不通过 op 数量、相邻关系或 window family 命名 cluster。闭合的普通 `for/if` 可以作为
一个 stage 内的有序单元；嵌套作者 Level、外部 write/unknown effect 没有重排合同，仍拒绝。

默认以 contraction 建立消费边界。若跨阶段的 numeric values 本身已占满目标向量寄存器，
scheduler 另枚举由真实循环 carry 依赖确定的边界：把独立算术留在 producer，将其较小结果
传给 consumer。只有传递的向量驻留量严格下降才采用；所有阶段仍须通过同一 effect、shape、
依赖与展开验证，最终资源 pass 仍可拒绝该候选。它不删除作者算术、隐式降低精度或取消
所请求的流水，不把较小驻留量当作实际加速证据。

静态 `physical-share` / `local-value-copy` / `control-value` 私有数组须证明全部 storage aliases
留在当前迭代内。具有闭合平坦 descriptor 配方、已知容量且满足私有存储预算的对象，可以由
前一阶段填充、后一阶段读取；scheduler 保留 payload 的原始内存顺序，写出首次至最后访问的
阶段区间。expander 按该区间分配版本，而不是给所有 buffer 复制固定 depth 份。
descriptor 在使用阶段从当前迭代的 slot 重建，不进入数值 carry，也不重新执行源级 birth。
普通 `FieldView` 与 slice 一样只是字段地址和 origin 的纯描述符，可在消费阶段以对应
坐标重建；这项规则不允许重放 field payload load、allocation 或 state/stage birth。
RISC-V 的 `local_slot` 明确记录 backing、index、byte stride 和版本容量；verifier 检查索引范围、
对齐、大小及原 owner/lifetime，alias 与资源分析沿 slot 回到真实 allocation。x86 使用相同
阶段区间与 expander，物化 memref backing/view。非平坦、嵌套或无法在预算内版本化的私有数组
仍按其原作用域保持同阶段；作者 state/stage birth 不参与这一选择。展开后重新核算全部资源。

`PipelineRISCVLevels` 与 x86 的 `PipelineLevels` 使用同一个 `expandPhysicalPipeline`。
depth 1 顺序执行；depth 2/3/4 的操作进入所选 stage，并按各个值的生产阶段与最后消费阶段
保存必要的版本。共同展开器使用 SCF 与 IRMapping 形成 guarded fill、steady loop 和 ordered drain；
空循环不执行 producer，短循环只消费已经初始化的版本。所有 source carries 按原顺序更新，
跨阶段值及坐标版本来自实际依赖。RISC-V 的局部 shape 身份也从同一 IRMapping 克隆。

x86 的 `ScheduleLevels` 写入 stage/order/invariant；expander 只读取并验证该决定。
LLVM intrinsic 的纯度来自其内建 effect、termination 和 synchronization 属性，不能由名称
前缀判断。版本化和同阶段私有存储都保留完整读写顺序；分支可向本次迭代的后续
操作交付 buffer，但不能将本次新建的地址传到本次迭代的回边。stack slot 分配在展开后运行，
向量/谓词驻留重新分析。x86 谓词构造与其使用保持同阶段，避免多个廉价掩码版本占满
AVX-512 的有限 k 寄存器。RISC-V 同样在展开后完成 local storage 与寄存器资源闭合。

`backend_options["pipeline_depth"]` 在两目标上接受整数 1/2/3/4，默认 1，与 GPU `num_stages`
无隐式映射。当前版本化限于编译器私有静态存储，不接纳逃逸 descriptor 或 async transfer。
显式深度不能建立合法 producer/consumer
关系时给出诊断。Physical IR 保留 `weft.pipeline.expanded_depth`、每阶段操作数与版本数量；这些是编译
决定，实际性能仍由相同计时边界验证。

### `ShareRISCVLayeredWindows`

该 pass 读取 typed storage geometry、field owner、logical point、consumer layout 与 affine
record relation，物化 layered stream、record load 和 field-to-register replica load；在 pure
conversion、无中间 write/unknown effect 时，它还能证明多个 layer 共享一个 RVV storage
window。成功时都以真实 physical operation替换 extract。当前仅对 full-validity layer做共享；
tail cohort没有证明安全时保持原程序。

消费 layout 的 lane span 大于 storage layer 时，可将外侧 lane factors 精确拆为 time，
按完整 layer 宽度读取、共享 raw bytes，再用显式 `lane_repack` 恢复消费 layout。
此规则要求 full validity、连续坐标、精确整除与合法载体；不扩大字段读取范围。

完整一维 grouped/layered 字段的数值读取也沿用该路径：先按 storage layer 选择
time/lane carrier，显式生成 layered stream，再 repack 到消费 carrier。用于后续
register gather 的整表读取不交给 emitter 解码；最终 leaf materialization 产生的
field read 同样调用这套物化规则。字段范围、位序、读取位置及原数值类型保持不变。

多 record 的直接字段读取先将 record 坐标保留为 register replica，仅将一个 storage
layer 放入 lane，再显式 repack 到消费布局。连续 encoded axis 拆成 record/element
两轴时保留原 byte address、record stride 与实际有效 record 数；尾部 record 的 raw
读取受有效域控制，metadata 的 strided load 同样只读取实际存在的 record。

fragment supply 使用同一 sharing owner：在完整 chunk/cohort、无 snapshot/干扰写入的前提下，
将最多四次、无内部 birth/嵌套 control 的完整 layer issue 显式展开，保留 K issue 顺序与
carry。只允许 pure/read 和已闭合 MMA 私有更新；不复制作者状态声明。随后每个 raw read
最多比较 32 个此前候选，要求相同 field/load/parent point、child domain/active/partition、
fragment type/access，并由相对 offset 的 group/layer 商余数证明同一 byte window。
流水剥离后的正起点区间也可以展开，但至多四次 issue 的区间必须包含至少一个完整共享
window；展开次数取实际上下界之差，保留原 induction coordinate 与 carry。
共享结果是支配消费者的真实 SSA；decode 的 layer shift 不合并，完整 live set 仍由资源
owner 核算。final resource marker 未清除时不执行这项结构改写。

已物化的 replica load 若只被一组不重叠、完整覆盖的 `rvv_issue_slice` 消费，可以按
这些已选 part 投影 supply。每个投影保留 element type、SEW、LMUL、VL、validity 和
原有 window/decode map，只在同 block 且不跨 write/unknown effect 时移到对应 issue。
这缩短完整输入集合的寄存器存活期；共享 raw byte 可能在不同 issue 重新读取，最终资源
分析仍核算各个真实 load、decode 和 accumulator，不让 terminal emitter 隐式流式化整个 tuple。

### `MaterializeRISCVReplicaStorageLoads`

该 pass 位于最终一次 memory planning 之后。它读取已经闭合的 typed field/storage relation、
affine index、time/lane/replica分解、consumer layout与target facts，按target固定优先级选择
保留indexed、unit或strided field-to-register form。成功时把extract物化为
`rvv.replica_storage_load`；选中的op明确携带window plan、每个physical part的source window
以及exact load leaf，layered relation不会伪装成strided load。

连续 affine 索引可以覆盖多个完整 packed group。跨 group 时要求 scalar base 按 group
对齐、各 layer 正好组成一个完整 byte，并将 group 坐标保留在 raw-byte window offset 中；
layer 编号只取本 group 内的坐标。verifier 同时检查完整 raw span 与单个 layer 的窗口边界，
不能对跨 group offset 只取 layer 余数。完整地址关系成立时，无需先生成向量除余和 gather。

这个 pass只闭合现有 field-to-register edge，不创建logical axis、Level或source traversal。
单一 physical window 的两层 byte partition 在选择时闭合为一次 mask 或 shift，
消除零位移与最高层的冗余 mask；更多 layer 或多 part 窗口保留统一映射及原有联合供给。
partial materializer若需把一个已选unit load投影到issue window，只能机械投影原有plan与leaf，
不能重新选择memory form。
投影后的正 lane extent 可以为 1；它仍是已有 RVV carrier 的一个显式窗口，须满足
`vl == lane factors 的乘积` 及完整 part/window 映射。若随后转为 scalar，terminal 按已选
part 坐标提取 lane 0；这不改变 layout propagation 对普通 singleton 值默认选 scalar 的规则。
同一 block 内的单寄存器 natural/unit 读取可以复用一个已经存在的完整连续窗口。该规则
向前检查至多 32 个候选，要求同一 field SSA、record 坐标、常量逻辑范围与无写入/未知
effect 的区间；相同范围共享 SSA，连续子范围用已有 `rvv_issue_slice`，不增加更宽读取。
子范围只缩小最外侧的有效 lane axis，并保持其余坐标与 validity。跨动态 issue、其它 memory
节点与无法闭合的范围不由这条规则处理；完整资源分析仍核算共享后的存活区间。

同一 block 内两个完整一维 natural strided window，若 field owner、Encoding、type 与坐标
相同，步长都是 2、静态 base 相差 1，且每个 time part 覆盖相同的相邻交错区间，可以选择
`rvv_segment_pair_load`。匹配至多向后检查 32 个 load，并拒绝中间 write/unknown effect；
必须证明两路读取的并集没有越界、element alignment、NF×LMUL 与完整 tuple 临时资源合法。
两个结果保持原 layout，后续 widening、加法和 reduction 不变；emitter 只拼写已选 segment
load 与 tuple component binding，不重新选择 memory form。

### `PlanRISCVPartialTopologies` 与 `MaterializeRISCVPartialAccumulators`

单边拥有独立输出轴的 batched integer reduction 可以采用 widening-dot，然后逐输出
物化 partial。早期 operation selection 不将全部 scalar 输出元组当作同时存活的向量累加器；
双边分别提供 M/N 输出轴的 outer product 仍检查局部寄存器微块。实际载入、partial、spill
与其它活跃值的资源上界在生成 Physical Program 后重新计算。

两者分别实现在 `lib/Target/Partial/PlanTopologies.cpp` 与 `Materialize.cpp`，共同的
typed topology/几何工具在 `Support.h/.cpp`，add-tree 的规划与物化共用 `AddTrees.cpp`；
planner 与 materializer 不通过 include 实现片段形成第二份算法。

planner 读取 contraction/reduction 的 typed axes、time/lane/replica 分解、storage window、
consumer use 与 target resource budget，唯一写入 product carrier、issue slices、partial-set、
combine/final-reduction topology、selected local instructions 和联合资源合同。顺序 fused contraction
若跨多个 issue，必须得到显式 `sequential_partial_plan`；该 plan 给出每个 issue 的 source-part
投影、每个 operand 已选的 `slice` 或 `storage-rematerialize` supply、唯一 widened accumulator 与
最终 reduction，而不是让 composite dot 按 issue 数预留一组临时 partial。storage rematerialize
只对 typed grouped/layered field window 及其保持 axis 的 pure producer closure 合法；它投影已经选定的
storage plan，不能重选 memory form。当前该 fused program 只在 lhs/rhs 的 issue slice 与 reduction-only accumulator
具有同一完整 physical-part 映射时合法；带互异 free-axis replicas 的 outer contraction 由 planner
明确选择 `sequential_per_stream`，materializer 不得在失败后自行降级。

单归约轴的 tail issue 在 plan 中选择显式 VL 的 widened-accumulate leaf。
materializer 按 `source_parts/lane_offsets` 的真实逻辑坐标和两侧有效 extent 构造
每个 issue 的截断长度，不能把第一个片段的 VL 复制给后续片段。累加使用保持非活动
lane 的已选操作；融合 partial 最终归约使用各 issue 有效长度的最大值，独立 partial
使用各自长度。空 partial 返回加法 identity，不读取 VL=0 时未定义的 reduction 结果。
长度是 Index SSA operand，verifier 检查完整 part 数量及 carrier 容量上界。

materializer 只核验仍存在的 SSA graph 与 plan 一致，并实例化真实
`rvv_issue_slice → rvv_widen_accumulate → rvv_finalize_widen_dot` 或对应 partial-set program。
它不能重新选择 carrier、combine tree、instruction 或 resource groups。没有闭合 plan 的已选
topology 直接失败；terminal emitter 只按 `source_parts` 投影已有 vector binding，并拼写已选
accumulate/reduction leaf。nested、layered 与 scaled/reduced-scaled plan 同样必须在 planner
中冻结 issue types、scale supply、narrow-scale type、storage-window/decode instruction、partial
reduction、finalization leaf 与同时存活的 resource groups；materializer 只能按这些字段创建
operation，不能从 shape 或当前 SSA spelling 再次推断。

nested planner 的最大 issue window 若不能闭合资源，可以沿同一物理轴依次减半，
只考虑能精确划分原逻辑 extent 的窗口，至多检查 64 个候选，优先选择 issue 数最少的合法项。
这不改变作者的 reduction axes、scale 位置或整数宽度。选定的 window、issue 数与资源估计
写入原有 nested plan；全部不合法时报告尝试数并失败，materializer 和 emitter 不重试。
窗口缩到一个元素时，带轴坐标仍保留其轴；memory pass 必须沿已支持的仿射表达式
物化无轴标量地址基值，不能把单元素 shaped value 直接冒充 scalar。

需要 partial-repack 的 layout、nested 或 add-tree plan 还必须保存已选 repack leaf，包含
addressable carrier 与临时资源；无需 repack 的位置使用显式空项。materializer 核验并消费
该 leaf，不重新选择 `split`/`slice` 或载体宽度。

early add-tree 不能同时接管已有 sequential 或 independent typed plan 的 dot；这些叶子
会先被各自的 materializer 改写，原 dot 的 carrier 不能成为第二份计划的依据。
同一个 early add-tree 子树也只保留一个已选 owner；父节点保留原标量加法。
后续 extraction pass 仍可在实际生成的 i32 partial 上选择互不冲突的归并，不复用已失效计划。

带 Level-local issue unroll 的 scaled contraction 在展开前只有一个 loop-carried
contribution。planner 必须从该 seed 冻结完整 slots、product carrier、scale supply、
combine topology 与 resource groups；materializer 再按该 slots 机械展开。scale 可以是
typed scalar carrier 或已终结的 signed-i32 scalar，两者都必须由 verifier 证明为单一
replica；不能因展开时的 SSA 拼写不同而重选 topology。
多个独立输出 replica 的 Level-scaled partial 保留一个向量合并结果，再做最终标量提取，
避免每条 scale 链分别提取后在标量端重建同宽加法。已有 add-tree pass 仍可在向量内部
缩短 MAC 依赖链；整数宽度、各项 scale、原浮点更新边界与完整资源核算保持不变。

nested scale 与 indexed operand 共用具体 field 供应时，planner 保留表示转换路线，
不能仅因 replica 数较多就选择会重新读取字段的 scalar rematerialization。deferred scale
的 issue 克隆复用已有 field extract、Read-only supply 及共有 pointwise decode；
width/layout conversion 仍可在各 use 点重新物化，避免为共享而延长宽载体的 lifetime。
复用以同一个 SSA source 为键，不以字段名称、格式名称或相似地址猜测等价。

若完整scale已经有其它consumer、单轴full值可驻留一个vector register，且issue窗口
至少有四项，nested plan可选择`shared-before-issue`：保持原scale SSA，在各issue中以
显式numeric lookup取得连续窗口，不再克隆其算术链。该选择冻结在plan中，并额外预算
共享源、窗口与索引的存活资源；最终resource pass仍检查完整live set。它不改变原scale
的计算宽度或乘法/reduction结合位置，也不把没有共享收益的小窗口强制搬到vector上。

当每个 logical slot 本身跨多个 issue 时，planner还必须冻结 issue operand supply 与完整
product carrier。materializer先用显式`rvv_widen_multiply/rvv_widen_accumulate`形成每个slot，
再由`rvv_partial_collect`按SSA operand顺序建立typed slot集合，之后才执行partial reduction与
scale combine。`rvv_partial_collect`不生成新的数学运算；它使同时存活的独立product、slot顺序、
birth/lifetime和资源总量成为可验证的IR合同，terminal translator不能从原dot重新构造这棵程序。
该机制同时用于 Level-local 和显式 replica 的 scaled slot。多 K issue 合并进窄 accumulator
必须有原有 `fused_streams_legal` 证明；sequential plan 保存单个 slot 的 issue 资源，scaled
plan 保存完整 slot 集合与 scale/combine 资源，两者分别验证，不能以局部预算替代完整 live set。

单 issue 的 reduced-scaled 集合超过寄存器预算时，planner 在原 slot 顺序上选择可整除的
有界 chunk，把 direct/reduced/scale-combined set 类型与单 issue operand projection 一起冻结。
候选 source-lane 合并按实际 producer/consumer 边界计算 product→repack、repack→reduce、
reduce→scale-combine 的同时存活资源，不能仅确认每个集合单独可容纳；直接 scale 路线则计算
repack→widen-scale。每个边界保留其局部临时量，已经死亡的前一集合不重复计入后续峰值。
超限候选在 source-lane 枚举阶段排除，让已有 chunk 路线继续形成合法的有界集合。
materializer 按这些类型逐块生成 issue slice、widen product、reduction、scale combine，再以
同宽 i32 merge 汇合完成的块；不要求所有宽乘积同时存活。slice 显式标出实际消费的 source part，
资源 pass 可只 reload 需要的片段。整数尺度不跨浮点更新边界，完整 SSA 活跃集仍须通过最终预算检查。

independent i16 partial tree 若没有同宽跨 stream 合并的合法证明，首层 pairwise combine
必须 widening 到 i32，后续层保持 i32，再做最终 reduction；这同样适用于四路及更多
power-of-two streams，不能仅为两路特殊处理。选择前同时计入 operand/source-set live set、
第一层 source/widened-result live set 和最终收敛资源，并遵守 target 的单载体 widening 限制。
已冻结计划的 verifier 拒绝未经证明仍在 i16 中合并的首层；materializer 只消费这些 typed stages。

上述 plan attributes 是第二层内部、一次 lowering 中的瞬态冻结结果。完成物化后它们必须删除；
final verifier拒绝任何残留plan，terminal translator也不读取它们。只有单stream的closed widening-dot
leaf可以直接保留typed topology、lane/source-part relation与exact leaf，而不保留待解释program plan。

### `CoalesceRISCVPartialExtractions`

issue 展开后，重用上述 add-tree planner/materializer 合并已完成 reduction 与 scale 的
signed-i32 singleton partial，再做一次 scalar extract。只接受同一 block 内、单一 consumer、
同一 reduction axis 与完整 RVV layout 的 `rvv.partial-finalize.extract`；不接纳 i16 partial、
尚未 reduction 的 lane 集合、原始 dot 或 scale 前移，不改变 source Level/carry 或浮点结合。
局部搜索最多访问32个add节点和32个partial leaf；plan只保留互不重叠的最大可接受树，防止
子树改写使父计划的leaf身份失效。物化只生成既有 `rvv_partial_merge` 与finalize，完整资源
分析仍重新计算延长的partial lifetime；局部预算检查不是全局live-set合法性的替代。

对已归约的signed-i32/VL1 partial，另比较scalar scale/combine与显式
`rvv_partial_packed_scale`：后者按完整slot permutation把2至8个结果装入m1的有效lane，
与同轴、同序、完整的i32 scale向量相乘，再归约为原来的单slot。输入允许有界的partial-set
与scale-window列表：分别按operand顺序连接slot与scale lane，数量必须相等并适合target的
m1容量；slot permutation索引连接后的partial序列，不合并或丢弃各slot的独立权重。
输入/输出partial类型、
原scale位置、整数宽度与terms归属不变；不接纳tail、多个输出slot或跨轴scale。
scale的LMUL调整保留显式conversion，terminal仅拼写已选pack/multiply/reduce。
同block、single-use、顺序明确的相邻scale/combine可成对选择，最多检查32对；每对在第二个
原consumer处完成，避免把全部partial延长到最终merge。满unroll且两个窗口可容纳于m1时，
完整的一维scale可在至多两个register groups内供各issue共享，仍保留原读取和数值结合位置。
已选scale slice若来自同一数值SSA的连续同轴窗口，operation selector可合为一个完整窗口。
完整且整组对齐的issue slice由同一selector选择`rvv.issue-slice.split`，verifier按target
VLMAX、full active lanes、整数LMUL比和offset对齐闭合，terminal只拼写`vget`。
slot pack统一使用最终VL：每次slide保留已构造前缀，未构造lane在唯一的multiply/reduce之前
全部被后续slide覆盖；输入的未定义尾lane不会进入最终数值运算。
局部选择先检查target、坐标与资源合同，再比较包含pack、seed、reduction和VL设置的成本；
可scalar-rematerialize的供应以无vector extract的成本下界比较。供应搜索最多32个节点，
包括静态一维`iota`：同逻辑range/axis的scalar replica使用既有`register.iota`，
不能因初始vector坐标表示漏掉这条合法供应；查询与物化接受相同的producer集合。
估计以`weft.riscv.packed_scale_cost = [packed, scalar]`保存在selected op中，不作为合法性证明。
单scale窗口声明三个temporary vector groups，多窗口拼接声明四个；最终live-set仍由资源pass重算。

该pass位于普通编译与layout-input恢复共用的收尾入口，并由`triton-opt`公开供重放。已带
final resource marker的图不重新选择；独立副本须先清marker和统计后才能重放此改写。

未选择 packed scale 的完整 i32/VL1 scale-combine，若有 4–8 个偶数 slot 并收敛到一个
结果，可用两个既有 scale-combine result slots 承载独立短链，再以既有 pairwise combine
收敛。原 slot permutation、每项 scale、整数位宽和 scale 所在位置保持不变；只重组同一
闭合 i32 加法的物理依赖。两条链多占一个结果载体和一次向量 issue，最长 MAC 链减半，
最终完整 live set 仍重新核算。该选择不处理浮点、未归约的 lane 或 tail，不把 scale 前移。
参考 Triton `ReduceOpToLLVM.cpp:228-299` 在同线程 register 集合中物化 reduction tree；
这里的节点是 CPU 的独立 partial carriers，组合拓扑显式保存在 Physical IR 中。

### `MaterializeRISCVLocalProjections`

局部 issue slice 已展开后，若其完整输入来自未被改写的私有数值快照，pass 以逐 part/lane
坐标证明该 slice 是快照内的一个矩形，再物化 `subview → slice → load`。只读证明覆盖原
完整读取到当前使用点之间的 effects；只穿透行主序、线性载体相同的 reshape 与完整 local
view。后续 memory pass 为小窗口选择真实访存，避免把整个快照载入寄存器后才取片段。
不规则或跨写入的投影保留原路径，不改变作者的读取值、归约范围或私有存储归属。

### `SpecializeRISCVMemory`

将动态 dense stride 的局部访问候选物化为真实 `scf.if`，guard 只读取不可变的 ABI stride。
候选 physical loop 有界且不包含 storage birth 或作者 Level 时，可在整个 issue traversal
之外共同判断相关 stride。连续分支中的 unit access 必须由支配的 `stride == 1` 证明；
另一分支保留 strided access，读取范围、执行顺序及 loop carry 均不变。final verifier
拒绝仍含动态访存选择的 leaf，emitter 只拼写已选形式。
完整 vector tile 的 memory descriptor 必须精确覆盖其静态逻辑形状，且 VL 等于该目标
SEW/LMUL 的 VLMAX，才能选择 `.full` transfer。stride specialization 保留这一证明；
最终 verifier 重算形状和容量，terminal 直接拼写已选 VLMAX intrinsic。数值的 full validity
本身不能证明内存覆盖，也不能免除尾部 guard。

若 `scf.if` 的向量结果仅被各自的 stream finalize 消费，可在所选 arm 的末尾收敛，
使标量结果跨越 join。每条实际路径仍只收敛一次，不把 finalize 移入尚未闭合的 reduction loop。
完整 issue 的归约 step 必须同时证明 input 为 full 且 active 等于常量 VL，才能选择无需
tail passthrough 的 full-step leaf；其它 step 保留未更新 lane。整块证明来自实际 point
及源循环的 lower/upper/step，不能用名义 shape 或期望的 benchmark 尺寸代替。

### `SelectRISCVIntegerMACs`

在 byte-window 分析与 memory specialization 之后、leaf finalization 之前，选择同 block、
单 use 的整数 pointwise 乘加。普通 vector/scalar 模乘加可形成 `rvv.vmacc.vx`，同型
vector/vector 模乘加可形成 `rvv.vmacc.vv`；精确 widening
来源和同结果 type 的 add 可形成合法的 widening vector/vector 或 vector/scalar MAC。
保留原 signedness、位宽、axes、active domain 和加法边界；读写、未知 effect 或 region
阻止跨越，最终资源重新核算。不增加 reduction，也不改变最后的归约位置。
混合 signed/unsigned 的保值 widening multiply 加同型 accumulator 可选择
`rvv.vwmaccsu.vv`，两种输入顺序分别记录 exact leaf。有限位宽输入使用已经闭合的
整数 carrier；不能省略其截断、符号扩展或 zero extension。

这项融合不在前期反复运行的 operation-family selection 中执行。地址分析需要看见
`base * width + lane` 以及分段坐标的显式关系；过早形成不可分的 MAC 会隐藏连续访存证明，
迫使后端保留逐元素 gather。访存形态确定后，仍存活的索引与普通数值表达式共用该 MAC 选择。
vector/scalar 乘法的两端若来自同 signedness 的精确扩展，先检查 widening MAC 的完整
坐标与目标合法性；不能因为扩展后的同宽 MAC 也合法就先选它，留下冗余向量 widen。
没有合法 widening 关系时，同宽 MAC 继续按原输入选择，不改变显式窄化或累加精度。
新建的 scalar scale 扩展同时记录 scalar implementation，由随后 leaf finalization
闭合；不能依赖已经结束的 operation selector 再访问它。
局部 issue 资源核算只在 accumulator 为最后一次寄存器 use 时复用其目的组；候选仍需经过
完整 region 活跃区间、spill 与最终资源验证，不以单条指令的组数代替整个程序的预算。

### `SimplifyRISCVControl` 的整数常量

显式 signed/unsigned scalar 的常量 add/sub/mul、按位运算及合法位移，在声明位宽内用
APInt 求值，保留 wrap 和有符号右移语义。不折叠浮点、不跨 storage birth，也不把非法
位移当作合法常量。固定 storage offset 因而能在访存规划前形成显式常量 SSA。
layered replica read 可据该 SSA 的准确余数选择原 encoded group 内的 layer；shift 与
mask 写入 Physical IR，verifier 重新核对 offset、顺序和 layer 域。最高 byte layer
省去冗余 mask，零移位直接省去；动态 offset 仍保留原动态计算，emitter 不自行推断。

### `FinalizeRISCVLeaves`

full-valid 整数广播若仅增加连续的末尾 lane axes，且每个元素重复 2/4/8 次后
不超过 64 位，可选择 `rvv.axis-broadcast.packed`：无符号扩宽、按位重复常量乘法、
重解释回原元素类型。源/目标 part、LMUL 和坐标关系必须精确匹配；负数也复制其完整
位模式。单 use、同 block 的 i8→i16 保值扩宽可以移到该广播之后，减少宽 gather；
不跨算术或内存 effect，不改变数值转换边界。选择、verifier 和资源分析共享同一条件，
emitter 只输出已选序列。

`rvv.axis-broadcast.scalar-runs` 将标量坐标的连续重复段用 splat/slide 组成输出；
`rvv.axis-broadcast.repeat-tile` 按已证明的源 lane 模数关系重复完整 tile。坐标证明最多
4096 单元、64 个结果 part，每个 part 至多八个标量段或八次 tile 重复。scalar runs 保留
一个载体的拼接临时量；tile 重复直接复用目的寄存器。两者均不读取额外内存，也不改变
tail 的逻辑有效域；数值与存储边界仍由原始程序约束。

index residue 折叠后成为常量且按 layer 对齐的 base，若窗口完全位于一个 layer，可将其
layer 坐标移入既有 decode map，base 保留 group origin。实际 byte 地址与读取范围不变，
group 对齐证明进入 window plan；零位移与最高有效 layer 的冗余 mask 随之消失，不由
emitter 重新判断。该折叠在资源闭合前完成，不改变 operand/result 的 layout 或 leaf 类别。

此收尾还将 conversion 输入侧的 encoded 读取显式物化为 `field_read`：
`local_load`/`time_to_lane` 的字段供应直接产生已选结果；其它转换先读取原表示，再转换 numeric SSA。
复用既有 access 与 result layout，不追穿已经计算的 numeric SSA，不改原 load/快照身份。
`field_read` 的 Read effect、读取点和结果存活区间进入后续 CSE、snapshot 与资源核算；
final pure `time_to_lane` 只允许已闭合的 numeric RVV pack，不保留 emitter 重读字段的例外。
已选的 register Extract 也是 numeric consumer：若它的 input 仍是 encoded storage
projection，先形成同表示的 `field_read`；register gather 的 result 不能重新解释成地址。
`PlanRISCVMemory` 对完整的一维寄存器表，可将已证明在表内的宽整数索引逐级精确缩窄到
data SEW，再选择 `vrgather`；索引与输出的轴、shape、lane/replica 分布必须一致。
范围证明读取 iota、广播及保值整数表达式，不改变原索引的其它 consumer；未知范围不缩窄。
尚未解码的字段必须已具有完整 numeric table 供应；当前此选择仅接纳按字节寻址的 natural
字段，joined/layered 字段继续按已选索引解码，不能因 index 位宽匹配就假定整表已经可读。

读取 `ImplementationAttr` 与已经确定的 value carrier/SEW/operand form，为 pointwise、cast、
reduce、state transfer等仍未闭合的 operation写入 exact instruction/spelling leaf，并删除
`ImplementationAttr`。结构 family只在 `SelectRISCVOperations` 产生。由
`LowerRISCVComposites`新建的window step、RVV contract step、fragment和layout-conversion op在
创建时已有自己的exact leaf；本pass只终结仍保留generic operation的那一组。两者作用于互斥的
physical op集合，不为同一个operation重复选择instruction。

子字节整数 widening 根据真实 RVV carrier signedness 选择 identity 或 reinterpret：
Encoding 的 `ui1` 是 unsigned byte，普通 signless predicate 是 signed byte，不能只按
逻辑位宽假定两者相同。普通 integer cast 与显式非饱和 narrow 均可选择二倍缩窄 leaf；
该 leaf 先保留 source signedness 截取低位，再按 destination signedness 重解释同一位模式，
不改变原转换位置、截断结果或 lane mapping。

在资源闭合前，整数 RVV pointwise 的 `x+0`、`0+x`、`x-0` 可直接复用输入；
输入与结果的完整 type（含 axes、validity、layout）必须一致。零值证明最多访问 32 个
producer，仅穿过整数常量、整数转换、纯布局转换、splat/broadcast 和整数乘零，
不把浮点恒等式或轴消除混入该规则。失去用途的广播链由后续同一条 CSE/dead-layout 主链清理。
同样的有界零值证明适用于 pointwise widening integer MAC 的 accumulator：零初值可由
相同 typed operands 的 widening multiply 起链，去掉零载体和无效累加，后续加法保持不变。
循环内固定全域的 scalar dot 归约保留原 MAC 形式：其零种子可跨迭代复用，两种形式的
稳态乘积工作均为一条指令，不能仅用删除种子认定更优。保留这条依赖也避免向原生调度器
暴露逐个交错的 multiply/reduction，造成每个乘积都切换 SEW。尾部活动长度相关的零种子
仍可由同域 widening multiply 消除；显式长度和 undisturbed 合同不进入这项选择。

无符号 RVV 整数值除以或模一个直接标量、非零二次幂常量时，分别选择同 type 的
`shr(log2(divisor))` 或 `and(divisor-1)` 与对应 exact leaf；不改 Canonical IR。
lhs/result 的完整 type 必须相同，rhs 必须是同元素类型的直接整数常量，因而不存在
广播/validity 变化或有符号负数舍入歧义。零除数、非二次幂、动态除数及有符号除法不匹配。
同元素类型的有符号或无符号 wrap 乘法也可将单 bit 标量乘数改为左移，允许标量在任一侧。
左移保留相同位宽的低位结果，包括乘数只设置符号位的情况；不处理浮点、饱和或 widening
乘法。常量计算使用定宽 APInt，不靠宿主有符号移位；`weft-constant-arithmetic` 输出实际
选中的商、余数和乘积数量。emitter 沿用已选 shift/and 的拼写，资源由同一后续主链闭合。

已选 unsigned widening vector-scalar multiply 的结果若只被同宽右移消费，右移量是
输入 SEW 的直接整数常量，且右移也只被非饱和、rtz narrow 消费，narrow 的完整结果 type
恰好恢复原 vector type，则三步选择为单个 `rvv_multiply_high_scalar`。匹配只检查这条
固定 use-def 链；u8/u16 的 doubled-width unsigned product 不溢出，前置 wrap、后续
累加/归约与全部 axes 不变。emitter 只拼写该已选 leaf，资源由原有收尾重新核算。

同 block 的 unsigned `shr` 若只被非饱和、rtz 的二倍窄化消费，可选择
`rvv_narrow_shift_right`，精确拼写 `vnsrl.wv/wx`。输入与结果须有相同完整坐标和
validity，输入 SEW/LMUL 为结果的两倍；旧 shift 的 exact leaf 也须匹配该 operand form。
向量 shift amount 以显式 narrow 保留指令所读的低 `log2(input SEW)` 位，标量 amount
保持原输入元素类型。单个已物化 scalar 输入由既有 extract/splat 表达，amount 只允许
补入完整 singleton physical axes；不隐式改变其它 broadcast 或 register mapping。
融合留在原 shift 位置，不移动读取。相同 result type 的 scalar-splat/narrow-shift 对，
在输入与窄结果的 register groups 相等时，可将后一个纯 splat 提至其 numeric producer 后：
窗口最多 32 个操作，须跨同型 narrow-shift，且不能跨 region 或任何 effect。实际载体寿命
仍由后续资源主链核算。`weft-narrow-shift` 输出融合数和重排的 splat 数；emitter 不识别
producer 模式。每次融合消除一个已有 shift，不复制 shift producer，工作表有输入 DAG 上界。

单 slot 的已归约 i32 partial，若仅由一个 singleton i32 scale 相乘后立即提取为标量，
可选择已有 scalar mul，不再做 VL1 vector multiply。原 partial 必须来自完整一维 i16
carrier 的 widening reduction；以完整 i16 元素范围、lane 数和 scale 的静态整数范围
证明最终乘积始终在 i32 内。不改变 reduction、scale 的结合位置或数值宽度；仅将纯
partial extract 放到原 combine 处，系数仍在原消费点读取，不能把 deferred read 向后移。
singleton shaped scale 使用显式 index-zero extract，资源继续走同一收尾主链。

### `SelectRISCVScalarLoadPrimes`

该pass消费已经实例化的boolean physical parameter，并且只在`FinalizeRISCVLeaves`之后运行。
启用时，它在每个physical loop中沿单result、pure、同block use-def查找第一个送入
`rvv_partial_set/collect` 或 `rvv_finalize_widen_dot` 的 grouped/layered storage edge。合法edge必须已经具有完整window plan、exact
vector-load leaf；`rvv_replica_storage_load`还必须只有一个source window。pass把该op的leaf改为
对应的`scalar-prime` form，不改变type、SSA、control或storage geometry。启用但整个module没有
合法edge时当前candidate失败，而不是把参数静默当成零。
同一选择也覆盖送入 fragment MMA 的完整二至四 atom 连续 byte 供给；packing 与 full validity
必须由已有 fragment storage-load 合同证明。它读取整个供给的最后一个有效 byte，再发原有 loads。

这个leaf固定为同一raw window末端的一次有序scalar byte read加原vector load。它是有限的局部
memory spelling，不携带future-iteration、distance、buffer或pipeline schedule；若需要跨iteration
prefetch，必须先在RISC-V IR中建立真实producer/carry/control，不能扩展本pass暗做。

### `MaterializeRISCVReadSnapshots`

encoded `load` 的 field/projection 可以延迟实现，但不能跨越可能修改原 bytes 的写入。
该 pass 沿读取结果的 use-def 检查中间 effects、参数 alias set 和嵌套循环；存在干扰时，
在原读取位置插入显式 `local_alloc`，将它作为 `load.snapshot_storage` operand，选择
完整 record byte-copy leaf。之后的 field 投影读取这份私有存储，原 Value 的 Encoding、
shape、axes 和 Level 归属保持不变。私有 local writes 不与 pinned View 混为一类。

共同的 `Memory.h` 分析提供 storage provenance、局部别名边和读写干扰关系。快照、内存供应、
局部对象作用域及表示改写消费这些事实，不各自维护 operation 名单或根据 descriptor spelling
猜测底层存储。嵌套循环须包含较早迭代的写入；分支内读取只检查实际通向该 consumer 的分支
前缀。来源查询有界，无法闭合时保守报告可能干扰，不把查询失败解释成没有写入。

只读是 effect 事实，不是推测执行许可。循环外提还须证明读取所在循环至少执行一次，并保留
未知副作用与容量检查等执行边界；零次迭代不能因此产生新的访存。坐标构造的无内存副作用
也不使它获得通用 Pure/CSE 权限。

已知只执行一次的 physical loop 保留其原供应位置；外提不能减少动态执行次数，反而
会延长解码后数值的驻留。该选择不删除作者 Level 或改写其 birth/handoff。

快照要求连续、完整且非 interleaved 的 record 区间；未满足的干扰读取必须拒绝。
容量和 lifetime 由 `LocalType` 承载，资源 pass 核算分配；动态容量通过显式 size 和容量检查闭合。
没有干扰的读取不增加本地复制。final verifier 拒绝仍跨干扰写入但没有快照的 encoded load。
完整 pinned storage 的不重叠证明可以作为 `load.snapshot_borrow` SSA operand；verifier
同时检查实际干扰写入和读取结果的作用域，来源分析保留原存储与私有快照两种可能来源。
该证明只控制是否复制，不改变作者读取点、Value 生命周期或 ABI。预算内固定容量继续用
固定分配，不能为了可借用而改成逐 record 的 VLA；动态分配才按同一个谓词选择所需容量。
固定快照若直接供给向量读取，保留私有存储提供的无别名供给关系：C 工具链无法从条件
指针恢复完整的范围证明，借用会额外限制向量读取与写入之间的调度。此项原生成本尚不能
闭合时保留已有复制实现；仅有显式标量投影的固定快照才启用借用。分析沿共同存储别名边
检查实际 Read effect 和物理结果表示，不按 Encoding、算子或机器名称决定。
copy leaf 在 target 合法的 m1/m2/m4 中按精确 transfer 次数与 scratch 占用选择，并记录
全部所需参数。它完成显式 exact-window transfer，不把热路径交给带未知向量 clobber
的库函数调用。
已有 snapshot storage operand 的 load 保留这个已验证的 copy leaf；重放 memory planning
不能把它改回普通 load，清 resource marker 也不改变 snapshot 的归属和读取合同。
多维完整 record 的别名保护统一由这一 pass 实现；字段规划不再为相同读取另建逐字段
byte decode 快照。bit-level 非自然布局所需的显式字节投影仍由字段规划处理。

动态快照之后的有序消费者可移入现有 native helper，隔开动态栈分配与纯计算 frame。
分配、借用条件、copy 和原读取点保持在原作用域；消费者只移动一次，不复制 partial birth
或 buffer 身份。完整域与 masked 分支分别保留自身 traversal、captures 和 carries。
快照的两条来源都可证明与被检查 writer 不重叠：借用来源经过原范围检查，copy 来源是私有
allocation。该事实由共同存储分析提供给后续 alias 查询，而非直接从地址字符串加限定。
helper return 只在 payload-write 收集时区别于存储写入；跨控制边界移动仍受其退出 effect 限制。

### `CloseRISCVLeafResources`

从已选 op 的 typed operands/results 统一 leaf 的 operand/result resource fields，不冻结
kernel 的 final resource marker，不改变 leaf instruction 或表示；依赖当前 SSA 最后 use 的
broadcast 临时量同步重算。随后 CSE 与
dead-layout 清理可合并仅因旧资源字段不同而未合并的相同纯供应；真实 SSA live set、spill
和峰值仍在 `MaterializeRISCVResources` 中重新核算，不能在清理后复用旧统计。

### `FuseRISCVPhysicalIssueLoops`

仅消费 nested partial materializer 写在生成的 `scf.for` 上的 `weft.riscv.issue_window`
（axis、extent），不从作者循环或名称猜测 issue。候选必须具有相同边界、步长与属性，
两个平坦 body 只含纯操作或 Read effect；第二个循环的初值和外部 operand 必须支配第一个
循环，循环之间不能跨越副作用。每次只合并两个遍历，每个 kernel 最多检查 32 对候选；
合并结果不再参与扩大合并，候选标记在 pass 结束时消费掉。

候选副本先 CSE，再按 SSA 读取前沿安排独立消费者，避免机械展开后的共享供应跨越后续
issue 才被消费；每条原有累加依赖保持不变。仅当实际静态 Read 数减少、且原有
region-aware 资源分析的峰值不超过 target 寄存器预算时接受。候选数、共享 Read 数、
资源拒绝数与最大已选峰值写入候选诊断，并接入 pass statistics。这个估计不代替 effect/数值/type
合法性，也不保证 intrinsic C 后的 LLVM 寄存器分配没有 spill；有限物理绑定仍由 kernel
旁的配置提供并通过真实运行选择。最终资源 pass 重新核算实际选择的 Physical graph。

### `ContractRISCVIntegerPointwise`

同宽 unsigned RVV 的低位提取 `((x + c) >> s) & (2^k - 1)`，若通过有界
use-def 查询证明 `c` 是 `2^(s+k)` 的倍数，可只在该 consumer 上去掉 addend。
也识别等价的二次幂 `div/mod`，以便在最终 leaf 选择前合并索引供应。mask 保留，
所以原位宽的 wrap 不改变结果；`s+k` 不得超过元素宽度，全部 axes、layout 和
validity 保持相同。不穿过改变类型的数值操作，不移动任何读取。共享 add/shift 的
其它 consumers 仍使用原值；后续 CSE 和资源分析处理重复供应及实际驻留。

对同 block、单 use 的整数位字段拼接 `(low | (high << c)) ± bias`，先证明 `low` 是
至多 `c` 个 unsigned bits 的值，且 `0 < c < SEW`。窄 unsigned 值的数值 widening 可以提供
这项证明；未知位域不改写。所有参与运算的 vector type 相同，bias 与 multiplier 保持同一
元素宽度，不越过 region、write 或未知 effect。

非重叠位域的 OR 等于同宽加法，因此 pass 形成真实的 `low ± bias` 与
`rvv_integer_mac(high, 2^c, biased_low)`，以两条 RVV 指令代替 shift、OR、bias 三条指令。
模算术保持包括 overflow 在内的相同整数结果；scalar 常量查询不穿过改变位宽的 cast。
低位范围从数值 widening 的原始整数类型取得，不能因为结果已经扩展为 u8/u16 而丢失
原有位宽证明。OR 结果到 bias 之间允许一个单 use、同布局同位宽的 signedness cast；
此时在两个输入上显式保留该 bit-pattern reinterpret，再执行同宽模运算。改变位宽、
坐标或布局的 cast 不属于这一关系。
新增普通 binary 继续经过 operation selection 和 leaf finalization；MAC 的精度、布局与
资源由独立 verifier 核对。不根据 kernel 或存储格式名称匹配。

### `ContractRISCVFloatingPoint`

在资源冻结前将同 block、单 use 的直接浮点 `mul + add` 选择成 `fma` 局部 op。
两次运算的精度必须相同，不能绕过 cast/widen/narrow、跨 region 或 write/未知 effect。
当前支持 f32/f64 scalar，以及同布局 RVV vector/vector、vector/scalar；其它
形状继续使用原有两条运算。选择不依赖 kernel、量化格式或扩展名称。
向量结果使用 scalar addend 时，成本必须计入每个 packet 的广播和破坏式目的寄存器准备。
accumulator-tied FMA leaf 将输出与 vector accumulator 绑定；即使广播可共享，也仍需为后续 packet
或循环迭代复制该初值。因此 `splat/copy + FMA` 没有比原 `mul + .vf add/subtract`
减少 issue，且增加向量驻留；同成本保留原实现。不能把尚未物化的广播外提、不同的
寄存器绑定或仅有语义复用次数，当作已经减少的物理工作。
对完整 RVV carrier、标量乘数和标量 addend 的加法，可选择另一种精确 leaf `rvv.vfmadd.vf`：
只有向量乘数在这里最后使用、没有共享 storage identity，且同一完整 Value 有多个 carrier
parts，或直接 enclosing SCF loop 有至少两次已证明的静态迭代时，才物化显式 `rvv_splat`。
后一种情况还要求 addend 定义在该循环外，才能将广播放到循环前。选择的 `rvv.splat.shared`
leaf 将相同 scalar、SEW/LMUL 和完整 VL 的所有 parts 映射到一个只读 packet；emitter 按该决定
只生成一次广播，成本与驻留查询也只计这个 packet，而完整输出仍计全部 parts。
`vfmadd` 覆盖向量乘数，保留广播值，实际工作由每个 part 两条算术变为一次广播加每个 part
一条 FMA。tail carrier、减法及仍有消费者的乘数不使用这项选择，不改变作者 Level 或复用范围。
已有向量 accumulator 的融合继续进行；已经物化的 `fma` 不由这个隐式 pass 改写。
选择还要核对三个输入的共同驻留前沿，沿共享 register identity 去重并计入完整 time parts。
若这个必要输入集已经超过目标寄存器预算，保留原来的两条运算，使 addend 能在乘法输入死亡后
再 reload；不能先生成所有输入同时需要、后续 spill 无法修复的 FMA。
独立的 `fma` 保存 typed operands 和 exact leaf，资源 pass 重新核算实际活跃值；
它沿用原 pointwise 的逐 part 活动长度，没有 closed FMA issue 的调度屏障或 loop-unroll 限制。

### `SelectRISCVScalarPackets`

在资源冻结前，按消费者的物理表示闭合小型 scalar 供应。完整 natural 字段的 unit
windows 若只有一个同 block 的 RVV→scalar extract 消费者，可在原读取位置改为
`rvv_regular_repeat_scalar_load`。只接受 2–16 个 i8/i16/i32 元素，且原始 active lanes 小于
`VLEN × LMUL / SEW` 的实际载体容量；满载的 vector window 保留向量读取，避免仅凭
静态指令减少就增加 scalar memory issues。逐个 scalar coordinate
反查原 vector part/lane，并核对所有 window 恰好覆盖同一连续字节范围。多维结果保留
原 axes、shape 和 scalar part 顺序，不能把转化后的读取延迟到消费者处，也不能增加读取。
这消除只为标量消费者建立的 vector load、slide 和 extract，原整数计算与后续 reduction 不变。

若原窗口均已填满载体，pass 可以把多个相邻窗口合为一次完整 RVV 读取，继续由显式
extract 供应 scalar consumers。只有整个逻辑域恰好装满一个合法的至多 m1 carrier 时选择；
各轴改为 lane 映射，window/part/decode map 同步写回真实 load op。读取字节集合、base、
alignment 和读取位置保持一致；不会扩大读取、跨写入合并，也不增加 scalar memory issues。

同一 pass 还为连续 K 的 scalar tile load 选择已有 packet 表示：只在完整投影覆盖且
所有消费者构成一条有序 contraction carry 时冻结 packet，并核对 scalar 驻留上界。
这些选择都来自 storage、use-def 和坐标关系，不按算子或量化格式选择。

参考 Triton `CoalesceUtils.cpp:17-94` 按访问连续性、shape 和消费分布选择读取宽度；
Weft 的 CPU 消费者可以持有 scalar replicas，因而这里直接供应这些标量，不引入 SIMT
线程布局。实际读取与表示选择保存在 Physical IR，terminal 不重新分析消费者。

### `MaterializeRISCVResources`

资源冻结前，可提前准备同 block 的单消费者标量浮点乘法因子：限普通 scalar
`field/widen/cast/mul` 纯链，最多8个节点、向前最多64个操作；完整表达式及内部次序不变。
只能越过独立的 pure/read 操作，不能越过输入定义、region、write 或未知 effect，不能
把原字段读取向后移动。窗口须跨 reduction；RVV leaf 的 operand/result register groups
最大值作为有界 issue 估计，优先取能够覆盖该 scalar 链的最近窗口，以减少额外驻留。
窗口不能越过同时消费多个非常量整数 scalar 的 RVV op，避免准备字段地址/转换时延长
多系数 GPR live set；不能提供完整 scalar 链 issue 覆盖的窗口不选。
结果用既有 `register_materialize physical-share` 在选定点物化，terminal 不再决定调度。
每 block 最多4个准备操作，已有 scalar/ScalarTuple 浮点 physical-share 也计入预算，
清 resource marker 后重放不重复放大预算。`weft-scalar-prepare` 输出选中数和预计重叠的
register issues；估计不是周期预测，也不覆盖 effect、数值或最终资源合法性。

用 region-aware SSA live interval和 leaf temporary计算 vector/fragment peak，并把每个 leaf 的
operand/result per-issue resource groups写实。numeric RVV 的所有 time parts 进入实际驻留，
storage projection不计为numeric；显式零拷贝 alias 共用原值的完整 live interval。block argument
从 entry 即存活，loop init 只有整条 alias 链及底层 storage 都无其它 use 时才与 carry 合并计数。
spill 的字节数保存同一完整驻留值，不再次重复乘 time。首次核算前，可将同 block、single-use 的纯整数计算链
移到唯一 consumer 前：候选仅含 full RVV `binary/cast/widen/narrow`，最多32个节点，
输入边界仅接受常量或已物化的 numeric read；不移动读取点，也不穿过 deferred field/address。
只考虑跨 partial set 的最多256个操作窗口，窗口内不得有 region、write或未知 effect。
移动必须降低 `Σ resident_groups × live_interval_length` 且不增加该 block 的实际 peak；
否则恢复原位置。估计不替代合法性，候选数、选中数和驻留成本减量由
`weft-resource-sink` 诊断输出。该改写不复制计算、不改整数结合树、Level或storage归属。

fragment MMA 的 copy/reuse 在最终 use-def 图上明确选择。只有同 block 定义或当前 block
argument 的 accumulator 恰好只有该次 use 时才允许 reuse；其它情况保留 typed copy 资源。
该 policy 与按 atom 数计算的 scratch/copy 上界同时进入 leaf，不能由 emitter 看 C 变量名决定。
资源超限时，只对同一 block 内可合法保存的普通
`ValueType` 插入显式 local slot、`spill` 和各 use-site `reload`；fragment spill、跨 block spill
和任意 pure-producer rematerialization尚未实现时判当前 module非法。
pure 或仅有 Read effect 的 producer 均可提供被 spill 的数值 SSA；原 producer 保持在
原位置，只保存其已经产生的值并在 use-site reload，不复制读取，也不延迟读取。
encoded storage projection 本身不是 spill 对象。捕获索引 spill 后，同一 consumer 的相同
索引复用一次 reload；`field_read` 的纯地址投影在该读取点按它自己的索引 operands 重绑，
严格保留 selector、type 与来源关系，重新核算资源，不克隆数值 producer 或扩大读取字节。

若 CSE 共享已有 reload 后扩大驻留区间并再次超出预算，可以把该 reload 分到同一 block 的
各 consumer 前；同一 consumer 的重复 operands 仍共享一次 reload。此改写只接受来自
私有 allocation、由同 block 中唯一且先于 reload 的 spill 写入、其余用户均为 reload 的
slot；`reload_slice` 与保持同一载体的纯 reshape 同样沿实际 consumer 分开读取。
该值须在超限点仅驻留而非必须作为输入存活。不新增 slot 或重新执行原数值 producer；
使用点本身所需资源超限不能借此通过。改写仍受初始 SSA 候选数量的尝试上限约束并重新核算峰值。

kernel 创建时 `resources_materialized=false`；只有资源分析、必要 spill 与峰值汇总全部闭合后，
本 pass 才把它设为 `true`。初始化为零的 peak 不是闭合证据。layout canonicalization 仅在该
marker 为 `false` 时允许改变 producer lifetime 的 backward rematerialization；marker 为
`true` 后只能做不改变资源合同的等价 conversion CSE。本 pass 对已经闭合的 kernel 幂等跳过。

动态 local object 必须紧邻一个 typed `local_capacity_guard`。guard 显式消费本次请求字节数
和同一词法作用域路径中仍存活的动态字节数（含对齐余量），在扣除静态存储后检查共同
target budget，不给每个对象平均分配配额。嵌套循环每轮结束、分支退出时恢复父作用域
计数；while 的 before/after 属于同一迭代作用域，计数通过 condition/after 参数显式传递。
最终 verifier 从 allocation、作用域和 SSA 加法链重新核对 live bytes，kernel resource
summary 保存由这些 guards 保证的上界。静态存储保留量取词法作用域的同时存活峰值，
包含对齐余量；私有 storage alias 不得从其分配作用域逃逸。缺少闭合上界时拒绝生成 C VLA。

私有 Value 的一次已选 RVV packet 达到 16 bytes 时，初始 local allocation 至少沿用
RISC-V ABI 已有的 16-byte 栈边界；不增加动态栈重对齐。其余更高对齐由下述 pass 选择，
二者的字节与 padding 都由同一资源分析核算。

### `AlignRISCVLocalStorage`

根据已经选择的连续 RVV 访存宽度，提高静态 `physical-share` 私有缓冲区的对齐。
有显式 cache 模型时以 cache line 为上限；没有 cache 模型时使用目标 VLEN 的字节宽度，
不把缺少 cache 参数解释为只能使用 scalar 对齐。实际选择不超过访问宽度，且必须是
二次幂。作者对象、外部 workspace、动态容量和 spill slot 保留各自的存储合同。
每次提高对齐前重新计算同时存活的局部字节数与 padding；存在动态对象、寄存器超限或
栈预算不足时不进行这一可选优化。对齐写入 `LocalType`，emitter 只输出其 `_Alignas`。

### `VerifyFinalRISCV`

terminal eligibility 与 required-leaf 由 `RISCVOps.td` 的 `WeftRISCVTerminal` /
`WeftRISCVLeaf` traits 唯一定义；final verifier 不另维护 op 名单。terminal dispatcher
只登记实际实现，不能用自身分支补齐未选择的 leaf。

最终 verifier 拒绝：残留 canonical op、非 terminal RISC-V op、`unassigned` layout/access、
`implementation`、未展开 schedule、未选 leaf、丢失 Level birth/handoff、绕过 runtime
`memory_view` 的 ABI edge、`resources_materialized=false`，以及超出 target 的
register/fragment/local-storage 使用。
寄存器 summary 还必须与重新运行的同一 SSA 驻留分析相符，不能只凭旧统计低于预算通过。

operation/type verifier继续检查 logical domain、axis projection、Encoding field、conversion、
window、fragment role/packing和 exact leaf之间的局部合同；final verifier还把descriptor
storage facts与Encoding declaration、leaf widening/memory要求与target capability逐项核对。
成功只说明这份 RISC-V module满足
当前已实现的 terminal op 集合；没有实现的 spill、pipeline或extension结构必须在前序 pass
明确失败。

## 3. 可观察性

每个 pass 的输入和输出都是 program。用 MLIR pass instrumentation dump 同一 module时，应
直接看到 type/layout变化、`convert_layout` 插入或删除、memory attributes、loop/window/
fragment rewrite、pipeline expansion与spill/reload。只打印 analysis table或 assignment
dictionary不能证明程序已经物理化。

独立 parse/verify 和指定 pass/CSE 的二次 diff=0 只说明该样本在该流程上稳定，不证明
全部 pass 正确或幂等。尤其 final resource marker 会跳过 layout canonicalization 的主要
rematerialization 分支。机械重放的完整解释与真机验收边界见
[物理优化方法](optimization-principles.md#12-机械验收的证明范围)。

## 4. 与 Triton/TileLang 的机制关系

下列路径相对于仓库根目录位于同级reference checkout `../ref/`。Triton 的 layout encoding、`ttg.convert_layout`、Coalesce、AccelerateMatmul、
RemoveLayoutConversions 与 Pipeline同样在 TTGIR 上改写真实程序：

- `../ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:71`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:441`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp:42`
- `../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp:51`

TileLang也把layout、buffer与pipeline结果落到真实block/loop，再由LowerTileOp和
InjectSoftwarePipeline重写body：
`../ref/tilelang/src/transform/layout_inference/layout_inference.cc:1196`、
`lower_tile_op.cc:1080`、`inject_pipeline.cc:3608`。

Weft复用的是 typed representation、explicit conversion 与 real rewrite；逻辑值不因此获得
thread/warp/CTA ownership。
