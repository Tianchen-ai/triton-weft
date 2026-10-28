# 维护双语工具书

工具书采用 MkDocs Material 与静态语言文件。
英文使用 `topic.md`，对应中文使用 `topic.zh.md`，
共享代码、图片和对应导航结构。

## 构建与预览

在仓库根目录运行：

```sh
python -m pip install -r third_party/weft/requirements-docs.txt
python -m mkdocs build --strict -f mkdocs.yml --site-dir /tmp/weft-handbook
python -m mkdocs serve -f mkdocs.yml
```

英文位于站点根路径，中文位于 `/zh/`，
语言选择器连接对应章节，strict build 检查内部页面与锚点。

## 同步更新两种版本

1. 修改两种语言，对应同一行为和代码。
2. 内部链接使用中性文件名，如 `../compiler/runtime.md`，由本地化插件解析当前语言。
3. 工具书外的仓库文件使用源码链接。
4. 增加章节时更新 `mkdocs.yml` 中两种语言的导航。
5. 打开渲染页面，检查代码、表格、图片与窄屏布局。

直接说明已有行为，将安装条件和测量设置放在对应命令或结果旁边。

## 图片与规格

共享资产位于 `docs/weft/assets`，论文图片提供双语图注，
并在[图片来源](../experiments/results.md)登记。
公开测量 CSV 位于 Bench 的 `docs/paper`，
执行输出仍放入 Bench 忽略跟踪的 `output`。

详细语义和 pass 合同位于 `docs/specification`，
工具书讲解用法与结构。确认的语义修改同步更新对应合同和实现。

## 发布

`Build Triton and Weft` 执行双语 strict build，
默认分支的 handbook job 通过 GitHub Pages 发布至
`https://tianchen-ai.github.io/triton-weft/`。
