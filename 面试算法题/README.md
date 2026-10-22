# 面试算法题

本目录存放可独立打开的交互式算法题图谱。

- `index.html`：按公司筛选和搜索近期 Agent / 大模型相关面经算法题；每道题的 Java 参考解法默认折叠。
- [`AI模型与Agent手撕题.md`](AI模型与Agent手撕题.md)：Softmax、Attention、LoRA、向量检索、RRF、Tool 校验、ReAct Loop 等 AI 工程手撕题。
- [`2026-10 算法与手写面试题及参考答案`](2026-10-算法与手写面试题与参考答案.md)：15 道主题改写练习，附 [Python 实现与 40 项离线测试](python/README.md)，非公司官方真题。

本地可直接用浏览器打开 `index.html`。执行 `python3 scripts/build_site.py` 时，构建脚本会将本目录复制到 `dist/面试算法题/`，GitHub Pages 可通过 `/面试算法题/` 访问该页面。
