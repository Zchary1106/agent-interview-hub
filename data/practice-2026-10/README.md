# 2026-10 主题练习数据

- [`questions.json`](questions.json)：156 道题的唯一编辑输入，保留原题 ID、题干、主题、来源、日期、答案、得分点、追问、易错点及代码入口。新增 `position`、`document_path` 和算法 `code_file` 用于目录映射。
- [`answer-references.json`](answer-references.json)：26 条答案机制资料；支持技术解释，不证明公司曾问过某题。
- 面经来源只维护在 [`../interviews.json`](../interviews.json)，本批由 `collection_batch == "2026-10-22"` 选出 63 条，不再复制第二份来源目录。
- [阅读索引](../../通用知识/2026-10-Agent面试题索引.md)及分主题文档由同步器生成。

## 真实性与日期

所有题目均按公开面经报道主题改写，非公司官方真题；参考答案为编辑综合草稿，非公司标准答案，未经独立专家逐题复核。

- `curated_at: 2026-10-22`：来源与题目快照日期。
- `answer_date: 2026-10-23`：补充参考答案的日期，不是再次抓取来源的日期。
- `authentic_company_question_verified: false` / `question_text_verified_against_primary: false` 必须保留。
- B 表示曾定位公开原帖，可能仅预览或最近一次抓取不可读；C 表示仅聚合线索。两个等级都不代表作者亲历或逐题真实。
- 来源迁入时统一保守标为 `source_type: aggregation`、`confidence: low`。`published_at` 优先使用有记录的原帖发帖日，否则仅保留月份；原始占位日、报道面试日和归属疑点另存，不混同。
- 不读取或复制付费全文。不把聚合线索计为独立面试频率，不把本批加入 `question_signals.json` 的高频排序。

## 编辑与复现

使用 Python 3.11+，从仓库根目录运行：

```bash
# 只编辑上述 JSON 输入，再重新生成主题文档和交互题库分组
python3 scripts/sync_practice_2026_10.py
python3 scripts/sync_practice_2026_10.py --check
python3 scripts/validate_data.py
python3 scripts/validate_practice_2026_10.py
python3 -m pip install -r requirements.txt 'pytest>=8,<10'
python3 -m pytest -q 面试算法题/python/test_algorithm_answers.py
python3 scripts/build_site.py
```

同步器只替换根目录 `data.json` 中的「2026-10 跨公司主题练习（非真题）」分组，不覆盖旧公司题目。站点构建沿用现有 Markdown 发现机制和算法目录复制机制，不更改页面框架。不要直接修改生成后的主题 Markdown；CI 会检查漂移。
