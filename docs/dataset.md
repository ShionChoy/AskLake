# AskLake 数据集

AskLake 当前使用两类数据：IMDb 非商用结构化数据，以及构建图谱时从英文 Wikipedia
实时获取的剧情文本。数据集专属的连接器与配置统一位于 `datasets/imdb/`，引擎层不硬编码
IMDb 字段。

## 数据流

```text
IMDb 官方 TSV.gz
  └─ scripts/download_data.sh
       └─ data/imdb/raw/                 原始数据，Git 忽略
            └─ make build-imdb
                 └─ data/imdb/parquet/  DuckDB 查询的工作集，Git 忽略

IMDb Parquet + Wikipedia API
  └─ make build-graph
       └─ data/imdb/graph/triples.jsonl 图谱持久化文件，Git 忽略
```

`make build-imdb` 默认只保留投票数达到阈值的电影，以适配 16 GB 开发机。可通过
`MIN_VOTES` 调整工作集；`make build-imdb-full` 会生成电影、电视剧和电视电影的完整工作集。

## IMDb 结构化数据

下载脚本从 `https://datasets.imdbws.com/` 获取以下文件：

- `title.basics.tsv.gz`
- `title.ratings.tsv.gz`
- `title.crew.tsv.gz`
- `name.basics.tsv.gz`
- `title.principals.tsv.gz`
- `title.akas.tsv.gz`
- `title.episode.tsv.gz`

构建后的主要关系是标题、评分、演职人员、姓名和主创信息。`tconst` 是标题标识，`nconst`
是人员标识。实际语义描述、同义词和 few-shot 示例见 `datasets/imdb/semantic.yaml`。

IMDb 数据仅允许个人、非商业用途，不允许随项目重新分发。因此 `data/` 始终由 Git 忽略，
仓库只保留下载和转换代码。使用条款以
[IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/) 为准。

## Wikipedia 图谱文本

`make build-graph` 先从 IMDb Parquet 确定热门影片，再通过 Wikidata 的 P345 IMDb ID 映射
解析英文 Wikipedia 条目并获取剧情段落。图谱由两部分组成：

- 从 IMDb 确定性生成的类型、年份、导演、演员和角色关系；
- 由 LLM 从当前 Wikipedia 剧情中提取的主题与场景关系。

图谱本体位于 `datasets/imdb/graph/ontology.yaml`。默认由进程内图存储加载；也可加载到用户
自行管理的 Neo4j 服务。Wikipedia 文本采用 CC BY-SA 4.0，构建产物应保留来源引用并遵守
署名和相同方式共享要求。

## 合成 CRM 评测集

`datasets/crm/` 是一个确定性生成的第二数据集，用于验证同一引擎能否仅替换数据配置后
工作。运行 `make build-crm` 会在 `data/crm/parquet/` 生成客户、订阅、工单和地区表；
`make eval-real-crm` 使用对应 gold set。它属于评测能力，不是产品演示代码。

## 数据安全

- 不提交 `data/`、Parquet、DuckDB 文件或抓取到的 Wikipedia 文本。
- 自然语言提问只把 schema、语义上下文和必要的候选值发给 LLM；原始表数据留在本机。
- 发布或共享图谱构建结果前，分别核对 IMDb 非商业条款和 Wikipedia CC BY-SA 要求。

## 数据分类与执行策略

治理配置位于 `datasets/imdb/governance.yaml`，采用互相独立的五类标签：保密性、许可、
个人信息、可信度和内容属性。IMDb 影片表属于 `public_source + imdb_noncommercial`，并不等于
可自由再分发；人物字段额外标记为 `public_personal`；`isAdult` 是内容标签；Wikipedia 关系
保留 `wikipedia_cc_by_sa_4`；LLM 提取的主题和场景标记为 `llm_inferred` 并强制引用来源。

`numVotes` 只用于评估样本质量和排序，不再决定访问权限。策略最终落实为可执行规则：角色
允许的操作/表/列/图关系、成人内容行过滤、个人字段遮罩、SQL 复杂度和结果上限、批量导出
限制、来源引用及许可提示。策略或视图构建失败会中止启动，不会回退为无过滤访问。
