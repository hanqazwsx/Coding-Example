coding_agent — 后端工程自动化智能编码代理
coding_agent 为自研自动化编码调度框架，内置有限状态机推理、工具调用、技能路由、层级上下文压缩、长短时记忆、多智能体集群编排与全链路安全审计能力，专为笔记本运行环境优化资源占用。
核心能力一览
表格
能力模块	功能说明
有限状态机推理引擎	六阶段认知执行闭环：初始化 → 思考决策 → 执行动作 → 结果观测 → 复盘反思 → 任务完成
工具调用能力	基于 Pydantic 完成参数校验的工具注册中心，统一采用ToolResult结果封装格式
语义技能路由	BM25 稀疏检索 + Chroma 稠密向量检索混合方案，实现用户意图与业务技能精准匹配
上下文压缩	滑动窗口机制结合大模型摘要归纳，高效缩减上下文 Token 用量
静态内容缓存	自研内存缓存方案，弥补 DeepSeek 接口不支持提示词缓存的短板
短时记忆	先进先出的定长双端队列，充当会话工作内存
长时记忆	Chroma 向量数据库，跨会话持久存储经过压缩的历史执行经验
多智能体编排	主从架构分叉合并调度，依托线程池ThreadPoolExecutor实现并发执行
安全审计体系	注入攻击过滤、Shell 命令沙箱、命令白名单管控、全流程审计留痕
全链路链路追踪	针对状态流转、工具调用、路由匹配全流程耗时埋点统计
技术栈
开发语言：Python 3.10 及以上版本
核心依赖：langchain、langchain-openai、pydantic、tenacity 重试组件
向量数据库：chromadb（支持持久化存储 / 纯内存运行两种模式）
检索方案：rank-bm25（稀疏关键词）+ chromadb（稠密向量）混合检索
向量嵌入模型：sentence-transformers 本地模型，默认权重：all-MiniLM-L6-v2
大模型后端：DeepSeek 接口deepseek-chat，兼容 OpenAI 调用格式
运行架构：全同步架构，基于线程实现并行，完美适配 Jupyter 笔记本环境
项目目录结构
plaintext
coding_agent/
├── __init__.py              # 包版本与项目说明文档
├── config.py                # 全局配置单例，基于.env环境文件加载参数
├── main.py                  # 程序入口：演示模式/交互模式/评测模式
├── requirements.txt         # Python依赖清单
├── .env.example             # 环境变量配置模板
├── setup.bat                # Windows一键环境部署脚本
├── setup.sh                 # Linux/Mac一键环境部署脚本
│
├── core/                    # 核心引擎模块
│   ├── fsm.py               # 有限状态机实现（6种状态，可自定义状态流转规则）
│   ├── query_loop.py        # 大模型推理主循环，内置tenacity失败重试机制
│   ├── compressor.py        # 层级上下文压缩器（滑动窗口+智能摘要）
│   └── cache_manager.py     # 静态内容缓存管理器（系统提示词、工具定义缓存）
│
├── tools/                   # 工具调用层
│   ├── schema.py            # 工具规格ToolSpec、统一返回体ToolResult的Pydantic模型
│   ├── registry.py          # 单例工具注册中心，支持增删改查，内置3款基础工具
│   └── executor.py          # 参数校验、工具执行、执行前后安全钩子逻辑
│
├── skills/                  # 技能目录与意图路由
│   ├── catalog.py           # 三级技能目录：领域 > 能力 > 具体技能（共8项技能）
│   └── router.py            # BM25+Chroma混合语义意图路由模块
│
├── memory/                  # 记忆系统
│   ├── short_term.py        # 定长FIFO双端队列短时消息缓存
│   ├── long_term.py         # Chroma向量库，用于历史经验召回
│   └── reflection.py        # 基于DeepSeek的复盘流水线，复盘结果写入长时记忆
│
├── agents/                  # 多智能体编排
│   ├── orchestrator.py      # 主调度智能体：任务拆解 + 分叉合并任务分发
│   └── worker.py            # 工作子智能体：独立执行子任务
│
├── security/                # 安全防护层
│   ├── filter.py            # 输入注入检测（命令注入、路径穿越攻击识别）
│   ├── sandbox.py           # Shell沙箱（命令白名单、执行超时、工作目录锁定）
│   └── audit.py             # 追加式审计日志，JSONL格式持久化记录
│
└── monitoring/              # 运行监控模块
    ├── tracer.py            # 全链路追踪（状态跳转、工具调用、路由全流程耗时）
    └── eval.py              # 任务评测套件，预置4项测试任务，自动输出评测指标
安装部署教程
前置要求
Python 3.10 及以上版本
pip 包管理工具
步骤 1：创建虚拟环境
Windows 系统：
batch
python -m venv venv
call venv\Scripts\activate.bat
Linux / Mac 系统：
bash
运行
python3 -m venv venv
source venv/bin/activate
步骤 2：安装项目依赖
bash
运行
pip install --upgrade pip
pip install -r coding_agent/requirements.txt
pip install chromadb sentence-transformers
也可直接调用一键部署脚本完成安装：
bash
运行
# Windows执行
coding_agent\setup.bat

# Linux/Mac执行
bash coding_agent/setup.sh
步骤 3：配置环境变量
bash
运行
cp coding_agent/.env.example coding_agent/.env
# 编辑.env文件，填入你的DeepSeek密钥
.env配置示例：
ini
DEEPSEEK_API_KEY=sk-你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
快速上手
演示模式（无需 API 密钥）
内置 3 套预设演示场景，完整展示有限状态机流转、技能路由、工具调用全流程：
bash
运行
cd d:/MyTest
python -m coding_agent.main demo
交互模式（需配置 DeepSeek 密钥）
完整大模型驱动的人机对话交互：
bash
运行
python -m coding_agent.main interactive
交互示例：
plaintext
用户：读取config.py文件并总结文件内容
代理：即将读取config.py并为你整理内容摘要
  【技能路由匹配：代码审阅】
  【工具调用：读取文件 执行成功】

用户：新建用户认证接口
代理：将为你完成认证接口开发
  【技能路由匹配：接口创建】
  【工具调用：写入文件 执行成功】
评测模式
自动运行 4 项标准测试任务，输出完整评测指标报告：
bash
运行
python -m coding_agent.monitoring.eval
评测输出样例：
plaintext
==================================================
评测报告
==================================================
  任务完成数：    4/4
  任务完成率：     100.0%
  工具调用成功率：  100.0%
  平均交互轮次：  3.2
  单轮平均耗时：  40毫秒
  总耗时：        161毫秒
==================================================
核心模块详解
1. 有限状态机（FSM）—— core/fsm.py
固定六阶段认知循环流转逻辑：
plaintext
初始化 → 思考决策 → 执行动作 → 结果观测 → 复盘反思 → 任务完成
                 ↑←←←←←←←←←←←←←←←←←←←←←←←←←|
每条状态转移规则包含当前状态、目标状态、执行判断条件（可调用函数）
FSM.transition(context) 读取上下文，遍历匹配规则，自动执行首个符合条件的状态跳转
全局上下文字典_context全程保存流转数据，调用reset()可清空上下文，隔离不同任务数据
2. 工具调用体系 — tools/
注册中心：单例模式管理(工具定义, 执行函数)映射关系，内置文件读取、文件写入、Shell 命令执行三款工具
数据模型：ToolSpec定义工具入参（名称、类型、必填项）；ToolResult为统一返回结构体，包含执行状态、结果、报错信息、耗时、工具名、扩展元数据
执行器：严格按照规格校验入参，执行前后挂载安全校验钩子，自动捕获运行异常，标准化返回ToolResult对象
3. 技能路由模块 — skills/
技能目录：三级层级架构，包含 4 大业务领域（后端、前端、数据处理、运维）、6 项业务能力、8 个落地技能
路由算法（混合检索）
BM25 稀疏检索：关键词匹配，权重占比 0.4
Chroma 稠密向量检索：基于 sentence-transformers 语义相似度匹配，权重占比 0.6
分数加权融合，输出最优匹配技能与推荐入参
4. 上下文压缩 — core/compressor.py
滑动窗口默认保留最近 6 轮完整对话原文
更早的历史对话交由 DeepSeek 大模型进行精简摘要
压缩后的摘要以系统角色消息注入上下文，大幅降低 Token 消耗
5. 记忆系统 — memory/
短时记忆：定长双端队列deque(maxlen=50)，先进先出自动淘汰旧数据，支持读取最近 N 条会话记录
长时记忆：Chroma 向量数据库，持久化存储向量化的复盘经验
自动复盘机制：短时消息达到 20 条阈值时，调用大模型生成结构化复盘内容（目标、执行动作、执行结果、经验总结），存入长时记忆；短时记忆仅保留最新 4 条内容保证会话连贯，其余清空
6. 多智能体集群 — agents/
主调度智能体：调用 DeepSeek 将高层需求拆解为 2~4 个子任务（JSON 结构化输出），通过线程池（最大 4 并发）分发任务
工作子智能体：每个子智能体独立配备专属有限状态机、推理循环、短时记忆，依托as_completed()收集所有子任务执行结果
7. 安全防护模块 — security/
输入过滤器：正则规则识别命令注入（;、|、反引号、命令替换符）、路径穿越（../）攻击
Shell 沙箱：配置命令白名单（默认支持 ls、cat、grep、find、head、tail、wc、echo 等），封禁 curl、wget、nc、ssh 等网络命令；严格校验;/&&/||/|拼接的所有子命令；强制锁定运行目录为项目根路径，命令最长执行时限 30 秒
审计日志器：JSONL 格式追加写入日志，完整记录用户输入、工具调用、状态跳转、安全拦截事件，每积累 100 条日志自动落盘刷新
8. 运行监控 — monitoring/
追踪器 Tracer：通过上下文管理器 + 装饰器完成耗时埋点，日志按日期生成 JSONL 文件，可选同步存入 Chroma 的trace_logs追踪数据表
评测器 Evaluator：运行预置测试用例，统计任务完成率、工具调用成功率、平均交互轮次、平均耗时，支持导出 JSON 评测报告
配置参数总表
环境变量配置（.env）
表格
变量名	默认值	说明
DEEPSEEK_API_KEY	无	DeepSeek 接口密钥，大模型交互必备
DEEPSEEK_BASE_URL	https://api.deepseek.com/v1	兼容 OpenAI 格式的接口地址
DEEPSEEK_MODEL	deepseek-chat	调用的模型名称
CHROMA_PERSIST_DIR	./chroma_db	Chroma 向量库持久化存储路径
LOG_LEVEL	INFO	日志等级（DEBUG/INFO/WARNING/ERROR）
程序参数配置（config.py）
表格
参数名	默认值	说明
llm_temperature	0.3	大模型生成随机性温度值
llm_max_tokens	4096	单轮大模型响应最大 Token 数
fsm_max_iterations	15	单个任务有限状态机最大循环次数
query_loop_max_retries	3	LLM 接口调用最大重试次数（tenacity）
compressor_keep_last_n	6	滑动窗口保留原始对话轮数
short_term_maxlen	50	短时记忆最大消息条数
reflection_threshold	20	触发自动复盘的短时消息阈值
orchestrator_max_workers	4	多智能体最大并发工作线程数
tool_whitelist	[read_file, write_file, shell_exec]	允许调用的工具白名单
tool_blacklist	[]	禁用工具黑名单
shell_allowed_commands	[ls, cat, grep,...]	Shell 命令执行白名单
现存局限与注意事项
笔记本环境资源优化策略
线程池最大并发限制为 4 线程，避免抢占笔记本运行资源
Chroma 支持持久化磁盘存储、纯内存运行两种模式，内存模式可规避磁盘 IO 开销
短时记忆容量上限 50 条，长时记忆仅存储精简复盘内容，严控内存占用
所有大模型调用配置指数退避重试，最多重试 3 次，提升接口稳定性
DeepSeek 提示词缓存替代方案
DeepSeek 接口暂不支持 Anthropic 风格的cache_control缓存请求头，项目通过cache_manager.py自研内存缓存方案：
系统提示词、工具定义等静态内容一次性构建并缓存至字典
get_or_build(key, builder_fn)方法实现缓存命中复用，避免重复构建内容
可查看缓存命中 / 未命中统计数据
支持基于 SHA-256 内容指纹，手动触发缓存失效
Shell 沙箱防绕过机制
security/sandbox.py沙箱针对各类攻击做完整防护：
表格
攻击方式	防护手段		
分号;命令拼接	拆分所有子命令逐一校验		
&&/`		` 逻辑拼接命令	拆分所有子命令逐一校验
管道符 `	` 串联高危命令	管道全部子命令逐个校验	
执行网络请求命令	关键词匹配拦截封禁命令		
目录路径越权跳转	强制锁定工作目录为项目根目录		
进程卡死长时间占用	强制 30 秒执行超时终止		
编码混淆命令	暂未实现检测，为后续迭代方向		
演示模式限制
纯 FSM 演示模式通过静态映射表_skill_to_tool_map关联技能与工具，不会接入大模型生成入参，因此工具调用会触发参数校验报错，属于正常现象。演示仅用于展示状态流转、技能路由、安全钩子的执行流程，无需调用大模型密钥。
完整运行示例
下方为交互会话完整链路日志，完整展示全流程执行逻辑：
plaintext
用户：列出项目内所有Python文件

  【安全过滤器】注入检测：校验通过
  【技能路由匹配】匹配技能：编写Dockerfile，匹配得分0.807（当前最优匹配项）
  【有限状态机流转】初始化 → 思考决策 → 执行动作 → 结果观测 → 复盘反思 → 任务完成
  【工具执行器】执行shell_exec("ls *.py") → 执行成功（耗时57毫秒）
  【审计日志】本次工具调用记录编号#3
  【链路追踪】状态流转共计5轮迭代，15次状态切换
  【记忆状态】短时记忆：2条消息 | 长时记忆：暂无历史经验

代理：共检索到8个Python文件，明细如下：
  - main.py：程序入口
  - config.py：全局配置文件
  - core/fsm.py：有限状态机核心引擎
  ...
代码逻辑审计报告（问题均已修复）
完成全部 29 个源码文件全量审计，问题清单与整改结果如下：
表格
序号	目标文件	问题描述	风险等级	处理状态
1	core/fsm.py	FSM 重置方法未清空上下文字典，造成跨任务数据污染	高危	已修复
2	security/sandbox.py	命令可通过;/&&拼接绕过白名单校验	高危	已修复
3	security/filter.py	单分号;无法被注入规则识别	高危	已修复
4	agents/worker.py	@lc_tool装饰器封装工具逻辑失效	高危	已修复
5	main.py	FSM 演示模式绕过 Shell 沙箱安全校验	中危	已修复
6	monitoring/eval.py、main.py	Windows 系统 CP-437/1252 编码下特殊 Unicode 字符程序崩溃	中危	已修复
7	main.py	技能名称与注册工具名称无法对应匹配	中危	已修复
8	多文件多处	存在大量未使用导入包（State、search_skills、json、builtins 等）	低危	已修复
9	tools/schema.py	to_dict()序列化方法丢失 metadata 元数据字段	低危	已修复
10	config.py	模块顶层执行日志 basicConfig，产生非预期全局副作用	低危	已修复
其余模块无异常：同步 / 异步架构兼容性、DeepSeek 接口报文格式、循环依赖问题均检测正常。
