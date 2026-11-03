# 第一周周报
> 省流:
> - 学习使用API
> - 学习markdown
> - 抖音调研
> - 华社合伙人萌新成长秘籍
# 主要学习内容
- ## 在Python里装学习用API
  - 目标:一条命令跑出模型回答 + 知道这次花了多少钱 + git 里有第一个存档且不含密钥
  - 装工具,配身份:
    - 1.winget install --id Git.Git -e 失败（连不上 GitHub）→ 改用清华镜像手动装 Git-2.56.0.4-64-bit.exe
      > https://mirrors.tuna.tsinghua.edu.cn/github-release/git-for-windows/git/Git%20for%20Windows%20v2.56.0.windows.4/Git-2.56.0.4-64-bit.exe
    - 2.git安装向导里：默认分支选 main；PATH 选“Git from the command line and also from 3rd-party software”
    - 3.配身份:git config --global user.name / user.email
      > git config --global user.name  "你的名字"
      > 
      > git config --global user.email "你的邮箱"
      > 
      > 做完的标志：git --version 有输出、git config --list 里能看到你的名字和邮箱。
    - 4.GitHub setting - emile - Keep my email addresses private ，并把本地 user.email 换成 noreply 地址（避免手机号邮箱进提交记录)
  - 建项目 + venv + 依赖:
    - 1.在 PyCharm 项目 Learn 里建虚拟环境 .venv
    - 2.换清华源 python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    - 3.python -m pip install openai python-dotenv
    - 4.用 python -c "import sys; print(sys.prefix)" 确认当前解释器来自 .venv
  - 拿 key + 跑通:
    - 选阿里云百炼，模型 qwen-plus
    - Copy-Item .env.example .env → 记事本填 API_KEY / BASE_URL / MODEL
    - python hello.py → 模型回答 + token + 耗时 + 花费，并自动写 usage.csv
    - python extract.py → 两句输入抽成 JSON，缺时间的一例正确填了 null

  - 工程习惯（额外收获）
    - 记账逻辑抽成公共模块 usage_log.py，脚本 import 即自动记账（原先是写死在 hello.py 里，导致 extract.py 漏记）
    - 新增 report.py：汇总总次数、token、花费、按模型、按天
    - 删掉手写的 账本.md，数字交给 usage.csv，结论写进 README.md 的观察日志
    - 脚本输出全部改成 GBK 安全字符（去掉 ¥ 和 emoji），否则中文终端会 UnicodeEncodeError
  > ## 必须会的 6 件事
  > cd 切换目录，以及“相对路径是相对当前目录”的概念
  > 
  > python -m pip ...（避免装错解释器）
  > 
  > venv 是什么、怎么激活（提示符出现 (.venv)）
  > 
  > .env 放密钥、.gitignore 把 .env 挡住
  > 
  >  git status / add / commit / log 四条
  > 
  >  一次 API 调用的三件套：messages、usage、花费换算
  > ## 其它
  > winget 报 InternetOpenUrl() failed 0x80072efd	连不上 GitHub 下载源	改用清华镜像手动下载安装包
  > 
  > 终端卡在 (END)，像“输出重复了很多次”	git config --list 交给分页器 less 显示	按 q 退出；以后用 --no-pager
  > 
  > 命令输出和输入混在一行	上一条输出还没刷完就敲了下一条	等提示符回来再敲；必要时 cls(清屏)
  > 
  > 将终端指向目标文件夹:右键-打开于-终端
  > 
  > 输出了奇怪的中文（鏈▶I构浠朵负）	UTF-8 文件被 GBK 终端/编辑器解码
  > 
  > 脚本打印 ¥ / emoji 时崩	GBK 编码里没有 ¥、✅	输出改成中文 + ASCII
  > 
  > python -m pip install -U pip可将pip更新的最新版本

---
- ## 学习使用markdown(写周报刚学的)
  - 标题:字号变大、加粗	# 大标题、## 小标题
  - 列表:圆点列表	行首 - 内容	         
  - 编号:1. 2. 3.	行首 1. 内容	          
  - 加粗:重点	**重点**	            
  - 链接:	[文字](网址)	          
  - 分割线:一条横线	---	                  
  - 引用:左侧竖线的引言块	> 一句话

---
# 其它内容
- ## 抖音调研
  - 主要任务
    > 收集:直播间背景图,直播间数据,小黄车数据,特色视频
    > 
    > 填写:配置数据,粉丝数改变,特殊情况,姓名,地区,UID
    >
    > 提交:特色视频,周三直播间背景图,周六全部直播间背景图以及数据截图
    >
    > 修改数据:周四和周六
  - 需要注意
    > 文字识别易将0(零)识别成O(英文大写字母O)
    >
    > 钛钽品牌名别写错
    >
    > 详细配置要写全,包括品牌

---
- ## 萌新成长秘籍
  > 这个是真懒得写了(已力竭)
  >
  > <img width="730" height="282" alt="image" src="https://github.com/user-attachments/assets/e60d9c64-2d7f-448b-aa25-2de51dd81ba4" />


 
