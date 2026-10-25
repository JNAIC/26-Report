# aiuwei_week01
> aiuwei是伟er在智能系统俱乐部的别称，怎么样是不是超级有创意

## 本周阅读：
- 1.MadaKV:Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference
- 2.DeepSeekHarness技术入门与架构原理


## 本周工作：
- 1.参加俱乐部面试，阅读了MadaKV:Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference这篇论文，了解了多模态大模型长上下文推理的 KV 缓存压缩的方法

  - 问题
  多模态大模型处理长上下文时，KV 缓存占满显存、拖慢解码
  现有淘汰方法只认文本或固定先删图像分不清头在看图还是在读字，会误删关键信息。
  - 关键观察
  注意力稀疏，20% token 覆盖约 90% 分数
  但文本 token 注意力集中、视觉 token 分散
  不同注意力头偏好不同模态
  - 方法
  
    - 横向：MPA（模态偏好适配）
    用末尾 token 当探针，算每个头更偏图还是偏字，再按比例分配该层的缓存预算
    - 纵向：HCC（分层压缩补偿）
    按重要性阈值取 token，超支/节省的差额按剩余层数摊还，保证总预算守恒、抑制级联误差
  - 结果
  LLaVA-v1.5、Qwen2.5-VL 在 MileBench 上：
  解码延迟改善 1.3–1.5 倍
  KV 缓存显存降 80%–95%，精度基本不降
  TN 任务比纯文本淘汰方法高 6.11 个点
  - 局限
  没测更大模型（34B/70B）和极长上下文
  只覆盖图像+文本，无视频音频

- 2.阅读了DeepSeekHarness技术入门与架构原理，对小鲸鲸有了进一步的了解，完成了导读的学习

  - 导读：为什么现在需要Harness

    - 0.1 Model只会生成Token，但Agent要完成任务
    - 0.2 为什么DSH值得单独写一本书
    - 0.3 本书的学习路径

      - step1 先跑起来
      - step2 理解4套Preset(Standard/PTC/Minimal/Creative)
      - step3 亲手加一个能力(从Skill、Tool或Hook开始)
      - step4 理解 Everything is a Plugin
      - step5 理解 Cordis(Context/Service/Event/Effect/Lifecycle)
      - step6 追踪一次 Agent Turn(Turn/Step/Tool Call/Session Event)
      - step7 认识高级能力(Code Mode/Subagent/Sandbox/Extension)
      - step8 最后设计自己的Harness(用Capability、Preset、Policy和Runtime视角组织Agent)

## 碎碎叨：
- 1.关于俱乐部的笔试+面试的准备，我借助dsh写了本地个性化html文件辅助学习，不知道需不需要开源到这个仓库，但好像刚入部的同学都是佼佼者有自己一套的学习方法，对这个应该也不感兴趣
- 2.现阶段加入了信息安全俱乐部和智能系统俱乐部，就是可能加两个俱乐部对精力也是极大的消耗，可能会造成啥也学不精的结果，但是呢其实我更像做WEB X AI的岗位，比如LLM 应用渗透测试、AI 供应链安全 、自动化渗透智能体之类的方向，但是还没有一个明确的规划，也比较迷茫，其实我觉得如果把这个作为人生目标的话可能还是要读研深造一些，其实也没有考虑好，反正先把课内绩点刷高没有坏处
- 3.关于博客我后续可能会搞两个博客，一个是安全方向的，一个是ai方向的，当然，我建好之后会把ai方向的博客公开让大家学习
- 4.俱乐部能不能像学习隔壁俱乐部搞个网站啊，羡慕。其实既然俱乐部主要采用GitHub那么可以试一试用GitHub Page静态部署一个网站并实现评论功能，感觉周报放在仓库里别人不是能及时回复和评论
- 5.好想要学时，能不能给我好多好多学时，我想要学时我不想听讲座不想听讲座我想要学时，~~能不能偷摸搞个活动划给我10个学时~~
- 6.关于近期学习计划暂定为上午去上课的同时刷点相关课程的视频→下午去信安学习安全和课内的→然后把电脑放在啊啊学长的工位上，希望啊啊学长会同意(小声)→晚自习学一些课内的→晚上在俱乐部和宿舍与dsh甜蜜双排学ai(因为是谷期要库库蹬)
- 7.信安那边已经制定好11月目标，但是由于对俱乐部需要学习的内容没有清晰的目标，暂定在第一次例会结束后制定11月目标
