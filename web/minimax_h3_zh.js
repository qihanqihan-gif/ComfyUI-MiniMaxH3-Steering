import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

const NODE_TRANSLATIONS = {
  MiniMaxH3PromptDirector: {
    title: "MiniMax H3 提示词导演（顺序接口）",
    fields: {
      prompt: "用户意图（描述要生成的视频）",
      task_type: "任务类型",
      duration_seconds: "时长（秒）",
      shot_count: "分镜数（0=自动）",
      rewrite_mode: "改写模式",
      output_language: "输出语言",
      api_base_url: "API 地址",
      api_model: "模型 ID",
      api_key: "API 密钥（推荐留空，改用环境变量）",
      temperature: "温度",
      max_tokens: "最大生成令牌数（最高 131072）",
      timeout_s: "超时（秒）",
      lmstudio_after_use: "跑完后 LM Studio 模型处理",
      lmstudio_gpu_offload: "LM Studio GPU 加载策略",
      api_reasoning: "模型思考",
      json_mode: "JSON 模式",
      analysis_mode: "视觉分析模式",
      model_profile: "多模态模型兼容配置",
      frame_sequence_limit: "直连视频帧上限",
      frame_selection_mode: "直连视频选帧",
      frame_selection_spec: "直连自定义选帧",
      api_failure_policy: "API / JSON 失败策略",
      reference_fidelity: "参考视频跟随档",
      ref_image_1: "导演识图素材 1（不传给 H3）",
      ref_image_2: "导演识图素材 2（不传给 H3）",
      ref_image_3: "导演识图素材 3（不传给 H3）",
      ref_image_4: "导演识图素材 4（不传给 H3）",
      ref_image_5: "导演识图素材 5（不传给 H3）",
      ref_image_6: "导演识图素材 6（不传给 H3）",
      ref_image_7: "导演识图素材 7（不传给 H3）",
      ref_image_8: "导演识图素材 8（不传给 H3）",
      ref_image_9: "导演识图素材 9（不传给 H3）",
      video_frame_sequence: "参考视频抽帧 / 图像序列（批次，仅供 API）",
      video_timeline_manifest: "参考视频时间线（可选）",
      system_module: "创作规则（接规则组合节点）",
      module_manifest: "规则解析清单（接规则组合节点第 4 输出）",
      enhanced_prompt: "增强后提示词（接官方节点 prompt）",
      report: "诊断报告",
      reference_sheet: "参考素材分析表（JSON）",
      prompt_ir: "提示词中间表示（Prompt IR JSON）",
    },
    options: {
      task_type: {
        T2VA: "纯文字生视频",
        I2VA: "首帧图生视频（接 ImageToVideo first_frame）",
        FL2VA: "首尾帧锚定（接 first_frame + last_frame）",
        L2VA: "尾帧锚定",
        Ref2VA: "参考素材（ReferenceToVideo；不是首帧）",
      },
      rewrite_mode: {
        strict: "严格遵循",
        balanced: "平衡",
        creative: "创意发挥",
        transcribe: "忠实转译（不扩写）",
      },
      output_language: {
        中文: "中文（实验；官方主体规范为英文）",
        English: "English（推荐）",
      },
      lmstudio_after_use: {
        keep_loaded: "保持加载（默认）",
        unload_used_model: "卸载本次使用的模型",
        unload_and_wait_for_vram: "卸载并等待显存释放",
      },
      lmstudio_gpu_offload: {
        max: "全部加载到 GPU",
        "0.90": "GPU 90%",
        "0.75": "GPU 75%",
        "0.50": "GPU 50%",
        auto: "自动（默认）",
        off: "关闭（CPU）",
      },
      api_reasoning: {
        auto: "自动（Qwen/Gemma 分阶段；DeepSeek 低思考）",
        off: "关闭思考",
        on: "开启思考",
      },
      json_mode: {
        auto_retry: "自动重试（默认）",
        force: "强制 JSON",
        off: "关闭（纯文本）",
      },
      analysis_mode: {
        auto: "自动（DeepSeek 视频帧同次联合；本地模型按素材数分阶段）",
        single: "单次（参考图 + 视频帧同次直传）",
        staged: "分阶段（保留给同素材 A/B）",
      },
      model_profile: {
        auto: "自动识别模型（默认）",
        gemma: "Gemma",
        qwen: "Qwen",
        cloud: "云端兼容模式",
        qwen3_8: "Qwen3.8（紧凑结构化输出）",
        deepseek_vision: "DeepSeek Vision（连续帧联合理解）",
      },
      frame_selection_mode: {
        uniform_full: "均匀全程（包含首尾，默认）",
        uniform_no_edges: "均匀避开首尾（约 10%–90%）",
        custom_indices: "自定义帧序号（支持 -1=最后一帧）",
        custom_percent: "自定义百分比（0–100）",
      },
      api_failure_policy: {
        stop: "停止工作流（默认，防止原提示词误直通）",
        passthrough: "直通原提示词（旧行为）",
      },
      reference_fidelity: {
        auto: "自动（Ref2VA 视频按结构跟随）",
        locked: "锁定跟随（动作 / 姿势 / 镜头 / 时点）",
        structural: "结构跟随（主要阶段与切镜）",
        loose: "松散参考（允许重新编排）",
      },
    },
  },
  MiniMaxH3CloudDirector: {
    title: "MiniMax H3 云端多模态导演",
    fields: {
      prompt: "用户意图（描述视频、素材分工与保留/替换规则）",
      task_type: "任务类型",
      duration_seconds: "时长（秒）",
      shot_count: "分镜数（0=自动）",
      rewrite_mode: "改写模式",
      output_language: "输出语言",
      cloud_provider: "云端连接 / 兼容预设",
      api_model: "模型 ID（官方预设可留空；OpenAI 兼容必填）",
      cloud_base_url: "OpenAI 兼容 Base URL（仅该连接使用）",
      my_preset: "已保存连接名（仅 OpenAI 兼容；由底部管理器选择）",
      gemini_video_route: "Gemini 原生视频传输方式",
      gemini_video_fps: "Gemini 原生视频采样 FPS（0=官方默认 1 FPS）",
      temperature: "生成温度（预设可按供应商忽略）",
      max_tokens: "Prompt IR 最大输出令牌",
      timeout_s: "单次调用超时（秒）",
      api_reasoning: "思考强度（由云端预设转译）",
      analysis_mode: "参考图与视频帧的上传方式",
      frame_sequence_limit: "直连视频帧上限",
      frame_selection_mode: "直连视频选帧",
      frame_selection_spec: "直连自定义选帧",
      reference_fidelity: "参考视频跟随档",
      ref_image_1: "参考图 1（按连接顺序编号；仅供 API）",
      ref_image_2: "参考图 2（按连接顺序编号；仅供 API）",
      ref_image_3: "参考图 3（按连接顺序编号；仅供 API）",
      ref_image_4: "参考图 4（按连接顺序编号；仅供 API）",
      ref_image_5: "参考图 5（按连接顺序编号；仅供 API）",
      ref_image_6: "参考图 6（按连接顺序编号；仅供 API）",
      ref_image_7: "参考图 7（按连接顺序编号；仅供 API）",
      ref_image_8: "参考图 8（按连接顺序编号；仅供 API）",
      ref_image_9: "参考图 9（按连接顺序编号；仅供 API）",
      video_frame_sequence: "参考视频帧 / 图像序列（同一 <Video 1> 时间线；仅供 API）",
      cloud_video: "Gemini 原生视频（与视频帧二选一；仅供 API）",
      comfy_video: "Comfy 原生 VIDEO（未裁剪文件；与上两路二选一）",
      video_timeline_manifest: "参考视频时间线（可选）",
      system_module: "创作规则（接规则组合节点）",
      module_manifest: "规则解析清单（接规则组合节点第 4 输出）",
      enhanced_prompt: "增强后提示词（接官方节点 prompt）",
      report: "云端调用与校验报告",
      reference_sheet: "参考素材分析表（JSON）",
      prompt_ir: "提示词中间表示（Prompt IR JSON）",
    },
    options: {
      task_type: {
        T2VA: "纯文字生视频",
        I2VA: "首帧图生视频（接 ImageToVideo first_frame）",
        FL2VA: "首尾帧锚定（接 first_frame + last_frame）",
        L2VA: "尾帧锚定",
        Ref2VA: "参考素材（ReferenceToVideo；不是首帧）",
      },
      rewrite_mode: {
        strict: "严格遵循",
        balanced: "平衡",
        creative: "创意发挥",
        transcribe: "忠实转译（不扩写）",
      },
      output_language: {
        中文: "中文（实验；官方主体规范为英文）",
        English: "English（推荐）",
      },
      cloud_provider: {
        deepseek: "DeepSeek · 连续帧联合理解（实验）",
        gemini: "Gemini · 原生视频 / 抽帧 A/B",
        doubao: "豆包 · 火山方舟（未实测预设）",
        claude: "Claude · Anthropic 原生 Messages（未实测）",
        glm: "GLM · 智谱（视觉探针已测）",
        minimax: "MiniMax 文本 Chat（已暂停视觉入口）",
        custom: "OpenAI 兼容（临时填写或选择已保存连接）",
        my_presets: "旧版已保存连接（打开后自动迁移）",
      },
      api_reasoning: {
        auto: "自动（由连接预设决定）",
        off: "关闭思考",
        on: "开启较高思考",
      },
      analysis_mode: {
        auto: "自动（有视频帧时与参考图联合上传）",
        single: "联合上传（参考图 + 所选视频帧同次发送）",
        staged: "分阶段上传（分别分析后汇总，用于 A/B）",
      },
      gemini_video_route: {
        auto: "自动（同时核算视频 Base64 与参考图；超预算走 Files API）",
        inline: "强制内联（≤15 MiB）",
        file_api: "强制 Files API（调用后自动删除远端临时文件）",
      },
      frame_selection_mode: {
        uniform_full: "均匀全程（包含首尾，默认）",
        uniform_no_edges: "均匀避开首尾（约 10%–90%）",
        custom_indices: "自定义帧序号（支持 -1=最后一帧）",
        custom_percent: "自定义百分比（0–100）",
      },
      reference_fidelity: {
        auto: "自动（Ref2VA 视频按结构跟随）",
        locked: "锁定跟随（动作 / 姿势 / 镜头 / 时点）",
        structural: "结构跟随（主要阶段与切镜）",
        loose: "松散参考（允许重新编排）",
      },
    },
  },
  MiniMaxH3CloudVideoInput: {
    title: "MiniMax H3 云端原生视频输入",
    fields: {
      video: "选择 / 上传视频（仅运行时送往 Gemini）",
      cloud_video: "云端视频运行时对象（接云端导演）",
      report: "文件类型与体积报告",
    },
  },
  MiniMaxH3CompileValidate: {
    title: "MiniMax H3 提示词编译与校验",
    fields: {
      prompt_or_ir: "H3 提示词或 Prompt IR JSON",
      task_type: "任务类型",
      duration_seconds: "目标时长（秒）",
      fail_on_error: "发现错误时停止工作流",
      final_prompt: "最终 H3 提示词",
      validation_report: "确定性校验报告",
      normalized_ir: "规范化 Prompt IR",
      is_valid: "是否通过校验",
    },
    options: {
      task_type: {
        AUTO: "自动识别",
        T2VA: "纯文字生视频",
        I2VA: "首帧图生视频",
        FL2VA: "首尾帧生视频",
        L2VA: "尾帧图生视频",
        Ref2VA: "参考素材生视频",
      },
    },
  },
  MiniMaxH3VideoContext: {
    title: "MiniMax H3 参考视频准备",
    fields: {
      video_frame_sequence: "参考视频帧（按时间顺序）",
      fps: "源帧率（FPS）",
      duration_seconds: "参考片段时长（0=自动）",
      frame_sequence_limit: "代表帧数（0=全部）",
      frame_selection_mode: "代表帧分布",
      shot_boundaries: "真实切镜点（可选）",
      selected_frames: "代表帧（接导演参考视频帧）",
      timeline_manifest: "参考视频时间线（接导演同名接口）",
      report: "构建报告",
    },
    options: {
      frame_selection_mode: {
        head: "开头 N 帧",
        uniform: "均匀覆盖完整时间线（保首尾）",
      },
    },
  },
  MiniMaxH3PromptModuleLoader: {
    title: "MiniMax H3 创作规则组合 (热加载, 实验)",
    fields: {
      scope: "作用域",
      module_1: "优先规则 1",
      module_2: "优先规则 2",
      module_3: "优先规则 3",
      custom_instructions: "自定义规则（当前工作流）",
      module_4: "优先规则 4",
      module_5: "优先规则 5",
      module_folder: "规则库文件夹",
      render_profile: "规则渲染档（A/B 实验）",
      external_rule_pack: "外部规则包（接文件直载节点）",
      system_prompt_module: "已解析创作规则（接导演节点）",
      module_preview: "规则预览",
      module_diagnostics: "规则诊断",
      module_manifest: "规则解析清单（接导演节点）",
    },
    options: {
      scope: { 全部: "全部", T2VA: "T2VA", I2VA: "I2VA", FL2VA: "FL2VA", L2VA: "L2VA", Ref2VA: "Ref2VA" },
      render_profile: {
        standard: "标准（当前正文；默认）",
        compact: "紧凑（仅契约摘要；实验）",
        strong: "强化（正文 + 契约复核；实验）",
      },
      "（无）": "（无）",
    },
  },
  MiniMaxH3ModuleFolderLoader: {
    title: "MiniMax H3 创作规则文件直载 (轻量, 热加载)",
    fields: {
      module_file: "模块文件 1（按路径分类）",
      module_file_2: "模块文件 2",
      module_file_3: "模块文件 3",
      module_file_4: "模块文件 4",
      module_file_5: "模块文件 5",
      system_prompt_module: "原样规则文本（可直接接导演节点）",
      diagnostics: "直载诊断",
      rule_pack: "结构化规则包（接规则组合节点）",
    },
    options: {
      "（未选择）": "（未选择）",
    },
  },
  MiniMaxH3Steering: {
    title: "MiniMax H3 Steering (加载时方向操控, 实验)",
    fields: {
      clip: "文本编码器 (CLIP)",
      direction_file: "方向文件",
      steer_refusal: "削弱拒绝方向",
      steer_safety: "削弱安全方向",
      lam: "强度 λ",
      layer_band: "层区间",
    },
    options: {
      direction_file: {
        "auto (refusal/safety)": "自动（拒绝 / 安全双方向）",
      },
    },
  },
  MiniMaxH3OpenCache: {
    title: "MiniMax H3 开放缓存（实验）",
    fields: {
      model: "模型",
      threshold: "复用阈值",
      start_percent: "开始比例",
      end_percent: "结束比例",
      max_consecutive_skips: "最多连续跳步",
      cache_device: "缓存设备",
      verbose: "详细日志",
      MODEL: "已应用缓存的模型",
    },
    options: {
      cache_device: {
        gpu: "显卡（更快，约多占一个隐藏状态）",
        cpu: "内存（较慢，跳步时仍需传回显卡）",
      },
    },
  },
  MiniMaxH3ReferenceMediaPrep: {
    title: "MiniMax H3 参考素材预处理",
    fields: {
      images: "图像 / 视频帧",
      media_kind: "素材类型",
      target_width: "目标宽度",
      target_height: "目标高度",
      resize_mode: "尺寸策略",
      allow_upscale: "允许放大",
      max_frames: "最大帧数（0=不限制）",
      frame_selection: "抽帧策略",
      pad_value: "补边颜色（0黑 / 1白）",
      resize_chunk_size: "分批缩放帧数",
      width: "输出宽度",
      height: "输出高度",
      frames: "输出帧数",
      report: "处理报告",
    },
    options: {
      media_kind: {
        reference_image: "参考图片（只取第1张）",
        reference_video: "参考视频 / 图像序列",
      },
      resize_mode: {
        contain_pad: "完整等比缩放 + 补边（推荐）",
        preserve_area: "完整等比缩放到目标面积",
        stretch: "直接拉伸到画布",
        cover_crop: "等比覆盖 + 居中裁剪",
      },
      frame_selection: {
        uniform: "均匀抽帧（保留首尾范围）",
        head: "从开头截取（官方默认行为）",
      },
    },
  },
  MiniMaxH3ReferenceInspector: {
    title: "MiniMax H3 参考素材检查器",
    fields: {
      generation_width: "生成宽度",
      generation_height: "生成高度",
      generation_frames: "生成帧数",
      strict_mode: "危险处理",
      ref_image_size: "官方参考图尺寸策略（用于估算）",
      reference_images: "参考图片",
      reference_video: "参考视频 / 图像序列",
      reference_audio: "参考音频",
      report: "检查报告",
      profile_context: "性能计时上下文",
    },
    options: {
      strict_mode: {
        report_only: "只报告，不中断",
        raise_on_danger: "发现危险时停止工作流",
      },
      ref_image_size: {
        match: "匹配生成画面面积（较快）",
        max: "最大参考尺寸（身份更准、较慢）",
      },
    },
  },
  MiniMaxH3PerformanceProfiler: {
    title: "MiniMax H3 性能分析器",
    fields: {
      model: "模型",
      use_cuda_events: "使用 CUDA 精确计时",
      detailed_log: "逐次模型调用日志",
      profile_context: "参考条件计时上下文（可选）",
      MODEL: "已启用分析的模型",
    },
  },
  MiniMaxH3ImageToVideo: {
    title: "MiniMax H3 图像生成视频",
    fields: {
      clip: "文本编码器",
      vae: "视频 VAE",
      prompt: "提示词",
      width: "宽度",
      height: "高度",
      length: "帧数（自动对齐 17k+5）",
      first_frame: "首帧图像（可选）",
      last_frame: "尾帧图像（可选）",
      positive: "正向条件",
      LATENT: "音视频潜空间",
    },
  },
  MiniMaxH3ReferenceToVideo: {
    title: "MiniMax H3 参考素材生成视频",
    fields: {
      clip: "文本编码器",
      vae: "视频 VAE",
      audio_vae: "音频 VAE",
      prompt: "提示词",
      width: "宽度",
      height: "高度",
      length: "帧数（自动对齐 17k+5）",
      ref_image_size: "参考图尺寸策略",
      ref_images: "参考图像（可多张）",
      ref_image: "参考图",
      ref_videos: "参考视频（可多个）",
      ref_video: "参考视频",
      ref_video_audios: "参考视频配套原声",
      ref_video_audio: "参考视频配套原声",
      ref_audios: "独立参考音频",
      ref_audio: "独立参考音频",
      positive: "正向条件",
      LATENT: "音视频潜空间",
    },
    options: {
      ref_image_size: {
        match: "匹配生成画面面积（较快）",
        max: "最大参考尺寸（身份更准、较慢）",
      },
    },
  },
  EmptyMiniMaxH3LatentAV: {
    title: "创建 MiniMax H3 空白音视频潜空间",
    fields: {
      width: "宽度",
      height: "高度",
      length: "帧数（自动对齐 17k+5）",
      LATENT: "音视频潜空间",
    },
  },
  MiniMaxH3SigmaShift: {
    title: "MiniMax H3 Sigma 偏移",
    fields: {
      model: "模型",
      shift_video: "视频 Sigma 偏移",
      shift_audio: "音频 Sigma 偏移",
      MODEL: "模型",
    },
  },
  ResolutionSelector: {
    title: "分辨率选择器",
    fields: {
      aspect_ratio: "宽高比",
      megapixels: "百万像素",
      multiple: "尺寸倍数",
      width: "宽度",
      height: "高度",
    },
    options: {
      aspect_ratio: {
        "1:1 (Square)": "1:1（方形）",
        "2:3 (Portrait Photo)": "2:3（竖版照片）",
        "3:2 (Photo)": "3:2（横版照片）",
        "3:4 (Portrait Standard)": "3:4（标准竖版）",
        "4:3 (Standard)": "4:3（标准横版）",
        "9:16 (Portrait Widescreen)": "9:16（竖版宽屏）",
        "16:9 (Widescreen)": "16:9（宽屏）",
        "21:9 (Ultrawide)": "21:9（超宽屏）",
      },
    },
  },
  PrimitiveFloat: {
    title: "浮点数",
    fields: {
      value: "数值",
      FLOAT: "浮点数",
    },
  },
  ComfyMathExpression: {
    title: "数学表达式",
    fields: {
      expression: "表达式",
      values: "输入数值",
      value: "数值",
      FLOAT: "浮点数",
      INT: "整数",
      BOOL: "布尔值",
    },
  },
  ImageScaleToTotalPixels: {
    title: "按总像素缩放图像",
    fields: {
      image: "图像",
      upscale_method: "缩放算法",
      megapixels: "目标百万像素",
      resolution_steps: "尺寸对齐步长",
      IMAGE: "图像",
    },
    options: {
      upscale_method: {
        "nearest-exact": "精确最近邻",
        bilinear: "双线性",
        area: "区域采样",
        bicubic: "双三次",
        lanczos: "Lanczos",
      },
    },
  },
}

const WORKFLOW_TITLE_TRANSLATIONS = {
  "Resolution Selector (Size)": "分辨率选择器（尺寸）",
  "Float (Duration)": "浮点数（时长）",
  "Image to Video (MiniMax H3)": "图像生成视频（MiniMax H3）",
  "Reference to Video (MiniMax H3)": "参考素材生成视频（MiniMax H3）",
  "Note: Size Settings Reference": "说明：尺寸设置参考",
  "Note: MiniMax H3": "说明：MiniMax H3",
}

const GROUP_TITLE_TRANSLATIONS = {
  "User Inputs": "用户输入",
  Models: "模型",
  Sampling: "采样",
  Conditioning: "条件控制",
  "Decoding and create video": "解码并生成视频",
}

const SIZE_REFERENCE_ZH = `| 百万像素 | 宽高比 | 输出尺寸（32 倍数） |
|---|---|---|
| 0.2 | 16:9 | 608 × 352 |
| 0.3 | 16:9 | 736 × 416 |
| 0.4 | 16:9 | 864 × 480 |
| 0.5 | 16:9 | 960 × 544 |
| 0.6 | 16:9 | 1056 × 608 |
| 0.7 | 16:9 | 1152 × 640 |
| 0.8 | 16:9 | 1216 × 672 |
| 0.9 | 16:9 | 1280 × 736 |
| 0.98 | 16:9 | 1344 × 768 |
| 1.0 | 16:9 | 1376 × 768 |
| 1.2 | 16:9 | 1504 × 832 |
| 1.5 | 16:9 | 1664 × 928 |
| 1.8 | 16:9 | 1824 × 1024 |
| 2.0 | 16:9 | 1920 × 1088 |`

const MINIMAX_NOTE_ZH = `## MiniMax H3

MiniMax H3 是同时生成视频与原生立体声音频的多模态模型。

### 当前工作流

- 未连接首帧/尾帧：文生视频（T2VA）
- 连接首帧或尾帧：首尾帧图生视频（FL2VA）
- 宽度和高度必须分别是 32 的倍数
- 时长按 24 FPS 换算，并向上对齐到 17k+5 帧
- 官方约 5 秒设置会对齐为 124 帧（约 5.17 秒）

建议先用 0.2–0.4 百万像素验证模型，再逐步提高分辨率。`

const NOTE_BODY_TRANSLATIONS = {
  "Note: Size Settings Reference": SIZE_REFERENCE_ZH,
  "说明：尺寸设置参考": SIZE_REFERENCE_ZH,
  "Note: MiniMax H3": MINIMAX_NOTE_ZH,
  "说明：MiniMax H3": MINIMAX_NOTE_ZH,
}

function isChineseLocale() {
  const candidates = [
    app.ui?.settings?.getSettingValue?.("Comfy.Locale"),
    app.ui?.settings?.getSettingValue?.("Comfy.Language"),
    navigator.language,
    ...(navigator.languages || []),
  ]
  return candidates.some((value) => String(value || "").toLowerCase().startsWith("zh"))
}

function lookupField(fields, name) {
  if (!fields || !name) return null
  const fullName = String(name)
  const leafName = fullName.split(".").pop()
  const candidates = [
    fullName,
    fullName.replace(/_\d+$/, ""),
    leafName,
    leafName.replace(/_\d+$/, ""),
  ]
  for (const candidate of candidates) {
    if (fields[candidate]) return fields[candidate]
  }
  return null
}

function localizeNode(node, config, defaultTitles = []) {
  if (!node || !config) return

  const currentTitle = String(node.title || "")
  const allowedTitles = new Set([
    "",
    config.title,
    node.type,
    node.comfyClass,
    ...defaultTitles.filter(Boolean),
  ])
  if (allowedTitles.has(currentTitle)) node.title = config.title

  for (const widget of node.widgets ?? []) {
    const label = lookupField(config.fields, widget.name)
    if (label) widget.label = label

    const optionLabels = config.options?.[widget.name]
    if (optionLabels && widget.options) {
      if (!widget.options.__minimaxH3OriginalGetOptionLabel) {
        widget.options.__minimaxH3OriginalGetOptionLabel = widget.options.getOptionLabel
      }
      const original = widget.options.__minimaxH3OriginalGetOptionLabel
      widget.options.getOptionLabel = (value) => {
        const key = value == null ? "" : String(value)
        if (Object.prototype.hasOwnProperty.call(optionLabels, key)) {
          return optionLabels[key]
        }
        return original ? original(value) : key
      }
    }
  }

  for (const slot of [...(node.inputs ?? []), ...(node.outputs ?? [])]) {
    // 完整字段名优先（本插件 1-based 端口 ref_image_1..9 等直接命中；
    // 只用完整名直查，避免去 _数字 后缀的回退丢掉端口编号）
    const directLabel = config.fields?.[slot.name]
    if (directLabel) {
      slot.label = directLabel
      slot.localized_name = directLabel
      continue
    }
    // Autogrow 编号端口：ref_image_0/ref_video_1/ref_video_audio_2/ref_audio_3
    // 显示为 1-based 编号（与提示词 <Picture i>/<Video k>/<Audio j> 一致），
    // 且 ref_video_audio_k 明确标注"配套同编号视频"。
    const slotLeafName = String(slot.name || "").split(".").pop()
    const numberedMatch = slotLeafName.match(/^(ref_(?:image|video|video_audio|audio))_(\d+)$/)
    if (numberedMatch) {
      const kind = numberedMatch[1]
      const base = config.fields?.[kind]
      const n = Number(numberedMatch[2]) + 1
      if (kind === "ref_video_audio") {
        slot.label = `参考视频端口 ${n} 的配套原声`
      } else {
        slot.label = `${base ?? kind}端口 ${n}`
      }
      slot.localized_name = slot.label
      continue
    }
    const label = lookupField(config.fields, slot.name)
    if (!label) continue
    slot.label = label
    slot.localized_name = label
  }
}

function localizeWorkflowText(node) {
  if (!node) return
  const originalTitle = String(node.title || "")
  const translatedTitle = WORKFLOW_TITLE_TRANSLATIONS[originalTitle]
  const noteBody = NOTE_BODY_TRANSLATIONS[originalTitle]
  if (translatedTitle) node.title = translatedTitle

  if (noteBody) {
    const textWidget = (node.widgets ?? []).find((widget) => typeof widget.value === "string")
    if (textWidget) textWidget.value = noteBody
  }
}

function localizeGroups() {
  for (const group of app.graph?._groups ?? []) {
    const translated = GROUP_TITLE_TRANSLATIONS[group.title]
    if (translated) group.title = translated
  }
}

function installInspectorReportPreview(nodeType) {
  if (nodeType.prototype.__minimaxH3ReportPreviewInstalled) return
  nodeType.prototype.__minimaxH3ReportPreviewInstalled = true
  const originalOnExecuted = nodeType.prototype.onExecuted
  nodeType.prototype.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments)
    const report = Array.isArray(message?.text) ? message.text[0] : message?.text
    if (typeof report !== "string") return result

    if (!this.__minimaxH3ReportElement && this.addDOMWidget) {
      const textarea = document.createElement("textarea")
      textarea.readOnly = true
      textarea.style.width = "100%"
      textarea.style.minHeight = "220px"
      textarea.style.resize = "vertical"
      textarea.style.fontFamily = "monospace"
      textarea.style.fontSize = "12px"
      textarea.style.whiteSpace = "pre-wrap"
      textarea.style.boxSizing = "border-box"
      this.addDOMWidget("__minimax_h3_report", "textarea", textarea, {
        serialize: false,
        hideOnZoom: false,
      })
      this.__minimaxH3ReportElement = textarea
      const width = Math.max(this.size?.[0] ?? 0, 420)
      const height = Math.max(this.size?.[1] ?? 0, 430)
      this.setSize?.([width, height])
    }
    if (this.__minimaxH3ReportElement) {
      this.__minimaxH3ReportElement.value = report
    }
    return result
  }
}

function installModelRefreshButton(nodeType) {
  // 移植自 LingBot：按钮 → 后端 /minimaxh3lab/api/models → 更新 api_model COMBO 选项
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    const node = this
    const findWidget = (name) => (node.widgets ?? []).find((w) => w.name === name)
    const baseWidget = findWidget("api_base_url")
    const keyWidget = findWidget("api_key")
    const modelWidget = findWidget("api_model")
    const timeoutWidget = findWidget("timeout_s")
    if (!baseWidget || !modelWidget) return result

    const button = node.addWidget("button", "刷新模型列表", null, async () => {
      if (button.__fetching) return
      button.__fetching = true
      const idleName = button.name
      button.name = "正在获取模型列表…"
      node.setDirtyCanvas?.(true, true)
      try {
        const response = await api.fetchApi("/minimaxh3lab/api/models", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            api_base_url: String(baseWidget.value ?? "").trim(),
            api_key: String(keyWidget?.value ?? "").trim(),
            timeout_s: Number(timeoutWidget?.value ?? 15),
          }),
        })
        const payload = await response.json()
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`)
        const models = Array.isArray(payload.models) ? payload.models.map((m) => String(m?.id ?? "")).filter(Boolean) : []
        if (!models.length) throw new Error("API 未返回可用模型")
        // 保留 auto 选项，其余按服务端顺序
        const values = ["auto", ...models.filter((m) => m !== "auto")]
        modelWidget.options = { ...(modelWidget.options || {}), values }
        if (!values.includes(String(modelWidget.value ?? ""))) modelWidget.value = "auto"
        button.name = `刷新模型列表（${models.length}）`
        console.info("[MiniMaxH3Lab] models:", models)
      } catch (error) {
        button.name = idleName
        console.error("[MiniMaxH3Lab] 刷新模型失败:", error?.message || error)
        alert(`刷新模型失败：${error?.message || error}`)
      } finally {
        button.__fetching = false
        node.setDirtyCanvas?.(true, true)
      }
    }, { serialize: false })
    button.serialize = false
    button.serializeValue = () => undefined
    return result
  }
}

const CLOUD_CREDENTIAL_PROVIDERS = {
  deepseek: {
    provider: "deepseek",
    credentialId: "deepseek_default",
    label: "DeepSeek",
    defaultModel: "deepseek-v4-flash-vision-exp",
    probeLabel: "模型列表鉴权探针（不生成内容）",
  },
  gemini: {
    provider: "gemini",
    credentialId: "gemini_default",
    label: "Gemini",
    defaultModel: "gemini-3.1-flash-lite",
    probeLabel: "小图 + 严格 JSON 真实视觉探针",
  },
  // 2026-08-28：常用云端 LLM 预设
  doubao: {
    provider: "volcengine",
    credentialId: "volcengine_default",
    label: "豆包 · 火山方舟",
    defaultModel: "doubao-seed-evolving",
    probeLabel: "小图 + 严格 JSON 视觉探针",
  },
  claude: {
    provider: "anthropic",
    credentialId: "anthropic_default",
    label: "Claude · Anthropic",
    defaultModel: "claude-sonnet-4-6",
    probeLabel: "小图 + 严格 JSON 真实视觉探针",
  },
  glm: {
    provider: "zhipu",
    credentialId: "zhipu_default",
    label: "GLM · 智谱",
    defaultModel: "glm-5.3-flash",
    probeLabel: "小图 + 严格 JSON 视觉探针",
  },
  minimax: {
    provider: "minimax",
    credentialId: "minimax_default",
    label: "MiniMax 官方 LLM",
    defaultModel: "MiniMax-M2.7",
    probeLabel: "视觉入口已暂停（官方文本 Chat 未证明支持多图）",
  },
  custom: {
    provider: "custom",
    credentialId: "custom_default",
    label: "OpenAI 兼容",
    defaultModel: "",
    probeLabel: "Base URL + 模型 ID + 小图严格 JSON 探针",
  },
  my_presets: {
    provider: "custom",
    credentialId: "custom_default",
    label: "OpenAI 兼容（旧版已保存连接）",
    defaultModel: "",
    probeLabel: "旧工作流兼容入口",
  },
}

function cloudCredentialConfig(provider) {
  return CLOUD_CREDENTIAL_PROVIDERS[String(provider || "deepseek").toLowerCase()]
    || CLOUD_CREDENTIAL_PROVIDERS.deepseek
}

function nodeCloudCredentialConfig(node) {
  const widget = node?.widgets?.find((item) => item?.name === "cloud_provider")
  const providerId = String(widget?.value || "deepseek").toLowerCase()
  const config = { ...cloudCredentialConfig(providerId) }
  if (providerId === "custom" || providerId === "my_presets") {
    config.baseUrl = String(node?.widgets?.find((item) => item?.name === "cloud_base_url")?.value || "").trim()
    config.presetName = String(node?.widgets?.find((item) => item?.name === "my_preset")?.value || "").trim()
  }
  return config
}

const CLOUD_WIDGET_SCHEMA_PROP = "minimax_h3_cloud_widget_schema"
const CLOUD_WIDGET_SCHEMA_VERSION = 5
const CLOUD_WIDGET_NAMES = [
  "prompt", "task_type", "duration_seconds", "shot_count", "rewrite_mode",
  "output_language", "cloud_provider", "api_model", "temperature", "max_tokens",
  "timeout_s", "api_reasoning", "analysis_mode", "frame_sequence_limit",
  "frame_selection_mode", "frame_selection_spec", "cloud_base_url", "my_preset",
  "gemini_video_route", "gemini_video_fps", "reference_fidelity",
]

function normalizeCloudDirectorWidgetValues(rawValues, declaredVersion = 0) {
  const raw = Array.isArray(rawValues) ? [...rawValues] : []
  if (!raw.length) return raw
  // 2026-08-28 的短暂插入式布局：base_url 插在 api_model 前，my_preset
  // 插在 temperature 前。把它重排为“所有新字段追加在末尾”的 v2 布局。
  const looksLikeInsertedV1 = raw.length >= 18 && declaredVersion < 2 && (
    typeof raw[8] === "string" || typeof raw[11] === "number"
  )
  if (looksLikeInsertedV1) {
    return [
      ...raw.slice(0, 7),
      raw[8], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15], raw[16], raw[17],
      raw[7], raw[9],
    ]
  }
  return raw
}

function repairCloudDirectorWidgetValues(node, info) {
  const declaredVersion = Number(info?.properties?.[CLOUD_WIDGET_SCHEMA_PROP] || 0)
  const values = normalizeCloudDirectorWidgetValues(info?.widgets_values, declaredVersion)
  if (!values.length) return
  const defaults = {
    prompt: "", task_type: "Ref2VA", duration_seconds: 5, shot_count: 0,
    rewrite_mode: "balanced", output_language: "English", cloud_provider: "deepseek",
    api_model: "", temperature: 0.2, max_tokens: 16384, timeout_s: 600,
    api_reasoning: "auto", analysis_mode: "auto", frame_sequence_limit: 48,
    frame_selection_mode: "uniform_full", frame_selection_spec: "",
    cloud_base_url: "", my_preset: "", gemini_video_route: "auto", gemini_video_fps: 0,
    reference_fidelity: "auto",
  }
  const allowed = {
    task_type: new Set(["T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA"]),
    rewrite_mode: new Set(["strict", "balanced", "creative", "transcribe"]),
    output_language: new Set(["English", "中文"]),
    api_reasoning: new Set(["auto", "off", "on"]),
    analysis_mode: new Set(["auto", "single", "staged"]),
    gemini_video_route: new Set(["auto", "inline", "file_api"]),
    reference_fidelity: new Set(["auto", "locked", "structural", "loose"]),
    frame_selection_mode: new Set(["uniform_full", "uniform_no_edges", "custom_indices", "custom_percent"]),
  }
  const byName = Object.fromEntries(CLOUD_WIDGET_NAMES.map((name, index) => [name, values[index]]))
  const finite = (name, min, max, integer = false) => {
    const parsed = Number(byName[name])
    if (!Number.isFinite(parsed)) return defaults[name]
    const bounded = Math.min(max, Math.max(min, parsed))
    return integer ? Math.round(bounded) : bounded
  }
  const normalized = {
    ...defaults,
    prompt: typeof byName.prompt === "string" ? byName.prompt : "",
    task_type: allowed.task_type.has(byName.task_type) ? byName.task_type : defaults.task_type,
    duration_seconds: finite("duration_seconds", 4, 15),
    shot_count: finite("shot_count", 0, 20, true),
    rewrite_mode: allowed.rewrite_mode.has(byName.rewrite_mode) ? byName.rewrite_mode : defaults.rewrite_mode,
    output_language: allowed.output_language.has(byName.output_language) ? byName.output_language : defaults.output_language,
    cloud_provider: byName.cloud_provider === "my_presets"
      ? "custom"
      : (typeof byName.cloud_provider === "string" && byName.cloud_provider ? byName.cloud_provider : defaults.cloud_provider),
    api_model: typeof byName.api_model === "string" ? byName.api_model : "",
    temperature: finite("temperature", 0, 2),
    max_tokens: finite("max_tokens", 256, 131072, true),
    timeout_s: finite("timeout_s", 30, 1800, true),
    api_reasoning: allowed.api_reasoning.has(byName.api_reasoning) ? byName.api_reasoning : defaults.api_reasoning,
    analysis_mode: allowed.analysis_mode.has(byName.analysis_mode) ? byName.analysis_mode : defaults.analysis_mode,
    frame_sequence_limit: finite("frame_sequence_limit", 1, 600, true),
    frame_selection_mode: allowed.frame_selection_mode.has(byName.frame_selection_mode)
      ? byName.frame_selection_mode : defaults.frame_selection_mode,
    frame_selection_spec: typeof byName.frame_selection_spec === "string" ? byName.frame_selection_spec : "",
    cloud_base_url: typeof byName.cloud_base_url === "string" ? byName.cloud_base_url : "",
    my_preset: typeof byName.my_preset === "string" ? byName.my_preset : "",
    gemini_video_route: allowed.gemini_video_route.has(byName.gemini_video_route)
      ? byName.gemini_video_route : defaults.gemini_video_route,
    gemini_video_fps: finite("gemini_video_fps", 0, 24),
    reference_fidelity: allowed.reference_fidelity.has(byName.reference_fidelity)
      ? byName.reference_fidelity : defaults.reference_fidelity,
  }
  if (normalized.cloud_provider !== "custom") {
    normalized.cloud_base_url = ""
    normalized.my_preset = ""
  }
  for (const [name, value] of Object.entries(normalized)) {
    const widget = node?.widgets?.find((item) => item?.name === name)
    if (widget) widget.value = value
  }
  info.widgets_values = CLOUD_WIDGET_NAMES.map((name) => normalized[name])
  info.properties ||= {}
  info.properties[CLOUD_WIDGET_SCHEMA_PROP] = CLOUD_WIDGET_SCHEMA_VERSION
  node.properties ||= {}
  node.properties[CLOUD_WIDGET_SCHEMA_PROP] = CLOUD_WIDGET_SCHEMA_VERSION
}

function nodeCloudModel(node, config) {
  const widget = node?.widgets?.find((item) => item?.name === "api_model")
  return String(widget?.value || "").trim() || config.defaultModel
}

function setCloudModelOverride(node, value) {
  const widget = node?.widgets?.find((item) => item?.name === "api_model")
  if (!widget) return false
  widget.value = String(value || "")
  node.graph?.change?.()
  node.setDirtyCanvas?.(true, true)
  return true
}

function setCloudWidgetValue(node, name, value) {
  const widget = node?.widgets?.find((item) => item?.name === name)
  if (!widget) return false
  widget.value = value
  node.graph?.change?.()
  node.setDirtyCanvas?.(true, true)
  return true
}

async function cloudCredentialRequest(path, options = {}) {
  const response = await api.fetchApi(path, options)
  let payload = {}
  try {
    payload = await response.json()
  } catch (_) {
    payload = {}
  }
  if (!response.ok || payload?.ok === false) {
    throw new Error(payload?.error || `HTTP ${response.status}`)
  }
  return payload
}

async function fetchCloudCredentialStatus(config) {
  const query = new URLSearchParams({
    provider: config.provider,
    credential_id: config.credentialId,
  })
  const payload = await cloudCredentialRequest(`/minimaxh3lab/cloud/credential/status?${query}`)
  return payload.status || {}
}

function createCloudModelPicker(node, config, onModelCountChanged, options = {}) {
  const overlay = document.createElement("div")
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100000", background: "rgba(0,0,0,.62)",
    display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
  })
  const panel = document.createElement("div")
  Object.assign(panel.style, {
    width: "min(760px, 94vw)", maxHeight: "90vh", overflow: "auto",
    background: "#202124", color: "#eee", border: "1px solid #505157",
    borderRadius: "12px", boxShadow: "0 18px 60px rgba(0,0,0,.55)", padding: "22px",
    fontFamily: "system-ui, sans-serif",
  })
  overlay.appendChild(panel)

  const titleRow = document.createElement("div")
  Object.assign(titleRow.style, { display: "flex", alignItems: "center", gap: "12px" })
  const title = document.createElement("h2")
  title.textContent = `选择 ${config.label} 模型`
  Object.assign(title.style, { margin: "0", flex: "1", fontSize: "20px" })
  const close = document.createElement("button")
  close.textContent = "×"
  Object.assign(close.style, {
    width: "36px", height: "36px", border: "0", borderRadius: "8px",
    background: "#34363b", color: "#fff", fontSize: "24px", cursor: "pointer",
  })
  titleRow.append(title, close)
  panel.appendChild(titleRow)

  const summary = document.createElement("p")
  const currentModel = () => String((
    typeof options.currentModel === "function" ? options.currentModel() : nodeCloudModel(node, config)
  ) || "").trim()
  summary.textContent = `当前：${currentModel() || "未填写"}；预设默认：${config.defaultModel || "无"}`
  Object.assign(summary.style, { color: "#b7bac3", margin: "8px 0 14px" })
  panel.appendChild(summary)

  const filters = document.createElement("div")
  Object.assign(filters.style, { display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center" })
  const search = document.createElement("input")
  search.type = "search"
  search.placeholder = "搜索模型 ID / 显示名称"
  Object.assign(search.style, {
    flex: "1", minWidth: "260px", padding: "9px 11px", borderRadius: "7px",
    border: "1px solid #5b5e66", background: "#17181b", color: "#fff",
  })
  const recommendedLabel = document.createElement("label")
  Object.assign(recommendedLabel.style, { display: "flex", alignItems: "center", gap: "6px" })
  const recommendedOnly = document.createElement("input")
  recommendedOnly.type = "checkbox"
  recommendedOnly.checked = config.provider !== "custom"
  recommendedLabel.append(recommendedOnly, document.createTextNode("优先只看 Director 候选"))
  filters.append(search, recommendedLabel)
  panel.appendChild(filters)

  const select = document.createElement("select")
  select.size = 14
  Object.assign(select.style, {
    width: "100%", marginTop: "12px", minHeight: "320px", padding: "7px",
    borderRadius: "8px", border: "1px solid #5b5e66", background: "#17181b",
    color: "#fff", fontFamily: "ui-monospace, Consolas, monospace", fontSize: "13px",
  })
  panel.appendChild(select)

  const feedback = document.createElement("div")
  Object.assign(feedback.style, { minHeight: "24px", margin: "10px 0", color: "#b9dcff" })
  panel.appendChild(feedback)
  const note = document.createElement("p")
  note.textContent = "模型出现在供应商列表，只代表当前账户可见；★ 表示适合文本 Prompt IR 的候选，不等于已经完成本插件的多图/严格 JSON 实测。"
  Object.assign(note.style, { color: "#e9bd73", fontSize: "13px", lineHeight: "1.55" })
  panel.appendChild(note)

  const actions = document.createElement("div")
  Object.assign(actions.style, { display: "flex", flexWrap: "wrap", gap: "9px" })
  const makeButton = (text, background = "#3568d4") => {
    const button = document.createElement("button")
    button.textContent = text
    Object.assign(button.style, {
      padding: "9px 15px", border: "0", borderRadius: "7px", background,
      color: "#fff", cursor: "pointer", fontWeight: "600",
    })
    actions.appendChild(button)
    return button
  }
  const useSelected = makeButton("使用所选模型")
  const usePreset = makeButton("恢复连接预设默认", "#6b5b95")
  const refresh = makeButton("重新获取", "#3c7f62")
  panel.appendChild(actions)

  let models = []
  const render = () => {
    const query = String(search.value || "").trim().toLowerCase()
    const current = currentModel()
    const rows = models.filter((item) => {
      if (recommendedOnly.checked && !item.recommended) return false
      const haystack = `${item.id || ""} ${item.display_name || ""}`.toLowerCase()
      return !query || haystack.includes(query)
    })
    select.replaceChildren()
    for (const item of rows) {
      const option = document.createElement("option")
      option.value = item.id
      const display = item.display_name && item.display_name !== item.id
        ? ` — ${item.display_name}` : ""
      const inputLimit = item.input_token_limit
        ? `in ${Number(item.input_token_limit).toLocaleString()}` : ""
      const outputLimit = item.output_token_limit
        ? `out ${Number(item.output_token_limit).toLocaleString()}` : ""
      const limits = inputLimit || outputLimit ? ` [${[inputLimit, outputLimit].filter(Boolean).join(" / ")}]` : ""
      option.textContent = `${item.recommended ? "★ " : "  "}${item.id}${display}${limits}`
      option.selected = item.id === current
      select.appendChild(option)
    }
    feedback.textContent = `显示 ${rows.length} / ${models.length} 个模型`
  }
  const load = async () => {
    refresh.disabled = true
    useSelected.disabled = true
    feedback.textContent = `正在通过本机后端获取 ${config.label} 模型列表…`
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/models", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: config.provider,
          credential_id: config.credentialId,
          base_url: config.baseUrl || "",
          preset_name: config.presetName || "",
          timeout_s: 30,
        }),
      })
      models = Array.isArray(payload.models) ? payload.models.filter((item) => item?.id) : []
      models.sort((left, right) => Number(Boolean(right.recommended)) - Number(Boolean(left.recommended))
        || String(left.id).localeCompare(String(right.id)))
      onModelCountChanged?.(models.length)
      render()
      if (!models.length) feedback.textContent = "服务没有返回模型列表；仍可手动填写模型 ID。"
    } catch (error) {
      models = []
      select.replaceChildren()
      feedback.textContent = `获取失败：${error?.message || error}`
    } finally {
      refresh.disabled = false
      useSelected.disabled = false
    }
  }
  search.oninput = render
  recommendedOnly.onchange = render
  select.ondblclick = () => useSelected.click()
  useSelected.onclick = () => {
    if (!select.value) {
      feedback.textContent = "请先选择一个模型。"
      return
    }
    if (typeof options.onSelect === "function") {
      options.onSelect(select.value)
    } else {
      setCloudModelOverride(node, select.value)
    }
    summary.textContent = `当前：${select.value}；预设默认：${config.defaultModel || "无"}`
    feedback.textContent = `已选择模型：${select.value}`
  }
  if (options.hidePresetReset) usePreset.style.display = "none"
  usePreset.onclick = () => {
    setCloudModelOverride(node, "")
    summary.textContent = `当前：${config.defaultModel}（预设）；预设默认：${config.defaultModel}`
    feedback.textContent = "已清空模型覆盖，运行时使用连接预设默认模型。"
    render()
  }
  refresh.onclick = load
  const dismiss = () => overlay.remove()
  close.onclick = dismiss
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) dismiss()
  })
  document.body.appendChild(overlay)
  search.focus()
  load()
}

function credentialSourceLabel(source) {
  const value = String(source || "")
  if (value.startsWith("environment:")) return "环境变量"
  if (value.startsWith("local_file:")) return "本地凭据文件"
  return "未配置"
}

function createCloudCredentialModal(config, modelId, onStatusChanged) {
  const overlay = document.createElement("div")
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100000", background: "rgba(0,0,0,.62)",
    display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
  })
  const panel = document.createElement("div")
  Object.assign(panel.style, {
    width: "min(680px, 94vw)", maxHeight: "90vh", overflow: "auto",
    background: "#202124", color: "#eee", border: "1px solid #505157",
    borderRadius: "12px", boxShadow: "0 18px 60px rgba(0,0,0,.55)", padding: "22px",
    fontFamily: "system-ui, sans-serif",
  })
  overlay.appendChild(panel)

  const titleRow = document.createElement("div")
  Object.assign(titleRow.style, { display: "flex", alignItems: "center", gap: "12px" })
  const title = document.createElement("h2")
  title.textContent = "MiniMax H3 云端连接"
  Object.assign(title.style, { margin: "0", flex: "1", fontSize: "20px" })
  const close = document.createElement("button")
  close.textContent = "×"
  Object.assign(close.style, {
    width: "36px", height: "36px", border: "0", borderRadius: "8px",
    background: "#34363b", color: "#fff", fontSize: "24px", cursor: "pointer",
  })
  titleRow.append(title, close)
  panel.appendChild(titleRow)

  const subtitle = document.createElement("p")
  subtitle.textContent = `${config.label}${config.provider === "custom" ? "连接" : "官方 API"} · 凭据不会写入节点、工作流、history 或生成媒体元数据`
  Object.assign(subtitle.style, { color: "#b7bac3", margin: "8px 0 18px" })
  panel.appendChild(subtitle)

  const statusBox = document.createElement("div")
  Object.assign(statusBox.style, {
    padding: "12px 14px", background: "#2a2c31", borderRadius: "8px", lineHeight: "1.65",
    marginBottom: "16px", whiteSpace: "pre-wrap", wordBreak: "break-all",
  })
  panel.appendChild(statusBox)

  const label = document.createElement("label")
  label.textContent = `${config.label} API Key（留空不会改变已保存值；本机 loopback 可不填）`
  Object.assign(label.style, { display: "block", marginBottom: "7px", fontWeight: "600" })
  panel.appendChild(label)

  const inputRow = document.createElement("div")
  Object.assign(inputRow.style, { display: "flex", gap: "8px" })
  const keyInput = document.createElement("input")
  keyInput.type = "password"
  keyInput.autocomplete = "new-password"
  keyInput.placeholder = "粘贴新的 Key；重新打开窗口时不会回填旧 Key"
  Object.assign(keyInput.style, {
    flex: "1", minWidth: "0", padding: "10px 12px", borderRadius: "7px",
    border: "1px solid #5b5e66", background: "#17181b", color: "#fff",
  })
  const reveal = document.createElement("button")
  reveal.textContent = "显示"
  Object.assign(reveal.style, {
    padding: "0 14px", borderRadius: "7px", border: "1px solid #5b5e66",
    background: "#34363b", color: "#fff", cursor: "pointer",
  })
  inputRow.append(keyInput, reveal)
  panel.appendChild(inputRow)

  const warning = document.createElement("p")
  warning.textContent = "本版按你的取舍使用本机明文 JSON。复制整个 ComfyUI/user 目录、让他人访问本机文件或开放 ComfyUI 服务时，凭据仍可能暴露。"
  Object.assign(warning.style, { color: "#e9bd73", fontSize: "13px", lineHeight: "1.55" })
  panel.appendChild(warning)

  const feedback = document.createElement("div")
  Object.assign(feedback.style, { minHeight: "24px", margin: "8px 0", color: "#b9dcff" })
  panel.appendChild(feedback)

  const actions = document.createElement("div")
  Object.assign(actions.style, { display: "flex", flexWrap: "wrap", gap: "9px" })
  const makeButton = (text, background = "#3568d4") => {
    const button = document.createElement("button")
    button.textContent = text
    Object.assign(button.style, {
      padding: "9px 15px", border: "0", borderRadius: "7px", background,
      color: "#fff", cursor: "pointer", fontWeight: "600",
    })
    actions.appendChild(button)
    return button
  }
  const save = makeButton("保存 / 替换")
  const test = makeButton("测试已保存连接", "#3c7f62")
  const clear = makeButton("清除本地 Key", "#984d4d")
  panel.appendChild(actions)

  let status = {}
  const renderStatus = (nextStatus) => {
    status = nextStatus || {}
    const configured = status.configured ? "已配置" : "未配置"
    const source = credentialSourceLabel(status.source)
    statusBox.textContent = [
      `状态：${configured}（${source}）`,
      `本地存储：${status.storage_path || "等待后端返回"}`,
      status.environment_configured
        ? "环境变量优先：清除本地文件后，环境变量仍会继续生效。"
        : "环境变量：未配置；保存后使用本地凭据文件。",
    ].join("\n")
    onStatusChanged?.(status)
  }
  const refresh = async () => {
    feedback.textContent = "正在读取配置状态…"
    try {
      renderStatus(await fetchCloudCredentialStatus(config))
      feedback.textContent = ""
    } catch (error) {
      feedback.textContent = `读取失败：${error?.message || error}`
    }
  }
  const setBusy = (busy) => {
    save.disabled = busy
    test.disabled = busy
    clear.disabled = busy
  }

  reveal.onclick = () => {
    const show = keyInput.type === "password"
    keyInput.type = show ? "text" : "password"
    reveal.textContent = show ? "隐藏" : "显示"
  }
  save.onclick = async () => {
    const apiKey = String(keyInput.value || "").trim()
    if (!apiKey) {
      feedback.textContent = "请输入新的 API Key；空值不会覆盖已保存凭据。"
      return
    }
    setBusy(true)
    feedback.textContent = "正在保存到本机用户配置…"
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/credential/save", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: config.provider,
          credential_id: config.credentialId,
          api_key: apiKey,
        }),
      })
      keyInput.value = ""
      keyInput.type = "password"
      reveal.textContent = "显示"
      renderStatus(payload.status)
      feedback.textContent = status.environment_configured
        ? "已保存，但当前环境变量优先于本地 Key。"
        : "已保存。Key 不会进入当前工作流。"
    } catch (error) {
      feedback.textContent = `保存失败：${error?.message || error}`
    } finally {
      setBusy(false)
    }
  }
  clear.onclick = async () => {
    if (!confirm(`清除本地保存的 ${config.label} Key？环境变量不会被删除。`)) return
    setBusy(true)
    feedback.textContent = "正在清除本地凭据…"
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/credential/clear", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider: config.provider, credential_id: config.credentialId }),
      })
      renderStatus(payload.status)
      feedback.textContent = payload.status?.configured
        ? "本地 Key 已清除；环境变量仍在生效。"
        : "本地 Key 已清除。"
    } catch (error) {
      feedback.textContent = `清除失败：${error?.message || error}`
    } finally {
      setBusy(false)
    }
  }
  test.onclick = async () => {
    setBusy(true)
    feedback.textContent = `正在执行 ${config.label}：${config.probeLabel}…`
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/credential/test", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: config.provider,
          credential_id: config.credentialId,
          model_id: modelId,
          base_url: config.baseUrl || "",
        }),
      })
      const probe = payload.vision_probe === "strict_json_passed"
        ? `；视觉结构化探针：通过（${payload.probe_model || modelId}）`
        : ""
      feedback.textContent = `连接成功；凭据来源：${credentialSourceLabel(payload.source)}；模型数：${payload.model_count ?? 0}${probe}`
      await refresh()
    } catch (error) {
      feedback.textContent = `连接测试失败：${error?.message || error}`
    } finally {
      setBusy(false)
    }
  }
  const dismiss = () => overlay.remove()
  close.onclick = dismiss
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) dismiss()
  })
  document.body.appendChild(overlay)
  keyInput.focus()
  refresh()
}

function createCloudPresetManager(node, onPresetChanged) {
  const overlay = document.createElement("div")
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100000", background: "rgba(0,0,0,.62)",
    display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
  })
  const panel = document.createElement("div")
  Object.assign(panel.style, {
    width: "min(860px, 96vw)", maxHeight: "92vh", overflow: "auto",
    background: "#202124", color: "#eee", border: "1px solid #505157",
    borderRadius: "12px", boxShadow: "0 18px 60px rgba(0,0,0,.55)", padding: "22px",
    fontFamily: "system-ui, sans-serif",
  })
  overlay.appendChild(panel)

  const titleRow = document.createElement("div")
  Object.assign(titleRow.style, { display: "flex", alignItems: "center", gap: "12px" })
  const title = document.createElement("h2")
  title.textContent = "OpenAI 兼容连接管理"
  Object.assign(title.style, { margin: "0", flex: "1", fontSize: "20px" })
  const close = document.createElement("button")
  close.textContent = "×"
  Object.assign(close.style, {
    width: "36px", height: "36px", border: "0", borderRadius: "8px",
    background: "#34363b", color: "#fff", fontSize: "24px", cursor: "pointer",
  })
  titleRow.append(title, close)
  panel.appendChild(titleRow)

  const explanation = document.createElement("p")
  explanation.textContent = "这里统一管理 OpenAI 兼容连接：可把表单临时用于当前节点，也可保存为连接以便复用。每个已保存连接拥有独立 Key 槽；Key 只存本机用户配置，不进入工作流。"
  Object.assign(explanation.style, { color: "#b7bac3", lineHeight: "1.6", margin: "8px 0 16px" })
  panel.appendChild(explanation)

  const body = document.createElement("div")
  Object.assign(body.style, {
    display: "grid", gridTemplateColumns: "minmax(220px, .8fr) minmax(340px, 1.4fr)", gap: "16px",
  })
  const list = document.createElement("select")
  list.size = 13
  Object.assign(list.style, {
    width: "100%", minHeight: "330px", padding: "8px", borderRadius: "8px",
    border: "1px solid #5b5e66", background: "#17181b", color: "#fff",
  })
  body.appendChild(list)

  const form = document.createElement("div")
  Object.assign(form.style, { display: "flex", flexDirection: "column", gap: "11px" })
  const makeField = (labelText, placeholder) => {
    const wrapper = document.createElement("label")
    Object.assign(wrapper.style, { display: "flex", flexDirection: "column", gap: "6px", fontWeight: "600" })
    wrapper.appendChild(document.createTextNode(labelText))
    const input = document.createElement("input")
    input.type = "text"
    input.placeholder = placeholder
    Object.assign(input.style, {
      padding: "10px 12px", borderRadius: "7px", border: "1px solid #5b5e66",
      background: "#17181b", color: "#fff", fontFamily: "ui-monospace, Consolas, monospace",
    })
    wrapper.appendChild(input)
    form.appendChild(wrapper)
    return input
  }
  const nameInput = makeField("预设名称", "例如：我的 Gemini 代理")
  const baseUrlInput = makeField("Base URL", "https://example.com/v1")
  const modelInput = makeField("模型 ID", "vision-model-id")
  const credentialInfo = document.createElement("div")
  Object.assign(credentialInfo.style, {
    padding: "10px 12px", borderRadius: "7px", background: "#2a2c31", color: "#b9dcff",
    lineHeight: "1.55", wordBreak: "break-all",
  })
  credentialInfo.textContent = "新预设保存后才会生成独立凭据槽。"
  form.appendChild(credentialInfo)
  body.appendChild(form)
  panel.appendChild(body)

  const feedback = document.createElement("div")
  Object.assign(feedback.style, { minHeight: "24px", margin: "12px 0 8px", color: "#b9dcff" })
  panel.appendChild(feedback)
  const actions = document.createElement("div")
  Object.assign(actions.style, { display: "flex", flexWrap: "wrap", gap: "9px" })
  const makeButton = (text, background = "#3568d4") => {
    const button = document.createElement("button")
    button.textContent = text
    Object.assign(button.style, {
      padding: "9px 15px", border: "0", borderRadius: "7px", background,
      color: "#fff", cursor: "pointer", fontWeight: "600",
    })
    actions.appendChild(button)
    return button
  }
  const fresh = makeButton("新建（沿用当前地址/模型）", "#6b5b95")
  const save = makeButton("保存 / 更新")
  const useSelected = makeButton("使用已保存连接", "#3c7f62")
  const useTemporary = makeButton("仅临时用于当前节点", "#3f6d8a")
  const chooseModel = makeButton("获取 / 选择模型", "#506a91")
  const configureKey = makeButton("配置当前表单的 Key / 测试", "#8a6d35")
  const remove = makeButton("删除预设", "#984d4d")
  const reload = makeButton("刷新列表", "#4f5967")
  panel.appendChild(actions)

  let presets = []
  let selected = null
  const clearForm = (useCurrentNode = true) => {
    selected = null
    list.value = ""
    nameInput.value = ""
    const nodeProvider = String(node?.widgets?.find((item) => item?.name === "cloud_provider")?.value || "")
    baseUrlInput.value = useCurrentNode && nodeProvider === "custom"
      ? String(node?.widgets?.find((item) => item?.name === "cloud_base_url")?.value || "") : ""
    modelInput.value = useCurrentNode && nodeProvider === "custom"
      ? String(node?.widgets?.find((item) => item?.name === "api_model")?.value || "") : ""
    credentialInfo.textContent = "未保存时使用 OpenAI 兼容临时 Key 槽；保存后生成独立 Key 槽。"
    nameInput.focus()
  }
  const selectPreset = (preset) => {
    selected = preset || null
    if (!selected) return clearForm()
    nameInput.value = selected.name || ""
    baseUrlInput.value = selected.base_url || ""
    modelInput.value = selected.model || ""
    credentialInfo.textContent = `独立凭据槽：${selected.credential_id || "等待后端生成"}`
  }
  const render = () => {
    const activeName = String(node?.widgets?.find((item) => item?.name === "my_preset")?.value || "")
    list.replaceChildren()
    for (const preset of presets) {
      const option = document.createElement("option")
      option.value = preset.id
      option.textContent = `${preset.name}${preset.name === activeName ? "  ✓ 当前" : ""} — ${preset.model}`
      list.appendChild(option)
    }
    const preferred = presets.find((item) => item.name === activeName)
      || presets.find((item) => item.id === selected?.id)
      || presets[0]
    if (preferred) {
      list.value = preferred.id
      selectPreset(preferred)
    } else {
      clearForm(true)
    }
  }
  const load = async () => {
    feedback.textContent = "正在读取本机预设…"
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/presets")
      presets = Array.isArray(payload.presets) ? payload.presets : []
      render()
      feedback.textContent = presets.length ? `已读取 ${presets.length} 个预设。` : "尚无预设，可以直接新建。"
    } catch (error) {
      feedback.textContent = `读取失败：${error?.message || error}`
    }
  }
  list.onchange = () => selectPreset(presets.find((item) => item.id === list.value))
  fresh.onclick = () => clearForm(true)
  save.onclick = async () => {
    const record = {
      name: String(nameInput.value || "").trim(),
      base_url: String(baseUrlInput.value || "").trim(),
      model: String(modelInput.value || "").trim(),
    }
    if (!record.name || !record.base_url || !record.model) {
      feedback.textContent = "名称、Base URL、模型 ID 都必须填写。"
      return
    }
    save.disabled = true
    feedback.textContent = "正在保存预设（不含 Key）…"
    try {
      const payload = await cloudCredentialRequest("/minimaxh3lab/cloud/presets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
      })
      selected = payload
      await load()
      list.value = payload.id
      selectPreset(presets.find((item) => item.id === payload.id) || payload)
      feedback.textContent = `预设“${payload.name}”已保存；它拥有自己的独立 Key 槽。`
    } catch (error) {
      feedback.textContent = `保存失败：${error?.message || error}`
    } finally {
      save.disabled = false
    }
  }
  useSelected.onclick = () => {
    if (!selected) {
      feedback.textContent = "请先选择并保存一个预设。"
      return
    }
    setCloudWidgetValue(node, "cloud_provider", "custom")
    setCloudWidgetValue(node, "my_preset", selected.name)
    setCloudWidgetValue(node, "api_model", selected.model)
    setCloudWidgetValue(node, "cloud_base_url", selected.base_url)
    feedback.textContent = `当前节点已使用已保存连接“${selected.name}”；地址和模型已同步显示。`
    onPresetChanged?.(selected)
    render()
  }
  useTemporary.onclick = () => {
    const baseUrl = String(baseUrlInput.value || "").trim()
    const model = String(modelInput.value || "").trim()
    if (!baseUrl || !model) {
      feedback.textContent = "临时连接也需要填写 Base URL 和模型 ID。"
      return
    }
    setCloudWidgetValue(node, "cloud_provider", "custom")
    setCloudWidgetValue(node, "my_preset", "")
    setCloudWidgetValue(node, "cloud_base_url", baseUrl)
    setCloudWidgetValue(node, "api_model", model)
    feedback.textContent = "已作为临时 OpenAI 兼容连接写入当前节点；不会新增预设。"
    onPresetChanged?.(null)
    render()
  }
  chooseModel.onclick = () => {
    const baseUrl = String(baseUrlInput.value || "").trim()
    if (!baseUrl) {
      feedback.textContent = "请先填写 Base URL，再获取模型列表。"
      return
    }
    createCloudModelPicker(node, {
      provider: "custom",
      credentialId: selected?.credential_id || "custom_default",
      label: selected ? `OpenAI 兼容 · ${selected.name}` : "OpenAI 兼容临时连接",
      defaultModel: String(modelInput.value || "").trim(),
      baseUrl,
      presetName: "",
    }, null, {
      currentModel: () => String(modelInput.value || "").trim(),
      onSelect: (model) => {
        modelInput.value = model
        feedback.textContent = `已把模型 ${model} 填入当前表单；保存或临时使用后生效。`
      },
      hidePresetReset: true,
    })
  }
  configureKey.onclick = () => {
    const baseUrl = String(baseUrlInput.value || "").trim()
    const model = String(modelInput.value || "").trim()
    if (!baseUrl || !model) {
      feedback.textContent = "请先填写 Base URL 和模型 ID，再配置或测试连接。"
      return
    }
    createCloudCredentialModal({
      provider: "custom",
      credentialId: selected?.credential_id || "custom_default",
      label: selected ? `已保存连接 · ${selected.name}` : "OpenAI 兼容临时连接",
      defaultModel: model,
      probeLabel: "小图 + 严格 JSON 真实视觉探针",
      baseUrl,
    }, model)
  }
  remove.onclick = async () => {
    if (!selected || !confirm(`删除预设“${selected.name}”？对应本地 Key 不会自动删除。`)) return
    try {
      await cloudCredentialRequest("/minimaxh3lab/cloud/presets/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selected.name }),
      })
      if (String(node?.widgets?.find((item) => item?.name === "my_preset")?.value || "") === selected.name) {
        setCloudWidgetValue(node, "my_preset", "")
        onPresetChanged?.(null)
      }
      selected = null
      await load()
      feedback.textContent = "预设已删除；其 Key 槽未自动清除，避免误删仍在使用的凭据。"
    } catch (error) {
      feedback.textContent = `删除失败：${error?.message || error}`
    }
  }
  reload.onclick = load
  const dismiss = () => overlay.remove()
  close.onclick = dismiss
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) dismiss()
  })
  document.body.appendChild(overlay)
  load()
}

function isDirectorInputConnected(input) {
  return input?.link !== null && input?.link !== undefined
}

const DIRECTOR_FRAME_WIDGETS = [
  "frame_sequence_limit", "frame_selection_mode", "frame_selection_spec",
]

function refreshDirectorTimelineAuthority(node) {
  if (!node) return
  const timelineInput = (node.inputs ?? []).find((item) => item?.name === "video_timeline_manifest")
  const controlled = isDirectorInputConnected(timelineInput)
  const directLabels = {
    frame_sequence_limit: "直连视频帧上限",
    frame_selection_mode: "直连视频选帧",
    frame_selection_spec: "直连自定义选帧",
  }
  const timelineLabels = {
    frame_sequence_limit: "代表帧数【由上游时间线控制】",
    frame_selection_mode: "选帧方式【由上游时间线控制】",
    frame_selection_spec: "自定义选帧【由上游时间线控制】",
  }
  for (const name of DIRECTOR_FRAME_WIDGETS) {
    const widget = (node.widgets ?? []).find((item) => item?.name === name)
    if (!widget) continue
    widget.label = (controlled ? timelineLabels : directLabels)[name]
    widget.disabled = controlled
  }
  node.setDirtyCanvas?.(true, true)
}

function installDirectorTimelineAuthority(nodeType) {
  if (nodeType.prototype.__minimaxH3TimelineAuthorityInstalled) return
  nodeType.prototype.__minimaxH3TimelineAuthorityInstalled = true
  const refreshSoon = (node) => queueMicrotask(() => refreshDirectorTimelineAuthority(node))
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    refreshSoon(this)
    return result
  }
  const originalOnConfigure = nodeType.prototype.onConfigure
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure?.apply(this, arguments)
    refreshSoon(this)
    return result
  }
  const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange
  nodeType.prototype.onConnectionsChange = function () {
    const result = originalOnConnectionsChange?.apply(this, arguments)
    refreshSoon(this)
    return result
  }
}

function installVideoContextHelpButton(nodeType) {
  if (nodeType.prototype.__minimaxH3VideoContextHelpInstalled) return
  nodeType.prototype.__minimaxH3VideoContextHelpInstalled = true
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    const button = this.addWidget("button", "ℹ 接线与片段范围说明", null, () => {
      alert([
        "这个节点准备一段参考视频，保留代表帧对应的真实时间；它不会调用模型。",
        "",
        "接线：视频加载节点的 IMAGE → 本节点“参考视频帧”；本节点两个输出分别 → 导演的“参考视频帧”和“参考视频时间线”。",
        "",
        "片段范围由上游视频加载 / 裁剪节点决定。本节点不会裁掉视频；例如 24 FPS 下输入 360 帧，就是 15 秒参考片段。",
        "",
        "常用时只需确认源帧率，参考片段时长保持 0，代表帧数保持 48，分布选“均匀覆盖完整时间线”。真实切镜点可以留空。",
        "",
        "连接时间线后，导演里的三个直连选帧设置会标为由上游控制，避免重复选帧。",
      ].join("\n"))
    }, { serialize: false })
    button.serialize = false
    button.serializeValue = () => undefined
    return result
  }
}

function directorInputSourceLabel(node, input) {
  const links = node?.graph?.links
  const link = links?.[input?.link] ?? links?.get?.(input?.link)
  const source = node?.graph?.getNodeById?.(link?.origin_id)
  return String(source?.title || source?.type || source?.comfyClass || "上游节点")
}

function collectDirectorMediaReferences(node) {
  const inputs = node?.inputs || []
  const pictures = []
  for (let port = 1; port <= 9; port += 1) {
    const input = inputs.find((item) => item?.name === `ref_image_${port}`)
    if (!isDirectorInputConnected(input)) continue
    pictures.push({
      kind: "picture",
      token: `<Picture ${pictures.length + 1}>`,
      port,
      source: directorInputSourceLabel(node, input),
    })
  }
  const videoInputs = ["video_frame_sequence", "cloud_video", "comfy_video"]
    .map((name) => inputs.find((item) => item?.name === name))
    .filter(isDirectorInputConnected)
  const video = videoInputs.length ? {
    kind: "video",
    token: "<Video 1>",
    ports: videoInputs.map((input) => String(input?.label || input?.name || "视频输入")),
    sources: videoInputs.map((input) => directorInputSourceLabel(node, input)),
  } : null
  return { pictures, video }
}

function inspectDirectorMediaReferences(prompt, references) {
  const text = String(prompt || "")
  const pictureTags = [...text.matchAll(/<Picture\s+(\d+)>/gi)].map((match) => Number(match[1]))
  const videoTags = [...text.matchAll(/<Video\s+(\d+)>/gi)].map((match) => Number(match[1]))
  const pictureSet = new Set(pictureTags)
  const videoSet = new Set(videoTags)
  const warnings = []
  const invalidPictures = [...new Set(pictureTags)]
    .filter((number) => number < 1 || number > references.pictures.length)
  if (invalidPictures.length) {
    warnings.push(`不存在的图像标签：${invalidPictures.map((number) => `<Picture ${number}>`).join("、")}`)
  }
  const missingPictures = references.pictures.filter((item, index) => !pictureSet.has(index + 1))
  if (missingPictures.length) {
    warnings.push(`已连接但未显式引用：${missingPictures.map((item) => item.token).join("、")}`)
  }
  const invalidVideos = [...new Set(videoTags)].filter((number) => number !== 1 || !references.video)
  if (invalidVideos.length) {
    warnings.push(`与当前输入不匹配的视频标签：${invalidVideos.map((number) => `<Video ${number}>`).join("、")}`)
  }
  if (references.video && !videoSet.has(1)) warnings.push("已连接视频但未显式引用：<Video 1>")
  return {
    warnings,
    missingTokens: [
      ...missingPictures.map((item) => item.token),
      ...(references.video && !videoSet.has(1) ? ["<Video 1>"] : []),
    ],
  }
}

function setDirectorPromptValue(node, value, cursorPosition) {
  const widget = node?.widgets?.find((item) => item?.name === "prompt")
  if (!widget) return false
  widget.value = value
  const input = widget.inputEl
  if (input) {
    input.value = value
    input.dispatchEvent(new Event("input", { bubbles: true }))
    input.dispatchEvent(new Event("change", { bubbles: true }))
    queueMicrotask(() => {
      input.focus?.()
      input.setSelectionRange?.(cursorPosition, cursorPosition)
    })
  }
  node.graph?.change?.()
  node.setDirtyCanvas?.(true, true)
  return true
}

function insertDirectorPromptToken(node, text, cursorState) {
  const widget = node?.widgets?.find((item) => item?.name === "prompt")
  if (!widget) return false
  const current = String(widget.value || "")
  const rawPosition = Number(cursorState.position)
  const position = Number.isFinite(rawPosition)
    ? Math.max(0, Math.min(current.length, rawPosition))
    : current.length
  const before = current.slice(0, position)
  const after = current.slice(position)
  const leftSpace = before && !/\s$/.test(before) ? " " : ""
  const rightSpace = after && !/^\s/.test(after) ? " " : ""
  const insertion = `${leftSpace}${text}${rightSpace}`
  const nextPosition = position + insertion.length
  cursorState.position = nextPosition
  return setDirectorPromptValue(node, before + insertion + after, nextPosition)
}

function createDirectorMediaReferenceModal(node) {
  const promptWidget = node?.widgets?.find((item) => item?.name === "prompt")
  if (!promptWidget) {
    alert("当前节点没有找到用户意图文本框。")
    return
  }
  const promptInput = promptWidget.inputEl
  const cursorState = {
    position: Number.isFinite(Number(promptInput?.selectionStart))
      ? Number(promptInput.selectionStart)
      : String(promptWidget.value || "").length,
  }
  const references = collectDirectorMediaReferences(node)
  const overlay = document.createElement("div")
  Object.assign(overlay.style, {
    position: "fixed", inset: "0", zIndex: "100000", background: "rgba(0,0,0,.62)",
    display: "flex", alignItems: "center", justifyContent: "center", padding: "24px",
  })
  const panel = document.createElement("div")
  Object.assign(panel.style, {
    width: "min(780px, 94vw)", maxHeight: "90vh", overflow: "auto",
    background: "#202124", color: "#eee", border: "1px solid #505157",
    borderRadius: "12px", boxShadow: "0 18px 60px rgba(0,0,0,.55)", padding: "22px",
    fontFamily: "system-ui, sans-serif",
  })
  overlay.appendChild(panel)
  const titleRow = document.createElement("div")
  Object.assign(titleRow.style, { display: "flex", alignItems: "center", gap: "12px" })
  const title = document.createElement("h2")
  title.textContent = "插入 / 检查素材引用"
  Object.assign(title.style, { margin: "0", flex: "1", fontSize: "20px" })
  const close = document.createElement("button")
  close.textContent = "×"
  Object.assign(close.style, {
    width: "36px", height: "36px", border: "0", borderRadius: "8px",
    background: "#34363b", color: "#fff", fontSize: "24px", cursor: "pointer",
  })
  titleRow.append(title, close)
  panel.appendChild(titleRow)

  const explanation = document.createElement("p")
  explanation.textContent = "点击下方素材即可把官方标签插入用户意图。这里按实际连线密集编号，不读取缩略图，也不会改写你的文字。"
  Object.assign(explanation.style, { margin: "8px 0 16px", color: "#b7bac3", lineHeight: "1.55" })
  panel.appendChild(explanation)

  const list = document.createElement("div")
  Object.assign(list.style, { display: "grid", gap: "9px" })
  panel.appendChild(list)
  const audit = document.createElement("div")
  Object.assign(audit.style, {
    marginTop: "16px", padding: "12px", borderRadius: "8px", lineHeight: "1.55",
    border: "1px solid #4e5159", background: "#18191c", whiteSpace: "pre-wrap",
  })
  panel.appendChild(audit)
  const actions = document.createElement("div")
  Object.assign(actions.style, {
    marginTop: "14px", display: "flex", flexWrap: "wrap", gap: "10px", justifyContent: "flex-end",
  })
  const insertMissing = document.createElement("button")
  insertMissing.textContent = "插入全部未引用素材"
  const done = document.createElement("button")
  done.textContent = "完成"
  for (const button of [insertMissing, done]) {
    Object.assign(button.style, {
      border: "0", borderRadius: "7px", padding: "9px 14px", cursor: "pointer",
      background: button === done ? "#3667d6" : "#3b3d43", color: "#fff",
    })
  }
  actions.append(insertMissing, done)
  panel.appendChild(actions)

  const addReferenceRow = (token, detail) => {
    const row = document.createElement("button")
    Object.assign(row.style, {
      width: "100%", display: "grid", gridTemplateColumns: "120px 1fr", gap: "14px",
      alignItems: "center", textAlign: "left", border: "1px solid #4e5159",
      borderRadius: "8px", padding: "11px 13px", cursor: "pointer",
      background: "#292b30", color: "#fff",
    })
    const label = document.createElement("strong")
    label.textContent = token
    label.style.color = "#8fb5ff"
    const description = document.createElement("span")
    description.textContent = detail
    description.style.color = "#d0d2d8"
    row.append(label, description)
    row.onclick = () => {
      insertDirectorPromptToken(node, token, cursorState)
      renderAudit()
    }
    list.appendChild(row)
  }
  for (const item of references.pictures) {
    addReferenceRow(item.token, `物理端口“参考图 ${item.port}” · 来源：${item.source}`)
  }
  if (references.video) {
    addReferenceRow(
      references.video.token,
      `输入：${references.video.ports.join(" + ")} · 来源：${references.video.sources.join(" + ")}`,
    )
  }
  if (!references.pictures.length && !references.video) {
    const empty = document.createElement("div")
    empty.textContent = "当前节点没有检测到已连接的参考图或参考视频。"
    Object.assign(empty.style, {
      border: "1px dashed #555860", borderRadius: "8px", padding: "16px", color: "#b7bac3",
    })
    list.appendChild(empty)
  }

  const renderAudit = () => {
    const result = inspectDirectorMediaReferences(String(promptWidget.value || ""), references)
    if (result.warnings.length) {
      audit.textContent = `非阻断提醒：\n• ${result.warnings.join("\n• ")}\n\n即使没有显式标签，已连接素材仍会发送给提示词模型。`
      audit.style.borderColor = "#8a6b2f"
      audit.style.color = "#f0d394"
    } else if (references.pictures.length || references.video) {
      audit.textContent = "检查通过：当前已连接素材都使用了有效的官方标签。"
      audit.style.borderColor = "#356b4a"
      audit.style.color = "#9cdeb5"
    } else {
      audit.textContent = "没有可检查的素材连线。"
      audit.style.borderColor = "#4e5159"
      audit.style.color = "#b7bac3"
    }
    insertMissing.disabled = result.missingTokens.length === 0
    insertMissing.style.opacity = insertMissing.disabled ? ".5" : "1"
    insertMissing.dataset.tokens = JSON.stringify(result.missingTokens)
  }
  insertMissing.onclick = () => {
    const tokens = JSON.parse(insertMissing.dataset.tokens || "[]")
    if (!tokens.length) return
    insertDirectorPromptToken(node, tokens.join(" "), cursorState)
    renderAudit()
  }
  const dismiss = () => {
    overlay.remove()
    promptInput?.focus?.()
  }
  close.onclick = dismiss
  done.onclick = dismiss
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) dismiss()
  })
  document.body.appendChild(overlay)
  renderAudit()
}

function installDirectorMediaReferenceButton(nodeType) {
  if (nodeType.prototype.__minimaxH3MediaReferenceInstalled) return
  nodeType.prototype.__minimaxH3MediaReferenceInstalled = true
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    const node = this
    const button = node.addWidget("button", "📎 插入 / 检查素材引用", null, () => {
      createDirectorMediaReferenceModal(node)
    }, { serialize: false })
    button.serialize = false
    button.serializeValue = () => undefined
    return result
  }
}

function installCloudCredentialButton(nodeType) {
  if (nodeType.prototype.__minimaxH3CloudCredentialInstalled) return
  nodeType.prototype.__minimaxH3CloudCredentialInstalled = true
  const originalOnConfigure = nodeType.prototype.onConfigure
  nodeType.prototype.onConfigure = function (info) {
    const result = originalOnConfigure?.apply(this, arguments)
    repairCloudDirectorWidgetValues(this, info)
    return result
  }
  const originalOnSerialize = nodeType.prototype.onSerialize
  nodeType.prototype.onSerialize = function (info) {
    this.properties ||= {}
    this.properties[CLOUD_WIDGET_SCHEMA_PROP] = CLOUD_WIDGET_SCHEMA_VERSION
    const result = originalOnSerialize?.apply(this, arguments)
    if (info) {
      info.properties ||= {}
      info.properties[CLOUD_WIDGET_SCHEMA_PROP] = CLOUD_WIDGET_SCHEMA_VERSION
    }
    return result
  }
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    const node = this
    const modelButton = node.addWidget("button", "🔄 刷新 / 选择云端模型", null, () => {
      const config = nodeCloudCredentialConfig(node)
      if (config.provider === "custom" && config.presetName) {
        createCloudPresetManager(node, () => refreshButtonStatus())
        return
      }
      createCloudModelPicker(node, config, (count) => {
        modelButton.name = `🔄 选择 ${config.label} 模型（${count}）`
        node.setDirtyCanvas?.(true, true)
      })
    }, { serialize: false })
    modelButton.serialize = false
    modelButton.serializeValue = () => undefined
    const mediaHelp = node.addWidget("button", "ℹ 参考图 / 视频如何发送", null, () => {
      alert([
        "参考图：按实际连接顺序编号为 <Picture 1>、<Picture 2>……，只发送给提示词 API。",
        "参考视频帧：整个 IMAGE 批次都属于同一个 <Video 1> 的按时间排列帧。直连时由导演的三个选帧设置取样；若同时连接“参考视频准备”的时间线，则以上游代表帧为准。",
        "DeepSeek 600 帧：仅作为官方 DeepSeek 的边界实验档。节点会优先缩图压缩并在报告中给出请求/实际帧数、payload、prompt token 与网络耗时；其他连接仍自动收敛到 300。",
        "Gemini 原生视频：可用“云端原生视频输入”连接 cloud_video，也可把 Comfy 核心 Load Video 的未裁剪文件型 VIDEO 直接接入 comfy_video。auto 会核算视频 Base64 与参考图的联合体积，超预算时走 Files API；远端临时文件在生成调用后自动删除。原生视频与 IMAGE 帧批次必须二选一，便于做可信 A/B。",
        "自动 / 联合上传：参考图与所选视频帧放进同一次多模态请求，便于模型同时理解目标身份和动作时间线。",
        "分阶段上传：先分别分析参考图与视频时间线，再汇总生成 Prompt IR；主要用于 A/B 或单次联合请求不稳定时。",
        "参考视频时间线：把“参考视频准备”的两个输出成对接入导演，可保留真实源帧索引、时间码和切镜分段。此时导演不会再做第二套人工选帧。",
        "重要：以上素材只供云端 API 识别，不会自动传给 MiniMax H3；生成节点仍需另行连接对应图片、视频和音频。",
      ].join("\n\n"))
    }, { serialize: false })
    mediaHelp.serialize = false
    mediaHelp.serializeValue = () => undefined
    const button = node.addWidget("button", "🔑 管理云端连接", null, () => {
      const config = nodeCloudCredentialConfig(node)
      if (config.provider === "custom") {
        createCloudPresetManager(node, () => refreshButtonStatus())
        return
      }
      createCloudCredentialModal(config, nodeCloudModel(node, config), (status) => {
        button.name = status?.configured
          ? `🔑 ${config.label} 已配置（${credentialSourceLabel(status.source)}）`
          : `🔑 管理 ${config.label} 连接（未配置）`
        node.setDirtyCanvas?.(true, true)
      })
    }, { serialize: false })
    button.serialize = false
    button.serializeValue = () => undefined
    const refreshButtonStatus = () => {
      const config = nodeCloudCredentialConfig(node)
      if (config.provider === "custom") {
        const mode = config.presetName ? `已保存：${config.presetName}` : "临时连接"
        modelButton.name = config.presetName
          ? `🗂 OpenAI 兼容连接 / 模型（${config.presetName}）`
          : "🔄 获取 / 选择 OpenAI 兼容模型"
        button.name = `🔑 管理 OpenAI 兼容连接（${mode}）`
        node.setDirtyCanvas?.(true, true)
        return
      }
      button.name = `🔑 正在检查 ${config.label}…`
      fetchCloudCredentialStatus(config).then((status) => {
        button.name = status?.configured
          ? `🔑 ${config.label} 已配置（${credentialSourceLabel(status.source)}）`
          : `🔑 管理 ${config.label} 连接（未配置）`
        node.setDirtyCanvas?.(true, true)
      }).catch(() => {
        button.name = `🔑 管理 ${config.label}（状态不可用）`
        node.setDirtyCanvas?.(true, true)
      })
    }
    const providerWidget = node.widgets?.find((item) => item?.name === "cloud_provider")
    if (providerWidget && !providerWidget.__minimaxH3CloudCredentialCallback) {
      providerWidget.__minimaxH3CloudCredentialCallback = true
      providerWidget.__minimaxH3LastProvider = String(providerWidget.value || "deepseek")
      const originalCallback = providerWidget.callback
      providerWidget.callback = function () {
        const callbackResult = originalCallback?.apply(this, arguments)
        const config = nodeCloudCredentialConfig(node)
        const modelWidget = node.widgets?.find((item) => item?.name === "api_model")
        const nextProvider = String(providerWidget.value || "deepseek")
        const changed = nextProvider !== providerWidget.__minimaxH3LastProvider
        providerWidget.__minimaxH3LastProvider = nextProvider
        if (changed && modelWidget) modelWidget.value = ""
        if (changed && nextProvider !== "custom") {
          setCloudWidgetValue(node, "my_preset", "")
          setCloudWidgetValue(node, "cloud_base_url", "")
        }
        modelButton.name = config.provider === "custom"
          ? "🔄 获取 / 选择 OpenAI 兼容模型"
          : `🔄 刷新 / 选择 ${config.label} 模型`
        queueMicrotask(refreshButtonStatus)
        return callbackResult
      }
    }
    refreshButtonStatus()
    return result
  }
}

app.registerExtension({
  name: "MiniMaxH3Lab.ChineseDisplay",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "MiniMaxH3ReferenceInspector") {
      installInspectorReportPreview(nodeType)
    }
    if (nodeData.name === "MiniMaxH3PromptDirector") {
      installModelRefreshButton(nodeType)
      installDirectorMediaReferenceButton(nodeType)
      installDirectorTimelineAuthority(nodeType)
    }
    if (nodeData.name === "MiniMaxH3CloudDirector") {
      installCloudCredentialButton(nodeType)
      installDirectorMediaReferenceButton(nodeType)
      installDirectorTimelineAuthority(nodeType)
    }
    if (nodeData.name === "MiniMaxH3VideoContext" && isChineseLocale()) {
      installVideoContextHelpButton(nodeType)
    }
    if (!isChineseLocale()) return
    const config = NODE_TRANSLATIONS[nodeData.name]
    if (!config) return

    const originalDisplayName = nodeData.display_name
    nodeData.display_name = config.title
    nodeType.title = config.title

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments)
      localizeNode(this, config, [originalDisplayName, nodeData.name])
      localizeWorkflowText(this)
      return result
    }

    const originalOnConfigure = nodeType.prototype.onConfigure
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments)
      localizeNode(this, config, [originalDisplayName, nodeData.name])
      localizeWorkflowText(this)
      return result
    }
  },

  nodeCreated(node) {
    if (!isChineseLocale()) return
    localizeWorkflowText(node)
  },

  afterConfigureGraph() {
    if (!isChineseLocale()) return
    for (const node of app.graph?._nodes ?? []) localizeWorkflowText(node)
    localizeGroups()
    app.graph?.setDirtyCanvas?.(true, true)
  },
})


// ===== v10: 多槽模块节点「刷新列表」按钮 + 联动下拉 =====
app.registerExtension({
  name: "MiniMaxH3.RefreshButtons",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isMod = nodeData.name === "MiniMaxH3PromptModuleLoader";
    if (!isMod) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const node = this;
      const refreshFiles = async () => {
        try {
          const curFolder = node.widgets?.find(w => w.name === "module_folder")?.value || "";
          const resp = await api.fetchApi("/minimaxh3lab/api/module_files", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ kind: "mod", folder: curFolder }),
          });
          const data = await resp.json();
          const all = data.all_folders || {};
          const folderKeys = Object.keys(all);
          for (const w of node.widgets || []) {
            if (w.name === "module_folder") {
              w.options.values = folderKeys.length ? folderKeys : [""];
            } else if (w.name && /^module_\d$/.test(w.name)) {
              const mods = data.modules || [];
              if (mods.length) w.options.values = mods.map(m => m.title_zh);
            }
          }
          node.setDirtyCanvas(true, true);
        } catch (e) {
          console.error("[MiniMaxH3] 刷新列表失败", e);
        }
      };
      const btn = node.addWidget("button", "\u{1f504} 刷新列表", null, refreshFiles);
      btn.serialize = false; // 按钮不入工作流序列化
      // module_folder 变化时自动刷新模块下拉
      const folderW = node.widgets?.find(w => w.name === "module_folder");
      if (folderW) {
        const origCb = folderW.callback;
        folderW.callback = (v) => {
          origCb?.(v);
          refreshFiles();
        };
      }
      return r;
    };
  },
});
