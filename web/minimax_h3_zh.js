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
      frame_sequence_limit: "API 最多发送序列帧数",
      frame_selection_mode: "序列帧快捷选择方式",
      frame_selection_spec: "自定义帧序号 / 百分比",
      api_failure_policy: "API / JSON 失败策略",
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
      system_module: "提示词模块（接模块节点）",
      module_manifest: "模块清单（接模块节点第 4 输出）",
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
      api_model: "模型 ID 覆盖（留空使用预设）",
      temperature: "生成温度（预设可按供应商忽略）",
      max_tokens: "Prompt IR 最大输出令牌",
      timeout_s: "单次调用超时（秒）",
      api_reasoning: "思考强度（由云端预设转译）",
      analysis_mode: "参考图与视频帧的上传方式",
      frame_sequence_limit: "参考视频最多上传帧数",
      frame_selection_mode: "参考视频选帧方式",
      frame_selection_spec: "自定义帧序号 / 百分比",
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
      system_module: "提示词模块（接模块节点）",
      module_manifest: "模块清单（接模块节点第 4 输出）",
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
        gemini: "Gemini · 原生多模态（G0 抽帧）",
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
      frame_selection_mode: {
        uniform_full: "均匀全程（包含首尾，默认）",
        uniform_no_edges: "均匀避开首尾（约 10%–90%）",
        custom_indices: "自定义帧序号（支持 -1=最后一帧）",
        custom_percent: "自定义百分比（0–100）",
      },
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
  MiniMaxH3PromptModuleLoader: {
    title: "MiniMax H3 提示词模块 (热加载, 实验)",
    fields: {
      scope: "作用域",
      module_1: "模块 1",
      module_2: "模块 2",
      module_3: "模块 3",
      custom_instructions: "自定义规则（当前工作流）",
      system_prompt_module: "提示词模块（接导演节点）",
      module_preview: "模块预览",
      module_diagnostics: "模块诊断",
      module_manifest: "模块清单（接导演节点）",
    },
    options: {
      scope: { 全部: "全部", T2VA: "T2VA", I2VA: "I2VA", FL2VA: "FL2VA", L2VA: "L2VA", Ref2VA: "Ref2VA" },
      "（无）": "（无）",
    },
  },
  MiniMaxH3ModuleFolderLoader: {
    title: "MiniMax H3 模块文件夹加载器 (独立, 热加载)",
    fields: {
      module_file: "模块文件 1（按路径分类）",
      module_file_2: "模块文件 2",
      module_file_3: "模块文件 3",
      module_file_4: "模块文件 4",
      module_file_5: "模块文件 5",
      system_prompt_module: "提示词模块（接导演节点）",
      diagnostics: "加载诊断",
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
}

function cloudCredentialConfig(provider) {
  return CLOUD_CREDENTIAL_PROVIDERS[String(provider || "deepseek").toLowerCase()]
    || CLOUD_CREDENTIAL_PROVIDERS.deepseek
}

function nodeCloudCredentialConfig(node) {
  const widget = node?.widgets?.find((item) => item?.name === "cloud_provider")
  return cloudCredentialConfig(widget?.value)
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

function createCloudModelPicker(node, config, onModelCountChanged) {
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
  summary.textContent = `当前：${nodeCloudModel(node, config)}；预设默认：${config.defaultModel}`
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
  recommendedOnly.checked = true
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
    const current = nodeCloudModel(node, config)
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
      const limits = item.input_token_limit
        ? ` [in ${Number(item.input_token_limit).toLocaleString()}]` : ""
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
          timeout_s: 30,
        }),
      })
      models = Array.isArray(payload.models) ? payload.models.filter((item) => item?.id) : []
      models.sort((left, right) => Number(Boolean(right.recommended)) - Number(Boolean(left.recommended))
        || String(left.id).localeCompare(String(right.id)))
      onModelCountChanged?.(models.length)
      render()
      if (!models.length) feedback.textContent = "供应商没有返回可用于 generateContent 的模型。"
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
    setCloudModelOverride(node, select.value)
    summary.textContent = `当前：${select.value}；预设默认：${config.defaultModel}`
    feedback.textContent = `已写入模型覆盖：${select.value}`
  }
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
  subtitle.textContent = `${config.label} 官方 API · 凭据不会写入节点、工作流、history 或生成媒体元数据`
  Object.assign(subtitle.style, { color: "#b7bac3", margin: "8px 0 18px" })
  panel.appendChild(subtitle)

  const statusBox = document.createElement("div")
  Object.assign(statusBox.style, {
    padding: "12px 14px", background: "#2a2c31", borderRadius: "8px", lineHeight: "1.65",
    marginBottom: "16px", whiteSpace: "pre-wrap", wordBreak: "break-all",
  })
  panel.appendChild(statusBox)

  const label = document.createElement("label")
  label.textContent = `${config.label} API Key（留空不会改变已保存值）`
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
        }),
      })
      const probe = payload.vision_probe
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

function installCloudCredentialButton(nodeType) {
  if (nodeType.prototype.__minimaxH3CloudCredentialInstalled) return
  nodeType.prototype.__minimaxH3CloudCredentialInstalled = true
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated?.apply(this, arguments)
    const node = this
    const modelButton = node.addWidget("button", "🔄 刷新 / 选择云端模型", null, () => {
      const config = nodeCloudCredentialConfig(node)
      createCloudModelPicker(node, config, (count) => {
        modelButton.name = `🔄 选择 ${config.label} 模型（${count}）`
        node.setDirtyCanvas?.(true, true)
      })
    }, { serialize: false })
    modelButton.serialize = false
    modelButton.serializeValue = () => undefined
    const mediaHelp = node.addWidget("button", "ℹ 参考图与视频帧上传说明", null, () => {
      alert([
        "参考图：按实际连接顺序编号为 <Picture 1>、<Picture 2>……，只发送给提示词 API。",
        "参考视频帧：整个 IMAGE 批次都属于同一个 <Video 1> 的按时间排列帧；节点按“最多上传帧数”和“选帧方式”取样。",
        "自动 / 联合上传：参考图与所选视频帧放进同一次多模态请求，便于模型同时理解目标身份和动作时间线。",
        "分阶段上传：先分别分析参考图与视频时间线，再汇总生成 Prompt IR；主要用于 A/B 或单次联合请求不稳定时。",
        "重要：以上素材只供云端 API 识别，不会自动传给 MiniMax H3；生成节点仍需另行连接对应图片、视频和音频。",
      ].join("\n\n"))
    }, { serialize: false })
    mediaHelp.serialize = false
    mediaHelp.serializeValue = () => undefined
    const button = node.addWidget("button", "🔑 管理云端连接", null, () => {
      const config = nodeCloudCredentialConfig(node)
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
      const originalCallback = providerWidget.callback
      providerWidget.callback = function () {
        const callbackResult = originalCallback?.apply(this, arguments)
        const config = nodeCloudCredentialConfig(node)
        const modelWidget = node.widgets?.find((item) => item?.name === "api_model")
        const modelValue = String(modelWidget?.value || "").trim()
        const knownOtherDefault = Object.values(CLOUD_CREDENTIAL_PROVIDERS).some(
          (item) => item.provider !== config.provider && item.defaultModel === modelValue,
        )
        if (modelWidget && knownOtherDefault) modelWidget.value = ""
        modelButton.name = `🔄 刷新 / 选择 ${config.label} 模型`
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
    }
    if (nodeData.name === "MiniMaxH3CloudDirector") {
      installCloudCredentialButton(nodeType)
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
