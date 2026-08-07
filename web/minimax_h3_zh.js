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
      api_key: "API 密钥",
      temperature: "温度",
      max_tokens: "最大令牌数",
      timeout_s: "超时（秒）",
      lmstudio_after_use: "跑完后 LM Studio 模型处理",
      lmstudio_gpu_offload: "LM Studio GPU 加载策略",
      api_reasoning: "模型思考",
      json_mode: "JSON 模式",
      analysis_mode: "视觉分析模式",
      ref_image_1: "参考资产 1",
      ref_image_2: "参考资产 2",
      ref_image_3: "参考资产 3",
      ref_image_4: "参考资产 4",
      ref_image_5: "参考资产 5",
      ref_image_6: "参考资产 6",
      ref_image_7: "参考资产 7",
      ref_image_8: "参考资产 8",
      ref_image_9: "参考资产 9",
      system_module: "提示词模块（接模块节点）",
      enhanced_prompt: "增强后提示词（接官方节点 prompt）",
      report: "诊断报告",
      reference_sheet: "参考素材分析表（JSON）",
    },
    options: {
      task_type: {
        T2VA: "纯文字生视频",
        I2VA: "参考图驱动",
        FL2VA: "首帧锚定",
        Ref2VA: "参考素材",
      },
      rewrite_mode: {
        strict: "严格遵循",
        balanced: "平衡",
        creative: "创意发挥",
      },
      output_language: {
        中文: "中文",
        English: "English",
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
        auto: "自动（默认）",
        off: "关闭思考",
        on: "开启思考",
      },
      json_mode: {
        auto_retry: "自动重试（默认）",
        force: "强制 JSON",
        off: "关闭（纯文本）",
      },
      analysis_mode: {
        auto: "自动（1-2 图单次，3-9 图分阶段，推荐）",
        single: "单次（多图直传）",
        staged: "分阶段（逐素材分析+合并）",
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
    },
    options: {
      scope: { 全部: "全部", T2VA: "T2VA", I2VA: "I2VA", FL2VA: "FL2VA", Ref2VA: "Ref2VA" },
      "（无）": "（无）",
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
    const numberedMatch = String(slot.name || "").match(/^(ref_(?:image|video|video_audio|audio))_(\d+)$/)
    if (numberedMatch) {
      const base = config.fields?.[numberedMatch[1]]
      const n = Number(numberedMatch[2]) + 1
      slot.label = `${base ?? numberedMatch[1]} ${n}`
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

app.registerExtension({
  name: "MiniMaxH3Lab.ChineseDisplay",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "MiniMaxH3ReferenceInspector") {
      installInspectorReportPreview(nodeType)
    }
    if (nodeData.name === "MiniMaxH3PromptDirector") {
      installModelRefreshButton(nodeType)
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
