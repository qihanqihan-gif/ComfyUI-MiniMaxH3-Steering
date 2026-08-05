# Example workflow note

The plugin does not redistribute the community Work-Fisher workflow.

A local-only derivative of that workflow was used for development testing
(three-node chain: Media Prep → Inspector → native ReferenceToVideo, with the
Profiler between model loader and guider). It is not part of this repository —
recreate it from the official Work-Fisher workflow if needed.

Its active reference-image path is:

```text
Load Image
  -> MiniMax H3 Reference Media Prep
  -> MiniMax H3 Reference Inspector
  -> native MiniMax H3 Reference to Video (ref_image_0)
```

The inspector's `profile_context` is connected to `MiniMax H3 Performance
Profiler`, and the profiler is inserted between the model loader and the native
guider/scheduler.

For reference-video testing, change Media Prep to `reference_video`, disconnect
the inspector's image pass-through from `ref_image_0`, and connect its video
pass-through to `ref_video_0`.
