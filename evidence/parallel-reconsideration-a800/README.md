# NeMo AutoModel A800 pilot

The training code was commit `304075221269453836905f41213e40bd21a094ba`. The target was Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`. We used the first 2,048 `train_sft` rows of HuggingFaceH4/ultrachat_200k at revision `8049631c405ae6576f93f445c6b8166f76f5505a`, keeping each row through its first assistant response. This gives the stock Qwen3 tokenizer an unambiguous answer-only loss mask. The SHA-256 of that JSONL was `e0fb0d3ccf44fc590e756c2d00897b118cc6d984a9f55cd770332d2dcc950111`.

The single A800 80 GB PCIe node ran driver 580.105.08, PyTorch 2.10.0+cu128, Transformers 5.18.0, BF16 autocast and FP32 draft parameters. The pilot used `examples/speculative/parallel_reconsideration/qwen3_0_6b_mvp.yaml`, with local target and data paths, a 32-epoch scheduler horizon, checkpointing every 1,500 steps, and consolidated export reserved for the end. `timed_am.py` delegates training to the native recipe and records its logging metrics to stdout.

The run was stopped manually after a clear decrease in training loss. The last log is step 37,410. The saved step 36,000 checkpoint was consolidated and reloaded for the CUDA forward test. The logged loss is the native mean over ten optimizer steps, while the gradient norm comes from the logged boundary step. The curve uses an EMA with span 31 over all logged loss means.
