# Configuration Files

Configs retained for the synth-u VerbalTS + SAE pipeline.

```
configs/
├── cttp/
│   └── cttp_synth-u.yaml        # CTTP evaluation config (references ./checkpoints/longclip)
├── generators/
│   └── verbalts_synth-u.yaml    # VerbalTS training config for synth-u
└── README.md
```

## Path placeholders

`pretrain_model_path: ./checkpoints/longclip` must point to a local
LongCLIP-GmP-ViT-L-14 directory (see the top-level README, section
"LongCLIP (CLIP) text encoder").

Released CTTP runtime files (`model_configs.yaml`, `clip_model_best.pth` for
synth-u) are downloaded from `mldi-lab/ConTSG-Bench-Checkpoints` — see the
top-level README, section "Released CTTP resources (synth-u)".
