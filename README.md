# Run
python orchestrator.py
```

**W&B runs will look like:**
```
qwen-mt/
├── qwen2.5-3b-part1        (train)
├── qwen2.5-3b-part1-eval   (eval)
├── qwen2.5-3b-part2        (train, uses part1 adapter)
├── qwen2.5-3b-part2-eval   (eval)
├── qwen2.5-3b-part3        (train, uses part2 adapter)
├── qwen2.5-3b-part3-eval   (eval)
...
```

**Output structure:**
```
outputs/
├── qwen2.5-3b-part1/best_model/
├── qwen2.5-3b-part2/best_model/
├── qwen2.5-3b-part3/best_model/
└── progress.json