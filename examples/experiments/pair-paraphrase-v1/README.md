# PAWS-Wiki sentence deduplication experiment

Given two English sentences with high lexical overlap, classify whether they convey the same facts. Use the result to evaluate a possible filter for duplicate memory candidates. A real memory application needs its own validation.

The [Google Research dataset](https://github.com/google-research-datasets/paws) uses Wikipedia source sentences, controlled word swaps and back translation. Humans judged paraphrase and fluency. The [dataset license](https://github.com/google-research-datasets/paws/blob/master/LICENSE) permits use for any purpose and welcomes acknowledgment.

Only the pinned Labeled Final Parquet files are used. The preparer hashes all three files, groups pairs by transitively shared normalized sentences, removes groups that cross official splits, and samples one row per group with seed 20261109. Each split has equal label counts. Both final sets are derived from the official test split and have disjoint families. The grouping detects exact normalized matches and reversed pairs. It cannot detect every near duplicate or establish pretraining isolation.

Run these commands from the repository root with PyArrow, huggingface-hub and scikit-learn installed in the project virtualenv.

```sh
.venv/bin/python scripts/prepare_pair_paraphrase.py --download-only
.venv/bin/python scripts/prepare_pair_paraphrase.py
.venv/bin/python scripts/evaluate_pair_baselines.py --freeze
```

The freeze command reads train and dev only. It stores the selected model and its digest in `.local/pair-paraphrase/`. Use only the locally generated serialization. The receiver may score its own held-out split with the frozen model and without the training inputs.
