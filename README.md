# Latin-to-English Neural Machine Translator

A from-scratch neural machine translation model that translates classical Latin text into English, built with PyTorch.

The training data is the [`grosenthal/latin_english_translation`](https://huggingface.co/datasets/grosenthal/latin_english_translation) dataset from HuggingFace, which contains parallel Latin–English sentence pairs drawn from a wide range of classical and biblical sources.

---

## Model Architecture

The model is a **Transformer encoder-decoder**, implemented from scratch using `torch.nn.Transformer` primitives.

| Component | Detail |
|---|---|
| Architecture | Transformer encoder-decoder |
| Tokenizer | Byte-Pair Encoding (BPE), shared vocab |
| Vocabulary size | 16,000 tokens |
| Encoder/Decoder layers | 3 each |
| Attention heads | 8 |
| Positional encoding | Sinusoidal |
| Optimizer | Adam (lr=1e-4) with ReduceLROnPlateau |
| Loss | Cross-entropy with label smoothing (0.1) |
| Inference | Greedy decoding |

Both source (Latin) and target (English) share a single BPE tokenizer, trained on the combined corpus. A `LengthBucketSampler` groups sequences of similar length together to reduce padding waste during training.

---

## Progress: 256 vs 512 Embedding Dimension

Two models were trained for 40 epochs — one with `emb_dim=256` and one with `emb_dim=512`. The difference in outcome is stark.

Full sample outputs on the validation set are in [output_256.txt](output_256.txt) and [output_512.txt](output_512.txt).

### emb_dim=256 — Model Collapse

The smaller model collapsed during training, producing the same repeated phrase regardless of input:

```
Latin:    "vale."
English:  "Farewell."
Model:    "And all nations shall see their face, and shall see their face,
           and their eyes shall be turned away from them."

Latin:    "hoc malunt Thebae."
English:  "This rather Thebes desires."
Model:    "And all nations shall see their face, and shall see their face,
           and their eyes shall be turned away from them."
```

Every single input — from a one-word sentence to a long complex clause — produces this identical output. The model learned to minimise loss by latching onto a high-frequency phrase pattern rather than learning to translate.

### emb_dim=512 — Learning Real Translations

The larger model shows genuine translation ability, capturing structure, vocabulary, and meaning across a variety of sentence types:

```
Latin:    "et alius angelus exivit de templo quod est in caelo habens
           et ipse falcem acutam"
English:  "And another angel came out of the temple which is in heaven,
           he also having a sharp sickle."
Model:    "And the angel went forth out of the temple, that was in heaven,
           hath the false things, he himself held false charge."

Latin:    "et pergens ad Emor patrem suum accipe mihi inquit
           puellam hanc coniugem"
English:  "And going to Hemor his father, he said: Get me this damsel to wife."
Model:    "And to Emmanuor his father, he said: Take this damsel my wife,
           this wife of wife:"

Latin:    "pes fatui facilis in domum proximi et homo peritus
           confundetur a persona potentis"
English:  "The foot of a fool is soon in his neighbour's house:
           but a man of experience will be abashed at the person of the mighty."
Model:    "Thy foot is foolishly in the house: and a man that is experienced
           in thy neighbour's house hold."
```

The 512-dim model demonstrates word-level alignment, correct syntactic ordering, and semantic fidelity — a significant improvement over the collapsed 256-dim model. Remaining errors are largely in rare vocabulary and long-range dependencies.
