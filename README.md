# Latin-to-English Encoder-Decoder Translator

A from-scratch transformer based translation model that translates classical Latin text into English, built with PyTorch.

The training data is the [`grosenthal/latin_english_translation`](https://huggingface.co/datasets/grosenthal/latin_english_translation) dataset from HuggingFace, which contains parallel Latin–English sentence pairs drawn from a wide range of classical and biblical sources.

---

## Model Architecture

The model is a **Transformer encoder-decoder**, implemented from scratch using `torch.nn.Transformer` primitives.

| Component              | Detail                                                                   |
| ---------------------- | ------------------------------------------------------------------------ |
| Architecture           | Transformer encoder-decoder                                              |
| Tokenizer              | Byte-Pair Encoding (BPE) or `bert-base-multilingual-cased`, shared vocab |
| Vocabulary size        | 16,000 (BPE) / ~120,000 (BERT)                                           |
| Encoder/Decoder layers | 3 each                                                                   |
| Attention heads        | 8                                                                        |
| Positional encoding    | Sinusoidal                                                               |
| Optimizer              | Adam (lr=1e-4) with ReduceLROnPlateau                                    |
| Loss                   | Cross-entropy with label smoothing (0.1)                                 |
| Inference              | Greedy decoding, Beam Search                                             |

Both source (Latin) and target (English) share a single tokenizer. A `LengthBucketSampler` groups sequences of similar length together to reduce padding waste during training.

---

## Results

Three variants of the model were trained for 40 epochs and evaluated on the test set. The scores in the table below are computed with NLTK's `sentence_bleu` (averaged over the test set, with smoothing), expressed on the 0–1 scale.

| Model                                                 | NLTK BLEU |
| ----------------------------------------------------- | --------- |
| emb_dim=256, BPE tokenizer                            | 0.048     |
| emb_dim=512, BPE tokenizer                            | 0.060     |
| emb_dim=512, `bert-base-multilingual-cased` tokenizer | 0.1379    |

Holding the architecture fixed and swapping the from-scratch BPE tokenizer for the pre-trained multilingual BERT WordPiece tokenizer **more than doubles BLEU**. The qualitative outputs below show why — the gap is even larger than the BLEU number suggests.

### sacrebleu — Greedy vs. Beam Search

To compare against published numbers, the best model (emb_dim=512, `bert-base-multilingual-cased` tokenizer) was also evaluated with [sacrebleu](https://github.com/mjpost/sacrebleu), which computes **corpus-level** BLEU on the 0–100 scale and uses a fixed internal tokenizer. The two rows differ only in the decoding strategy.

| Inference         | sacrebleu BLEU | 1/2/3/4-gram precision   | BP    |
| ----------------- | -------------- | ------------------------ | ----- |
| Greedy            | 18.40          | 51.8 / 22.8 / 12.7 / 8.4 | 0.977 |
| Beam search (k=5) | 19.15          | 53.9 / 24.4 / 13.9 / 9.2 | 0.947 |

Beam search improves BLEU by **+0.75** by finding higher-probability sequences — precision rises at every n-gram order. The trade-off is a stronger brevity penalty (0.947 vs. 0.977), reflecting beam search's well-known bias toward shorter outputs.

Note that the NLTK and sacrebleu numbers are **not directly comparable**: NLTK averages per-sentence BLEU (which crushes short sentences with few matches) while sacrebleu aggregates n-gram counts across the whole corpus before computing precisions. The same model will score noticeably higher under sacrebleu.

Full sample outputs are in [results/output_256.txt](results/output_256.txt), [results/output_512.txt](results/output_512.txt), and [results/output_512_ver2.txt](results/output_512_ver2.txt).

### emb_dim=256 — Model Collapse (Greedy inference)

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

Every single input — from a one-word sentence to a long complex clause — produces this identical output. The model learned to minimize loss by latching onto a high-frequency phrase pattern rather than learning to translate.

### emb_dim=512 — Learning Real Translations (Greedy inference)

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

### emb_dim=512, BERT tokenizer — Cleaner Subwords, Fluent Output (Greedy inference)

Keeping the architecture fixed and switching only the tokenizer to `bert-base-multilingual-cased` produces noticeably more fluent translations, often within a few word choices of the reference:

```
Latin:    "ipsi autem sicut Adam transgressi sunt pactum
           ibi praevaricati sunt in me"
English:  "But they, like Adam, have transgressed the covenant,
           there have they dealt treacherously against me."
Model:    "But they themselves, as Adam, went over the covenant,
           there they transgressed in me."

Latin:    "et veniebant de cunctis populis ad audiendam sapientiam
           Salomonis et ab universis regibus terrae qui audiebant
           sapientiam eius"
English:  "And they came from all nations to hear the wisdom of Solomon,
           and from all the kings of the earth, who heard of his wisdom."
Model:    "And there came of all the people to hear the wisdom of Solomon,
           and from all the kings of the earth, that heard his wisdom."

Latin:    "et ad populum sic locutus est haec dicit Dominus Deus
           Israhel trans fluvium habitaverunt patres vestri ab initio
           Thare pater Abraham et Nahor servieruntque diis alienis"
English:  "And he spoke thus to the people: Thus saith the Lord the God
           of Israel: Your fathers dwelt of old on the other side of the
           river, Thare the father of Abraham, and Nachor: and they
           served strange gods."
Model:    "And thus spoke to the people: Thus saith the Lord the God of
           Israel: Your fathers dwelt beyond the river, from the beginning
           of Thare, the father of Abraham, and Nachor: and they served
           strange gods."
```

Two things stand out compared with the from-scratch BPE runs above. First, the visible subword spacing (`fooli sh ly`, `as s`, `fe et`) is gone — the pre-trained WordPiece vocabulary covers the target English far more cleanly. Second, rare proper nouns (`Salomon`, `Thare`, `Abraham`, `Nahor`) survive the encoder–decoder round-trip intact, which the from-scratch BPE model frequently mangled. The remaining errors are subtler: dropped or rearranged clauses rather than collapsed syntax.

---

## Future Work

### Pre-trained Encoder / Embeddings

Use a pre-trained BERT model as encoder to improve representation quality.

### Flash Attention

Replace the standard `torch.nn.Transformer` attention with [FlashAttention](https://github.com/Dao-AILab/flash-attention) to speed up both training and inference. FlashAttention computes exact attention in tiled blocks that fit in SRAM, avoiding the O(N²) materialization of the attention matrix in HBM — yielding wall-clock speedups and a smaller memory footprint, especially at longer sequence lengths. This would let the model train on longer contexts within the same GPU budget and reduce per-token latency at decode time.
