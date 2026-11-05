# qwen2.5-dpo-ultrafeedback

Fine-tuning [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) with Direct Preference Optimization (DPO) on UltraFeedback preference pairs, small enough to run on a free Colab T4 GPU.

This was a learning project. I built the pipeline step by step, wrote the DPO loss in PyTorch to understand how it behaves, and then trained with Hugging Face TRL. My first run overfitted badly, so most of the work went into diagnosing and fixing that.

## Results

Each run was scored on the same 200 held-out validation pairs. The untrained model starts at a loss of 0.693, which is what DPO gives when the model hasn't changed from its reference.

| Run | Training pairs | Learning rate | Epochs | Best validation loss | Accuracy at best step |
|---|---|---|---|---|---|
| 1 | 1,500 | 5e-6 | 2 | 0.730 (0.912 at the end) | 0.61 |
| 2 | 1,500 | 1e-6 | 1 | 0.625 | 0.62 |
| 3 | 2,000 | 2e-6 | 1 | 0.616 | 0.64 |
| 4 | About 400 new pairs, continuing from run 3 | 2e-6 | Stopped at the best step | **0.604** | 0.665 |

Accuracy here means the share of pairs where the model moved toward the preferred answer compared with the original model. It doesn't mean the share of answers that were correct.

**Final test on 200 unseen pairs:** loss **0.606**, accuracy **68.5%**

The validation pairs were used to choose between runs, so the test result is the fairer number. With 200 pairs, the accuracy has a rough 95% range of 62% to 75%. The final model matched its validation loss on unseen data, which suggests the improvement carries over beyond the pairs used to choose it.

The notebook contains runs 3 and 4. Runs 1 and 2 were earlier experiments in the same project.

## What I learned

- **Run 1 overfitted.** Training loss fell to 0.05 while validation loss rose to 0.91. The drop came when the model saw the same pairs again in epoch 2.
- **Accuracy alone can mislead.** Run 1 reached 61% accuracy while its validation loss got worse, because a few large wrong moves can outweigh many small right ones.
- **A lower learning rate, fewer passes and new data fixed it.** Validation loss then stayed below the baseline, and training and validation loss stayed close together.
- **The reference model has to stay fixed.** Run 4 continued from run 3 but kept the original model as its reference, so the losses stayed comparable across runs.

## Method

1. Load `trl-lib/ultrafeedback_binarized` and take a random pool of 6,000 pairs.
2. Keep pairs where the chosen answer scored higher and both conversations fit in 512 tokens. Long pairs are dropped rather than cut, because a cut answer may no longer match its score.
3. Hold out 200 pairs for validation.
4. Record baseline answers on held-out prompts.
5. Train with TRL's `DPOTrainer`, using `beta=0.1`, a batch size of 8 and precomputed reference log-probabilities.
6. Compare the validation metrics and generated answers before and after training.

The loss looks like this:

```text
advantage = (model_chosen - reference_chosen) - (model_rejected - reference_rejected)
loss = -log(sigmoid(beta * advantage))
```

## Sample output changes

The answers changed in mixed ways on three held-out prompts:

- **Open question:** the answer dropped the "As an AI language model" opening and became more direct and structured.
- **Summary:** the answer became longer and included more of the source text, but it also invented details, such as a student finance department and who the stakeholders were.
- **Arithmetic:** the answer stayed the same and stayed wrong.

The numbers show a shift toward the preferred style of answer, but they don't show that the answers became more accurate.

## Limitations

- The training set is small, so the improvement is modest.
- The results come from 200-pair samples, so small differences between runs may be noise.
- UltraFeedback scores came from GPT-4 rather than human raters, so they can carry its biases.
- Dropping long pairs leans the data toward shorter answers.
- DPO shapes which answers the model prefers, not what it knows. An arithmetic prompt was wrong before training and stayed wrong afterward.
- A more detailed answer isn't always a more accurate one. The final model added unsupported details to a summary.

## Run it

Open `Qwen_DPO.ipynb` in Google Colab, choose a T4 GPU runtime, then install the versions used here:

```bash
pip install trl==1.17.1 transformers==5.22.1 datasets==5.1.0 accelerate==1.15.0
```

Colab deletes local files when the runtime ends, so copy any trained model to Google Drive before closing it.

## Credits

- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) by the Qwen team
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), using the binarized version from `trl-lib`
- [TRL](https://github.com/huggingface/trl) by Hugging Face
- [DPO paper](https://arxiv.org/abs/2305.18290) by Rafailov et al.
