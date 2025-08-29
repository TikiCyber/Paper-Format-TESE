#!/usr/bin/env python
"""
token_attribution.py
Run Integrated-Gradients & SHAP on sentences for a BERT sentiment model.

Usage
-----
$ pip install torch transformers captum shap matplotlib
$ python token_attribution.py "Great, another Monday stuck in traffic..." \
                              --model textattack/bert-base-uncased-SST-2
"""
import argparse, pathlib, json, warnings
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from captum.attr import LayerIntegratedGradients
import shap

# ---------- helpers ----------------------------------------------------------
def integrated_gradients(sentence, model, tokenizer, class_idx=1, steps=50):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")
    baseline = tokenizer("", return_tensors="pt",
                         max_length=inputs["input_ids"].size(1),
                         padding="max_length", truncation=True)

    def fwd(inp_emb, att_mask):
        return model(inputs_embeds=inp_emb,
                     attention_mask=att_mask).logits[:, class_idx]

    lig = LayerIntegratedGradients(fwd, model.bert.embeddings)
    attributions, _ = lig.attribute(
        inputs_embeds=model.bert.embeddings(**inputs),
        baselines=model.bert.embeddings(**baseline),
        additional_forward_args=(inputs["attention_mask"],),
        n_steps=steps,
        return_convergence_delta=True
    )
    word_scores = attributions.sum(dim=-1).squeeze(0).detach()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, word_scores


def plot_ig(tokens, scores, outfile="ig_bar.png"):
    scores = scores / torch.norm(scores)
    plt.figure(figsize=(max(6, 0.45 * len(tokens)), 1.8))
    plt.bar(range(len(tokens)), scores, color="salmon")
    plt.xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def shap_text(sentence, hf_pipeline, tokenizer, out_html="shap_vis.html"):
    explainer = shap.Explainer(hf_pipeline, shap.maskers.Text(tokenizer))
    sv = explainer([sentence])
    shap.save_html(out_html, shap.plots.text(sv[0], display=False))
    return sv[0]


# ---------- main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", help="Input sentence for attribution")
    parser.add_argument("--model", default="textattack/bert-base-uncased-SST-2",
                        help="HF model ID (binary or multi-label sentiment)")
    parser.add_argument("--outdir", default="attribution_out",
                        help="Directory to store results")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model “{args.model}”…")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    pipe = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, return_all_scores=False)

    # IG ----------------------------------------------------------------------
    print("Running Integrated Gradients…")
    tokens, ig_scores = integrated_gradients(args.sentence, mdl, tok)
    plot_ig(tokens, ig_scores, outdir / "ig_bar.png")
    ig_dict = {tok: float(s) for tok, s in zip(tokens, ig_scores)}
    (outdir / "ig_scores.json").write_text(json.dumps(ig_dict, indent=2))

    # SHAP --------------------------------------------------------------------
    print("Running SHAP…")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sv = shap_text(args.sentence, pipe, tok, out_html=outdir / "shap_vis.html")

    print(f"\nDone!  • IG bar plot → {outdir/'ig_bar.png'}"
          f"\n       • IG raw scores → {outdir/'ig_scores.json'}"
          f"\n       • SHAP HTML vis → {outdir/'shap_vis.html'}")

if __name__ == "__main__":
    main()