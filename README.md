# Assignment 3: Text Generation Using GPT-2

**ETMMGA304 – Generative AI and Its Applications** · MCA, K.R. Mangalam University
· Lab 3 (lab manual pages 10–12)

Compares greedy decoding with nucleus sampling on two prompts, fine-tunes only GPT-2's last Transformer block on study-tip examples (prompt tokens masked with -100), and compares the base and tuned models on held-out topics.

**Settings:** openai-community/gpt2; max_new_tokens 45; sampling T ∈ {0.6, 1.0}, top_p 0.9, seeds {42, 43}; fine-tune transformer.h[-1] with AdamW lr 2e-5 for 3 epochs.

## Files

```
GenAI-Assignment-3/
├── README.md
├── LAB_RECORD.md        # observation tables, results, viva answers
├── requirements.txt
├── lab3_gpt2_text_generation.py
├── lab3_gpt2_text_generation.ipynb
├── data/                # created automatically (datasets)
└── lab_outputs/         # created automatically (results)
```

## Setup

Python 3.10 or newer is required.

```bash
git clone https://github.com/Aman13082001/GenAI-Assignment-3.git
cd GenAI-Assignment-3
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Hardware:** Runs on CPU. The first run downloads GPT-2 (about 550 MB).

### CPU or CUDA GPU

On Windows, `pip install -r requirements.txt` installs the CPU build of PyTorch.
For an NVIDIA GPU, install the CUDA build first (choose the CUDA version for
your driver at [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check with `python -c "import torch; print(torch.cuda.is_available())"`.

## Run

```bash
python lab3_gpt2_text_generation.py # options: --epochs 3 --skip-decoding
```

Outputs:

- `lab_outputs/study_tip_gpt2/` (fine-tuned model + tokenizer)
- `lab_outputs/lab3_decoding.csv`
- `lab_outputs/lab3_held_out.csv`

## Google Colab

Open [lab3_gpt2_text_generation.ipynb](lab3_gpt2_text_generation.ipynb) in Colab (File → Open notebook → GitHub, then paste this repo's URL). Choose **Runtime → Change runtime type → T4 GPU** if you need a GPU, then run the cells in order. Download `lab_outputs/` before the session ends.

## Lab record

The observation tables, expected ranges and viva answers are in
[LAB_RECORD.md](LAB_RECORD.md). Fill the tables in from your own run.
