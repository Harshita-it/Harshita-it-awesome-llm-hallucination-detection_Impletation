# Token-Level Uncertainty and Semantic Entropy for Wrong-Answer Detection

**Course:** AI Tools for Research — Lab 02  
**Author:** Harshita Kushwaha  
**Enrollment:** PDE2026001  
**Institution:** Indian Institute of Information Technology, Allahabad  
**Research task:** Compare token-level predictive uncertainty with multi-sample semantic entropy for identifying wrong answers in closed-book TriviaQA.

> **Repository status — reproducibility scaffold / partial pilot, not a completed benchmark.** The supplied Colab notebook contains a 50-question exploratory pilot and data-preparation outputs. It does not contain a completed 400-question calibration run or 600-question test evaluation. The NLI environment check previously failed with `AttributeError: 'Tee' object has no attribute 'isatty'`; the stream wrapper has been patched in this repository, but a fresh GPU rerun is still required. No final AUROC/AUPRC/ECE/Brier values are claimed. Do not cite pilot labels as final test results.

## 1. Research questions
1. How well do token-level uncertainty scores distinguish wrong from correct short answers?
2. How does answer-span scoring compare with whole-output aggregation, and how much does answer length confound the scores?
3. What detection/cost trade-off results from using a token-level score to gate expensive semantic-entropy estimation?

## 2. Methods represented in the code
- **Token-level uncertainty:** teacher-forced token log-probabilities and predictive entropy; aggregations include mean NLL, sum NLL, maximum NLL, mean entropy and maximum entropy.
- **Semantic entropy:** ten sampled answers are intended to be grouped by bidirectional NLI entailment; entropy is computed over semantic clusters.
- **Correctness proxy:** normalized exact match against TriviaQA aliases. This is a benchmark answerability proxy, not a universal definition of hallucination.
- **Controls planned in protocol:** answer length and a seeded random score.

## 3. Repository map
```text
AITR_Lab02_PDE2026001_TokenUncertainty/
├── README.md
├── requirements.txt                 # broad install constraints; not a frozen lockfile
├── environment_snapshot.txt         # versions observed in the saved Colab output
├── configs/protocol_v1.yaml         # draft protocol; not frozen
├── data/README.md                   # dataset acquisition and data-handling policy
├── notebooks/
│   ├── 00_colab_environment_test.ipynb
│   └── 01_full_experiment_workbook.ipynb  # original supplied notebook; exploratory/incomplete
├── src/uncertainty/                 # data, scoring, prompting, text normalization, seeds
├── scripts/
│   ├── 00_env_test.py               # environment diagnostic
│   ├── 01_build_subset.py           # deterministic TriviaQA subset and split
│   ├── 02_generate.py               # greedy generation, token scoring, 10 samples
│   ├── 03_semantic_entropy.py      # DeBERTa NLI clustering and semantic entropy
│   ├── 04_build_table.py            # join raw records into per-question CSV
│   ├── 05_handcheck.py              # blind label check and error analysis
│   ├── 06_evaluate.py               # audit, metrics, bootstrap and cascade analysis
│   ├── 07_make_latex_tables.py      # result-derived LaTeX tables
│   ├── 08_sampling_check.py         # sampling truncation diagnostic
│   ├── 09_repro_check.py            # compare repeat generation runs
│   ├── 10_extension_length_adjusted.py
│   ├── 11_freeze_protocol.py        # hash and tag the protocol
│   ├── 12_audit_outputs.py          # pre-evaluation audit
│   └── 13_results_outline.py        # evidence-based results/discussion outline
├── tests/                           # CPU helper tests
├── results/                         # intentionally contains no fabricated final metrics
├── logs/                            # run logs (generated locally)
├── paper/                           # LaTeX manuscript source and bibliography
└── *.md                              # protocol, audit, provenance and status records
```

## 4. Environment and installation
The observed Colab session used Python 3.13.16, Linux, a Tesla T4 GPU, CUDA 13.0, PyTorch 2.14.1+cu130, Transformers 5.22.2, Datasets 5.2.0, Accelerate 1.15.0, NumPy 2.4.6, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, PyYAML (version recorded in the generated environment report), and Matplotlib 3.11.2. These are **observed environment details**, not a guarantee that the exact combination is installable on every machine. The original runtime's exact `pip freeze` should be retained when the environment test is rerun.

Recommended platform: Google Colab with a GPU (T4-class or better). The model uses FP16 and the NLI model is large; CPU-only execution is not the intended full-pipeline configuration.

```bash
git clone <YOUR_PUBLISHED_GITHUB_URL>
cd AITR_Lab02_PDE2026001_TokenUncertainty
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For Colab, select **Runtime → Change runtime type → GPU**, restart the runtime, clone/upload this repository, then install dependencies. Avoid blindly reinstalling the Colab-provided PyTorch/CUDA build; check `torch.version.cuda` and `torch.cuda.is_available()` first.

## 5. Reproduction commands

The **complete staged runbook** is in [`EXPERIMENT_PIPELINE.md`](EXPERIMENT_PIPELINE.md). It gives the exact order, commands, output files, test-split gate and current execution limitations. The short commands below are an index; follow the runbook rather than skipping gates.
Run from the repository root.

### 5.1 Static and CPU checks
```bash
python -m compileall -q src scripts tests
python tests/run_tests.py
```
These tests cover helper functions only. Passing them does not validate model loading, NLI clustering, or experimental claims.

### 5.2 Environment gate
```bash
python scripts/00_env_test.py
```
The diagnostic writes its report under `results/` and log under `logs/`. **Stop if any required stage reports FAIL.** For a cheaper CPU-only diagnostic, inspect `python scripts/00_env_test.py --help` and use its documented flags; skipping NLI does not validate the semantic-entropy component.

### 5.3 Dataset acquisition and split creation
```bash
python scripts/01_build_subset.py
```
This downloads TriviaQA validation `rc.nocontext` through Hugging Face Datasets, deduplicates by `question_id`, constructs the declared pool, and writes only question IDs/split assignments plus a manifest. It does not save question text or answer aliases. The config currently declares pool size 1,000, calibration fraction 0.4, and pilot size 50. The protocol is still `draft_ready_to_freeze`; freeze it only after completing the pre-test checklist and recording a protocol hash/tag.

The previously observed saved notebook output showed 17,944 rows before cleaning, 7,984 duplicate IDs removed, and 9,960 remaining. These are historical notebook outputs and must be rechecked on a fresh download.

### 5.4 Full experiment sequence

After the pilot and hand-check pass, run the remaining calibration set:

```bash
python scripts/02_generate.py --split calibration
python scripts/03_semantic_entropy.py --split calibration
python scripts/04_build_table.py
```

Complete `experiment_protocol.md` Section 14 and freeze before touching test rows:

```bash
python scripts/11_freeze_protocol.py --freeze
```

Only then run the final test pipeline:

```bash
python scripts/02_generate.py --split test --allow-test
python scripts/03_semantic_entropy.py --split test --allow-test
python scripts/04_build_table.py
python scripts/12_audit_outputs.py --mode final
python scripts/06_evaluate.py --mode final --frozen
python scripts/10_extension_length_adjusted.py --frozen
python scripts/07_make_latex_tables.py
python scripts/13_results_outline.py
```

See `EXPERIMENT_PIPELINE.md` for the complete pilot, diagnostics, human verification, freeze and error-analysis sequence. These GPU/model commands remain **[TEST REQUIRED]** until executed successfully on the target runtime.

## 6. Dataset acquisition, access and handling
Dataset: TriviaQA, Hugging Face dataset identifier `mandarjoshi/trivia_qa`, configuration `rc.nocontext`, split `validation`. The project loads it at runtime:
```python
from datasets import load_dataset
ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
```
See [`data/README.md`](data/README.md) for source and terms caveat. No raw dataset is committed. The data page notes that the dataset authors do not own copyright in the questions/documents; the repository does not assert a license that was not verified. Check current upstream terms before redistribution. Do not commit raw questions, answer aliases, credentials, model weights or private data.

## 7. Configuration and reproducibility
All planned settings are in [`configs/protocol_v1.yaml`](configs/protocol_v1.yaml): dataset, model, prompt, greedy and sampling generation, scoring, NLI clustering, labels, metrics, bootstrap, controls and cascade gate. The file explicitly remains `draft_ready_to_freeze`. In particular, `margin_justification` and assumed GPU details remain unresolved. Do not call this a frozen protocol until the checklist is completed and the exact config/code commit is tagged.

Seeds currently specified include pool/split seed 42, sampling seed base 1000, and bootstrap seed 2026. Keep generated manifests, environment report, stdout logs, package freeze and Git commit hash with every run.

## 8. Current validation evidence and limitations
- Saved notebook: 92 cells, 72 code cells, no execution-count metadata; partial saved outputs exist.
- Dataset preparation and a 50-question pilot were observed in the notebook.
- Automatic pilot alias matching: 8 correct and 42 wrong out of 50. Exploratory only; not human-audited final labels.
- Semantic clustering was observed for the pilot; mean 24.4 directional NLI calls/question.
- A 30-row blind-check sheet was created, but no completed human labels or agreement statistic is evidenced.
- NLI environment diagnostic failed due to `Tee.isatty`; a later notebook cell also raised `NameError: RUN_PILOT is not defined`.
- No completed calibration run, final test run, paired bootstrap, final metric table, cascade evaluation, or public results archive is present.

See [`NOTEBOOK_EXECUTION_EVIDENCE.md`](NOTEBOOK_EXECUTION_EVIDENCE.md), [`baseline_validation.md`](baseline_validation.md), and [`SUBMISSION_STATUS.md`](SUBMISSION_STATUS.md).

## 9. Minimum reproduction checklist
- [ ] Clone the exact commit and record `git rev-parse HEAD`.
- [ ] Record GPU, CUDA, Python and `pip freeze`.
- [ ] Run CPU tests and environment gate; resolve all FAIL results.
- [ ] Verify dataset row counts, schema, duplicate handling and upstream terms.
- [ ] Complete correctness-label hand-check and document agreement.
- [ ] Finalize and hash the protocol before test evaluation.
- [ ] Run calibration only on calibration questions.
- [ ] Run the untouched test split once under the frozen protocol.
- [ ] Run evaluation, bootstrap intervals, calibration and controls.
- [ ] Archive metrics, logs, seeds and figures; exclude restricted/raw data.
- [ ] Reproduce the paper's tables and figures from the archived outputs.

## 10. Citation
If you use TriviaQA, cite the original dataset paper: Joshi et al., “TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension,” ACL 2017. See `paper/references.bib` for the bibliographic entry included with the manuscript.

## 11. AI assistance and integrity
AI tools assisted with code drafting, debugging suggestions, documentation and manuscript preparation. The student must review, understand and rerun the code, record human verification and disclose AI use according to institute rules. No result in this repository should be represented as final unless it is generated from the documented protocol and archived outputs.

## 12. Publication
This working folder is prepared as a Git repository, but **no public GitHub URL has been created or verified**. After creating a GitHub repository, replace `<YOUR_PUBLISHED_GITHUB_URL>` in the commands above and test the clone instructions from a clean environment.

## 13. Dependency reference files
- `requirements.txt` is the practical install specification and is intentionally not presented as an exact lockfile.
- `requirements-colab-observed.txt` records the versions visible in the supplied notebook environment snapshot. It is a reference only; CUDA/PyTorch compatibility must be checked on the target runtime.
- `environment_snapshot.txt` records the observed hardware/software context. Generate a fresh lock after a successful run with `python -m pip freeze > results/pip-freeze.txt`.

## 14. GitHub publication
See [`GITHUB_UPLOAD_GUIDE.md`](GITHUB_UPLOAD_GUIDE.md) for creating the remote repository, checking ignored files, pushing the project, and verifying the published URL. No GitHub URL is claimed until you create and verify the remote.

## 15. Repository verification
See [`REPOSITORY_STATUS.md`](REPOSITORY_STATUS.md) for the packaging checks and the distinction between static tests, pilot evidence, and the pending full GPU experiment.
