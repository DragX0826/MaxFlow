# MaxFlow: Kaggle Submission Guide

## 1. Prepare Data Package
I have already created a zip file for you: `maxflow-core.zip`.
It contains the model checkpoint and the library code.

**If you need to recreate it manually:**
1.  Navigate to `kaggle_submission/maxflow-core`.
2.  Select all files and folders inside.
3.  Zip them into `maxflow-core.zip`.

## 2. Upload to Kaggle
1.  Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2.  Click **New Dataset**.
3.  Drag and drop `maxflow-core.zip`.
4.  Name the dataset: `maxflow-core` (or similar).
5.  Create.

## 3. Create Notebook
1.  Go to [Kaggle Kernels](https://www.kaggle.com/code).
2.  Click **New Notebook**.
3.  In the sidebar (right or left), click **Add Data**.
4.  Search for your `maxflow-core` dataset and add it.
5.  **Verify**: You should see `/kaggle/input/maxflow-core` in the input section.

## 4. Run Pipeline
1.  Open `kaggle_one_click_pipeline.py` on your local machine.
2.  Copy the entire content.
3.  Paste it into the first cell of your Kaggle Notebook.
4.  **Run All**.

## 5. Result
The notebook will:
- Install dependencies (rdkit, torch_geometric, etc.).
- Load the model from your dataset.
- Download the test target (7SMV.pdb).
- Run inference.
- Generate metrics and plots (Figure 1, Figure 2).
- **Automatically zip all results and the model into `maxflow_results.zip`.**

## 6. Download Results
1.  Wait for the notebook to finish (check the "Output" tab or sidebar).
2.  Look for **`maxflow_results.zip`**.
3.  Download it. This file contains:
    - `maxflow_pretrained.pt` (The trained model)
    - `fig1_speed_accuracy.pdf` (Benchmark Plot)
    - `fig2_pareto.pdf` (Pareto Frontier Plot)
    - `results_table.tex` (Latex Table)
