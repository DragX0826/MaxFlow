# Kaggle Deployment Guide: MaxFlow v48.5 (Golden Edition)

本指南旨在幫助您在 Kaggle Notebook 環境中順利執行 **MaxFlow v48.5**。該版本已針對 T4 GPU 和 9 小時執行限制進行了深度優化。

## 1. 準備環境 (Environment Setup)

在 Kaggle 創建新的 Notebook：
1.  **Accelerator**: 選擇 **GPU T4 x2** (或單張 T4，代碼會自動適應)。
2.  **Internet**: 確保右側面板的 **Internet on** 已開啟（用於下載 ESM-2 權重）。
3.  **Persistence**: 建議開啟 **Files Only** 或 **Variables and Files** 以保留中間數據。

## 2. GitHub 整合流程 (GitHub Workflow) - **推薦方式**

使用 GitHub 可以更方便地同步代碼並在 Kaggle 中下載。

### 第一步：在本地推送到 GitHub
在您的電腦目錄（`d:\Drug\kaggle_submission`）執行：

```bash
git init
git add lite_experiment_suite.py *.pdb
git commit -m "MaxFlow v48.5 Golden Submission"
# 替換為您的 repo 網址
git remote add origin https://github.com/您的用戶名/MaxFlow_ICLR.git
git push -u origin main
```

### 第二步：在 Kaggle 下載並跑
在 Kaggle Notebook 中，直接克隆並執行：

```bash
# 1. 克隆倉庫
!git clone https://github.com/您的用戶名/MaxFlow_ICLR.git
%cd MaxFlow_ICLR

# 2. 安裝依賴
!pip install rdkit meeko biopython fair-esm scipy

# 3. 執行
!python lite_experiment_suite.py --target 7SMV --steps 300 --batch 16
```

## 3. 手動上傳方式 (Manual Upload)

在 Notebook 的第一個 Cell 中執行以下命令安裝核心依賴：

```bash
!pip install rdkit meeko biopython fair-esm scipy
```

## 4. 執行命令 (Execution)

### A. 標準測試 (單一目標)
先跑一個目標確認環境正常：

```bash
!python lite_experiment_suite.py --target 7SMV --steps 300 --batch 16
```

### B. 消融實驗 (Ablation Study)
跑出論文所需的比較數據：

```bash
!python lite_experiment_suite.py --ablation --steps 300 --batch 16
```

## 5. 斷線恢復 (Checkpoint & Resume)

MaxFlow v48.5 具備 **Segmented Training** 功能。
-   **機制**：程序每 100 步會自動保存 `maxflow_ckpt.pt`。
-   **恢復**：如果 9 小時超時斷線，您只需**再次運行相同的腳本命令**。代碼會自動檢測當前目錄下的 `maxflow_ckpt.pt` 並從中斷的地方繼續優化。

## 6. 結果提取 (Retrieving Assets)

運行結束後，所有報告會生成在 `/kaggle/working/` 下：
-   `MaxFlow_v48.5_Kaggle_Golden.zip`: 包含所有 PDF, PDB, Tex 表格。
-   `fig1_trilogy_*.pdf`: 3D 姿態演化圖。
-   `technical_report.md`: 最終數據匯總。

---
**💡 小貼士**：建議使用 Kaggle 的 **"Save Version" -> "Save & Run All (Commit)"**。這樣程序會在後台運行 9 小時，完成後您可以在 Viewer 中直接下載生成的 Zip 包。
