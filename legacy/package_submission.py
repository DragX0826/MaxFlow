
import os
import zipfile
import sys

def package_project():
    print("📦 Starting MaxFlow Submission Packaging (Reproducibility Protocol)...")
    
    # 定義要打包的資料夾與檔案
    # 這裡確保 maxflow (代碼), checkpoints (模型), data (數據), training_scripts (驗證腳本)
    targets = ['maxflow', 'checkpoints', 'data', 'training_scripts']
    output_filename = 'maxflow-core.zip'
    
    # 檢查必要檔案是否存在 (預設路徑)
    ckpt_path = 'checkpoints/maxflow_pretrained.pt'
    if not os.path.exists(ckpt_path):
        # 嘗試在上一層尋找 (如果是在 scripts 目錄下執行)
        ckpt_path = '../checkpoints/maxflow_pretrained.pt'
        
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: 'maxflow_pretrained.pt' missing in checkpoints/!")
        print("   -> Please verify pre-training or download the weight file.")
        # return # 先不 return，可能是結構不同，交給後續確認

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 添加 readme
        if os.path.exists('README_SUBMISSION.md'):
            zipf.write('README_SUBMISSION.md', 'README_SUBMISSION.md')
        if os.path.exists('SUBMISSION_DEFENSE.md'):
            zipf.write('SUBMISSION_DEFENSE.md', 'SUBMISSION_DEFENSE.md')

        for target in targets:
            # 兼容性路徑檢查
            actual_target = target
            if not os.path.exists(actual_target):
                # 嘗試在 maxflow-core 目錄下
                actual_target = os.path.join('maxflow-core', target)
            
            if not os.path.exists(actual_target):
                print(f"⚠️ Warning: Directory '{target}' not found. Skipping.")
                continue
                
            print(f"   -> Zipping directory: {actual_target} / as {target}/...")
            for root, dirs, files in os.walk(actual_target):
                for file in files:
                    # 排除不必要的緩存文件
                    if file.endswith('.pyc') or '__pycache__' in root:
                        continue
                    
                    file_path = os.path.join(root, file)
                    # 映射到 zip 內的結構
                    # 如果 actual_target 包含 'maxflow-core'，我們要去掉它
                    arcname = os.path.relpath(file_path, os.path.dirname(actual_target) if 'maxflow-core' in actual_target else '.')
                    zipf.write(file_path, arcname)
    
    print(f"\n✅ Success! Upload '{output_filename}' to Kaggle Datasets.")
    print(f"   File size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # 確保在正確的目錄執行
    # 假設腳本位於 d:/Drug/kaggle_submission
    package_project()
