# GitHub Flow 團隊開發流程指南

本文件旨在提供團隊一個清晰、統一的 Git 工作流程。我們採用 GitHub Flow，這是一套適用於持續部署與整合的簡易開發模式。

## 核心原則

> **`main` 分支永遠是穩定、可隨時部署的正式版本。**

所有開發工作都必須遵循此原則，確保 `main` 分支的程式碼品質。

可參考連結: [OpenBB](https://github.com/OpenBB-finance/OpenBB)

## 日常開發流程：六個步驟

### 1. 建立分支 (Create a Branch)

在開始任何新任務（新功能或錯誤修復）前，請務必從最新的 `main` 分支建立一個獨立的工作分支。

**分支命名規則：**

* **新功能**: `feature/<功能描述>` (例如: `feature/user-login`)
* **修復 Bug**: `fix/<問題描述>` (例如: `fix/submit-button-bug`)

```bash
# 1. 切換到主分支並同步最新版本
git switch main
git pull origin main

# 2. 建立並切換到你的新分支
git switch -c feature/add-user-avatar
```

### 2. 開發與提交 (Add Commits)

在新分支上進行開發。請將你的工作切分成數個小的、有意義的 Commit (提交)，讓每次的變更都清晰可循。

```bash
# 將檔案加入提交清單
git add .

# 提交變更，並撰寫清楚的訊息
git commit -m "feat: 完成使用者頭像上傳元件"
```

### 3. 推送並建立 Pull Request (Open a Pull Request)

當你完成一個階段的開發後，將分支推送到遠端倉庫，並在 GitHub 上建立一個 Pull Request (PR)，請求將你的程式碼合併回 `main` 分支。

```bash
# 第一次推送時使用 -u 參數，讓本地分支追蹤遠端分支
git push -u origin feature/add-user-avatar
```

推送後，請至 GitHub 專案頁面，點擊按鈕為你的分支建立 Pull Request。

**一個好的 PR 應包含：**

* 清晰的標題，說明 PR 的目的。
* 簡短的描述，解釋你做了什麼、為什麼這麼做。
* 在右側指派 (Assign) 給自己，並選擇審查者 (Reviewers)。

### 4. 審查與討論 (Discuss and Review)

PR 是團隊協作的核心。團隊成員會審查你的程式碼並提供回饋。請根據回饋進行修改、再次 Commit 與 Push。所有新的提交都會自動更新到同一個 PR 中。

### 5. 合併 (Merge)

當你的 PR 獲得批准 (Approve) 且所有自動化檢查都通過後，**請直接在 GitHub 的 PR 頁面上，點擊綠色的 "Merge pull request" 按鈕**，將其合併到 `main` 分支。

### 6. 清理 (Clean up)

PR 合併後，原本的開發分支已完成任務。為了保持倉庫整潔，請刪除分支。

```bash
# 1. GitHub 合併後，可直接點擊按鈕刪除遠端分支

# 2. 在本地，切換回 main 分支並同步
git switch main
git pull origin main

# 3. 刪除已用不到的本地分支
git branch -d feature/add-user-avatar
```

## 參考範例 (Real-world Example)

為了幫助你更具體地理解這個流程，這裡提供一個真實世界中的 Pull Request 範例。你可以觀察其中包含的元素，例如：清晰的標題、詳細的描述、自動化檢查、以及最終的合併紀錄。

## 版本發布流程(待執行)

**此流程由專案負責人執行。** 當 `main` 分支準備好發布新版本時，我們將為其建立一個永久的「版本標籤 (Tag)」。

### 1. 建立版本標籤 (Tag)

在本地電腦上，為 `main` 分支的最新 Commit 打上一個附註標籤 (Annotated Tag)。

**版本號命名規則：**
我們遵循**語意化版本 (Semantic Versioning, `v主版號.次版號.修訂號`)**。

* **主版號 (MAJOR)**: 不相容的 API 修改。
* **次版號 (MINOR)**: 向下相容的功能新增。
* **修訂號 (PATCH)**: 向下相容的錯誤修正。

```bash
# 確保 main 是最新的
git switch main
git pull origin main

# 建立標籤
git tag -a v1.1.0 -m "Release version 1.1.0"
```

### 2. 推送標籤 (Push Tag)

標籤需要被明確地推送到遠端倉庫。

```bash
# 推送剛剛建立的標籤
git push origin v1.1.0
```

### 3. 建立 GitHub Release

推送 Tag 後，請至 GitHub 專案的 "Releases" 頁面，點擊 "Draft a new release"，並選擇你剛推送的 Tag。在此頁面撰寫詳細的**版本更新日誌 (Changelog)**，然後發布。
