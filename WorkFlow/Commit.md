# Commit 訊息的藝術：撰寫專業、清晰且高效的 Git 提交日誌

## 一、為什麼 Commit 訊息如此重要？

很多人在學習 Git 時，只專注於指令操作，卻忽略了 Commit 訊息的品質。一個糟糕的 Commit 訊息（例如 `"fix bug"`, `"update"`, `"wip"`）會帶來許多問題。反之，一個好的 Commit 訊息能帶來以下巨大好處：

1. **加速 Code Review**：審查者光看訊息就能快速了解這次變更的目的與範疇，不需要從頭猜測或通讀所有程式碼，大幅提升審查效率。
2. **簡化問題追溯 (Debugging)**：當線上發生問題時，你可以透過 `git log` 或 `git blame` 快速瀏覽提交歷史，從訊息中定位到可能引入問題的變更，而不是大海撈針。
3. **自動產生版本日誌 (Changelog)**：如果團隊遵循一致的格式（例如 Conventional Commits），就可以使用工具自動從 Git 歷史中產生精美的版本更新日誌，省去手動整理的麻煩。
4. **促進團隊溝通**：Commit 歷史本身就是一份最精確的專案開發日誌。它告訴了所有成員（包括幾個月後的你自己）專案是如何演進的，每個變更背後的「為什麼」是什麼。
5. 可參考[This link](https://github.com/OpenBB-finance/OpenBB)

## 二、什麼是好的 Commit 訊息？Conventional Commits 規範

為了讓訊息標準化，社群發展出了一套廣受歡迎的規範——**Conventional Commits**。它不僅具備上述所有優點，更已成為許多開源專案與企業團隊的標準。

一個完整的 Conventional Commit 訊息結構如下：

```bash
<類型>[<範圍>]!: <標題>

[<內文>]

[<註腳>]
```

* **標題 (Header)**：**必需**。包含類型、可選的範圍和描述。
* **內文 (Body)**：**可選**。對變更的詳細描述。
* **註腳 (Footer)**：**可選**。通常用來記錄重大變更 (Breaking Change) 或關聯的 Issue 編號。

### 1. 標題 (Header)

這是 Commit 訊息中最重要的一行，格式為 `<類型>(<範圍>): <標題>`。

#### **類型 (Type)** - **必需**

用來指明這次提交的**性質**。常見的類型有：

* **`feat`**: **新功能 (Feature)**。例如：新增使用者登入功能。
* **`fix`**: **修復 Bug (Bug Fix)**。例如：修復登入按鈕無法點擊的問題。
* **`docs`**: **文件 (Documentation)**。僅修改文件，例如 README.md 或註解。
* **`style`**: **程式碼風格**。不影響程式碼運行的變動，例如：調整縮排、修正拼字、移除空白行。
* **`refactor`**: **重構 (Refactoring)**。既不是新增功能，也不是修復 bug 的程式碼結構調整。
* **`test`**: **測試**。新增或修改測試案例。
* **`chore`**: **雜務 (Chore)**。建構流程、輔助工具的變動，例如：修改 `.gitignore` 或更新套件版本。
* **`perf`**: **效能 (Performance)**。提升程式碼效能的變更。

#### **範圍 (Scope)** - **可選**

用來指明這次提交**影響的範圍**。它可以是某個模組、某個頁面或某個功能。

* **範例**：`feat(api): ...`, `fix(login-page): ...`, `docs(contributing): ...`

#### **標題 (Subject)** - **必需**

用簡潔的文字描述這次提交的**目的**。

* **以動詞原形開頭**：例如用 `add` 而不是 `added` 或 `adds`。
* **首字母小寫**。
* **結尾不加句號**。

**好的標題範例：**

```bash
feat(auth): add password reset functionality
fix(api): correct user data validation logic
```

### 2. 內文 (Body) - 可選

當標題無法完整說明變更時，就應該使用內文。內文與標題之間必須有一個**空白行**。

內文用來解釋：

* **變更的動機 (Why)**：為什麼需要這次變更？解決了什麼問題？
* **與之前的差異 (What)**：跟舊的實作方式有什麼不同？

**範例：**

```bash
fix(parser): handle multi-byte characters correctly

The previous implementation assumed single-byte characters, which caused
errors when processing strings containing emojis or CJK characters.

This commit updates the string iteration logic to correctly handle
multi-byte UTF-8 characters, ensuring proper parsing for all inputs.
```

### 3. 註腳 (Footer) - 可選

註腳與內文之間也必須有一個**空白行**。主要用於兩種情況：

#### **重大變更 (Breaking Changes)**

如果你的提交包含了不向下相容的 API 修改，必須在註腳中以 `BREAKING CHANGE:` 開頭進行說明。也可以在類型/範圍後面加上 `!` 來強調。

**範例：**

```bash
refactor(user)!: rename user ID field from `uid` to `id`

BREAKING CHANGE: The user ID field in the API response has been renamed
from `uid` to `id` to align with the database schema.
Clients consuming this API will need to update their data models.
```

#### **關聯 Issue**

如果這次提交是為了解決某個特定的 Issue，可以在註腳中引用它。

**範例：**

```bash
fix(checkout): prevent duplicate order submission

Closes: #123
```

## 三、實戰對比：好的 vs. 壞的 Commit

| 壞的 Commit 訊息 | 好的 Commit 訊息 | 優點 |
| :--- | :--- | :--- |
| `fixed stuff` | `fix(payment): resolve race condition in payment processing` | 清晰指出修復了什麼問題、在哪個模組 |
| `more work on login` | `feat(auth): implement OAuth2 with Google provider` | 明確說明新增了什麼具體功能 |
| `Update README` | `docs: add setup instructions for new developers` | 讓貢獻者知道文件更新的內容 |
| `refactored some code` | `refactor(utils): extract date formatting logic into a helper` | 說明了重構的具體操作與目的 |

## 四、總結與最佳實踐

1. **擁抱規範**：將 **Conventional Commits** 作為你的團隊標準。
2. **原子化提交**：一個 Commit 只做一件相關的事情。
3. **解釋「為什麼」**：在內文中，多著墨於變更的動機，而不僅僅是「做了什麼」。
4. **善用工具**：可以使用如 `commitizen` 等工具來輔助你撰寫符合規範的 Commit 訊息。

養成撰寫高品質 Commit 訊息的習慣，是一項投資報酬率極高的技能。它將為你和你的團隊在未來的專案維護中，省下無數的時間與精力。
