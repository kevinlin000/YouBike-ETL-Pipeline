# 文件語言策略

這個 repository 採用「中文說明為主、英文說明為輔」的文件策略。

English short version: Chinese is the primary language for narrative, interview delivery, and claim boundaries. English is supporting material for quick scanning, resume keywords, and non-Chinese reviewers.

## 為什麼中文優先

這個專案主要用途是作品集、面試說明和口頭展示。中文優先有三個好處：

- 面試時比較容易自然說清楚專案脈絡。
- 可以把 ML 邊界、demo mode 限制、後端 / AI 應用定位講得更精準。
- README 主線可以更貼近實際展示方式，而不是只寫成工具清單。

主要讀者需要的是：

- 清楚的專案故事線。
- 面試時可以安全使用的說法。
- dashboard demo flow。
- ML claim boundary。
- backend / AI application positioning。

這些內容放在 `README.md` 和中文優先文件最合理。

## 英文文件的角色

英文文件應該是輔助材料，不需要完整複製中文文件。

英文適合用來做：

- 非中文 reviewer 的快速掃描。
- 履歷 bullet 和關鍵字支撐。
- 短版 backup pitch。
- 高層次 API、architecture、model-serving context。

除非有特定職缺或 reviewer 需要，不建議把每份中文 walkthrough 都翻成等長英文版。那會增加維護成本，也容易讓兩份說法不同步。

## 目前分工

| Purpose | Primary file |
| --- | --- |
| Main portfolio entry | `README.md` |
| English overview | `README.en.md` |
| Backend / AI role framing | `docs/backend_ai_positioning.md` |
| Demo script | `docs/demo_walkthrough.md` |
| API contract detail | `docs/api_contract_walkthrough.md` |
| ML claim boundary | `docs/ml_modeling_audit.md`, `docs/lstm_evaluation_report.md` |

## 維護原則

之後新增文件時照這個規則：

1. 需要幫助面試表達、專案脈絡、claim boundary 的內容，用中文寫完整。
2. 需要給國際 reviewer、履歷、GitHub 快速掃描的內容，補短英文摘要。
3. 技術 identifiers、commands、endpoint names、metrics、code terms 維持英文。
4. 不維護兩份很長的平行版本，除非那份內容是公開 README 的核心入口。
