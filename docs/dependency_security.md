# Dependency Security 與 Supply-Chain 邊界

本文件說明 Python dependency manifests 的維護方式，以及本專案目前能保證與不能保證的 supply-chain security 邊界。

## Dependency Manifests

本專案保留多份 requirements，原因是展示場景不同：

| 檔案 | 用途 | 規則 |
| --- | --- | --- |
| `requirements.txt` | 歷史 notebook / analysis 環境 | 使用 exact pins，保留分析環境可追溯性 |
| `requirements-test.txt` | CI 與本機測試 | 使用明確版本下限，避免裸套件名稱 |
| `requirements-dev.txt` | 開發入口 | 只引用 `requirements-test.txt` |
| `requirements-dbt.txt` | dbt analytics scaffold | 保留 `dbt-core>=1.7,<1.8` 與 `dbt-mysql==1.7.0` 的相容性約束 |
| `requirements_app.txt` | FastAPI + Streamlit app | 使用明確版本下限，避免無約束安裝 |
| `api/requirements.txt` | API container/runtime | 使用明確版本下限，避免無約束安裝 |

## CI 檢查

CI 會執行：

```bash
make validate-dependencies
```

這個檢查會擋下：

- 裸套件名稱，例如 `fastapi`。
- direct URL、VCS 或 local path dependency，例如 `git+...`、`https://...`、`../local-package`。
- wildcard version，例如 `package==*`。
- 同一份 requirements 中重複宣告同一個 package。
- `requirements.txt` 中非 exact pin 的 dependency。
- 非預期的 `-r` include。

## 可選的 Vulnerability Audit

`make validate-dependencies` 是 deterministic manifest hygiene check，不會連外查 vulnerability database。

若要在本機做 vulnerability audit，可以另外安裝並執行：

```bash
python -m pip install pip-audit
python -m pip_audit -r requirements-test.txt
python -m pip_audit -r requirements_app.txt
```

這類檢查會依外部 advisory database 變動而改變結果，所以目前不放進 CI，避免 portfolio demo 因外部資料源短暫異常而不穩。

## 目前邊界

目前已做到：

- CI 會檢查 requirements hygiene。
- runtime / app requirements 不再使用裸套件名稱。
- analysis lock-style requirements 使用 exact pins。
- config / secret boundary 由 `make validate-config` 檢查。

目前未主張：

- 完整 SCA 平台。
- SBOM 產出。
- 套件簽章驗證。
- hash-pinned installs。
- 自動 dependency upgrade workflow。

## 面試時的說法

可以這樣說：

> 這個作品不是完整 supply-chain security project，但我有補基本 dependency hygiene gate。CI 會檢查 requirements 不可使用裸套件名稱、direct URL、VCS dependency、local path 或 wildcard version；完整 analysis requirements 使用 exact pins，runtime requirements 至少有明確版本下限。若要正式上線，我會再補 SCA、SBOM、hash-pinned install、dependency upgrade policy 和定期 vulnerability audit。

## English Summary

This project uses deterministic dependency-manifest validation in CI. The check blocks unversioned requirements, direct URL / VCS / local-path dependencies, wildcard versions, duplicate package declarations, and non-pinned entries in the historical analysis lock-style requirements file. It is a portfolio-level guardrail, not a full SCA, SBOM, signature-verification, or vulnerability-management platform.
