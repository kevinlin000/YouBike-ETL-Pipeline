# 安全政策

本專案是作品集展示，不是正在營運的正式服務。這份文件說明目前 repository 的安全邊界，以及如果發現問題時應如何回報。

## 支援範圍

目前維護範圍以 `main` branch 為準。安全相關文件與檢查包含：

- [`docs/configuration_security.md`](docs/configuration_security.md)：環境變數、secret handling、demo / production-like 設定邊界與 API threat model。
- [`docs/dependency_security.md`](docs/dependency_security.md)：dependency manifest validation 與 supply-chain 邊界。
- [`docs/operations.md`](docs/operations.md)：本機 demo、health/readiness、metrics、rollback 與故障排查。

## 已有防護

- `.env`、dbt local profiles 與本機 artifact 不應提交。
- CI 會執行 `make validate-config`，檢查常見 secret pattern 與設定邊界。
- CI 會執行 `make validate-dependencies`，避免無版本約束 dependency、direct URL / VCS / local path dependency 與 wildcard version。
- FastAPI 提供 request validation、`/health`、`/ready`、`X-Request-ID`、JSON request log 與 Prometheus-style `/metrics`。
- 推論 endpoint 支援可選 `X-API-Key` 與單節點 rate limit guardrail。
- Demo mode 不讀模型檔、不連 MySQL，適合展示 API contract 與 dashboard flow。

## 非目標

這個 repository 不主張已具備完整 production security controls，例如：

- 正式身份系統或 OAuth / OIDC。
- API gateway 或 WAF。
- 企業級 secret scanning / SCA。
- SBOM 與 hash-pinned install。
- secret rotation 與 incident response process。
- 多節點或分散式 rate limiting。

## 回報方式

請不要在 issue、PR 或公開留言中貼上真實 credential、token、database dump 或雲端資源資訊。

如果發現安全或設定問題，建議先用不含敏感資訊的方式描述：

- 受影響的檔案或 endpoint。
- 可重現步驟。
- 預期行為與實際行為。
- 是否涉及 secret、dependency、API validation、rate limit 或 demo / production-like 設定邊界。

若問題包含敏感資訊，請先移除或遮蔽敏感值後再回報。
