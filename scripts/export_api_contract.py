from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app.main import app


DEFAULT_OUTPUT_DIR = Path("docs")

API_EXAMPLES = """# YouBike FastAPI request examples
# 先啟動 demo API：
#   make api-demo
#
# 這些範例使用 API demo mode 的固定資料，用來驗證 request/response
# contract 與 request tracing；它們不是模型效果評估結果。

@base_url = http://127.0.0.1:8000
@request_id = portfolio-demo-001

### 服務基本資訊
GET {{base_url}}/
X-Request-ID: {{request_id}}

### Liveness
GET {{base_url}}/health
X-Request-ID: {{request_id}}

### Inference readiness
GET {{base_url}}/ready
X-Request-ID: {{request_id}}

### Prometheus-style metrics
GET {{base_url}}/metrics
X-Request-ID: {{request_id}}

### 站點清單
GET {{base_url}}/stations
X-Request-ID: {{request_id}}

### 單站預測
POST {{base_url}}/predict
Content-Type: application/json
X-Request-ID: {{request_id}}

{
  "station_no": "500101002",
  "bikes_available": 8,
  "temperature": 33,
  "rain": 3
}

### 單站預測：明確提供 lag window
POST {{base_url}}/predict
Content-Type: application/json
X-Request-ID: {{request_id}}

{
  "station_no": "500101002",
  "bikes_available": 8,
  "temperature": 33,
  "rain": 3,
  "recent_observations": [
    {"bikes_available": 10, "temperature": 31.5, "rain": 0},
    {"bikes_available": 9, "temperature": 32.2, "rain": 1.5},
    {"bikes_available": 8, "temperature": 33, "rain": 3}
  ]
}

### 多站點風險排序
POST {{base_url}}/stations/risk
Content-Type: application/json
X-Request-ID: {{request_id}}

{
  "temperature": 25,
  "rain": 0,
  "stations": [
    {
      "station_no": "500101001",
      "bikes_available": 4,
      "spaces_available": 16
    },
    {
      "station_no": "500101002",
      "bikes_available": 5,
      "spaces_available": 15
    },
    {
      "station_no": "500101003",
      "bikes_available": 18,
      "spaces_available": 2
    }
  ]
}

### Validation 範例：負數車輛應回 422
POST {{base_url}}/predict
Content-Type: application/json
X-Request-ID: {{request_id}}

{
  "station_no": "500101002",
  "bikes_available": -1,
  "temperature": 33,
  "rain": 3
}
"""


def build_openapi_schema() -> dict[str, Any]:
    return app.openapi()


def write_contract_files(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    openapi_path = output_dir / "openapi.json"
    examples_path = output_dir / "api_examples.http"

    openapi_path.write_text(
        json.dumps(build_openapi_schema(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    examples_path.write_text(API_EXAMPLES, encoding="utf-8")

    return [openapi_path, examples_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FastAPI OpenAPI schema and local API request examples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for openapi.json and api_examples.http.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    written_files = write_contract_files(args.output_dir)
    for path in written_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
