"""Create the `flights` table and load the CSV into PostgreSQL.

Usage:  python scripts/load_db.py
Requires the DB to be reachable (see docker-compose.yml for local Postgres).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402
from app.database import init_and_load, table_status  # noqa: E402


def main() -> None:
    print(f"Connecting to postgres://{config.DB_PARAMS['user']}@"
          f"{config.DB_PARAMS['host']}:{config.DB_PARAMS['port']}/{config.DB_PARAMS['dbname']}")
    if not config.CSV_PATH.exists():
        print(f"CSV not found at {config.CSV_PATH}. Run scripts/download_data.py first.")
        sys.exit(1)
    n = init_and_load(config.CSV_PATH)
    print(f"✓ loaded {n} rows into `flights`")
    print("status:", table_status())


if __name__ == "__main__":
    main()
