from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


class TextToSQLTool:
    """Converts simple analytics intents into safe SQL and executes them."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self._ensure_seed_data()

    def _ensure_seed_data(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sales (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer TEXT NOT NULL,
                    product TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    order_date TEXT NOT NULL
                )
                """
            )
            cur.execute("SELECT COUNT(*) FROM sales")
            count = cur.fetchone()[0]
            if count == 0:
                cur.executemany(
                    """
                    INSERT INTO sales (customer, product, quantity, unit_price, order_date)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        ("alice", "book", 2, 18.0, "2026-03-01"),
                        ("alice", "headphones", 1, 85.0, "2026-03-04"),
                        ("bob", "keyboard", 1, 65.0, "2026-03-02"),
                        ("carol", "book", 3, 18.0, "2026-03-05"),
                        ("dave", "monitor", 1, 210.0, "2026-03-08"),
                    ],
                )
            conn.commit()

    def text_to_sql(self, query: str) -> str:
        q = query.lower()

        customer_match = re.search(r"customer\s+([a-zA-Z0-9_]+)", q)
        if customer_match:
            customer = customer_match.group(1)
            return (
                "SELECT customer, product, quantity, unit_price, order_date "
                f"FROM sales WHERE lower(customer)=lower('{customer}') ORDER BY order_date DESC"
            )

        if "total revenue" in q or "sum" in q:
            return "SELECT ROUND(SUM(quantity * unit_price), 2) AS total_revenue FROM sales"

        if "top product" in q or "best selling" in q:
            return (
                "SELECT product, SUM(quantity) AS units "
                "FROM sales GROUP BY product ORDER BY units DESC LIMIT 5"
            )

        return "SELECT customer, product, quantity, unit_price, order_date FROM sales ORDER BY order_date DESC LIMIT 10"

    def run_query(self, sql: str) -> list[dict[str, Any]]:
        if not sql.strip().lower().startswith("select"):
            raise ValueError("Only SELECT queries are allowed.")

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return [dict(row) for row in rows]
