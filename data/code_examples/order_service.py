from __future__ import annotations

from dataclasses import dataclass

from pricing_engine import PricingEngine


@dataclass
class OrderItem:
    sku: str
    unit_price: float
    quantity: int


class OrderService:
    def __init__(self) -> None:
        self.pricing_engine = PricingEngine()

    def summarize_order(self, items: list[OrderItem]) -> dict[str, float]:
        gross_total = 0.0
        net_total = 0.0

        for item in items:
            gross_total += item.unit_price * item.quantity
            net_total += self.pricing_engine.calculate_total(item.unit_price, item.quantity)

        discount_total = round(gross_total - net_total, 2)
        return {
            "gross_total": round(gross_total, 2),
            "discount_total": discount_total,
            "net_total": round(net_total, 2),
        }
