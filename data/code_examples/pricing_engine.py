from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PriceRule:
    min_quantity: int
    discount_percent: float


class PricingEngine:
    def __init__(self) -> None:
        self.rules = [
            PriceRule(min_quantity=3, discount_percent=5.0),
            PriceRule(min_quantity=5, discount_percent=10.0),
            PriceRule(min_quantity=10, discount_percent=15.0),
        ]

    def calculate_total(self, unit_price: float, quantity: int) -> float:
        subtotal = unit_price * quantity
        discount_percent = self._best_discount(quantity)
        discount_value = subtotal * (discount_percent / 100.0)
        return round(subtotal - discount_value, 2)

    def _best_discount(self, quantity: int) -> float:
        matched = [rule.discount_percent for rule in self.rules if quantity >= rule.min_quantity]
        return max(matched) if matched else 0.0
