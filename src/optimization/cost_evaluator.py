# src/optimization/cost_evaluator.py
"""
CostEvaluator: compute integrated daily cost components in a logical,
traceable, and consistent way with procurement & logistics outputs.

Assumptions / conventions:
- procurement_cost_total: monetary cost ($) for one REVIEW CYCLE (the same review period used in procurement,
  e.g., if procurement aggregates u days, procurement_cost_total is cost for that order / cycle).
- distribution_cost_total: monetary inbound+outbound logistics cost for that same cycle.
- avg_inventory_value: monetary value ($) of safety stock or held inventory (an average $ amount).
- total_order_units: total number of units ordered in the cycle (sum of order_qty_units). Optional but recommended.
- avg_price_per_unit: optional average price per unit ($). If provided and total_order_units is missing,
  avg_inventory_value can be converted to units if necessary.
- function returns dailyized components (per calendar day) so that outputs are comparable across strategies.
"""

from typing import Optional
from config.settings import ProjectConfig as Cfg

class CostEvaluator:
    @staticmethod
    def calculate_integrated_daily_cost(
        procurement_cost_total: float,
        distribution_cost_total: float,
        avg_inventory_value: float,  # $ average inventory value (safety stock $ value + other)
        P_lim: int,
        U_lim: int,
        freshness_loss_days_avg: float = 0.0,
        extra_holding_days: float = 0.0,
        total_order_units: Optional[float] = None,
        avg_price_per_unit: Optional[float] = None
    ) -> dict:
        """
        Returns dict with dailyized cost components and total daily cost.
        - All *_Daily_* entries are expressed in $/day.
        - If total_order_units is provided, freshness penalty is computed as:
            FRESHNESS_PENALTY_PER_DAY ($ per unit-day) * avg_days_lost_per_unit * (units_per_day)
          where units_per_day = total_order_units / U_lim.
        - If total_order_units not provided but avg_price_per_unit provided, we estimate daily units
          from avg_inventory_value (as a fallback).
        """

        # Safety: cast to floats
        procurement_cost_total = float(procurement_cost_total or 0.0)
        distribution_cost_total = float(distribution_cost_total or 0.0)
        avg_inventory_value = float(avg_inventory_value or 0.0)
        U_lim = int(max(1, U_lim))

        # --- 1) Daily procurement & distribution (simple dailyization over U_lim)
        daily_procurement_cost = procurement_cost_total / float(U_lim)
        daily_distribution_cost = distribution_cost_total / float(U_lim)

        # --- 2) Holding cost logic (financial view)
        # a) Cycle stock value (monetary): assume order value for cycle = procurement_cost_total
        #    average cycle stock (monetary) = procurement_cost_total / 2 (linear consumption)
        cycle_stock_value = procurement_cost_total / 2.0

        # b) Total average inventory value carried (monetary)
        #    This uses cycle stock (avg in-transit/received) + avg_inventory_value (safety stock/other)
        total_inventory_value = cycle_stock_value + avg_inventory_value

        # c) Daily holding cost = avg value * daily holding rate (RATE is percent per day in config)
        daily_holding_rate = float(getattr(Cfg, 'DAILY_HOLDING_RATE_PCT', 0.0)) / 100.0
        base_holding_cost_daily = total_inventory_value * daily_holding_rate

        # d) Extra holding due to logistics delay (distributed over days of the cycle)
        #    If extra_holding_days > 0, we approximate daily extra holding cost as:
        #      extra_holding_cost_total = procurement_cost_total * daily_holding_rate * extra_holding_days
        #    Then convert to daily by dividing by U_lim (so it is added to per-day figure)
        extra_holding_cost_daily = 0.0
        if extra_holding_days and extra_holding_days > 0.0:
            extra_holding_cost_total = procurement_cost_total * daily_holding_rate * float(extra_holding_days)
            extra_holding_cost_daily = extra_holding_cost_total / float(U_lim)

        total_holding_cost_daily = base_holding_cost_daily + extra_holding_cost_daily

        # --- (inside calculate_integrated_daily_cost) ---

        # --- 3) Freshness penalty (daily) --- robust & anchored to price when possible
        freshness_penalty_daily = 0.0

        # Safety: cast config value
        cfg_fp = float(getattr(Cfg, 'FRESHNESS_PENALTY_PER_DAY', 0.0))

        # compute units_per_day if we have total_order_units (recommended)
        units_per_day = None
        if total_order_units and float(total_order_units) > 0:
            units_per_day = float(total_order_units) / float(U_lim)

        # compute avg_price_per_unit if provided
        avg_price_per_unit = float(avg_price_per_unit) if avg_price_per_unit is not None else None

        # Interpretation logic:
        # - If cfg_fp <= 1.0: treat as fraction of unit price per unit-day (e.g. 0.05 -> 5% of unit price per day)
        # - If cfg_fp > 1.0: treat as legacy absolute $ per unit-day (but if avg_price_per_unit available we'll prefer fraction mode)
        if cfg_fp <= 1.0:
            # fractional mode (recommended)
            penalty_rate = cfg_fp  # fraction of unit price per day
            if units_per_day is not None and avg_price_per_unit:
                freshness_penalty_daily = units_per_day * avg_price_per_unit * freshness_loss_days_avg * penalty_rate
            elif units_per_day is not None:
                # no price; use avg_inventory_value as monetary proxy to estimate price-per-unit
                est_price = (avg_inventory_value / max(1.0, float(total_order_units))) if total_order_units and float(total_order_units) > 0 else 0.0
                freshness_penalty_daily = units_per_day * est_price * freshness_loss_days_avg * penalty_rate
            else:
                # fallback: small proxy based on avg_inventory_value
                freshness_penalty_daily = (avg_inventory_value / float(max(1, U_lim))) * penalty_rate * freshness_loss_days_avg / max(1.0, (avg_price_per_unit if avg_price_per_unit else 1.0))
        else:
            # legacy absolute-dollar mode, but scale down if it seems absurd
            # cfg_fp is $ per unit-day. If avg_price_per_unit exists and cfg_fp >> avg_price we log a warning.
            if units_per_day is not None:
                freshness_penalty_daily = units_per_day * freshness_loss_days_avg * cfg_fp
            elif avg_price_per_unit:
                # estimate units/day from avg_inventory_value
                est_units = (avg_inventory_value / max(1.0, avg_price_per_unit)) / float(max(1, U_lim))
                freshness_penalty_daily = est_units * freshness_loss_days_avg * cfg_fp
            else:
                # last fallback proxy: treat cfg_fp as relative fraction scaled by 1/1000
                freshness_penalty_daily = (avg_inventory_value / float(max(1, U_lim))) * (cfg_fp / 1000.0) * freshness_loss_days_avg

        # --- 4) Aggregate daily total
        total_daily_cost = (
            daily_procurement_cost
            + daily_distribution_cost
            + total_holding_cost_daily
            + freshness_penalty_daily
        )

        # Round outputs for reporting
        out = {
            "P_lim": P_lim,
            "U_lim": U_lim,
            "Daily_Procurement_Cost": round(daily_procurement_cost, 2),
            "Daily_Distribution_Cost": round(daily_distribution_cost, 2),
            "Daily_Holding_Cost": round(total_holding_cost_daily, 2),
            "Daily_Freshness_Penalty": round(freshness_penalty_daily, 2),
            "Total_Daily_Cost": round(total_daily_cost, 2)
        }

        return out
