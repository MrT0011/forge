# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:16:47 2023

@author: Jeffrey
"""

import datetime


def pricing(
    injection_dates: list(datetime),
    withdrawal_dates: list(datetime),
    prices: {datetime: float},
    rate: float,
    maximum_volume_sotred: float,
    storage_costs: float,
):
    """
    input:
        injection_dates
        withdrawal_dates
        prices: commodity prices on injection/withdrawal dates
        rate: rate of the commodity inject/withdraw
        maximum_volume_sotred: maximum amount of commodity can be stored
        storage_costs: storage costs per day

    output:
        fair_price
    """

    if len(injection_dates) * rate >= maximum_volume_sotred:
        print("Exceeding storage capacity, cannot price")

    buy_price = []
    for day in injection_dates:
        buy_price += prices[day] * rate

    sell_price = []
    for day in withdrawal_dates:
        sell_price += prices[day] * rate

    total_storage_costs = (max(withdrawal_dates) - min(injection_dates)).days * storage_costs

    fair_price = sum(sell_price) - sum(buy_price) - total_storage_costs

    if fair_price <= 0:
        print("This is the fair price with no profit")
        return abs(fair_price)
    print("This is the profit from the contract")
    return fair_price
