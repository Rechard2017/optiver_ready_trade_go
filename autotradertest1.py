# 存储快照


# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import numpy as np
import pandas as pd
from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side


LOT_SIZE = 20
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS

# order is etf, hegde is future
class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.position = 0
        self.position_buyorder = 0
        self.position_sellorder = 0
        self.p_ask_id = 0
        self.p_bid_id = 0
        self.p_bid_dict = {}
        self.p_ask_dict = {}

        self.a_ask_id = 0
        self.a_bid_id = 0
        # self.a_ask_id_hedge = 0
        # self.a_bid_id_hedge = 0

        self.a_bid_dict = {}
        self.a_ask_dict = {}
        # self.a_bid_hedge_dict = {}
        # self.a_ask_hedge_dict = {}

        self.new_future_info = {}
        self.new_etf_info = {}

        self.passive_sign = True

        self.passi_buy_p = None
        self.passi_sell_p = None
        self.agg_buy_p = None
        self.agg_sell_p = None


    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())
        if client_order_id != 0 and (client_order_id in self.p_bid_dict.keys() or client_order_id in self.p_ask_dict.keys()):
            self.on_order_status_message(client_order_id, 0, 0, 0)

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled.
        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(f"hedge filled info id: {client_order_id},price: {price}, volumn: {volume}")

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        self.logger.info(f"order_book_update info instrument:{instrument},sequence_number:{sequence_number},position_now {self.position}")
        # self.logger.info(f"ask_prices:{ask_prices},bid_prices:{bid_prices}")
        # self.logger.info(f"ask_volumes:{ask_volumes},bid_volumes:{bid_volumes}")
        self.logger.info(f"position now: {self.position} buy_order:{self.position_buyorder}, sell_order:{self.position_sellorder}")

        if instrument == Instrument.FUTURE:
            self.logger.info(f"future part")
            self.new_future_info['ask_prices'] = ask_prices
            self.new_future_info['bid_prices'] = bid_prices
            self.new_future_info['ask_volumes'] = ask_volumes
            self.new_future_info['bid_volumes'] = bid_volumes

            price_adjustment = - (self.position // LOT_SIZE) * TICK_SIZE_IN_CENTS
            if abs(self.position) > 50:
                price_adjustment = -1 * np.sign(self.position) * (((abs(self.position)-50) // LOT_SIZE) * TICK_SIZE_IN_CENTS * 2 + 200)

            spread = 200
            new_bid_price = bid_prices[0] - spread + price_adjustment if bid_prices[0] != 0 else 0
            new_ask_price = ask_prices[0] + spread + price_adjustment if ask_prices[0] != 0 else 0
            
            if self.passive_sign:
                if self.p_bid_id != 0 and new_bid_price not in (self.p_bid_dict[self.p_bid_id]['price'], 0):
                    self.send_cancel_order(self.p_bid_id)
                    self.position_buyorder = self.position_buyorder - self.p_bid_dict[self.p_bid_id]['volume']
                    self.logger.info(f"send cancel order bid_id:{self.p_bid_id},new_bid_price:{new_bid_price} self.bid_price:{self.p_bid_dict[self.p_bid_id]['price']}")
                    self.p_bid_id = 0
                    self.passi_buy_p = None

                if self.p_ask_id != 0 and new_ask_price not in (self.p_ask_dict[self.p_ask_id]['price'], 0):
                    self.send_cancel_order(self.p_ask_id)
                    self.position_sellorder = self.position_sellorder - self.p_ask_dict[self.p_ask_id]['volume']
                    self.logger.info(f"send cancel order ask_id:{self.p_ask_id},new_ask_price:{new_ask_price} self.ask_price:{self.p_ask_dict[self.p_ask_id]['price']}")
                    self.p_ask_id = 0
                    self.passi_sell_p = None

                if self.p_bid_id == 0 and new_bid_price != 0 and self.position + self.position_buyorder + LOT_SIZE < POSITION_LIMIT:    #(self.agg_sell_p is None or self.agg_sell_p > new_bid_price) and 
                    self.p_bid_id = next(self.order_ids)
                    self.send_insert_order(self.p_bid_id, Side.BUY, new_bid_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                    self.position_buyorder = self.position_buyorder + LOT_SIZE
                    self.passi_buy_p = new_bid_price
                    
                    self.p_bid_dict[self.p_bid_id] = {}
                    self.p_bid_dict[self.p_bid_id]['price'] = new_bid_price
                    self.p_bid_dict[self.p_bid_id]['volume'] = LOT_SIZE
                    self.logger.info(f"send insert order bid_id:{self.p_bid_id}, new_bid_price:{new_bid_price}, Side.BUY:{Side.BUY}, volumn:{LOT_SIZE}, position_now: {self.position}")

                if self.p_ask_id == 0 and new_ask_price != 0 and self.position - self.position_sellorder - LOT_SIZE > -POSITION_LIMIT:  #(self.agg_buy_p is None or self.agg_buy_p < new_ask_price) 
                    self.p_ask_id = next(self.order_ids)
                    self.send_insert_order(self.p_ask_id, Side.SELL, new_ask_price, LOT_SIZE, Lifespan.GOOD_FOR_DAY)
                    self.position_sellorder = self.position_sellorder + LOT_SIZE
                    self.passi_sell_p = new_ask_price

                    self.p_ask_dict[self.p_ask_id] = {}
                    self.p_ask_dict[self.p_ask_id]['price'] = new_ask_price
                    self.p_ask_dict[self.p_ask_id]['volume'] = LOT_SIZE
                    self.logger.info(f"send insert order ask_id:{self.p_ask_id}, new_ask_price:{new_ask_price}, Side.SELL:{Side.SELL}, volumn:{LOT_SIZE}, position_now: {self.position}")
            else:
                if self.p_bid_id != 0:
                    self.send_cancel_order(self.p_bid_id)
                    self.position_buyorder = self.position_buyorder - self.p_bid_dict[self.p_bid_id]['volume']
                    self.logger.info(f"send cancel order bid_id:{self.p_bid_id},new_bid_price:{new_bid_price} self.bid_price:{self.p_bid_dict[self.p_bid_id]['price']}")
                    self.p_bid_id = 0
                    self.passi_buy_p = None

                if self.p_ask_id != 0:
                    self.send_cancel_order(self.p_ask_id)
                    self.position_sellorder = self.position_sellorder - self.p_ask_dict[self.p_ask_id]['volume']
                    self.logger.info(f"send cancel order ask_id:{self.p_ask_id},new_ask_price:{new_ask_price} self.ask_price:{self.p_ask_dict[self.p_ask_id]['price']}")
                    self.p_ask_id = 0
                    self.passi_sell_p = None

        # if instrument == Instrument.ETF:


        if instrument == Instrument.ETF:
            self.logger.info(f"etf part")
            self.new_etf_info['ask_prices'] = ask_prices
            self.new_etf_info['bid_prices'] = bid_prices
            self.new_etf_info['ask_volumes'] = ask_volumes
            self.new_etf_info['bid_volumes'] = bid_volumes
            # self.passive_sign = True

            # buy etf sell future
            aribitrage = self.new_future_info['bid_prices'][0] - ask_prices[0]
            if ask_prices[0] != 0 and self.new_future_info['bid_prices'][0] != 0:
                if  aribitrage  >= 300:
                    free_position = 50 - self.position
                    buy_amount = min(30, free_position, ask_volumes[0], self.new_future_info['bid_volumes'][0])
                    
                    # buy aggressive
                    if self.a_ask_id == 0 and buy_amount > 0: #and (self.passi_sell_p is None or self.passi_sell_p > ask_prices[0])
                        self.a_ask_id = next(self.order_ids)
                        self.send_insert_order(self.a_ask_id, Side.BUY, ask_prices[0], buy_amount, Lifespan.FILL_AND_KILL)
                        self.position_buyorder = self.position_buyorder + buy_amount
                        self.agg_buy_p = ask_prices[0]
                        
                        self.a_ask_dict[self.a_ask_id] = {}
                        self.a_ask_dict[self.a_ask_id]['price'] = ask_prices[0]
                        self.a_ask_dict[self.a_ask_id]['volume'] = buy_amount
                        self.logger.info(f"send insert agg order ask_id:{self.a_ask_id}, ask_price:{ask_prices[0]}, volumn:{buy_amount}, position_now: {self.position}")
                        # self.passive_sign = False
                elif aribitrage >= 200:
                    free_position = 50 - self.position
                    buy_amount = min(20, free_position, ask_volumes[0], self.new_future_info['bid_volumes'][0])
                    
                    # buy aggressive
                    if self.a_ask_id == 0 and buy_amount > 0: #and (self.passi_sell_p is None or self.passi_sell_p > ask_prices[0])
                        self.a_ask_id = next(self.order_ids)
                        self.send_insert_order(self.a_ask_id, Side.BUY, ask_prices[0], buy_amount, Lifespan.FILL_AND_KILL)
                        self.position_buyorder = self.position_buyorder + buy_amount
                        self.agg_buy_p = ask_prices[0]
                        
                        self.a_ask_dict[self.a_ask_id] = {}
                        self.a_ask_dict[self.a_ask_id]['price'] = ask_prices[0]
                        self.a_ask_dict[self.a_ask_id]['volume'] = buy_amount
                        self.logger.info(f"send insert agg order ask_id:{self.a_ask_id}, ask_price:{ask_prices[0]}, volumn:{buy_amount}, position_now: {self.position}")
                        # self.passive_sign = False

                elif aribitrage >= 100:
                    free_position = 50 - self.position
                    buy_amount = min(10, free_position, ask_volumes[0], self.new_future_info['bid_volumes'][0])
                    
                    # buy aggressive
                    if self.a_ask_id == 0 and buy_amount > 0: #and (self.passi_sell_p is None or self.passi_sell_p > ask_prices[0])
                        self.a_ask_id = next(self.order_ids)
                        self.send_insert_order(self.a_ask_id, Side.BUY, ask_prices[0], buy_amount, Lifespan.FILL_AND_KILL)
                        self.position_buyorder = self.position_buyorder + buy_amount
                        self.agg_buy_p = ask_prices[0]
                        
                        self.a_ask_dict[self.a_ask_id] = {}
                        self.a_ask_dict[self.a_ask_id]['price'] = ask_prices[0]
                        self.a_ask_dict[self.a_ask_id]['volume'] = buy_amount
                        self.logger.info(f"send insert agg order ask_id:{self.a_ask_id}, ask_price:{ask_prices[0]}, volumn:{buy_amount}, position_now: {self.position}")
                        # self.passive_sign = False



            # buy future sell etf
            aribitrage = bid_prices[0] - self.new_future_info['ask_prices'][0]
            if bid_prices[0] != 0 and self.new_future_info['ask_prices'][0] != 0:
                if aribitrage >= 300:
                    free_position = -50 - self.position
                    self.logger.info(f"sell etf free_position {free_position}")
                    sell_amount = min(30, -free_position, bid_volumes[0], self.new_future_info['ask_volumes'][0])

                    # sell aggressive
                    if self.a_bid_id == 0 and sell_amount > 0: #and (self.passi_buy_p is None or self.passi_buy_p < bid_prices[0])
                        self.a_bid_id = next(self.order_ids)
                        self.send_insert_order(self.a_bid_id, Side.SELL, bid_prices[0], sell_amount, Lifespan.FILL_AND_KILL)
                        self.position_sellorder = self.position_sellorder + sell_amount
                        self.agg_sell_p = bid_prices[0]
                        
                        self.a_bid_dict[self.a_bid_id] = {}
                        self.a_bid_dict[self.a_bid_id]['price'] = bid_prices[0]
                        self.a_bid_dict[self.a_bid_id]['volume'] = sell_amount
                        self.logger.info(f"send insert agg order bid_id:{self.a_bid_id}, bid_price:{bid_prices[0]}, volumn:{sell_amount}, position_now: {self.position}")
                        # self.passive_sign = False
                elif aribitrage >= 200:
                    free_position = -50 - self.position
                    self.logger.info(f"sell etf free_position {free_position}")
                    sell_amount = min(20, -free_position, bid_volumes[0], self.new_future_info['ask_volumes'][0])

                    # sell aggressive
                    if self.a_bid_id == 0 and sell_amount > 0: #and (self.passi_buy_p is None or self.passi_buy_p < bid_prices[0])
                        self.a_bid_id = next(self.order_ids)
                        self.send_insert_order(self.a_bid_id, Side.SELL, bid_prices[0], sell_amount, Lifespan.FILL_AND_KILL)
                        self.position_sellorder = self.position_sellorder + sell_amount
                        self.agg_sell_p = bid_prices[0]
                        
                        self.a_bid_dict[self.a_bid_id] = {}
                        self.a_bid_dict[self.a_bid_id]['price'] = bid_prices[0]
                        self.a_bid_dict[self.a_bid_id]['volume'] = sell_amount
                        self.logger.info(f"send insert agg order bid_id:{self.a_bid_id}, bid_price:{bid_prices[0]}, volumn:{sell_amount}, position_now: {self.position}")
                        # self.passive_sign = False
                elif aribitrage >= 100:
                    free_position = -50 - self.position
                    self.logger.info(f"sell etf free_position {free_position}")
                    sell_amount = min(10, -free_position, bid_volumes[0], self.new_future_info['ask_volumes'][0])

                    # sell aggressive
                    if self.a_bid_id == 0 and sell_amount > 0: #and (self.passi_buy_p is None or self.passi_buy_p < bid_prices[0])
                        self.a_bid_id = next(self.order_ids)
                        self.send_insert_order(self.a_bid_id, Side.SELL, bid_prices[0], sell_amount, Lifespan.FILL_AND_KILL)
                        self.position_sellorder = self.position_sellorder + sell_amount
                        self.agg_sell_p = bid_prices[0]
                        
                        self.a_bid_dict[self.a_bid_id] = {}
                        self.a_bid_dict[self.a_bid_id]['price'] = bid_prices[0]
                        self.a_bid_dict[self.a_bid_id]['volume'] = sell_amount
                        self.logger.info(f"send insert agg order bid_id:{self.a_bid_id}, bid_price:{bid_prices[0]}, volumn:{sell_amount}, position_now: {self.position}")
                        # self.passive_sign = False


    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info(f" order_filled info client_order_id:{client_order_id}, price:{price}, volume:{volume} ")

        if client_order_id in self.p_bid_dict.keys():
            self.position += volume
            hegde_id = next(self.order_ids)
            self.send_hedge_order(hegde_id, Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.logger.info(f'postion increase {volume} and now position {self.position}')
            self.logger.info(f'then send hegde to sell id: {hegde_id},Side.ASK:{Side.ASK}, price:{MIN_BID_NEAREST_TICK}, volume:{volume} ')
        elif client_order_id in self.p_ask_dict.keys():
            self.position -= volume
            hegde_id = next(self.order_ids)
            self.send_hedge_order(hegde_id, Side.BID, MAX_ASK_NEAREST_TICK, volume)
            self.logger.info(f'postion decrease {volume} and now position {self.position}')
            self.logger.info(f'then send hegde to buy id: {hegde_id},Side.BID:{Side.BID}, price:{MAX_ASK_NEAREST_TICK}, volume:{volume} ')
        elif client_order_id in self.a_ask_dict.keys():
            self.position += volume
            hegde_id = next(self.order_ids)
            self.send_hedge_order(hegde_id, Side.ASK, MIN_BID_NEAREST_TICK, volume)
            self.logger.info(f'agg postion increase {volume} and now position {self.position}')
            self.logger.info(f'then send hegde to sell id: {hegde_id},Side.ASK:{Side.ASK}, price:{MIN_BID_NEAREST_TICK}, volume:{volume} ')
        elif client_order_id in self.a_bid_dict.keys():
            self.position -= volume
            hegde_id = next(self.order_ids)
            self.send_hedge_order(hegde_id, Side.BID, MAX_ASK_NEAREST_TICK, volume)
            self.logger.info(f'agg postion decrease {volume} and now position {self.position}')
            self.logger.info(f'then send hegde to buy id: {hegde_id},Side.BID:{Side.BID}, price:{MAX_ASK_NEAREST_TICK}, volume:{volume} ')


    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info(f" order_status info client_order_id:{client_order_id}, fill_volume:{fill_volume}, remaining_volume:{remaining_volume}, fees:{fees} ")
        
        if remaining_volume == 0:
            if client_order_id == self.p_bid_id:
                self.position_buyorder = self.position_buyorder - self.p_bid_dict[client_order_id]['volume']
                self.p_bid_id = 0
                self.passi_buy_p = None
                self.p_bid_dict.pop(client_order_id, None)


            elif client_order_id == self.p_ask_id:
                self.position_sellorder = self.position_sellorder - self.p_ask_dict[client_order_id]['volume']
                self.p_ask_id = 0
                self.passi_sell_p = None
                self.p_ask_dict.pop(client_order_id, None)

            elif client_order_id == self.a_ask_id:
                self.position_buyorder = self.position_buyorder - self.a_ask_dict[client_order_id]['volume']
                self.a_ask_id = 0
                self.agg_buy_p = None
                self.a_ask_dict.pop(client_order_id, None)

            elif client_order_id == self.a_bid_id:
                self.position_sellorder = self.position_sellorder - self.a_bid_dict[client_order_id]['volume']
                self.a_bid_id = 0
                self.agg_sell_p = None
                self.a_bid_dict.pop(client_order_id, None)
        else:
            if client_order_id == self.p_bid_id:
                self.position_buyorder = self.position_buyorder - self.p_bid_dict[client_order_id]['volume'] + remaining_volume
                self.p_bid_dict[client_order_id]['volume'] = remaining_volume

            elif client_order_id == self.p_ask_id:
                self.position_sellorder = self.position_sellorder - self.p_ask_dict[client_order_id]['volume'] + remaining_volume
                self.p_ask_dict[client_order_id]['volume'] = remaining_volume

            elif client_order_id == self.a_ask_id:
                self.position_buyorder = self.position_buyorder - self.a_ask_dict[client_order_id]['volume'] + remaining_volume
                self.a_ask_dict[client_order_id]['volume'] = remaining_volume

            elif client_order_id == self.a_bid_id:
                self.position_sellorder = self.position_sellorder - self.a_bid_dict[client_order_id]['volume'] + remaining_volume
                self.a_bid_dict[client_order_id]['volume'] = remaining_volume

        
            
    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """

        self.logger.info(f" trade_ticks info instrument be traded :{instrument}, sequence_number:{sequence_number}")
        self.logger.info(f" ask_prices :{ask_prices}, ask_volumes:{ask_volumes}")
        self.logger.info(f" bid_prices :{bid_prices}, bid_volumes:{bid_volumes}")




