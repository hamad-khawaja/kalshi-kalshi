"""Central data aggregator for all market data sources."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from src.config import AssetConfig, BotSettings, StrategyConfig
from src.data.binance_feed import BinanceFeed
from src.data.binance_futures_feed import BinanceFuturesFeed
from src.data.chainlink_feed import ChainlinkFeed
from src.data.kalshi_client import KalshiRestClient
from src.data.kalshi_ws import KalshiWebSocket
from src.data.market_scanner import MarketScanner
from src.data.models import MarketSnapshot, Orderbook, OrderbookLevel
from src.strategy.fair_value import compute_fair_value_from_prices, parse_strike_price

logger = structlog.get_logger()


class DataHub:
    """Central data aggregator providing unified snapshots for the strategy layer.

    Combines data from:
    - Kalshi REST + WebSocket (orderbooks, market data)
    - Coinbase/Kraken WebSocket (price feeds)
    """

    def __init__(
        self,
        kalshi_rest: KalshiRestClient,
        kalshi_ws: KalshiWebSocket,
        feeds: dict[str, BinanceFeed],
        scanners: dict[str, MarketScanner],
        secondary_feeds: dict[str, BinanceFeed] | None = None,
        chainlink_feeds: dict[str, ChainlinkFeed] | None = None,
        futures_feed: BinanceFuturesFeed | None = None,
        strategy_config: StrategyConfig | None = None,
        asset_configs: list[AssetConfig] | None = None,
    ):
        self._kalshi_rest = kalshi_rest
        self._kalshi_ws = kalshi_ws
        self._feeds = feeds
        self._secondary_feeds = secondary_feeds or {}
        self._chainlink_feeds = chainlink_feeds or {}
        self._futures_feed = futures_feed
        self._scanners = scanners
        self._strategy_config = strategy_config or StrategyConfig()
        self._asset_configs = asset_configs or []
        # Build series_ticker -> symbol mapping for routing
        self._series_to_symbol: dict[str, str] = {
            ac.series_ticker: ac.symbol for ac in self._asset_configs
        }
        self._orderbook_cache: dict[str, Orderbook] = {}
        self._ws_subscribed_tickers: set[str] = set()
        self._ws_seq: dict[str, int] = {}  # Last processed seq per ticker

    def _ticker_to_symbol(self, market_ticker: str) -> str:
        """Map a market ticker like 'KXBTC15M-...' to its asset symbol ('BTC').

        Matches by checking if the ticker starts with a known series_ticker prefix.
        Falls back to the first asset symbol if no match.
        """
        upper = market_ticker.upper()
        for series, symbol in self._series_to_symbol.items():
            if upper.startswith(series):
                return symbol
        # Fallback: return first asset symbol
        if self._asset_configs:
            return self._asset_configs[0].symbol
        return "BTC"

    async def start(self) -> None:
        """Connect all data sources. Kalshi failures are non-fatal."""
        # Price feeds are critical, others are best-effort
        for symbol, feed in self._feeds.items():
            await feed.connect()
            logger.info("primary_feed_connected", symbol=symbol)

        best_effort = [
            ("kalshi_rest", self._kalshi_rest.connect()),
            ("kalshi_ws", self._kalshi_ws.connect()),
        ]
        for symbol, feed in self._secondary_feeds.items():
            best_effort.append((f"secondary_feed_{symbol}", feed.connect()))
        for symbol, feed in self._chainlink_feeds.items():
            best_effort.append((f"chainlink_feed_{symbol}", feed.start()))
        if self._futures_feed is not None:
            best_effort.append(("futures_feed", self._futures_feed.start()))

        for name, coro in best_effort:
            try:
                await coro
            except Exception:
                logger.warning("data_source_connect_failed", source=name)

        logger.info("data_hub_started", assets=list(self._feeds.keys()))

    async def stop(self) -> None:
        """Gracefully close all connections."""
        coros = [
            self._kalshi_rest.close(),
            self._kalshi_ws.close(),
        ]
        for feed in self._feeds.values():
            coros.append(feed.close())
        for feed in self._secondary_feeds.values():
            coros.append(feed.close())
        for feed in self._chainlink_feeds.values():
            coros.append(feed.stop())
        if self._futures_feed is not None:
            coros.append(self._futures_feed.stop())
        await asyncio.gather(*coros, return_exceptions=True)
        logger.info("data_hub_stopped")

    async def subscribe_market(self, ticker: str) -> None:
        """Subscribe to WebSocket channels for a market."""
        if ticker in self._ws_subscribed_tickers:
            return

        await self._kalshi_ws.subscribe_orderbook(
            ticker, lambda msg: self._on_orderbook_update(ticker, msg)
        )
        self._ws_subscribed_tickers.add(ticker)
        logger.info("data_hub_subscribed", ticker=ticker)

    def _on_orderbook_update(self, ticker: str, msg: dict) -> None:
        """Handle orderbook updates from WebSocket.

        Processes both orderbook_snapshot (full book) and orderbook_delta
        (incremental quantity changes). Uses seq numbers for deduplication
        since all channel callbacks fire for every message.
        """
        msg_type = msg.get("type", "")
        payload = msg.get("msg", {})
        actual_ticker = payload.get("market_ticker")
        if not actual_ticker:
            return

        # Dedup: only process each message once (all callbacks fire for every msg)
        seq = msg.get("seq", 0)
        last_seq = self._ws_seq.get(actual_ticker, -1)
        if seq <= last_seq:
            return
        self._ws_seq[actual_ticker] = seq

        now = datetime.now(timezone.utc)

        if msg_type == "orderbook_snapshot":
            self._handle_orderbook_snapshot(actual_ticker, payload, now)
        elif msg_type == "orderbook_delta":
            self._handle_orderbook_delta(actual_ticker, payload)

    def _handle_orderbook_snapshot(
        self, ticker: str, payload: dict, now: datetime
    ) -> None:
        """Parse a full orderbook snapshot into the cache."""
        yes_levels: list[OrderbookLevel] = []
        no_levels: list[OrderbookLevel] = []

        # Kalshi sends fixed-point contract counts as "*_dollars_fp"
        for entry in payload.get("yes_dollars_fp") or []:
            if isinstance(entry, list) and len(entry) >= 2:
                yes_levels.append(
                    OrderbookLevel(
                        price_dollars=Decimal(str(entry[0])),
                        quantity=int(Decimal(str(entry[1]))),
                    )
                )
        for entry in payload.get("no_dollars_fp") or []:
            if isinstance(entry, list) and len(entry) >= 2:
                no_levels.append(
                    OrderbookLevel(
                        price_dollars=Decimal(str(entry[0])),
                        quantity=int(Decimal(str(entry[1]))),
                    )
                )

        self._orderbook_cache[ticker] = Orderbook(
            ticker=ticker,
            yes_levels=yes_levels,
            no_levels=no_levels,
            timestamp=now,
        )
        logger.debug(
            "ws_orderbook_snapshot",
            ticker=ticker,
            yes_levels=len(yes_levels),
            no_levels=len(no_levels),
        )

    def _handle_orderbook_delta(self, ticker: str, payload: dict) -> None:
        """Apply an incremental delta to the cached orderbook."""
        cached = self._orderbook_cache.get(ticker)
        if cached is None:
            return  # No snapshot yet — can't apply delta

        side = payload.get("side")  # "yes" or "no"
        delta_raw = payload.get("delta_fp", 0)
        delta = int(Decimal(str(delta_raw)))

        price_str = payload.get("price_dollars")
        if price_str is None:
            return
        price = Decimal(str(price_str))

        levels = cached.yes_levels if side == "yes" else cached.no_levels

        # Find existing level at this price
        found = False
        for i, lvl in enumerate(levels):
            if lvl.price_dollars == price:
                new_qty = lvl.quantity + delta
                if new_qty > 0:
                    levels[i] = OrderbookLevel(
                        price_dollars=price, quantity=new_qty
                    )
                else:
                    levels.pop(i)
                found = True
                break

        if not found and delta > 0:
            # New price level — insert and keep sorted descending by price
            levels.append(OrderbookLevel(price_dollars=price, quantity=delta))
            levels.sort(key=lambda l: l.price_dollars, reverse=True)

    async def get_snapshot(self, market_ticker: str) -> MarketSnapshot | None:
        """Build a complete market snapshot for the strategy layer.

        Returns None if critical data (underlying price) is unavailable.
        """
        now = datetime.now(timezone.utc)

        # Resolve asset symbol for this ticker
        symbol = self._ticker_to_symbol(market_ticker)
        feed = self._feeds.get(symbol)
        if feed is None:
            logger.warning("snapshot_no_feed", symbol=symbol, ticker=market_ticker)
            return None

        # Underlying price data
        spot_price = feed.latest_price
        if spot_price is None:
            logger.warning("snapshot_no_spot_price", symbol=symbol)
            return None

        # Recent price history — single deque scan, then cheap list filters
        ticks_30min = feed.get_prices_since(1800)
        cutoff_5min = now.timestamp() - 900
        cutoff_1min = now.timestamp() - 60
        ticks_5min = [t for t in ticks_30min if t.timestamp.timestamp() >= cutoff_5min]
        ticks_1min = [t for t in ticks_30min if t.timestamp.timestamp() >= cutoff_1min]
        prices_1min = [t.price for t in ticks_1min]
        prices_5min = [t.price for t in ticks_5min]
        prices_30min = [t.price for t in ticks_30min]
        volumes_1min = [t.volume for t in ticks_1min]

        # Kalshi orderbook — use WS-maintained cache, REST fallback on cache miss
        orderbook = self._orderbook_cache.get(market_ticker)
        if orderbook is None:
            try:
                orderbook = await self._kalshi_rest.get_orderbook(market_ticker)
                self._orderbook_cache[market_ticker] = orderbook
            except Exception:
                logger.warning("snapshot_orderbook_error", ticker=market_ticker)
                orderbook = Orderbook(ticker=market_ticker, timestamp=now)

        # Kalshi market info — search across all scanners
        market = None
        for scanner in self._scanners.values():
            market = scanner.active_markets.get(market_ticker)
            if market:
                break
        time_to_expiry = 0.0
        volume = 0
        strike_price = None
        if market:
            # Use close_time (actual trading deadline), not expiration_time
            close = market.close_time or market.expected_expiration_time or market.expiration_time
            if close:
                if close.tzinfo is None:
                    close = close.replace(tzinfo=timezone.utc)
                time_to_expiry = max(0.0, (close - now).total_seconds())
            volume = market.volume
            # Parse strike price from market title
            if market.yes_sub_title:
                strike_price = parse_strike_price(market.yes_sub_title)

        # Compute statistical fair value when we have enough data
        import numpy as np
        stat_fair_value = None
        if strike_price and time_to_expiry > 0 and len(prices_5min) >= 20:
            price_arr = np.array([float(p) for p in prices_5min], dtype=np.float64)
            stat_fair_value = compute_fair_value_from_prices(
                spot_price=float(spot_price),
                strike_price=float(strike_price),
                price_history=price_arr,
                time_to_expiry_seconds=time_to_expiry,
                price_window_seconds=300.0,
            )

        # Cross-exchange data (secondary feed for this asset)
        secondary_feed = self._secondary_feeds.get(symbol)
        secondary_spot_price = None
        cross_exchange_spread = None
        cross_exchange_lead = None
        if secondary_feed and secondary_feed.latest_price is not None:
            secondary_spot_price = secondary_feed.latest_price
            # Spread: (secondary - primary) / primary as percentage
            cb_price = float(spot_price)
            bn_price = float(secondary_spot_price)
            if cb_price > 0:
                cross_exchange_spread = (bn_price - cb_price) / cb_price
            # Lead signal: compare short-term momentum across exchanges
            bn_ticks = secondary_feed.get_prices_since(15)
            cutoff_15s = now.timestamp() - 15
            cb_ticks = [t for t in ticks_1min if t.timestamp.timestamp() >= cutoff_15s]
            if len(bn_ticks) >= 2 and len(cb_ticks) >= 2:
                bn_start = float(bn_ticks[0].price)
                bn_end = float(bn_ticks[-1].price)
                cb_start = float(cb_ticks[0].price)
                cb_end = float(cb_ticks[-1].price)
                if bn_start > 0 and cb_start > 0:
                    bn_mom = (bn_end - bn_start) / bn_start
                    cb_mom = (cb_end - cb_start) / cb_start
                    # Positive lead = secondary moving up faster (bullish)
                    cross_exchange_lead = bn_mom - cb_mom

        # Taker buy/sell volume (real-time from secondary feed)
        taker_buy = None
        taker_sell = None
        if secondary_feed:
            buy, sell = secondary_feed.get_taker_volume_since(300)  # 5 min window
            if buy > 0 or sell > 0:
                # Convert from base asset to USD
                price_usd = float(spot_price)
                taker_buy = buy * price_usd
                taker_sell = sell * price_usd

        # Chainlink oracle data
        chainlink_oracle_price = None
        chainlink_divergence = None
        chainlink_round_updated = False
        chainlink_feed = self._chainlink_feeds.get(symbol)
        if chainlink_feed and chainlink_feed.latest_price is not None:
            chainlink_oracle_price = chainlink_feed.latest_price
            oracle_f = float(chainlink_oracle_price)
            if oracle_f > 0:
                chainlink_divergence = (float(spot_price) - oracle_f) / oracle_f
            chainlink_round_updated = chainlink_feed.round_just_updated

        # Binance Futures: funding rate + liquidation stats
        funding_rate = None
        predicted_funding_rate = None
        liquidation_long_usd = None
        liquidation_short_usd = None
        if self._futures_feed is not None:
            futures_symbol = symbol + "USDT"  # e.g. "BTCUSDT"
            funding_rate = self._futures_feed.get_funding_rate(futures_symbol)
            predicted_funding_rate = self._futures_feed.get_predicted_funding_rate(futures_symbol)
            long_liq, short_liq = self._futures_feed.get_liquidation_stats_since(futures_symbol, 300)
            if long_liq > 0 or short_liq > 0:
                liquidation_long_usd = long_liq
                liquidation_short_usd = short_liq

        # Cross-asset futures data (other asset's funding rate + liquidations)
        other_asset_funding_rate = None
        other_asset_liquidation_long_usd = None
        other_asset_liquidation_short_usd = None
        if self._futures_feed is not None:
            other_symbol = "ETH" if symbol == "BTC" else "BTC"
            other_futures = other_symbol + "USDT"
            other_asset_funding_rate = self._futures_feed.get_funding_rate(other_futures)
            other_long, other_short = self._futures_feed.get_liquidation_stats_since(other_futures, 300)
            if other_long > 0 or other_short > 0:
                other_asset_liquidation_long_usd = other_long
                other_asset_liquidation_short_usd = other_short

        # BTC momentum lead for non-BTC assets
        btc_momentum_lead = None
        if symbol != "BTC":
            btc_feed = self._feeds.get("BTC")
            if btc_feed and btc_feed.latest_price is not None:
                btc_ticks_60 = btc_feed.get_prices_since(60)
                btc_ticks_180 = btc_feed.get_prices_since(180)
                mom_60s = 0.0
                mom_180s = 0.0
                if len(btc_ticks_60) >= 2:
                    start_f = float(btc_ticks_60[0].price)
                    end_f = float(btc_ticks_60[-1].price)
                    if start_f > 0:
                        mom_60s = (end_f - start_f) / start_f
                if len(btc_ticks_180) >= 2:
                    start_f = float(btc_ticks_180[0].price)
                    end_f = float(btc_ticks_180[-1].price)
                    if start_f > 0:
                        mom_180s = (end_f - start_f) / start_f
                btc_momentum_lead = 0.4 * mom_60s + 0.6 * mom_180s

        # Compute time elapsed and window phase
        time_elapsed = max(0.0, 900.0 - time_to_expiry)
        cfg = self._strategy_config
        if time_elapsed < cfg.phase_observation_end:
            window_phase = 1
        elif time_elapsed < cfg.phase_confirmation_end:
            window_phase = 2
        elif time_elapsed < cfg.phase_active_end:
            window_phase = 3
        elif time_elapsed < cfg.phase_late_end:
            window_phase = 4
        else:
            window_phase = 5

        return MarketSnapshot(
            timestamp=now,
            market_ticker=market_ticker,
            spot_price=spot_price,
            spot_prices_1min=prices_1min,
            spot_prices_5min=prices_5min,
            spot_prices_30min=prices_30min,
            spot_volumes_1min=volumes_1min,
            orderbook=orderbook,
            implied_yes_prob=orderbook.implied_yes_prob,
            spread=orderbook.spread,
            strike_price=strike_price,
            statistical_fair_value=stat_fair_value,
            secondary_spot_price=secondary_spot_price,
            cross_exchange_spread=cross_exchange_spread,
            cross_exchange_lead=cross_exchange_lead,
            taker_buy_volume=taker_buy,
            taker_sell_volume=taker_sell,
            chainlink_oracle_price=chainlink_oracle_price,
            chainlink_divergence=chainlink_divergence,
            chainlink_round_updated=chainlink_round_updated,
            btc_momentum_lead=btc_momentum_lead,
            funding_rate=funding_rate,
            predicted_funding_rate=predicted_funding_rate,
            liquidation_long_usd=liquidation_long_usd,
            liquidation_short_usd=liquidation_short_usd,
            other_asset_funding_rate=other_asset_funding_rate,
            other_asset_liquidation_long_usd=other_asset_liquidation_long_usd,
            other_asset_liquidation_short_usd=other_asset_liquidation_short_usd,
            time_to_expiry_seconds=time_to_expiry,
            time_elapsed_seconds=time_elapsed,
            window_phase=window_phase,
            volume=volume,
        )
