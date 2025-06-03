//=== StrategyBBRSIReversal.mqh ===
#ifndef __StrategyBBRSIReversal_mqh__
#define __StrategyBBRSIReversal_mqh__

void EvaluateBB_RSI_Reversal(string sym, SymbolState* state) {
   // Ensure data is synchronized
   if (SeriesInfoInteger(sym, ExpectedTimeframe, SERIES_SYNCHRONIZED) == false)
      return;
   // Manage exit if a position exists
   if (PositionSelect(sym)) { HandleExit(sym, state); return; }
   // Enforce cooldown
   if (TimeCurrent() - state.lastTradeTime < ReentryCooldownBars * PeriodSeconds(_Period))
      return;

   // Fetch current bid/ask
   double bid, ask;
   if (!SymbolInfoDouble(sym, SYMBOL_BID, bid) || !SymbolInfoDouble(sym, SYMBOL_ASK, ask))
      return;

   // Create Bollinger Bands indicator handle
   int bbHandle = iBands(sym, ExpectedTimeframe, BBPeriod, BBDeviation, 0, PRICE_CLOSE);
   if (bbHandle == INVALID_HANDLE)
      return;

   double bufUpper[1], bufLower[1], bufMiddle[1];
   if (CopyBuffer(bbHandle, 1, 0, 1, bufUpper) != 1 ||
       CopyBuffer(bbHandle, 2, 0, 1, bufLower) != 1 ||
       CopyBuffer(bbHandle, 0, 0, 1, bufMiddle) != 1) {
      IndicatorRelease(bbHandle);
      return;
   }
   IndicatorRelease(bbHandle);

   double upper  = bufUpper[0];
   double lower  = bufLower[0];
   double middle = bufMiddle[0];
   double width  = (upper - lower) / middle;
   if (width < BBWidthThreshold)
      return;

   // Compute RSI for previous candle
   int rsiHandle = iRSI(sym, ExpectedTimeframe, RSI_Period, PRICE_CLOSE);
   if (rsiHandle == INVALID_HANDLE)
      return;
   double rsiArr[2];
   if (CopyBuffer(rsiHandle, 0, 0, 2, rsiArr) != 2) {
      IndicatorRelease(rsiHandle);
      return;
   }
   IndicatorRelease(rsiHandle);
   double prevRSI = rsiArr[1];

   // Get previous and current candle data
   double prevClose = iClose(sym, ExpectedTimeframe, 1);
   double prevHigh  = iHigh(sym, ExpectedTimeframe, 1);
   double prevLow   = iLow(sym, ExpectedTimeframe, 1);
   double currClose = iClose(sym, ExpectedTimeframe, 0);

   // Entry logic: bullish reversal
   bool isBuy = false;
   if (prevClose < lower && prevRSI < RSI_LowThreshold && currClose > prevHigh) {
      isBuy = true;
   }
   // Entry logic: bearish reversal
   else if (prevClose > upper && prevRSI > RSI_HighThreshold && currClose < prevLow) {
      isBuy = false;
   }
   else {
      return;
   }

   // ATR-based stop-loss and take-profit
   double atrArr[1];
   if (CopyBuffer(state.atrHandle, 0, 0, 1, atrArr) != 1 || atrArr[0] <= 0)
      return;
   double sl = isBuy
      ? NormalizeDouble(bid - ATRMultiplierSL * atrArr[0], _Digits)
      : NormalizeDouble(ask + ATRMultiplierSL * atrArr[0], _Digits);
   double tp = isBuy
      ? NormalizeDouble(bid + ATRMultiplierTP * atrArr[0], _Digits)
      : NormalizeDouble(ask - ATRMultiplierTP * atrArr[0], _Digits);

   // Open trade
   double lot = CalculateLotSize(sym, sl, isBuy);
   OpenTrade(sym, isBuy, sl, tp, lot, NULL);
   state.lastTradeTime = TimeCurrent();
}

#endif // __StrategyBBRSIReversal_mqh__
