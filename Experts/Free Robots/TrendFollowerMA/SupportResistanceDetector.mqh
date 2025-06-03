//=== SupportResistanceDetector.mqh ===
#ifndef __SupportResistanceDetector_mqh__
#define __SupportResistanceDetector_mqh__

//+------------------------------------------------------------------+
//| Detect support and resistance levels on higher timeframe         |
//+------------------------------------------------------------------+
void DetectSupportResistance(string symbol, ENUM_TIMEFRAMES timeframe, int lookbackBars, int pivotLeft, int pivotRight)
{
   // Get the current symbol state
   int symbolIndex = -1;
   for(int i = 0; i < ArraySize(Symbols); i++)
   {
      if(Symbols[i] == symbol)
      {
         symbolIndex = i;
         break;
      }
   }
   
   if(symbolIndex == -1)
      return;
      
   SymbolState *state = (SymbolState *)symbolStates.At(symbolIndex);
   
   // Check if we need to update (only once per H1 bar)
   datetime currentH1Time = iTime(symbol, timeframe, 0);
   if(currentH1Time == state.lastSRTime)
      return;
      
   state.lastSRTime = currentH1Time;
   
   // Clear previous levels
   for(int i = 0; i < ArraySize(state.srNamesUp); i++)
   {
      if(state.srNamesUp[i] != "")
         ObjectDelete(0, state.srNamesUp[i]);
   }
   
   for(int i = 0; i < ArraySize(state.srNamesDn); i++)
   {
      if(state.srNamesDn[i] != "")
         ObjectDelete(0, state.srNamesDn[i]);
   }
   
   ArrayFree(state.srNamesUp);
   ArrayFree(state.srNamesDn);
   
   // Get price data
   MqlRates rates[];
   if(CopyRates(symbol, timeframe, 0, lookbackBars, rates) != lookbackBars)
      return;
      
   // Find pivot highs and lows
   int upCount = 0;
   int dnCount = 0;
   
   for(int i = pivotLeft; i < lookbackBars - pivotRight; i++)
   {
      // Check for pivot high
      bool isPivotHigh = true;
      for(int j = i - pivotLeft; j <= i + pivotRight; j++)
      {
         if(j == i) continue; // Skip self
         
         if(rates[j].high >= rates[i].high)
         {
            isPivotHigh = false;
            break;
         }
      }
      
      // Check for pivot low
      bool isPivotLow = true;
      for(int j = i - pivotLeft; j <= i + pivotRight; j++)
      {
         if(j == i) continue; // Skip self
         
         if(rates[j].low <= rates[i].low)
         {
            isPivotLow = false;
            break;
         }
      }
      
      // Draw levels
      if(isPivotHigh)
      {
         string name = "SR_High_" + IntegerToString(upCount);
         ArrayResize(state.srNamesUp, upCount + 1);
         state.srNamesUp[upCount] = name;
         
         ObjectCreate(0, name, OBJ_HLINE, 0, 0, rates[i].high);
         ObjectSetInteger(0, name, OBJPROP_COLOR, clrCrimson);
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
         ObjectSetString(0, name, OBJPROP_TOOLTIP, "Resistance: " + DoubleToString(rates[i].high, _Digits));
         
         upCount++;
      }
      
      if(isPivotLow)
      {
         string name = "SR_Low_" + IntegerToString(dnCount);
         ArrayResize(state.srNamesDn, dnCount + 1);
         state.srNamesDn[dnCount] = name;
         
         ObjectCreate(0, name, OBJ_HLINE, 0, 0, rates[i].low);
         ObjectSetInteger(0, name, OBJPROP_COLOR, clrForestGreen);
         ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_DASH);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
         ObjectSetString(0, name, OBJPROP_TOOLTIP, "Support: " + DoubleToString(rates[i].low, _Digits));
         
         dnCount++;
      }
   }
   
   Print("✅ Updated ", upCount, " resistance and ", dnCount, " support levels for ", symbol);
}

//+------------------------------------------------------------------+
//| Check if price is near any support or resistance level           |
//+------------------------------------------------------------------+
bool IsNearSupportResistance(string symbol, double price, double proximityPips)
{
   // Get the current symbol state
   int symbolIndex = -1;
   for(int i = 0; i < ArraySize(Symbols); i++)
   {
      if(Symbols[i] == symbol)
      {
         symbolIndex = i;
         break;
      }
   }
   
   if(symbolIndex == -1)
      return false;
      
   SymbolState *state = (SymbolState *)symbolStates.At(symbolIndex);
   
   // Convert pips to price
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   
   // Determine pip value based on digits (for JPY pairs and others)
   double pipMultiplier = 1.0;
   if(digits == 3 || digits == 5) // Standard 5-digit broker (EURUSD = 5 digits, USDJPY = 3 digits)
      pipMultiplier = 10.0;
   else if(digits == 2 || digits == 4) // 4-digit broker
      pipMultiplier = 1.0;
      
   double proximityDistance = proximityPips * point * pipMultiplier;
   
   // Check resistance levels
   for(int i = 0; i < ArraySize(state.srNamesUp); i++)
   {
      if(state.srNamesUp[i] == "")
         continue;
         
      double level = ObjectGetDouble(0, state.srNamesUp[i], OBJPROP_PRICE);
      
      if(MathAbs(price - level) < proximityDistance)
      {
         Print("⚠️ Price ", price, " is near resistance level ", level, " (distance: ", 
               MathAbs(price - level) / (point * pipMultiplier), " pips)");
         return true;
      }
   }
   
   // Check support levels
   for(int i = 0; i < ArraySize(state.srNamesDn); i++)
   {
      if(state.srNamesDn[i] == "")
         continue;
         
      double level = ObjectGetDouble(0, state.srNamesDn[i], OBJPROP_PRICE);
      
      if(MathAbs(price - level) < proximityDistance)
      {
         Print("⚠️ Price ", price, " is near support level ", level, " (distance: ", 
               MathAbs(price - level) / (point * pipMultiplier), " pips)");
         return true;
      }
   }
   
   return false;
}

//+------------------------------------------------------------------+
#endif // __SupportResistanceDetector_mqh__
