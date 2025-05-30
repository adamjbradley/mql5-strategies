#property indicator_separate_window
#property indicator_buffers 1
#property indicator_plots   1

#property indicator_label1  "LWTI"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2

//--- input parameters
input int LWTIPeriod    = 25;   // Raw volume EMA period
input int LWTISmoothing = 20;   // EMA smoothing period

//--- indicator buffers
double rawVol[];
double emaRaw[];
double LWTIBuffer[];

int OnInit()
{
    //--- indicator buffers mapping
    SetIndexBuffer(0, LWTIBuffer, INDICATOR_DATA);

    //--- DO NOT set as series - use standard indexing for all arrays
    ArraySetAsSeries(rawVol, false);
    ArraySetAsSeries(emaRaw, false);
    ArraySetAsSeries(LWTIBuffer, false);

    //--- indicator name
    IndicatorSetString(INDICATOR_SHORTNAME, "Larry Williams Trade Indicator");
    
    //--- name for DataWindow
    PlotIndexSetString(0, PLOT_LABEL, "LWTI");
    IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

    return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
    // Check for minimum required bars
    if(rates_total < 2) return(0);
    
    // Check if input arrays are set as series
    bool inputIsSeries = ArrayGetAsSeries(close);
    
    // Ensure all arrays are properly sized
    ArrayResize(rawVol, rates_total);
    ArrayResize(emaRaw, rates_total);
    
    // Prevent division by zero
    double alpha1 = (LWTIPeriod > 0) ? 2.0 / (LWTIPeriod + 1.0) : 0.1;
    double alpha2 = (LWTISmoothing > 0) ? 2.0 / (LWTISmoothing + 1.0) : 0.1;
    
    // Determine starting point for calculation
    int start;
    
    // First time calculation or after parameter change
    if(prev_calculated <= 0)
    {
        // Initialize all bars with zero
        for(int i = 0; i < rates_total; i++)
        {
            rawVol[i] = 0;
            emaRaw[i] = 0;
            LWTIBuffer[i] = 0;
        }
        
        // Start calculation from the first bar
        start = 0;
    }
    else
    {
        // Continue calculation from where we left off
        start = prev_calculated - 1;
        // Ensure we don't go below valid index
        if(start < 0) start = 0;
    }
    
    // Main calculation loop
    for(int i = start; i < rates_total; i++)
    {
        // Get correct index for input arrays based on whether they are series or not
        int idx = inputIsSeries ? rates_total - 1 - i : i;
        
        // Calculate raw volume with price direction
        rawVol[i] = (close[idx] > open[idx]) ? tick_volume[idx] : 
                   ((close[idx] < open[idx]) ? -tick_volume[idx] : 0);
        
        // First EMA calculation
        if(i == 0)
            emaRaw[i] = rawVol[i];
        else
            emaRaw[i] = rawVol[i] * alpha1 + emaRaw[i-1] * (1 - alpha1);
        
        // Second EMA gives LWTI
        if(i == 0)
            LWTIBuffer[i] = emaRaw[i];
        else
            LWTIBuffer[i] = emaRaw[i] * alpha2 + LWTIBuffer[i-1] * (1 - alpha2);
    }

    return(rates_total);
}
