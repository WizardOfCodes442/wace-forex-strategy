//+------------------------------------------------------------------+
//|                                                     Research.mq5 |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Includes                                                         |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Indicators\Oscilators.mqh>
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input ENUM_TIMEFRAMES      TimeFrame   =  PERIOD_H1;
input double               MinProfit   =  10;
//---
input group                "---- RSI ----"
input int                  RSIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   RSIPrice    =  PRICE_CLOSE;   //Applied price
//---
input group                "---- CCI ----"
input int                  CCIPeriod   =  14;            //Period
input ENUM_APPLIED_PRICE   CCIPrice    =  PRICE_TYPICAL; //Applied price
//---
input group                "---- ATR ----"
input int                  ATRPeriod   =  14;            //Period
//---
input group                "---- MACD ----"
input int                  FastPeriod  =  12;            //Fast
input int                  SlowPeriod  =  26;            //Slow
input int                  SignalPeriod =  9;            //Signal
input ENUM_APPLIED_PRICE   MACDPrice   =  PRICE_CLOSE;   //Applied price
input int                  OptimizationAgents = 1;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
SState               sState;
STrajectory          Base;
STrajectory          Buffer[];
STrajectory          Frame[1];
CNet                 Agent;
CNet                 RTG;
//---
float                dError;
datetime             dtStudied;
//---
CSymbolInfo          Symb;
CTrade               Trade;
//---
MqlRates             Rates[];
CiRSI                RSI;
CiCCI                CCI;
CiATR                ATR;
CiMACD               MACD;
//---
CBufferFloat         bState;
CBufferFloat        *Result;
vector<float>        check;
vector<float>        AgentResult;
double               PrevBalance = 0;
double               PrevEquity = 0;
int                  History = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!Symb.Name(_Symbol))
      return INIT_FAILED;
   Symb.Refresh();
//---
   if(!RSI.Create(Symb.Name(), TimeFrame, RSIPeriod, RSIPrice))
      return INIT_FAILED;
//---
   if(!CCI.Create(Symb.Name(), TimeFrame, CCIPeriod, CCIPrice))
      return INIT_FAILED;
//---
   if(!ATR.Create(Symb.Name(), TimeFrame, ATRPeriod))
      return INIT_FAILED;
//---
   if(!MACD.Create(Symb.Name(), TimeFrame, FastPeriod, SlowPeriod, SignalPeriod, MACDPrice))
      return INIT_FAILED;
//---
   History = MathMax(ValueBars, NBarInPattern);
   if(!RSI.BufferResize(History) || !CCI.BufferResize(History) ||
      !ATR.BufferResize(History) || !MACD.BufferResize(History))
     {
      PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
      return INIT_FAILED;
     }
//---
   if(!Trade.SetTypeFillingBySymbol(Symb.Name()))
      return INIT_FAILED;
//--- load models
   float temp;
   if(!Agent.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *agent = new CArrayObj();
      if(!CreateDescriptions(agent))
        {
         delete agent;
         return INIT_FAILED;
        }
      if(!Agent.Create(agent))
        {
         delete agent;
         return INIT_FAILED;
        }
      delete agent;
     }
   if(!RTG.Load(FileName + "RTG.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *rtg = new CArrayObj();
      if(!CreateRTGDescriptions(rtg))
        {
         delete rtg;
         return INIT_FAILED;
        }
      if(!RTG.Create(rtg))
        {
         delete rtg;
         return INIT_FAILED;
        }
      delete rtg;
     }
//---
   RTG.SetOpenCL(Agent.GetOpenCL());
//---
   Agent.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the Agent does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   AgentResult = vector<float>::Zeros(NActions);
//---
   RTG.getResults(Result);
   if(Result.Total() != NRewards)
     {
      PrintFormat("The scope of the RTG-model does not match the rewards count (%d <> %d)", NRewards, Result.Total());
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   delete Result;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   if(!IsNewBar())
      return;
//---
   int bars = CopyRates(Symb.Name(), TimeFrame, iTime(Symb.Name(), TimeFrame, 1), History, Rates);
   if(!ArraySetAsSeries(Rates, true))
      return;
//---
   RSI.Refresh();
   CCI.Refresh();
   ATR.Refresh();
   MACD.Refresh();
   Symb.Refresh();
   Symb.RefreshRates();
//--- History data
   float atr = 0;
   bState.Clear();
   for(int b = ValueBars - 1; b >= 0; b--)
     {
      float open = (float)Rates[b].open;
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      bState.Add((float)(Rates[b].close - open));
      bState.Add((float)(Rates[b].high - open));
      bState.Add((float)(Rates[b].low - open));
      bState.Add((float)(Rates[b].tick_volume / 1000.0f));
      bState.Add(rsi);
      bState.Add(cci);
      bState.Add(atr);
      bState.Add(macd);
      bState.Add(sign);
     }
   if(!RTG.feedForward((CArrayFloat*)GetPointer(bState), 1, false,(CBufferFloat*)NULL))
      return;
//---
   for(int b = 0; b < (int)NBarInPattern; b++)
     {
      float open = (float)Rates[b].open;
      float rsi = (float)RSI.Main(b);
      float cci = (float)CCI.Main(b);
      atr = (float)ATR.Main(b);
      float macd = (float)MACD.Main(b);
      float sign = (float)MACD.Signal(b);
      if(rsi == EMPTY_VALUE || cci == EMPTY_VALUE || atr == EMPTY_VALUE || macd == EMPTY_VALUE || sign == EMPTY_VALUE)
         continue;
      //---
      int shift = b * BarDescr;
      sState.state[shift] = (float)(Rates[b].close - open);
      sState.state[shift + 1] = (float)(Rates[b].high - open);
      sState.state[shift + 2] = (float)(Rates[b].low - open);
      sState.state[shift + 3] = (float)(Rates[b].tick_volume / 1000.0f);
      sState.state[shift + 4] = rsi;
      sState.state[shift + 5] = cci;
      sState.state[shift + 6] = atr;
      sState.state[shift + 7] = macd;
      sState.state[shift + 8] = sign;
     }
   bState.AssignArray(sState.state);
//--- Account description
   sState.account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   sState.account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   double position_discount = 0;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
   int total = PositionsTotal();
   datetime current = TimeCurrent();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      double profit = PositionGetDouble(POSITION_PROFIT);
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += profit;
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += profit;
            break;
        }
      position_discount += profit - (current - PositionGetInteger(POSITION_TIME)) * multiplyer * MathAbs(profit);
     }
   sState.account[2] = (float)buy_value;
   sState.account[3] = (float)sell_value;
   sState.account[4] = (float)buy_profit;
   sState.account[5] = (float)sell_profit;
   sState.account[6] = (float)position_discount;
   sState.account[7] = (float)Rates[0].time;
//---
   bState.Add((float)((sState.account[0] - PrevBalance) / PrevBalance));
   bState.Add((float)(sState.account[1] / PrevBalance));
   bState.Add((float)((sState.account[1] - PrevEquity) / PrevEquity));
   bState.Add(sState.account[2]);
   bState.Add(sState.account[3]);
   bState.Add((float)(sState.account[4] / PrevBalance));
   bState.Add((float)(sState.account[5] / PrevBalance));
   bState.Add((float)(sState.account[6] / PrevBalance));
//--- Time label
   double x = (double)Rates[0].time / (double)(D'2024.01.01' - D'2023.01.01');
   bState.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_MN1);
   bState.Add((float)MathCos(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_W1);
   bState.Add((float)MathSin(2.0 * M_PI * x));
   x = (double)Rates[0].time / (double)PeriodSeconds(PERIOD_D1);
   bState.Add((float)MathSin(2.0 * M_PI * x));
//--- Prev action
   bState.AddArray(AgentResult);
//--- Latent representation
   RTG.getResults(Result);
   bState.AddArray(Result);
//---
   if(!Agent.feedForward(GetPointer(bState), 1, false, (CBufferFloat *)NULL))
      return;
   Agent.getResults(AgentResult);
   for(ulong i = 0; i < AgentResult.Size(); i++)
     {
      float rnd = ((float)MathRand() / 32767.0f - 0.5f) * 0.03f;
      float t = AgentResult[i] + rnd;
      if(t > 1 || t < 0)
         t = AgentResult[i] - rnd;
      AgentResult[i] = t;
     }
   AgentResult.Clip(0.0f, 1.0f);
//---
   PrevBalance = sState.account[0];
   PrevEquity = sState.account[1];
//---
   double min_lot = Symb.LotsMin();
   double step_lot = Symb.LotsStep();
   double stops = MathMax(Symb.StopsLevel(), 1) * Symb.Point();
   if(AgentResult[0] >= AgentResult[3])
     {
      AgentResult[0] -= AgentResult[3];
      AgentResult[3] = 0;
     }
   else
     {
      AgentResult[3] -= AgentResult[0];
      AgentResult[0] = 0;
     }
//--- buy control
   if(AgentResult[0] < 0.9*min_lot || (AgentResult[1] * MaxTP * Symb.Point()) <= stops || (AgentResult[2] * MaxSL * Symb.Point()) <= stops)
     {
      if(buy_value > 0)
         CloseByDirection(POSITION_TYPE_BUY);
     }
   else
     {
      double buy_lot = min_lot + MathRound((double)(AgentResult[0] - min_lot) / step_lot) * step_lot;
      double buy_tp = Symb.NormalizePrice(Symb.Ask() + AgentResult[1] * MaxTP * Symb.Point());
      double buy_sl = Symb.NormalizePrice(Symb.Ask() - AgentResult[2] * MaxSL * Symb.Point());
      if(buy_value > 0)
         TrailPosition(POSITION_TYPE_BUY, buy_sl, buy_tp);
      if(buy_value != buy_lot)
        {
         if(buy_value > buy_lot)
            ClosePartial(POSITION_TYPE_BUY, buy_value - buy_lot);
         else
            Trade.Buy(buy_lot - buy_value, Symb.Name(), Symb.Ask(), buy_sl, buy_tp);
        }
     }
//--- sell control
   if(AgentResult[3] < 0.9*min_lot || (AgentResult[4] * MaxTP * Symb.Point()) <= stops || (AgentResult[5] * MaxSL * Symb.Point()) <= stops)
     {
      if(sell_value > 0)
         CloseByDirection(POSITION_TYPE_SELL);
     }
   else
     {
      double sell_lot = min_lot + MathRound((double)(AgentResult[3] - min_lot) / step_lot) * step_lot;;
      double sell_tp = Symb.NormalizePrice(Symb.Bid() - AgentResult[4] * MaxTP * Symb.Point());
      double sell_sl = Symb.NormalizePrice(Symb.Bid() + AgentResult[5] * MaxSL * Symb.Point());
      if(sell_value > 0)
         TrailPosition(POSITION_TYPE_SELL, sell_sl, sell_tp);
      if(sell_value != sell_lot)
        {
         if(sell_value > sell_lot)
            ClosePartial(POSITION_TYPE_SELL, sell_value - sell_lot);
         else
            Trade.Sell(sell_lot - sell_value, Symb.Name(), Symb.Bid(), sell_sl, sell_tp);
        }
     }
//---
   int shift = BarDescr * (NBarInPattern);
   sState.rewards[0] = bState[shift];
   sState.rewards[1] = bState[shift + 1] - 1.0f;
   if((buy_value + sell_value) == 0)
      sState.rewards[2] -= (float)(atr / PrevBalance);
   else
      sState.rewards[2] = 0;
   for(ulong i = 0; i < NActions; i++)
      sState.action[i] = AgentResult[i];
   if(!Base.Add(sState))
      ExpertRemove();
  }
//+------------------------------------------------------------------+
//| Tester function                                                  |
//+------------------------------------------------------------------+
double OnTester()
  {
//---
   double ret = 0.0;
//---
   double profit = TesterStatistics(STAT_PROFIT);
   Frame[0] = Base;
   if(profit >= MinProfit)
      FrameAdd(MQLInfoString(MQL_PROGRAM_NAME), 1, profit, Frame);
//---
   return(ret);
  }
//+------------------------------------------------------------------+
//| TesterInit function                                              |
//+------------------------------------------------------------------+
void OnTesterInit()
  {
//---
   LoadTotalBase();
  }
//+------------------------------------------------------------------+
//| TesterPass function                                              |
//+------------------------------------------------------------------+
void OnTesterPass()
  {
//---
   ulong pass;
   string name;
   long id;
   double value;
   STrajectory array[];
   while(FrameNext(pass, name, id, value, array))
     {
      int total = ArraySize(Buffer);
      if(name != MQLInfoString(MQL_PROGRAM_NAME))
         continue;
      if(id <= 0)
         continue;
      if(ArrayResize(Buffer, total + (int)id, 10) < 0)
         return;
      ArrayCopy(Buffer, array, total, 0, (int)id);
     }
  }
//+------------------------------------------------------------------+
//| TesterDeinit function                                            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
//---
   int total = ArraySize(Buffer);
   printf("total %d", MathMin(total, MaxReplayBuffer));
   Print("Saving...");
   SaveTotalBase(1);
   Print("Saved");
  }
//+------------------------------------------------------------------+
