//+------------------------------------------------------------------+
//|                                                     Research.mq5 |
//|                                                   Copyright DNG® |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright DNG®"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#include "Trajectory.mqh"
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 100000;
input int                  LatentLayer    =  8;
input float                DiscountFactor  = 0.99f;
bool                 TrainMode = true;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Scheduler;
CFQF                 Actor;
//---
float                dError;
datetime             dtStudied;
//---
CBufferFloat         State;
CBufferFloat         *Result;
vector<float>        SchedulerResult;
vector<float>        check;
vector<float>        ActorResult;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
//--- load models
   float temp;
   if(!Scheduler.Load(FileName + "Sch.nnw", temp, temp, temp, dtStudied, true))
     {
      PrintFormat("Error of load scheduler model: %d", GetLastError());
      return INIT_FAILED;
     }
   if(!Actor.Load(FileName + "Act3.nnw", dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *scheduler = new CArrayObj();
      if(!CreateDescriptions(actor, scheduler))
        {
         delete actor;
         delete scheduler;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor))
        {
         delete actor;
         delete scheduler;
         return INIT_FAILED;
        }
      delete actor;
      delete scheduler;
      //---
     }
//---
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   Actor.SetOpenCL(Scheduler.GetOpenCL());
   Actor.SetUpdateTarget(Iterations+10);
//---
   Scheduler.getResults(Result);
   if(Result.Total() != AccountDescr)
     {
      PrintFormat("The scope of the scheduler does not match the account description (%d <> %d)", AccountDescr, Result.Total());
      return INIT_FAILED;
     }
//---
   Actor.GetLayerOutput(0, Result);
   int inputs = Result.Total();
   if(!Scheduler.GetLayerOutput(LatentLayer, Result))
     {
      PrintFormat("Error of load latent layer %d", LatentLayer);
      return INIT_FAILED;
     }
   if(inputs != Result.Total())
     {
      PrintFormat("Size of latent layer does not match input size of Actor (%d <> %d)", Result.Total(), inputs);
      return INIT_FAILED;
     }
//---
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   Actor.Save(FileName + "Act3.nnw", TimeCurrent(), true);
   delete Result;
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == 1001)
      Train();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool LoadTotalBase(void)
  {
   int handle = FileOpen(FileName + ".bd", FILE_READ | FILE_BIN | FILE_COMMON | FILE_SHARE_READ);
   if(handle < 0)
      return false;
   int total = FileReadInteger(handle);
   if(total <= 0)
     {
      FileClose(handle);
      return false;
     }
   if(ArrayResize(Buffer, total) < total)
     {
      FileClose(handle);
      return false;
     }
   for(int i = 0; i < total; i++)
      if(!Buffer[i].Load(handle))
        {
         FileClose(handle);
         return false;
        }
   FileClose(handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   uint ticks = GetTickCount();
   vector<float> account, reward;
   int bar;
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = (int)(((double)MathRand() / 32767.0) * (total_tr - 1));
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2));
      State.AssignArray(Buffer[tr].States[i].state);
      float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
      float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
      State.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
      State.Add(Buffer[tr].States[i].account[1] / PrevBalance);
      State.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
      State.Add(Buffer[tr].States[i].account[2] / PrevBalance);
      State.Add(Buffer[tr].States[i].account[4] / PrevBalance);
      State.Add(Buffer[tr].States[i].account[5]);
      State.Add(Buffer[tr].States[i].account[6]);
      State.Add(Buffer[tr].States[i].account[7] / PrevBalance);
      State.Add(Buffer[tr].States[i].account[8] / PrevBalance);
      //---
      bar = (HistoryBars - 1) * BarDescr;
      double cl_op = Buffer[tr].States[i + 1].state[bar];
      double prof_1l = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT) * cl_op /
                       SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      PrevBalance = Buffer[tr].States[i].account[0];
      PrevEquity = Buffer[tr].States[i].account[1];
      if(IsStopped())
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //---
      if(!Scheduler.feedForward(GetPointer(State), 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //---
      if(!Scheduler.GetLayerOutput(LatentLayer, Result))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //vector<float> temp;
      //Result.GetData(temp);
      //float delta;
      //if(check.Size()==temp.Size())
      //   delta=MathPow(check-temp,2.0f).Sum();
      //check=temp;
      //---
      if(!Actor.feedForward(Result, 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      //---
      ActorResult = vector<float>::Zeros(NActions);
      ActorResult[Buffer[tr].Actions[i]] = Buffer[tr].Revards[i];
      Result.AssignArray(ActorResult);
      if(!Actor.backProp(Result, DiscountFactor, NULL, 1, false))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
      if(GetTickCount() - ticks > 500)
        {
         string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Actor", iter * 100.0 / (double)(Iterations), Actor.getRecentAverageError());
         Comment(str);
         ticks = GetTickCount();
        }
     }
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   ExpertRemove();
//---
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector<float> GetNewState(float &prev_account[], int action, double prof_1l)
  {
   vector<float> result;
//---
   result.Assign(prev_account);
   switch(action)
     {
      case 0:
         result[5] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
      case 1:
         result[6] += (float)SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
      case 2:
         result[0] += result[4];
         result[1] = result[0];
         result[2] = result[0];
         for(int i = 3; i < AccountDescr; i++)
            result[i] = 0;
         break;
      case 3:
         result[7] += result[5] * (float)prof_1l;
         result[8] -= result[6] * (float)prof_1l;
         result[4] = result[7] + result[8];
         result[1] = result[0] + result[4];
         break;
     }
//--- return result
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector<float> GetAgentReward(int skill, vector<float> &discriminator, float &prev_account[])
  {
//--- prepare
   matrix<float> discriminator_matrix;
   discriminator_matrix.Init(1, discriminator.Size());
   discriminator_matrix.Row(discriminator, 0);
   discriminator_matrix.Reshape(NSkills, AccountDescr);
   vector<float> forecast = discriminator_matrix.Row(skill);
//--- check action
   int action = 3;
   float buy = forecast[5] - prev_account[5];
   float sell = forecast[6] - prev_account[6];
   if(buy < 0 && sell < 0)
      action = 2;
   else
      if(buy > sell)
         action = 0;
      else
         if(buy < sell)
            action = 1;
//--- calculate reward
   vector<float> result = vector<float>::Zeros(NActions);
   float mean = (forecast / discriminator_matrix.Mean(0)).Mean();
   result[action] = MathLog(MathAbs(mean));
//--- return result
   return result;
  }
//+------------------------------------------------------------------+
