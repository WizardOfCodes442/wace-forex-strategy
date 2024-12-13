//+------------------------------------------------------------------+
//|                                                    AsocRules.mqh |
//|                                              Copyright 2022, DNG |
//|                                https://www.mql5.com/ru/users/dng |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, DNG"
#property link      "https://www.mql5.com/ru/users/dng"
#property version   "1.00"
//---
#include "MyTreeNode.mqh"
//---
class CAssocRules : public CObject
  {
protected:
   CMyTreeNode          m_cRoot;
   CMyTreeNode          m_cBuyRules;
   CMyTreeNode          m_cSellRules;
   vectorf               m_vMin;
   vectorf               m_vStep;
   int                  m_iSections;
   matrixf               m_mPositions;
   matrixf               m_BuyPositions;
   matrixf               m_SellPositions;
   //---
   bool              NewPath(CMyTreeNode *root, matrixf &path);
   CMyTreeNode       *CheckPath(CMyTreeNode *root, vectorf &path);
   //---
   bool              PrepaerData(matrixf &data, matrixf &bin_data, vectorf &buy, vectorf &sell, const int sections = 10, const float min_sup = 0.03f);
   matrixf            CreatePath(vectorf &bin_data, matrixf &positions);
   matrixf            CreatePositions(vectorf &support, const float min_sup = 0.03f);
   bool              GrowsTree(CMyTreeNode *root, matrixf &bin_data, matrixf &positions);
   float             Probability(CMyTreeNode *root, vectorf &data, matrixf &positions);

public:
                     CAssocRules();
                    ~CAssocRules();
   //---
   bool              CreateRules(matrixf &data, vectorf &buy, vectorf &sell, int sections = 10, float min_freq = 0.03f, float min_prob = 0.3f);
   bool              Probability(vectorf &data, float &buy, float &sell);
   //--- methods for working with files
   virtual bool      Save(const int file_handle);
   virtual bool      Load(const int file_handle);
   virtual bool      Save(const string file_name);
   virtual bool      Load(const string file_name);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAssocRules::CAssocRules()   :  m_iSections(10)
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAssocRules::~CAssocRules()
  {
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::NewPath(CMyTreeNode *root, matrixf &path)
  {
   ulong total = path.Cols();
   if(total <= 0)
      return false;
   CMyTreeNode *parent = root;
   root.IncreaseSupport(path[1, 0]);
   for(ulong i = 0; i < total; i++)
     {
      CMyTreeNode *temp = parent.GetNext((ulong)path[0, i]);
      if(!temp)
        {
         temp = parent.AddNode((int)path[0, i], 0);
         if(!temp)
            return false;
        }
      temp.IncreaseSupport(path[1, i]);
      parent = temp;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::CreateRules(matrixf &data, vectorf &buy, vectorf &sell, int sections = 10, float min_sup = 0.03f, float min_conf = 0.3f)
  {
   if(data.Rows() <= 0 || data.Cols() <= 0 || sections <= 0 ||
      data.Rows() != buy.Size() || data.Rows() != sell.Size())
      return false;
//---
   matrixf binary_data;
   if(!PrepaerData(data, binary_data, buy, sell, sections))
      return false;
//---
   double k = 1.0 / (double)(binary_data.Rows());
   if(!GrowsTree(GetPointer(m_cRoot), binary_data * k, m_mPositions))
      return false;
//--- create buy rules
   vectorf supports = vectorf::Zeros(binary_data.Cols());
   binary_data = matrixf::Zeros(0, binary_data.Cols());
   if(!m_cRoot.Mining(supports, binary_data, m_cBuyRules.ID(), min_conf))
      return false;
   supports[m_cBuyRules.ID()] = 0;
   m_BuyPositions = CreatePositions(supports, min_sup);
   if(m_BuyPositions.Rows() > 0)
      if(!GrowsTree(GetPointer(m_cBuyRules), binary_data, m_BuyPositions))
         return false;
//--- create sell rules
   supports = vectorf::Zeros(binary_data.Cols());
   binary_data = matrixf::Zeros(0, binary_data.Cols());
   if(!m_cRoot.Mining(supports, binary_data, m_cSellRules.ID(), min_conf))
      return false;
   supports[m_cSellRules.ID()] = 0;
   m_SellPositions = CreatePositions(supports, min_sup);
   if(m_SellPositions.Rows() > 0)
      if(!GrowsTree(GetPointer(m_cSellRules), binary_data, m_SellPositions))
         return false;
//---
   m_cRoot.Clear();
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::PrepaerData(matrixf & data, matrixf & bin_data, vectorf & buy, vectorf & sell, const int sections = 10, const float min_sup = 0.03f)
  {
//---
   m_iSections = sections;
   m_vMin = data.Min(0);
   vectorf max = data.Max(0);
   vectorf delt = max - m_vMin;
   m_vStep = delt / sections + 1e-8;
   m_cBuyRules.ID(data.Cols() * m_iSections);
   m_cSellRules.ID(m_cBuyRules.ID() + 1);
   bin_data = matrixf::Zeros(data.Rows(), m_cSellRules.ID() + 1);
   for(ulong r = 0; r < data.Rows(); r++)
     {
      vectorf pos = (data.Row(r) - m_vMin) / m_vStep;
      if(!pos.Clip(0, m_iSections - 1))
         return false;
      for(ulong c = 0; c < pos.Size(); c++)
         bin_data[r, c * sections + (int)pos[c]] = 1;
     }
   if(!bin_data.Col(buy, m_cBuyRules.ID()) ||
      !bin_data.Col(sell, m_cSellRules.ID()))
      return false;
   vectorf supp = bin_data.Sum(0) / bin_data.Rows();
   m_mPositions = CreatePositions(supp, min_sup);
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
float CAssocRules::Probability(CMyTreeNode * root, vectorf & data, matrixf & positions)
  {
   if(m_cRoot.Total() <= 0)
      return 0;
   if(data.Size() != m_vMin.Size())
      return 0;
   vectorf pos = (data - m_vMin) / m_vStep;
   if(!pos.Clip(0, m_iSections - 1))
      return 0;
   vectorf bin_data = vectorf::Zeros(data.Size() * m_iSections);
   for(ulong c = 0; c < pos.Size(); c++)
      bin_data[c * m_iSections + (int)pos[c]] = 1;
   matrixf path = CreatePath(bin_data, positions);
   CMyTreeNode *temp = CheckPath(root, path.Row(0));
   if(!temp)
      return 0;
   if(temp.Total() > 0)
      return 0;
//---
   return temp.GetConfidence();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrixf CAssocRules::CreatePath(vectorf & bin_data, matrixf & positions)
  {
   ulong size = bin_data.Size();
//---
   ulong total = positions.Rows();
   int vect_pos = 0;
   matrixf path = matrixf::Zeros(2, total);
   for(ulong c = 0; c < total; c++)
     {
      ulong pos = (ulong)positions[c, 0];
      if(pos >= size)
         continue;
      if(bin_data[pos] == 0)
         continue;
      path[0, vect_pos] = (float)pos;
      path[1, vect_pos] = bin_data[pos];
      vect_pos++;
     }
   if(!path.Resize(2, vect_pos))
      return matrixf::Zeros(0, 0);
//---
   return path;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMyTreeNode *CAssocRules::CheckPath(CMyTreeNode * root, vectorf & path)
  {
   ulong total = path.Size();
   CMyTreeNode *node = root;
   for(ulong i = 0; i < total; i++)
     {
      node = node.GetNext((ulong)path[i]);
      if(!node)
         break;
     }
//---
   return node;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrixf CAssocRules::CreatePositions(vectorf &support, const float min_sup = 0.03f)
  {
   matrixf result = matrixf::Ones(support.Size(), 2);
   result = result.CumSum(0) - 1;
   if(!result.Col(support, 1))
      return matrixf::Zeros(0, 0);
   bool change = false;
   do
     {
      change = false;
      ulong total = result.Rows() - 1;
      for(ulong i = 0; i < total; i++)
        {
         if(result[i, 1] >= result[i + 1, 1])
            continue;
         if(result.SwapRows(i, i + 1))
            change = true;
        }
     }
   while(change);
   int i = 0;
   while(result[i, 1] >= min_sup)
      i++;
   if(!result.Resize(i, 2))
      return matrixf::Zeros(0, 0);
//---
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::GrowsTree(CMyTreeNode * root, matrixf & bin_data, matrixf &positions)
  {
   ulong rows = bin_data.Rows();
   for(ulong r = 0; r < rows; r++)
     {
      matrixf path = CreatePath(bin_data.Row(r), positions);
      ulong size = path.Cols();
      if(size <= 0)
         continue;
      if(!NewPath(root, path))
         return false;
     }
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::Probability(vectorf &data, float &buy, float &sell)
  {
   buy = Probability(GetPointer(m_cBuyRules), data, m_BuyPositions);
   sell = Probability(GetPointer(m_cSellRules), data, m_BuyPositions);
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::Save(const string file_name)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name, FILE_WRITE | FILE_BIN);
   bool result = Save(handle);
   FileClose(handle);
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::Load(const string file_name)
  {
   if(file_name == NULL)
      return false;
//---
   int handle = FileOpen(file_name, FILE_READ | FILE_BIN);
   bool result = Load(handle);
   FileClose(handle);
   return result;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::Save(const int file_handle)
  {
   if(!m_cBuyRules.Save(file_handle) ||
      !m_cSellRules.Save(file_handle))
      return false;
//---
   ulong rows = m_BuyPositions.Rows();
   ulong cols = m_BuyPositions.Cols();
   if(FileWriteLong(file_handle, rows) < sizeof(rows) ||
      FileWriteLong(file_handle, cols) < sizeof(cols))
      return false;
   for(ulong r = 0; r < rows; r++)
      for(ulong c = 0; c < cols; c++)
         if(FileWriteDouble(file_handle, m_BuyPositions[r, c]) < sizeof(m_BuyPositions[r, c]))
            return false;
//---
   rows = m_SellPositions.Rows();
   cols = m_SellPositions.Cols();
   if(FileWriteLong(file_handle, rows) < sizeof(rows) ||
      FileWriteLong(file_handle, cols) < sizeof(cols))
      return false;
   for(ulong r = 0; r < rows; r++)
      for(ulong c = 0; c < cols; c++)
         if(FileWriteDouble(file_handle, m_SellPositions[r, c]) < sizeof(m_SellPositions[r, c]))
            return false;
//---
   return true;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CAssocRules::Load(const int file_handle)
  {
   if(!m_cBuyRules.Load(file_handle) ||
      !m_cSellRules.Load(file_handle))
      return false;
//---
   ulong rows = FileReadLong(file_handle);
   ulong cols = FileReadLong(file_handle);
   if(!m_BuyPositions.Resize(rows, cols))
      return false;
   for(ulong r = 0; r < rows; r++)
      for(ulong c = 0; c < cols; c++)
         m_BuyPositions[r, c] = (float)FileReadDouble(file_handle);
//---
   rows = FileReadLong(file_handle);
   cols = FileReadLong(file_handle);
   if(!m_SellPositions.Resize(rows, cols))
      return false;
   for(ulong r = 0; r < rows; r++)
      for(ulong c = 0; c < cols; c++)
         m_SellPositions[r, c] = (float)FileReadDouble(file_handle);
//---
   return true;
  }
//+------------------------------------------------------------------+
