using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLTestProject
{
    public class CustomerData
    {
        public float Age { get; set; }
        public float Income { get; set; }
        public bool Label { get; set; } // True = Purchased, False = Not Purchased
    }

    public class CustomerPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PurchasePrediction { get; set; }
    }
}