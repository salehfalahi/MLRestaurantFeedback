using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLRestaurantFeedback.ML.Objects
{
    public class RestaurantFeedback
    {
        [LoadColumn(0)]
        public bool Label {  get; set; }
        [LoadColumn(1)]
        public string Text { get; set; }
    }
}
