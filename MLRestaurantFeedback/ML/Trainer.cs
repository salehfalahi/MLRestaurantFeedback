using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using MLRestaurantFeedback.ML.Base;
using MLRestaurantFeedback.ML.Objects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLRestaurantFeedback.ML
{
    public class Trainer : BaseML
    {
        public void Train(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                Console.WriteLine($"Failed to find training data file ({trainingFileName}");
                return;
            }
            IDataView trainingDataView = MlContext.Data.LoadFromTextFile<RestaurantFeedback>(trainingFileName);

   


            DataOperationsCatalog.TrainTestData dataSplit = MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

  

            TextFeaturizingEstimator dataProcessPipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(RestaurantFeedback.Text));
          

            var sdcaTrainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
         

            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline =
          dataProcessPipeline.Append(sdcaTrainer);


            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);

      
            MlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);
        

            IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);
    
          
            CalibratedBinaryClassificationMetrics modelMetrics=MlContext.BinaryClassification.Evaluate(
                data:testSetTransform,
                labelColumnName: nameof(RestaurantFeedback.Label),
                scoreColumnName: nameof(RestaurantPrediction.Score));
     

            Console.WriteLine($"Area Under Curve: {modelMetrics.AreaUnderRocCurve:P2}{Environment.NewLine}" +
                              $"Area Under Precision Recall Curve: {modelMetrics.AreaUnderPrecisionRecallCurve:P2}{Environment.NewLine}" +
                              $"Accuracy: {modelMetrics.Accuracy:P2}{Environment.NewLine}" +
                              $"F1Score: {modelMetrics.F1Score:P2}{Environment.NewLine}" +
                              $"Positive Recall: {modelMetrics.PositiveRecall:#.##}{Environment.NewLine}" +
                              $"Negative Recall: {modelMetrics.NegativeRecall:#.##}{Environment.NewLine}"
                              );
          

        }
    }
}
