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

            //داده‌های بازخورد رستوران را از یک فایل متنی بارگذاری کنید و آن‌ها را به‌صورت یک شیء IDataView ذخیره کنید. 


            DataOperationsCatalog.TrainTestData dataSplit = MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.2);

            //متغیر dataSplit اکنون حاوی دو مجموعه TrainSet و TestSet است که می‌توانید از آن‌ها برای آموزش و ارزیابی مدل استفاده کنید.

            TextFeaturizingEstimator dataProcessPipeline = MlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(RestaurantFeedback.Text));
            // یک تخمین‌گر (TextFeaturizingEstimator) ایجاد می‌کند که متن‌ها را از ستون Text در داده‌های بازخورد رستوران گرفته و آن‌ها را به ویژگی‌های عددی تبدیل می‌کند و در ستون Features ذخیره می‌کند.

            var sdcaTrainer = MlContext.BinaryClassification.Trainers.SdcaLogisticRegression();

            EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline =
          dataProcessPipeline.Append(sdcaTrainer);
            //  یک زنجیره تخمین‌گر(EstimatorChain) ایجاد می‌کند که شامل مراحل پردازش داده(از جمله استخراج ویژگی‌های متنی) و آموزش مدل(با استفاده از تخمین‌گر SdcaRegressionTrainer) است.

            ITransformer trainedModel = trainingPipeline.Fit(dataSplit.TrainSet);
            //مدل را بر اساس داده‌های آموزشی (TrainSet) آموزش می‌دهد و نتیجه آموزش (مدل آموزش‌دیده) را در متغیر trainedModel ذخیره می‌کند. پس از این مرحله، می‌توانید از trainedModel برای پیش‌بینی نتایج جدید استفاده کنید یا آن را برای ارزیابی با داده‌های تست (TestSet) استفاده کنید.
  
            MlContext.Model.Save(trainedModel, dataSplit.TrainSet.Schema, ModelPath);
            //  این کد مدل آموزش‌دیده(trainedModel) را در یک فایل با مسیر مشخص‌شده(ModelPath) ذخیره می‌کند.

            IDataView testSetTransform = trainedModel.Transform(dataSplit.TestSet);
            //بررسی دقت پیش‌بینی‌ها
          
            CalibratedBinaryClassificationMetrics modelMetrics=MlContext.BinaryClassification.Evaluate(
                data:testSetTransform,
                labelColumnName: nameof(RestaurantFeedback.Label),
                scoreColumnName: nameof(RestaurantPrediction.Score));
            //بر اساس داده‌های تست ارزیابی می‌کند و نتایج ارزیابی را در modelMetrics ذخیره می‌کند.

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
