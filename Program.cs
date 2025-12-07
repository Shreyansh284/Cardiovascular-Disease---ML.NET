using Microsoft.ML;
using Microsoft.ML.Data;
using System;

class Program
{
    static void Main()
    {
        var ml = new MLContext(seed: 1);

        // ✅ 1. Load Data
        var data = ml.Data.LoadFromTextFile<CardioData>(
            path: "D:\\DotNet\\cardiovascular-disease\\Data\\cardio_train.csv",
            hasHeader: true,
            separatorChar: ';');

        // ✅ 2. Outlier Filtering (Medical Cleaning)
        var cleanedData = ml.Data.FilterRowsByColumn(data, nameof(CardioData.ap_hi), 70, 250);
        cleanedData = ml.Data.FilterRowsByColumn(cleanedData, nameof(CardioData.ap_lo), 40, 150);
        cleanedData = ml.Data.FilterRowsByColumn(cleanedData, nameof(CardioData.weight), 30, 200);

        // ✅ 3. Train/Test Split
        var split = ml.Data.TrainTestSplit(cleanedData, testFraction: 0.2);

        // ✅ 4. Full Preprocessing + Training Pipeline
        var pipeline =
            ml.Transforms.ReplaceMissingValues(nameof(CardioData.weight))
            .Append(ml.Transforms.NormalizeMinMax(nameof(CardioData.age)))
            .Append(ml.Transforms.NormalizeMinMax(nameof(CardioData.height)))
            .Append(ml.Transforms.NormalizeMinMax(nameof(CardioData.weight)))
            .Append(ml.Transforms.NormalizeMinMax(nameof(CardioData.ap_hi)))
            .Append(ml.Transforms.NormalizeMinMax(nameof(CardioData.ap_lo)))

            .Append(ml.Transforms.Concatenate("Features",
                nameof(CardioData.age),
                nameof(CardioData.gender),
                nameof(CardioData.height),
                nameof(CardioData.weight),
                nameof(CardioData.ap_hi),
                nameof(CardioData.ap_lo),
                nameof(CardioData.cholesterol),
                nameof(CardioData.gluc),
                nameof(CardioData.smoke),
                nameof(CardioData.alco),
                nameof(CardioData.active)
            ))

            .Append(ml.BinaryClassification.Trainers.FastTree());

        // ✅ 5. Train Model
        var model = pipeline.Fit(split.TrainSet);

        // ✅ 6. Evaluate Model
        var predictions = model.Transform(split.TestSet);
        var metrics = ml.BinaryClassification.Evaluate(predictions);

        Console.WriteLine("\n====== MODEL METRICS ======");
        Console.WriteLine($"Accuracy: {metrics.Accuracy}");
        Console.WriteLine($"F1 Score: {metrics.F1Score}");
        Console.WriteLine($"Precision: {metrics.PositivePrecision}");
        Console.WriteLine($"Recall: {metrics.PositiveRecall}");

        Console.WriteLine("\nConfusion Matrix:");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

        // ✅ 7. Cross Validation
        var cvResults = ml.BinaryClassification.CrossValidate(cleanedData, pipeline, numberOfFolds: 5);
        Console.WriteLine("\nCross Validation Accuracies:");
        foreach (var r in cvResults)
            Console.WriteLine(r.Metrics.Accuracy);

        // ✅ 8. Save Model
        ml.Model.Save(model, split.TrainSet.Schema, "cardio_model.zip");
        Console.WriteLine("\nModel saved as cardio_model.zip");

        // ✅ 9. Load Model
        var loadedModel = ml.Model.Load("cardio_model.zip", out var schema);

        // ✅ 10. Prediction Engine
        var engine = ml.Model.CreatePredictionEngine<CardioData, CardioPrediction>(loadedModel);

        // ✅ 11. Sample Prediction
        var sample = new CardioData
        {
            age = 20000,
            gender = 1,
            height = 165,
            weight = 72,
            ap_hi = 140,
            ap_lo = 90,
            cholesterol = 2,
            gluc = 1,
            smoke = 0,
            alco = 0,
            active = 1
        };

        var result = engine.Predict(sample);

        Console.WriteLine("\n====== FINAL PREDICTION ======");
        Console.WriteLine($"Has Disease? {result.Prediction}");
        Console.WriteLine($"Probability: {result.Probability}");
    }
}
