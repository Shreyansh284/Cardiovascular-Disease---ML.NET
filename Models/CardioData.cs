using Microsoft.ML.Data;

public class CardioData
{
    [LoadColumn(0)] public float age { get; set; }
    [LoadColumn(1)] public float gender { get; set; }
    [LoadColumn(2)] public float height { get; set; }
    [LoadColumn(3)] public float weight { get; set; }
    [LoadColumn(4)] public float ap_hi { get; set; }
    [LoadColumn(5)] public float ap_lo { get; set; }
    [LoadColumn(6)] public float cholesterol { get; set; }
    [LoadColumn(7)] public float gluc { get; set; }
    [LoadColumn(8)] public float smoke { get; set; }
    [LoadColumn(9)] public float alco { get; set; }
    [LoadColumn(10)] public float active { get; set; }
    [LoadColumn(11)] public bool Label { get; set; } // cardio
}

public class CardioPrediction
{
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
