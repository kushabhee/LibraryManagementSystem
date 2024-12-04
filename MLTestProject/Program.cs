// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using MLTestProject;

string longString = "1234567890 long string that need to be truncated";
int maxLength = 8;
string linqString = new string(longString.Take(maxLength).ToArray());
Console.WriteLine(linqString);

Console.WriteLine("Hello, World!");
// Step 2: Create an ML context
var mlContext = new MLContext();

// Step 3: Load training data
var trainingData = mlContext.Data.LoadFromEnumerable(new[]
{
                new CustomerData { Age = 25, Income = 50000, Label = false },
                new CustomerData { Age = 45, Income = 100000, Label = true },
                new CustomerData { Age = 35, Income = 60000, Label = false },
                new CustomerData { Age = 50, Income = 120000, Label = true },
                new CustomerData { Age = 23, Income = 40000, Label = false },
            });

// Step 4: Define a data preparation and training pipeline
var pipeline = mlContext.Transforms.Concatenate("Features", "Age", "Income")
    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());

// Step 5: Train the model
var model = pipeline.Fit(trainingData);

// Step 6: Evaluate the model (optional, requires separate test data)
var predictions = model.Transform(trainingData);
var metrics = mlContext.BinaryClassification.Evaluate(predictions);
Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");

// Step 7: Use the model for predictions
var predictionEngine = mlContext.Model.CreatePredictionEngine<CustomerData, CustomerPrediction>(model);
var newCustomer = new CustomerData { Age = 40, Income = 95000 };
var prediction = predictionEngine.Predict(newCustomer);

Console.WriteLine($"Prediction: {(prediction.PurchasePrediction ? "Will Purchase" : "Will Not Purchase")}");
Console.ReadLine();