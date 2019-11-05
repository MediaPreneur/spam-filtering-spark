import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object spam {

  def main(args: Array[String]): Unit = {

    // start spark session
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Classification")
      .getOrCreate()

    // load data as spark-datasets
    val spam_training = spark.read.textFile("src/main/resources/spam_training.txt")
    val spam_testing = spark.read.textFile("src/main/resources/spam_testing.txt")
    val nospam_training = spark.read.text("src/main/resources/nospam_training.txt")
    val nospam_testing = spark.read.text("src/main/resources/nospam_testing.txt")

    // Add the label column which will be used to calculate the accuracy of the algorithm in the end.
    val spamTrainingLabeled = spam_training
      .withColumn("label", lit(1.0))
    val nospamTrainingLabeled = nospam_training
      .withColumn("label", lit(0.0))
    val spamTestLabeled = spam_testing
      .withColumn("label", lit(1.0))
    val nospamTestingLabeled = nospam_testing
      .withColumn("label", lit(0.0))

    // Union the datasets of spam and no-spam because we need both to train and later test the model.
    val fullTraining = spamTrainingLabeled.union(nospamTrainingLabeled)
    val fullTesting = spamTestLabeled.union(nospamTestingLabeled)

    // Split the lines into words.
    val tokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("words").setPattern("\\w+").setGaps(false)

    // Remove the english stopwords because are not useful to train the model.
    val stopwordsRemover = new StopWordsRemover().setInputCol("words").setOutputCol("stopwordsRemoved")

    // Create N-grams to have a more accurate solution.
    val ngram = new NGram().setN(1).setInputCol("stopwordsRemoved").setOutputCol("ngrams")

    // Hashing the words (from the output column of tokenizer) based on TF (term-frequency)
    val hashingTF = new HashingTF().setInputCol("ngrams").setOutputCol("rawFeatures").setNumFeatures(50)

    // Inverse document frequency (IDF) measures the importance of the term for the SMS text.
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")

    // Naive Bayes is the algorithm used to train the model.
    val naiveBayes = new NaiveBayes()

    // Pipeline which processes all the stages of our software.
    val pipeline = new Pipeline().setStages(Array(tokenizer, stopwordsRemover, ngram, hashingTF, idf, naiveBayes))

    // Fit the pipeline to train the model.
    val model = pipeline.fit(fullTraining)

    // Make predictions on the testing dataset.
    val predictions = model.transform(fullTesting)

    // The confusion matrix.
    predictions.groupBy("label", "prediction").count().show()

    // Select (prediction, label) and compute accuracy.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)

    println("Accuracy = " + (accuracy))

  }
}
