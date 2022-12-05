package it.unibo.andrp

import config.SparkProjectConfig
import utils.{ReadCSV, TimestampFormatter}

import org.apache.spark.sql.types.{DataType, StructType}
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

import scala.io.Source
import scala.util.Using

@main def run(): Unit =
    /*
     * Loading Spark and Hadoop.
     */
    val sparkSession = SparkProjectConfig.sparkSession("local[*]", 1)
    val sparkContext = sparkSession.sparkContext

    val data = MLUtils.loadLibSVMFile(sparkContext, "data/weatherAUSlibsvm.txt")

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Clear the default threshold.
    model.clearThreshold()

    // Compute raw scores on the test set.
    val scoreAndLabels = test.map { point =>
        val score = model.predict(point.features)
        (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()
    println(s"Area under ROC = $auROC")

    sparkSession.stop()

    // initialize global support vector

    // load data
    val data_csv = ReadCSV("data/weatherAUSfinal.csv")

    // divide data into chunks //TODO decidere in che modo dividere i dati

    // while hyp t =! hyp t-1

        // for on chunks/computer

            // split training and testing

            // add global support vectors to training set

            // train svm

            // get new support vectors

        // for on chunks/computer

            // add new support vectors to global support vectors