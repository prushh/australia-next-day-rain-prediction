package it.unibo.andrp

import config.SparkProjectConfig
import utils.TimestampFormatter
import model.{DataPoint, Dataset}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Encoders, Row}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

@main def run(): Unit =
    val sparkSession = SparkProjectConfig.sparkSession("local", 1)
    val sparkContext = sparkSession.sparkContext

    val datasetPath = "data/weatherAUS-final.csv"
    val datasetCSV = sparkSession.read
      .option("header", value = true)
      .option("timestampFormat", TimestampFormatter.timestampPattern)
      .csv(datasetPath)

    println("Dataset loaded!")

    val dataPoints = datasetCSV.rdd.map(row => {
        val features = row.toSeq.dropRight(1)
        val label = row.toSeq.last
        DataPoint(features, label)
    })

//    val (df, target) = readCSVIris("data/iris.csv")
//    //val (df, target) = ReadCSV("data/weatherAUS-final.csv")
//
//    val data = df zip target
//    val shuffledData = Random.shuffle(data).toArray
//    val (trainData, testData) = shuffledData.splitAt((data.length * 0.8).toInt)
//
//    val tmpTrainData = trainData.map((x, y) => DataPoint(x, y))
//    val tmpTestData = testData.map((x, y) => DataPoint(x, y))
//
//    val clf = LastSVM(LastSVM.linearKernel)
//    clf.train(tmpTrainData, 1.0, 1e-4)
//    println(clf.accuracy(tmpTestData))








