package it.unibo.andrp

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import config.SparkProjectConfig
import algorithm.knn.KNN
import model.DataPoint

import algorithm.getDataPoints
import org.apache.spark.rdd.RDD

@main
def run(): Unit = {

    // TODO: get params from array

    val master = "local[*]"
    val parallelism = 1


    val sparkSession = SparkProjectConfig.sparkSession(master, parallelism)
    val sparkContext = sparkSession.sparkContext

    val datasetPath = "data/weatherAUS-final.csv"
    val csvDataset = sparkSession.read
      .option("header", value = true)
      .csv(datasetPath)
      .limit(1000)

    val rddDataset = getDataPoints(csvDataset)
    val (trainingData, testData) = splitData(rddDataset, 0.8, 42L)

//    testData.collect().map(point => point.label = -1)
//
//    val k = 3
//    val prediction = KNN.knn(trainingData, testData.first(), k)
//    println(s"Prediction: $prediction")
    // algorithm.knn.computeClassification(sparkContext)(csvDataset)


    //    val (df, labels) = ReadCSV("data/weatherAUS-final.csv")
    //
    //    val data = df zip labels
    //    val dfDatapoint = Random.shuffle(data.map(x => DataPoint(x._1.toList, x._2)).toList).take(5000)
    //    val (train_data, test_data) = splitData(dfDatapoint, 0.9)
    //
    //
    //    val clfDT = new DecisionTreeMapReduce(sparkContext)
    //    clfDT.train(train_data)
    //    val accuracyDT = clfDT.accuracy(test_data)
    //    println(s"Accuracy Decision tree: $accuracyDT")

    //    val clfRF = new RandomForestMapReduce1(sparkContext)
    //    clfRF.train(train_data)
    //    val accuracyRF = clfRF.accuracy(test_data)
    //    println(s"Accuracy Random forest: $accuracyRF")
    //
    //    val clfRF2 = new RandomForestMapReduce2(sparkContext)
    //    clfRF2.train(train_data)
    //    val accuracyRF2 = clfRF2.accuracy(test_data)
    //    println(s"Accuracy Random forest: $accuracyRF2")
    //
    //    val clfGB = new GradientBoostingMapReduce(sparkContext)
    //    clfGB.train(train_data)
    //    val accuracyGB = clfGB.accuracy(test_data)
    //    println(s"Accuracy GradientBoosting: $accuracyGB")
}

def splitData(data: RDD[DataPoint], trainingRatio: Double, seed: Long): (RDD[DataPoint], RDD[DataPoint]) = {
    val splits = data.randomSplit(Array(trainingRatio, 1.0 - trainingRatio), seed)
    (splits(0), splits(1))
}








