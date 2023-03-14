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
    val k = 3
    val accuracy = KNN.accuracy(trainingData, testData.collect().toSeq, k)
    println(s"Accuracy: $accuracy")
    
}

def splitData(data: RDD[DataPoint], trainingRatio: Double, seed: Long): (RDD[DataPoint], RDD[DataPoint]) = {
    val splits = data.randomSplit(Array(trainingRatio, 1.0 - trainingRatio), seed)
    (splits(0), splits(1))
}








