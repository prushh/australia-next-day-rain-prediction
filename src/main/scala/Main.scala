package it.unibo.andrp

import algorithm.getDataPoints
import algorithm.knn.KNN
import algorithm.tree.{DecisionTree, GradientBoosting, RandomForest}
import config.SparkProjectConfig
import model.DataPoint

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

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


    /* KNN MAP REDUCE */
    val t0KNNMP = System.nanoTime
    val k = 3
    val accuracyKNNMP = KNN.accuracy(trainingData, testData.collect().toSeq, k)
    val durationKNNMP = (System.nanoTime - t0KNNMP) / 1e9d // 10^9 in order to be a double
    println(s"KNN Map Reduce:\n\t- Accuracy: $accuracyKNNMP,\n\t- Duration: $durationKNNMP")

    /* DECISION TREE MAP REDUCE*/
    val decisionTree = new DecisionTree(par = true)
    decisionTree.train(trainingData,None)
    val accuracyDTMP = decisionTree.accuracy(testData.collect().toList)
    println(s"Accuracy Decision tree: $accuracyDTMP")


    val rndf = new RandomForest(numTrees = 5,maxDepth = 2, par = true)
    rndf.train(trainingData)
    val accuracyForest = rndf.accuracy(testData.collect().toList)
    println(s"Accuracy Decision tree: $accuracyForest")

    val gb = new GradientBoosting(par = true)
    gb.train(trainingData)
    val accuracyGb = gb.accuracy(testData.collect().toList)
    println(s"Accuracy Decision tree: $accuracyGb")

}


def splitData(data: RDD[DataPoint], trainingRatio: Double, seed: Long): (RDD[DataPoint], RDD[DataPoint]) = {
    val splits = data.randomSplit(Array(trainingRatio, 1.0 - trainingRatio), seed)
    (splits(0), splits(1))

}








