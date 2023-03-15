package it.unibo.andrp

import algorithm.tree.{DecisionTree, DecisionTreeMapReduceIn, GradientBoosting, RandomForest}
import config.SparkProjectConfig
import model.DataPoint
import dataset.getDataPoints

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

@main
def run(): Unit = {

    // TODO: get params from array

    val master = "local[*]"
    val parallelism = 2

    val sparkSession = SparkProjectConfig.sparkSession(master, parallelism)
    val sparkContext = sparkSession.sparkContext

    val datasetPath = "data/weatherAUS-final.csv"
    val csvDataset = sparkSession.read
      .option("header", value = true)
      .csv(datasetPath)
      .limit(1000)

    val (trainingData, testData) = getDataPoints(csvDataset)

    val knnClassifier = algorithm.knn.classify(sparkContext) _
    time("[kNN]", knnClassifier(trainingData, testData))

    /* DECISION TREE MAP REDUCE*/
    //    val decisionTree = new DecisionTreeMapReduceIn(par=true)
    //    decisionTree.train(trainingData, None)
    //    val accuracyDTMP = decisionTree.accuracy(testData.collect().toList)
    //    println(s"Accuracy Decision tree: $accuracyDTMP")


    //    val rndf = new RandomForest(numTrees = 5,maxDepth = 2, par = true)
    //    rndf.train(trainingData)
    //    val accuracyForest = rndf.accuracy(testData.collect().toList)
    //    println(s"Accuracy Decision tree: $accuracyForest")
    //
    //    val gb = new GradientBoosting(par = true)
    //    gb.train(trainingData)
    //    val accuracyGb = gb.accuracy(testData.collect().toList)
    //    println(s"Accuracy Decision tree: $accuracyGb")

}








