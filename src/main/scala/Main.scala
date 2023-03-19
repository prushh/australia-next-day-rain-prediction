package it.unibo.andrp

import algorithm.tree.{DecisionTree, DecisionTreeMapReduceIn, GradientBoosting, RandomForest}
import config.SparkProjectConfig
import model.DataPoint
import dataset.getDataPoints

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.rogach.scallop.{ScallopConf, ScallopOption, intConverter, stringConverter}

import scala.util.Random
import scala.util.CommandLineParser


class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
    val master: ScallopOption[String] = opt[String]("master", default = Some("local[*]"))
    val datasetPath: ScallopOption[String] = opt[String]("datasetPath", default = Some("weatherAUS-final.csv"))
    val parallelism: ScallopOption[Int] = opt[Int]("parallelism", default = Some(1))
    verify()
}


given CommandLineParser[Array[String]] = CommandLineParser.identityParser

@main
def run(args: Array[String]): Unit = {
    /*
     * Checking arguments.
     */
    val conf = new Conf(args)
    val master = conf.master.getOrElse("default-value")
    val datasetPath = conf.datasetPath.getOrElse("default-value")
    val parallelism = conf.parallelism.getOrElse("default-value")

    println(s"master: $master")
    println(s"datasetPath: $datasetPath")

//    println("Configuration:")
//    println(s"- master: $master")
//    println(s"- dataset path: $datasetPath")
//    println(s"- execution type: $datasetPath")
//    println(s"- initialized Spark Context with parallelism: $parallelism")

    /*
     * Loading Spark and Hadoop.
     */
    val sparkSession = SparkProjectConfig.sparkSession(master, parallelism)
    val sparkContext = sparkSession.sparkContext

    /*
     * Loading the dataset.
     */
    val csvDataset = sparkSession.read
      .option("header", value = true)
      .csv(datasetPath)
      .limit(1000)

    println("Dataset loaded!")

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








