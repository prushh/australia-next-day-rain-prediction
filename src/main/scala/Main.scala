package it.unibo.andrp

import algorithm.tree.{DecisionTree, DecisionTreeMapReduceIn, GradientBoosting, RandomForest}
import config.SparkProjectConfig
import model.DataPoint
import dataset.getDataPoints

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

@main
def run(classifier: String, master: String, dataset: String, parallelism: Int, limit: Boolean): Unit = {
    /*
     * Checking arguments.
     */
    val datasetPath = s"data/$dataset"
    val numRows = if (limit) 1000 else 60000 // cardinality: 63754

    println("Configuration:")
    println(s"- algorithm: $classifier")
    println(s"- master: $master")
    println(s"- dataset path: $datasetPath")
    println(s"- initialized Spark Context with parallelism: $parallelism")

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
      .limit(numRows)

    println("Dataset loaded!")

    val (trainingData, testData) = getDataPoints(csvDataset)

    classifier match {
        case "knn" =>
            val knnClassifier = algorithm.knn.classify(sparkContext) _
            time("[kNN]", knnClassifier(trainingData, testData))
        // case "decision-tree" =>
        // case "gradient-boosting" =>
        // case "random-forest" =>
    }

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








