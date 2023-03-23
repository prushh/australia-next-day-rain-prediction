package it.unibo.andrp

import algorithm.tree.{DecisionTreeOld, GradientBoosting, RandomForest}
import config.SparkProjectConfig
import model.DataPoint
import dataset.getDataPoints
import SparkAlgorithm.{DecisionTreeS, RandomForestS}

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

@main def Main(classifier: String, master: String, dataset: String, parallelism: Int, limit: Boolean, simulation: Boolean): Unit = {
    /*
     * Checking arguments.
     */
    val datasetPath = s"data/$dataset"
    val numRows = if (limit) 1000 else 60000 // cardinality: 63754

    if (!simulation) {
        println("Configuration:")
        println(s"- algorithm: $classifier")
        println(s"- master: $master")
        println(s"- dataset path: $datasetPath")
        println(s"- initialized Spark Context with parallelism: $parallelism")
    }

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
    val testDataArray = testData.collect()

    if (simulation) {
        val knnClassifier = algorithm.knn.classify(sparkContext) _
        time("[knn]", knnClassifier(trainingData, testDataArray))

        val decisionTree = DecisionTreeOld(par = false)
        time("[decision-tree]", decisionTree.train(trainingData))
        println(s"[decision-tree] - accuracy: ${decisionTree.accuracy(testDataArray.toList)}")

        val decisionTreeSpark = DecisionTreeS()
        time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
        println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")

        val randomForest = RandomForest(par = false)
        time("[random-forest]", randomForest.train(trainingData))
        println(s"[random-forest] - accuracy: ${randomForest.accuracy(testDataArray.toList)}")

        // val randomForestSpark = RandomForestS()
        //        time("[random-forest] - spark", randomForestSpark.fit(trainingData))
        //        println(s"[random-forest] - spark - accuracy: ${randomForestSpark.accuracy(testData)}")
    } else {
        classifier match {
            case "knn" =>
                val knnClassifier = algorithm.knn.classify(sparkContext) _
                time("[kNN]", knnClassifier(trainingData, testDataArray))
            case "decision-tree" =>
                val decisionTree = DecisionTreeOld(par = false)
                time("[decision-tree]", decisionTree.train(trainingData))
                println(s"[decision-tree] - accuracy: ${decisionTree.accuracy(testDataArray.toList)}")

                val decisionTreeSpark = DecisionTreeS()
                time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
                println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")
            case "random-forest" =>
                val randomForest = RandomForest(par = false)
                time("[random-forest]", randomForest.train(trainingData))
                println(s"[random-forest] - accuracy: ${randomForest.accuracy(testDataArray.toList)}")

                val randomForestSpark = RandomForestS()
                time("[random-forest] - spark", randomForestSpark.fit(trainingData))
                println(s"[random-forest] - spark - accuracy: ${randomForestSpark.accuracy(testData)}")
        }
    }
}