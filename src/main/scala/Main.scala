package it.unibo.andrp

import algorithm.tree.{DecisionTreeOld, RandomForest}
import config.SparkProjectConfig
import dataset.getDataPoints
import SparkAlgorithm.{DecisionTreeS, RandomForestS}

object Main extends App {
    /*
     * Checking arguments.
     */
    private val master = args(0)
    private val datasetPath = args(1)
    private val classifier = args(2)
    private val parallelism = args(3).toInt
    private val limit = args(4).toBoolean
    private val simulation = args(5).toBoolean

    private val numRows = if (limit) 1000 else 60000 // cardinality: 63754

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
    private val sparkSession = SparkProjectConfig.sparkSession(master, parallelism)
    private val sparkContext = sparkSession.sparkContext

    /*
     * Loading the dataset.
     */
    private val csvDataset = sparkSession.read
      .option("header", value = true)
      .csv(datasetPath)
      .limit(numRows)

    println("Dataset loaded!")

    private val (trainingData, testData) = getDataPoints(csvDataset)
    private val testDataArray = testData.collect()

    if (simulation) {
        val knnClassifier = algorithm.knn.classify(sparkContext) _
        time("[knn]", knnClassifier(trainingData, testDataArray, true))

        val knnClassifierseq = algorithm.knn.classify(sparkContext) _
        time("[knn seq]", knnClassifierseq(trainingData, testDataArray, false))

        val decisionTree = new DecisionTreeOld(par = true)
        time("[decision-tree]", decisionTree.train(trainingData))
        println(s"[decision-tree] - accuracy: ${decisionTree.accuracy(testDataArray.toList)}")

        val decisionTreeSeq = new DecisionTreeOld(par = false)
        time("[decision-tree]", decisionTreeSeq.train(trainingData))
        println(s"[decision-tree seq] - accuracy: ${decisionTreeSeq.accuracy(testDataArray.toList)}")

        val decisionTreeSpark = new DecisionTreeS()
        time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
        println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")

        val randomForest = new RandomForest(par = true)
        time("[random-forest]", randomForest.train(trainingData))
        println(s"[random-forest] - accuracy: ${randomForest.accuracy(testDataArray.toList)}")

        val randomForestSeq = new RandomForest(par = false)
        time("[random-forest Seq]", randomForestSeq.train(trainingData))
        println(s"[random-forest Seq] - accuracy: ${randomForestSeq.accuracy(testDataArray.toList)}")

        val randomForestSpark = new RandomForestS()
        time("[random-forest] - spark", randomForestSpark.fit(trainingData))
        println(s"[random-forest] - spark - accuracy: ${randomForestSpark.accuracy(testData)}")
    } else {
        classifier match {
            case "knn" =>
                val knnClassifier = algorithm.knn.classify(sparkContext) _
                time("[kNN]", knnClassifier(trainingData, testDataArray, true))
            case "decision-tree" =>
                val decisionTree = new DecisionTreeOld(par = false)
                time("[decision-tree]", decisionTree.train(trainingData))
                println(s"[decision-tree] - accuracy: ${decisionTree.accuracy(testDataArray.toList)}")

                val decisionTreeSpark = new DecisionTreeS()
                time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
                println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")
            case "random-forest" =>
                val randomForest = new RandomForest(par = false)
                time("[random-forest]", randomForest.train(trainingData))
                println(s"[random-forest] - accuracy: ${randomForest.accuracy(testDataArray.toList)}")

                val randomForestSpark = new RandomForestS()
                time("[random-forest] - spark", randomForestSpark.fit(trainingData))
                println(s"[random-forest] - spark - accuracy: ${randomForestSpark.accuracy(testData)}")
        }
    }
}