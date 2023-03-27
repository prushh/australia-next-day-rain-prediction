package it.unibo.andrp

import config.SparkProjectConfig
import dataset.getDataPoints
import algorithm.spark.{DecisionTree, RandomForest}
import algorithm.Executions
import util.OutputWriter

object Main extends App {
    /*
     * Checking arguments.
     */
    private val master = args(0)
    private val datasetPath = args(1)
    private val simulation = if (args(2) == "sim=true") true else false
    private val limit = if (args(3) == "lim=true") 1000 else 10000

    private var classifier = ""
    private var execution = Executions.Sequential
    private var outputPath = ""
    private var numRun = 0
    private var label = ""

    if (!simulation) {
        classifier = args(4)
        execution = if (args(5) == "ex=parallel") Executions.Parallel else Executions.Sequential
    } else {
        outputPath = args(4)
        numRun = args(5).toInt
        label = args(6)
    }

    println("Configuration:")
    println(s"- master: $master")
    println(s"- dataset path: $datasetPath")
    if (!simulation) {
        println(s"- classifier: $classifier")
        println(s"- execution: $execution")
    } else {
        println(s"- output path: $outputPath")
        println(s"- num run: $numRun")
        println(s"- label used: $label")
    }

    /*
     * Loading Spark and Hadoop.
     */
    private val sparkSession = SparkProjectConfig.sparkSession(master)
    private val outputWriter = new OutputWriter(sparkSession, outputPath, label)

    /*
     * Loading the dataset.
     */
    private val csvDataset = sparkSession.read
      .option("header", value = true)
      .csv(datasetPath)
      .limit(limit)

    println("Dataset loaded!")

    private val (trainingData, testData) = getDataPoints(csvDataset)
    private val testDataArray = testData.collect()

    if (simulation) {
        for (idx <- 1 to numRun) {
            println(
                s"Run number: $idx " +
                  s"\n ===============================================================\n"
            )
            val knnClassifier = algorithm.knn.classify(trainingData, testDataArray) _
            val (accuracyKnn, timeKnn) = time(s"[knn - $execution]", knnClassifier(execution))
            outputWriter.addRow("knn", accuracyKnn, timeKnn, idx)

            val decisionTree = algorithm.tree.train(trainingData) _
            val (decisionTreeModel, timeDecisionTree) = time(s"[decision-tree - $execution]", decisionTree(execution))
            val evaluationDT = algorithm.tree.accuracy(decisionTreeModel) _
            val accuracyDecisionTree = evaluationDT(testDataArray)
            println(s"[decision-tree - $execution] - accuracy: $accuracyDecisionTree")
            outputWriter.addRow("DecisionTree", accuracyDecisionTree, timeDecisionTree, idx)

            val decisionTreeSpark = new DecisionTree()
            val (_, timeDecisionTreeSpark) = time("[decision-tree - spark]", decisionTreeSpark.fit(trainingData))
            val accuracyDecisionTreeSpark = decisionTreeSpark.accuracy(testData)
            println(s"[decision-tree - spark] - accuracy: ${decisionTreeSpark.accuracy(testData)}")
            outputWriter.addRow("DecisionTreeSpark", accuracyDecisionTreeSpark, timeDecisionTreeSpark, idx)

            val randomForest = algorithm.tree.trainRandomForest(trainingData) _
            val (randomForestModel, timeRandomForest) = time(s"[random-forest - $execution]", randomForest(execution))
            val evaluationRF = algorithm.tree.accuracy(randomForestModel) _
            val accuracyRandomForest = evaluationRF(testDataArray)
            println(s"[random-forest - $execution] - accuracy: $accuracyRandomForest")
            outputWriter.addRow("RandomForest", accuracyRandomForest, timeRandomForest, idx)

            val randomForestSpark = new RandomForest()
            val (_, timeRandomForestSpark) = time("[random-forest] - spark", randomForestSpark.fit(trainingData))
            val accuracyRandomForestSpark = randomForestSpark.accuracy(testData)
            println(s"[random-forest] - spark - accuracy: $accuracyRandomForestSpark")
            outputWriter.addRow("RandomForestSpark", accuracyRandomForestSpark, timeRandomForestSpark, idx)
        }

        outputWriter.saveToFile()
    } else {
        classifier match {
            case "knn" =>
                val knnClassifier = algorithm.knn.classify(trainingData, testDataArray) _
                time(s"[knn - $execution]", knnClassifier(execution))
            case "decision-tree" =>
                val decisionTree = algorithm.tree.train(trainingData) _
                val decisionTreeModel = time(s"[decision-tree - $execution]", decisionTree(execution))._1

                val evaluation = algorithm.tree.accuracy(decisionTreeModel) _
                println(s"[decision-tree - $execution] - accuracy: ${evaluation(testDataArray)}")


                val decisionTreeSpark = new DecisionTree()
                time("[decision-tree - spark]", decisionTreeSpark.fit(trainingData))
                println(s"[decision-tree - spark] - accuracy: ${decisionTreeSpark.accuracy(testData)}")
            case "random-forest" =>
                val randomForest = algorithm.tree.trainRandomForest(trainingData) _
                val randomForestModel = time(s"[random-forest - $execution]", randomForest(execution))._1

                val evaluation = algorithm.tree.accuracy(randomForestModel) _
                println(s"[random-forest - $execution] - accuracy: ${evaluation(testDataArray)}")

                val randomForestSpark = new RandomForest()
                time("[random-forest - spark]", randomForestSpark.fit(trainingData))
                println(s"[random-forest - spark] - accuracy: ${randomForestSpark.accuracy(testData)}")
        }
    }
}