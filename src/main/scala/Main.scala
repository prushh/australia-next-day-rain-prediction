package it.unibo.andrp

import algorithm.tree.{DecisionTreeOld, RandomForest}
import config.SparkProjectConfig
import dataset.getDataPoints
import SparkAlgorithm.{DecisionTreeS, RandomForestS}
import Utils.OutputWriter

object Main extends App {
  /*
   * Checking arguments.
   */
  private val master = args(0)
  private val datasetPath = args(1)
  private val outputPath = args(2)
  private val classifier = args(3)
  private val parallelism = args(4).toInt
  private val limit = args(5).toBoolean
  private val simulation = args(6).toBoolean
  private val numRun = args(7).toInt

  private val numRows = if (limit) 10000 else 15000 // cardinality: 63754

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
   * Creating outputWriter object
   */
  val ow = new OutputWriter(sparkSession, outputPath, parallelism)


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
    for (i <- 1 to numRun) {
      println(
        s"Run number: $i " +
          s"\n ===============================================================\n"
      )
      val knnClassifier = algorithm.knn.classify(sparkContext) _
      val (accKnn, timeKnn) = time("[knn]", knnClassifier(trainingData, testDataArray, true))
      ow.addRow("knn", accKnn, timeKnn, i)

      /*val knnClassifierseq = algorithm.knn.classify(sparkContext) _
      val (accKnnSeq, timeKnnSeq) = time("[knn seq]", knnClassifierseq(trainingData, testDataArray, false))
      ow.addRow("knnSeq", accKnnSeq, timeKnnSeq, i)*/

      val decisionTree = new DecisionTreeOld(par = true)
      val (_, timeDT) = time("[decision-tree]", decisionTree.train(trainingData))
      val accuracyDT = decisionTree.accuracy(testDataArray.toList)
      println(s"[decision-tree] - accuracy: ${accuracyDT}")
      ow.addRow("DecisionTree", accuracyDT, timeDT, i)

      /*val decisionTreeSeq = new DecisionTreeOld(par = false)
      val (_, timeDTSeq) = time("[decision-tree]", decisionTreeSeq.train(trainingData))
      val accuracyDTSeq = decisionTreeSeq.accuracy(testDataArray.toList)
      println(s"[decision-tree seq] - accuracy: ${accuracyDTSeq}")
      ow.addRow("DecisionTreeSeq", accuracyDTSeq, timeDTSeq, i)*/

      val decisionTreeSpark = new DecisionTreeS()
      val (_, timeDTSpark) = time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
      val accuracyDTSpark = decisionTreeSpark.accuracy(testData)
      println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")
      ow.addRow("DecisionTreeSpark", accuracyDTSpark, timeDTSpark, i)

      val randomForest = new RandomForest(par = true)
      val (_, timeRF) = time("[random-forest]", randomForest.train(trainingData))
      val accuracyRF = randomForest.accuracy(testDataArray.toList)
      println(s"[random-forest] - accuracy: ${accuracyRF}")
      ow.addRow("RandomForest", accuracyRF, timeRF, i)

      /*val randomForestSeq = new RandomForest(par = false)
      val (_, timeRFSeq) = time("[random-forest Seq]", randomForestSeq.train(trainingData))
      val accuracyRFSeq = randomForestSeq.accuracy(testDataArray.toList)
      println(s"[random-forest Seq] - accuracy: ${accuracyRFSeq}")
      ow.addRow("RandomForestSeq", accuracyRFSeq, timeRFSeq, i)*/

      val randomForestSpark = new RandomForestS()
      val (_, timeRFSpark) = time("[random-forest] - spark", randomForestSpark.fit(trainingData))
      val accuracyRFSpark = randomForestSpark.accuracy(testData)
      println(s"[random-forest] - spark - accuracy: ${accuracyRFSpark}")
      ow.addRow("RandomForestSpark", accuracyRFSpark, timeRFSpark, i)

    }
    ow.saveToFile()
    sparkSession.close()
  } else {
    classifier match {
      case "knn" =>
        val knnClassifier = algorithm.knn.classify(sparkContext) _
        time("[kNN]", knnClassifier(trainingData, testDataArray, true))

        val knnClassifierSeq = algorithm.knn.classify(sparkContext) _
        time("[kNN Seq]", knnClassifierSeq(trainingData, testDataArray, false))




      case "decision-tree" =>
        val decisionTree = new DecisionTreeOld(par = true)
        time("[decision-tree]", decisionTree.train(trainingData))
        println(s"[decision-tree] - accuracy: ${decisionTree.accuracy(testDataArray.toList)}")

        val decisionTreeSeq = new DecisionTreeOld(par = false)
        time("[decision-tree seq]", decisionTreeSeq.train(trainingData))
        println(s"[decision-tree seq] - accuracy: ${decisionTreeSeq.accuracy(testDataArray.toList)}")

      /*val decisionTreeSpark = new DecisionTreeS()
      time("[decision-tree] - spark", decisionTreeSpark.fit(trainingData))
      println(s"[decision-tree] - spark - accuracy: ${decisionTreeSpark.accuracy(testData)}")*/
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