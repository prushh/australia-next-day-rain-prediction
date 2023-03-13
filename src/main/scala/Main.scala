package it.unibo.andrp

import it.unibo.andrp.mapReduce.{DecisionTreeMapReduce, GradientBoostingMapReduce, RandomForestMapReduce1, RandomForestMapReduce2}
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext

import scala.util.Random


package object global {
  type DataFrame[A] = Vector[Vector[A]]
}
import global._
import utils._

object Main extends App {
  // load data
  //val (df, labels) = readCSVIris("data/iris.csv")
  val (df, labels) = ReadCSV("data/weatherAUS-final.csv")

  val data = df zip labels
  val dfDatapoint = Random.shuffle((data.map(x => DataPoint(x._1.toList, x._2))).toList).take(1000)
  val (train_data, test_data) = splitData(dfDatapoint, 0.9)


  //MAP REDUCE
  val sc = new SparkContext("local[*]", "Random Forest")
  /*val decisionTreeMapReduceIn = new DecisionTreeMapReduceIn()
  decisionTreeMapReduceIn.train(sc.parallelize(train_data),None)
  val accuracyDTMP = decisionTreeMapReduceIn.accuracy(test_data)
  println(s"Accuracy Decision tree: $accuracyDTMP")*/
  val rndf = new RandomForest(numTrees = 5,maxDepth = 2)
  rndf.train(sc.parallelize(train_data))
  val accuracyForest = rndf.accuracy(test_data)
  println(s"Accuracy Decision tree: $accuracyForest")
/*

  var clfDT = new DecisionTreeMapReduce(sc)
  clfDT.train(train_data)
  val accuracyDT = clfDT.accuracy(test_data)
  println(s"Accuracy Decision tree: $accuracyDT")

  val clfRF = new RandomForestMapReduce1(sc)
  clfRF.train(train_data)
  val accuracyRF = clfRF.accuracy(test_data)
  println(s"Accuracy Random forest: $accuracyRF")

  val clfRF2 = new RandomForestMapReduce2(sc)
  clfRF2.train(train_data)
  val accuracyRF2 = clfRF2.accuracy(test_data)
  println(s"Accuracy Random forest: $accuracyRF2")

  var clfGB = new GradientBoostingMapReduce(sc)
  clfGB.train(train_data)
  val accuracyGB = clfGB.accuracy(test_data)
  println(s"Accuracy GradientBoosting: $accuracyGB")

*/





  /*// Load data from file
    //val data = loadIrisData("iris.csv")

    // Split data into training and testing sets
    //val (trainData, testData) = splitData(data, 0.7)
    val numFeatures=4
    // Create a new random forest with 10 trees, maximum depth of 5, and 2 features per tree
    val forest = new RandomForest(numTrees = 3, maxDepth = 1)
    forest.train(train_data)

    val tree = new DecisionTreeGB(maxDepth = 4)
    val subFeatures = List.range(0, numFeatures)
    // Train the random forest on the training data
    tree.train(train_data,None)
    tree.printDecisionTree()
    val acc = tree.accuracy(test_data)
    println(s"Accuracy: $acc")

     //Make predictions on the testing data
    val predictions = test_data.map(forest.predict)

     //Evaluate the accuracy of the predictions
    val accuracy = calculateAccuracy(predictions, test_data.map(_.label))
    println(s"Accuracy forest: $accuracy")

    // Parametri dell'algoritmo
    val numTrees = 5
    val maxDepth = 3
    val minSplitSize = 10


    //XGoost
    var xgb = GradientBoostingClassifier(
      numIterations=15,
      learningRate=0.2,
      maxDepth=maxDepth,
      minSplitSize=minSplitSize,
      featureSubsetStrategy="sqrt",
      impurityFunc="gini")
    xgb.train(train_data)

    val acc_xgb = xgb.accuracyWithTolerance(test_data,0.5)
    println(s"Accuracy: $acc_xgb")*/


  // Utility function to split data into training and testing sets
  def splitData(data: List[DataPoint], trainFraction: Double): (List[DataPoint], List[DataPoint]) = {
    val trainSize = (trainFraction * data.size).toInt
    val trainData = data.take(trainSize)
    val testData = data.drop(trainSize)
    (trainData, testData)
  }
}









