package it.unibo.andrp

import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import scala.util.Random


package object global {
  type DataFrame[A] = Vector[Vector[A]]
}
import breeze.plot._
import global._
import utils._

object Main {
  def main(args: Array[String]): Unit = {
    // load data
    //val (df, labels) = readCSVIris("data/iris.csv")
    val (df, labels) = ReadCSV("data/weatherAUS-final.csv")

    val data = df zip labels
    val dfDatapoint = Random.shuffle((data.map(x => DataPoint(x._1.toList,x._2))).toList).take(5000)
    val (train_data,test_data) = splitData(dfDatapoint,0.9)
    // Load data from file
    //val data = loadIrisData("iris.csv")

    // Split data into training and testing sets
    //val (trainData, testData) = splitData(data, 0.7)
    val numFeatures=32
    // Create a new random forest with 10 trees, maximum depth of 5, and 2 features per tree
    val forest = new RandomForest(numTrees = 3, maxDepth = 1, numFeatures = numFeatures)
    forest.train(train_data)

    val tree = new DecisionTree(maxDepth = 4, numFeatures = numFeatures)
    val subFeatures = List.range(0, numFeatures)
    // Train the random forest on the training data
    tree.train(train_data,subFeatures)
    tree.printDecisionTree()
    val acc = tree.accuracy(test_data)
    println(s"Accuracy: $acc")

     //Make predictions on the testing data
    val predictions = test_data.map(forest.predict)

     //Evaluate the accuracy of the predictions
    val accuracy = calculateAccuracy(predictions, test_data.map(_.label))
    println(s"Accuracy forest: $accuracy")
  }



  // Utility function to split data into training and testing sets
  def splitData(data: List[DataPoint], trainFraction: Double): (List[DataPoint], List[DataPoint]) = {
    val trainSize = (trainFraction * data.size).toInt
    val trainData = data.take(trainSize)
    val testData = data.drop(trainSize)
    (trainData, testData)
  }

  // Utility function to calculate accuracy of predictions
  def calculateAccuracy(predictions: List[Int], labels: List[Int]): Double = {
    val correctCount = predictions.zip(labels).count { case (predicted, actual) => predicted == actual }
    correctCount.toDouble / labels.size
  }







  /*val sc = new SparkContext("local[*]", "Random Forest")

  // Parametri dell'algoritmo
  val numTrees = 5
  val maxDepth = 10
  val minSplitSize = 2

  // Caricamento del dataset e creazione di RDD
  val data = sc.textFile("data/weatherAUS-final.csv").map(line => {
      val split = line.split(",")
      val features = split.dropRight(1).map(_.toDouble).toVector
      val label = split.last.toInt
      DataPoint(features, label)
  }).cache()

  // Creazione degli alberi
  val trees = (1 to numTrees).map { _ =>
      val sample = data.sample(true, 1.0)
      val tree = new DecisionTree()
      tree.buildTree(sample.collect(), maxDepth, minSplitSize)
      tree
  }

  // Funzione di predizione
  def predict(point: DataPoint): Int = {
      val predictions = trees.map(tree => tree.predict(point, tree.root))
      predictions.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
  }

  // Esempio di utilizzo della funzione di predizione
  val testPoint = DataPoint(Vector(5.1, 3.5, 1.4, 0.2), -1)
  val prediction = predict(testPoint)
  println(s"Prediction for test point ${testPoint.features}: $prediction")
*/
}









