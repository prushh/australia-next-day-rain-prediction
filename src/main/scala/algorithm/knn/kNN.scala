package it.unibo.andrp
package algorithm.knn


import model.DataPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.math.Ordering.Implicits._

object KNN {


  def knnMapReduce(data: RDD[DataPoint], k: Int, query: DataPoint): Double = {
    // Map phase: calculate distances between the query point and all data points
    val distances = data.map { x =>
      (x, euclideanDistance(query, x))
    }

    // Reduce phase: collect the k nearest neighbors
    val nearestNeighbors = distances
      .takeOrdered(k)(Ordering.by { case (_, distance) => distance })
      .groupBy(_._1)
      .mapValues(_.size)

    // restituisco la classe con la maggioranza dei voti
    nearestNeighbors.maxBy { case (_, count) => count }._1.label
  }


  def knn(data: RDD[DataPoint], testPoint: Seq[Double], k: Int): Double = {
    val distances = data.map { point =>
      val dist = euclideanDistance(point, point)
      (point.label, dist)
    }

    val kNearestNeighbors = distances.takeOrdered(k)(Ordering[Double].on(x => x._2))

    val labels = kNearestNeighbors.map(_._1)

    majorityVote(labels)
  }

  def euclideanDistance(features1: DataPoint, features2: DataPoint): Double = {
    math.sqrt(features1.features.zip(features2.features).map { case (x, y) => math.pow(x - y, 2) }.sum)
  }

  def majorityVote(labels: Array[Double]): Double = {
    val labelCounts = labels.groupBy(identity).mapValues(_.length)
    labelCounts.maxBy(_._2)._1
  }

  def accuracy(data: RDD[DataPoint], testData: Seq[DataPoint], k: Int): Double = {
    val predictions = testData.map(testPoint => knnMapReduce(data, k, testPoint))
    val actualLabels = testData.map(_.label)
    val correctPredictions = predictions.zip(actualLabels).count {
      case (prediction, actualLabel) => {
        prediction == actualLabel
      }
    }
    val totalPredictions = testData.length
    correctPredictions.toDouble / totalPredictions.toDouble
  }
}

object Main {
  def main(args: Array[String]): Unit = {
    import org.apache.spark.{SparkConf, SparkContext}

    val conf = new SparkConf().setAppName("myApp").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val data = sc.parallelize(Seq(
      DataPoint(Seq(1.0, 2.0), 0.0),
      DataPoint(Seq(2.0, 3.0), 1.0),
      DataPoint(Seq(3.0, 4.0), 1.0),
      DataPoint(Seq(4.0, 5.0), 0.0)
    ))
    val testData = Seq(
      DataPoint(Seq(5.0, 6.0), 0.0),
      DataPoint(Seq(6.0, 7.0), 1.0)
    )

    val k = 3
    val accuracy = KNN.accuracy(data, testData, k)
    println(s"Accuracy: $accuracy")
  }
}


//import model.DataPoint
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//
//import scala.collection.mutable.ArrayBuffer
//import scala.math.sqrt
//
//class kNN(k: Int, trainData: RDD[DataPoint]) extends Serializable {
//
//    /**
//     * Find the k nearest neighbors of a query point.
//     */
//    private def findNeighbors(query: DataPoint): Array[DataPoint] = {
//        // Compute the Euclidean distance between the query point and each training point.
//        val distances = trainData.map(trainPoint => (trainPoint, distance(query.features, trainPoint.features)))
//
//        // Sort the distances in ascending order.
//        val sortedDistances = distances.sortBy(_._2)
//
//        // Take the k nearest neighbors.
//        val neighbors = sortedDistances.take(k).map(_._1)
//
//        neighbors
//    }
//
//    /**
//     * Compute the accuracy of the kNN model on a test dataset.
//     */
//    def accuracy(testData: RDD[DataPoint]): Double = {
//        val numCorrect = testData.map(testPoint => {
//            // Predict the label of the test point based on its nearest neighbors.
//            val predictedLabel = predictLabel(testPoint)
//            println(predictedLabel)
//
//            // Check if the predicted label matches the true label.
//            if (predictedLabel == testPoint.label) 1 else 0
//        }).sum()
//
//        // Compute the accuracy as the percentage of correctly classified points.
//        numCorrect / testData.count()
//    }
//
//    /**
//     * Predict the label of a test point based on its nearest neighbors.
//     */
//    private def predictLabel(testPoint: DataPoint): Double = {
//        val neighbors = findNeighbors(testPoint)
//
//        // Compute the mode of the labels of the nearest neighbors.
//        val labels = neighbors.map(_.label)
//        val labelCounts = labels.groupBy(identity).view.mapValues(_.length)
//        val modeLabel = labelCounts.maxBy(_._2)._1
//
//        modeLabel
//    }
//
//    /**
//     * Compute the Euclidean distance between two points.
//     */
//    private def distance(x: Seq[Double], y: Seq[Double]): Double = {
//        sqrt(x.zip(y).map { case (a, b) => math.pow(a - b, 2) }.sum)
//    }
//}
//
//object kNN {
//    def train(k: Int, data: RDD[DataPoint]): kNN = {
//        // Train the kNN model.
//        val knn = new kNN(k, data)
//        knn
//    }
//}

