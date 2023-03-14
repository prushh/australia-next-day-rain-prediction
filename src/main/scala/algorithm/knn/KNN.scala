package it.unibo.andrp
package algorithm.knn


import model.DataPoint
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer
import scala.math.Ordering.Implicits._

object KNN {

  def knn(data: RDD[DataPoint], testPoint: DataPoint, k: Int): Double = {
    val distances = data.map { point =>
      val dist = euclideanDistance(testPoint, point)
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
    val predictions = testData.map(testPoint => knn(data, testPoint, k))
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