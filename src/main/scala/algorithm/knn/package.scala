package it.unibo.andrp
package algorithm

import model.DataPoint
import config.AlgorithmConfig
import algorithm.knn.Distances.DistanceFunc

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

package object knn {
  def classify(sparkContext: SparkContext)(trainData: RDD[DataPoint], testData: Seq[DataPoint], par: Boolean = true): Double = {
    // Choose distance function
    val distance: DistanceFunc = AlgorithmConfig.Knn.DISTANCE_METHOD match {
      case Distances.Euclidean => Distances.euclidean
      case Distances.Manhattan => Distances.manhattan
    }
    var predictions: Seq[Double] = Seq()
    if (par) {
      // Compute predictions
      predictions = testData.map {
        testPoint =>
          _classify(
            trainData,
            testPoint,
            AlgorithmConfig.Knn.NUMBER_NEAREST_NEIGHBORS,
            distance
          )
      }
    } else {
      predictions = testData.map {
        testPoint =>
          _classifySeq(
            trainData.collect(),
            testPoint,
            AlgorithmConfig.Knn.NUMBER_NEAREST_NEIGHBORS,
            distance
          )
      }
    }


    // Get only labels
    val labels = testData map {
      _.label
    }

    // Count correct predictions by comparing them with the labels
    val correctPredictions = (predictions zip labels).count {
      case (prediction, label) => prediction == label
    }

    val totalPredictions = testData.length
    val accuracy = correctPredictions.toDouble / totalPredictions.toDouble

    println(s"[knn] - accuracy: $accuracy")
    accuracy
  }

  private def _classify(trainData: RDD[DataPoint], testPoint: DataPoint, k: Int, distance: DistanceFunc): Double = {
    // Compute distances
    val distances = trainData map { point =>
      val dist = distance(testPoint, point)
      (point.label, dist)
    }

    // Get kNNs based on distances value
    val kNearestNeighbors = (distances takeOrdered k) (Ordering[Double] on (_._2))
    // Get only labels
    val labels = kNearestNeighbors map (_._1)
    // Count occurrences for each label value
    val labelCounts = (labels groupBy identity).mapValues(_.length)
    // Get label with the highest count
    labelCounts.maxBy(_._2)._1
  }

  private def _classifySeq(trainData: Seq[DataPoint], testPoint: DataPoint, k: Int, distance: DistanceFunc): Double = {
    // Compute distances
    val distances = trainData map { point =>
      val dist = distance(testPoint, point)
      (point.label, dist)
    }

    // Get kNNs based on distances value
    val kNearestNeighbors = distances.sortBy(_._2).take(k)
    // Get only labels
    val labels = kNearestNeighbors map (_._1)
    // Count occurrences for each label value
    val labelCounts = (labels groupBy identity).mapValues(_.length)
    // Get label with the highest count
    labelCounts.maxBy(_._2)._1
  }
}
