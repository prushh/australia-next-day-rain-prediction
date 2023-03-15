package it.unibo.andrp
package algorithm

import model.DataPoint
import config.AlgorithmConfig
import algorithm.knn.Distance.DistanceFunc

import org.apache.spark.{RangePartitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

package object knn {
    def classify(sparkContext: SparkContext)(trainData: RDD[DataPoint], testData: Seq[DataPoint]): Unit = {
        // Choose distance function
        val distance: DistanceFunc = AlgorithmConfig.Knn.DISTANCE_METHOD match {
            case Distance.Euclidean => Distance.euclidean
            case Distance.Manhattan => Distance.manhattan
        }

        // Compute predictions
        val predictions = testData.map {
            testPoint =>
                _classify(
                    trainData,
                    testPoint,
                    AlgorithmConfig.Knn.NUMBER_NEAREST_NEIGHBORS,
                    distance
                )
        }

        // Get only labels
        val labels = testData map {
            _.label
        }

        // Count correct predictions comparing with labels
        val correctPredictions = (predictions zip labels).count {
            case (prediction, label) => prediction == label
        }

        val totalPredictions = testData.length
        val accuracy = correctPredictions.toDouble / totalPredictions.toDouble
        println(s"[kNN] - accuracy: $accuracy")
    }

    private def _classify(trainData: RDD[DataPoint], testPoint: DataPoint, k: Int, distance: DistanceFunc): Double = {
        val distances = trainData map { point =>
            val dist = distance(testPoint, point)
            (point.label, dist)
        }

        val kNearestNeighbors = (distances takeOrdered k)(Ordering[Double] on (_._2))

        val labels = kNearestNeighbors map (_._1)

        val labelCounts = (labels groupBy identity).view.mapValues(_.length)
        labelCounts.maxBy(_._2)._1
    }
}
