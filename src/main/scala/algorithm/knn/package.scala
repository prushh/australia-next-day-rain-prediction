package it.unibo.andrp
package algorithm

import model.DataPoint

import org.apache.spark.{RangePartitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.immutable.ListMap

package object knn {
    def splitData(data: RDD[DataPoint], trainingRatio: Double, seed: Long): (RDD[DataPoint], RDD[DataPoint]) = {
        val splits = data.randomSplit(Array(trainingRatio, 1.0 - trainingRatio), seed)
        (splits(0), splits(1))
    }

    def computeClassification(sparkContext: SparkContext)(csvDataset: DataFrame): Unit = {
        val rddDataset = getDataPoints(csvDataset)
        val (trainingData, testData) = splitData(rddDataset, 0.8, 42L)
//        val knn = kNN.train(3, trainingData)
//        val accuracy = knn.accuracy(testData)
//        println(s"Accuracy [kNN]: $accuracy")
    }
}
