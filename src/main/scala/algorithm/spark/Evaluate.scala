package it.unibo.andrp
package algorithm.spark

import model.DataPoint

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassificationModel, GBTClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

abstract class Evaluate {
    def run(trainData: RDD[DataPoint], testData: RDD[DataPoint]): Double = {
        this.fit(trainData)
        this.accuracy(testData)
    }

    def fit(trainData: RDD[DataPoint]): Unit

    def predict(testData: RDD[DataPoint]): Seq[Double]

    def accuracy(testData: RDD[DataPoint]): Double = {
        val testDataArray = testData.collect()

        val predictions = this.predict(testData)
        val correct = (predictions zip testDataArray).count {
            case (pred, point) => pred == point.label
        }
        correct.toDouble / testDataArray.length.toDouble
    }

}
