package it.unibo.andrp
package SparkAlgorithm


import model.DataPoint

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.PipelineModel
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, GBTClassificationModel, GBTClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}

abstract class MLAIalgorithms {

  // Metodo template che implementa la logica comune degli algoritmi
  def run(trainData: RDD[DataPoint], testData: RDD[DataPoint]): Unit = {

    // Chiamata del metodo specifico dell'algoritmo
    this.fit(trainData)

    val accuracy = this.accuracy(testData)
    println(s"Accuracy: $accuracy")

  }

  // Metodo astratto per l'addestramento del modello
  def fit(trainData: RDD[DataPoint]): Unit

  // Metodo astratto per la valutazione del modello
  def predict(testData: RDD[DataPoint]): Seq[Double]

  def accuracy(dataset: RDD[DataPoint]): Double = {
    val predictions = this.predict(dataset)
    val correct = predictions.zip(dataset.collect()).count { case (pred, dp) => pred == dp.label }
    correct.toDouble / dataset.collect().length.toDouble
  }

}
