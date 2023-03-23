package it.unibo.andrp
package algorithm.tree

import model.DataPoint

import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer
import scala.math.exp

class GradientBoosting(
                        val numIterations: Int = 10,
                        val learningRate: Double = 0.1,
                        val maxDepth: Int = 4,
                        val minSplitSize: Int = 10,
                        val featureSubsetStrategy: String = "all",
                        val impurityFunc: String = "gini",
                        val gamma: Int = 1,
                        val par: Boolean = false
                      ) {

  var trees: ArrayBuffer[DecisionTreeOld] = ArrayBuffer.empty[DecisionTreeOld]

  def train(
             data: RDD[DataPoint]
           ): Unit = {

    val n = data.collect().length
    var weights = Seq.fill(n)(1.0)
    var updated_data = data.collect().toList

    for (_ <- 0 until numIterations) {

      val tree = new DecisionTreeOld(
        maxDepth = maxDepth,
        minSplitSize = minSplitSize,
        featureSubsetStrategy = featureSubsetStrategy,
        impurityMeasure = impurityFunc,
        par = par
      )
      tree.train(data, Some(weights))
      trees += tree

      val predictions = data.collect().map(this.predict)
      val negativeGradients = computeGradients(updated_data.map(_.label), predictions.toList, weights.toList, mseLoss)
      val residuals: Seq[Double] = computeResiduals(updated_data.map(_.label), negativeGradients)

      updated_data = updateLabels(data.collect().toList, residuals)

      weights = weights.zipWithIndex.map { case (w, j) =>
        w * Math.exp(-learningRate * predict(data.collect()(j)))
      }.toList

      val sumWeights = weights.sum
      weights = weights.map(_ / sumWeights)
    }
  }

  def computeGradients(
                        predictions: List[Double],
                        labels: List[Double],
                        weights: List[Double],
                        lossFunction: (Double, Double) => Double
                      ): List[Double] = {

    val gradients = for ((prediction, label, weight) <- (predictions, labels, weights).zipped.toList) yield {
      weight * lossFunction(prediction, label)
    }

    gradients
  }

  def computeResiduals(
                        labels: Seq[Double],
                        predictions: Seq[Double]
                      ): Seq[Double] = {

    val n = labels.length
    val residuals = new Array[Double](n)

    for (i <- 0 until n) {
      residuals(i) = labels(i) - predictions(i)
    }

    residuals
  }

  def mseLoss(
               y: Double,
               ypred: Double
             ): Double = {
    math.pow(y - ypred, 2)
  }

  def updateLabels(
                    data: List[DataPoint],
                    residual: Seq[Double]
                  ): List[DataPoint] = {

    data.zip(residual).map { case (dp, res) =>
      DataPoint(dp.features, res)
    }

  }

  def predict(
               data: DataPoint
             ): Double = {

    val alpha = List.fill(trees.size)(1.0 / trees.size) //(0 to trees.size).map(i => learningRate / (1 + gamma * i))
    var prediction = 0.0

    for ((tree, alpha) <- trees.zip(alpha)) { // trees è la lista degli alberi del modello
      prediction += alpha * tree.predict(data) // learningRate è il parametro di apprendimento
    }

    prediction
  }

  def accuracy(
                             dataset: List[DataPoint],
                             tolerance: Double = 0.5
                           ): Double = {

    val predictions = dataset.map(this.predict)
    val numCorrect = predictions.zip(dataset.map(_.label)).count { case (pred, label) =>
      Math.abs(pred - label) <= tolerance
    }

    numCorrect.toDouble / predictions.length
  }

}