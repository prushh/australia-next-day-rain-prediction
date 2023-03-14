
package it.unibo.andrp

import it.unibo.andrp.algorithm.tree.DecisionTree
import it.unibo.andrp.model.DataPoint

import scala.math.exp
import scala.collection.mutable.ArrayBuffer

class GradientBoostingClassifier(
                                  val numIterations: Int=10,
                                  val learningRate: Double=0.1,
                                  val maxDepth: Int=4,
                                  val minSplitSize: Int=10,
                                  val featureSubsetStrategy: String="all",
                                  val impurityFunc: String="gini",
                                  val gamma: Int = 1
                                ) {


  var trees = ArrayBuffer.empty[DecisionTree]

  def train(
             data: Seq[DataPoint]
           ): Unit = {
    val n = data.length
    // Inizializzazione dei pesi
    var weights = Seq.fill(n)(1.0)
    val learningRateScaled = learningRate / n.toDouble

    // Initialize residuals

    val labels = data.map(_.label)
    val residuals = labels
    var updated_data= data.toList
    for (i <- 0 until numIterations) {
      // Compute negative gradients
      // Calcolo i negative gradients

      // Train a decision tree on the negative gradients
      val tree = new DecisionTree(
        maxDepth = maxDepth,
        minSplitSize = minSplitSize,
        featureSubsetStrategy = featureSubsetStrategy,
        impurityMeasure = impurityFunc
      )
      tree.train(updated_data, Some(weights))
      // Add the new tree to the list of trees
      trees += tree
      // Compute residuals and negative gradients
      val predictions = data.map(this.predict)

      val negativeGradients = computeGradients(updated_data.map(_.label),predictions.toList,weights.toList,mseLoss)
      val residuals:Seq[Double] = computeResiduals(updated_data.map(_.label), negativeGradients)

      updated_data = updateLabels(data.toList, residuals)

      // Aggiorno i pesi
      weights = weights.zipWithIndex.map { case (w, j) =>
        w * Math.exp(-learningRate * predict(data(j)))
      }.toList

      val sumWeights = weights.sum
      weights = weights.map(_ / sumWeights)




    }
  }
  def computeGradients(predictions: List[Double], labels: List[Double], weights: List[Double], lossFunction: (Double, Double) => Double): List[Double] = {
    val gradients = for ((prediction, label, weight) <- (predictions, labels, weights).zipped.toList) yield {
      weight * lossFunction(prediction, label)
    }
    gradients
  }

  def computeResiduals(labels: Seq[Double], predictions: Seq[Double]): Seq[Double] = {
    val n = labels.length
    val residuals = new Array[Double](n)
    for (i <- 0 until n) {
      residuals(i) = labels(i) - predictions(i)
    }
    residuals
  }
  def mseLoss(y: Double, ypred: Double): Double = {
    math.pow(y - ypred, 2)
  }



  def updateLabels(data: List[DataPoint], residual: Seq[Double]): List[DataPoint] = {
    data.zip(residual).map { case (dp, res) =>
      DataPoint(dp.features, res)
    }
  }

  def predict(data: DataPoint): Double = {
    // Calcolo la somma delle predizioni degli alberi
    val alpha = List.fill(trees.size)(1.0/trees.size)//(0 to trees.size).map(i => learningRate / (1 + gamma * i))
    var prediction = 0.0
    for ((tree,alpha) <- trees.zip(alpha)) { // trees è la lista degli alberi del modello
      prediction += alpha * tree.predict(data) // learningRate è il parametro di apprendimento
    }
    prediction
  }
  def normalizeList(l: List[Double]): List[Double] = {
    val minVal = l.min
    val maxVal = l.max
    l.map(x => 2 * (x - minVal) / (maxVal - minVal) - 1)
  }

  def accuracy(dataset: List[DataPoint]): Double = {
    val predictions = dataset.map(this.predict)
    val correct = predictions.zip(dataset).count { case (pred, dp) => pred == dp.label }
    correct.toDouble / dataset.length.toDouble
  }
  def accuracyWithTolerance(dataset: List[DataPoint], tolerance: Double): Double = {

    val predictions = dataset.map(this.predict)
    val numCorrect = predictions.zip(dataset.map(_.label)).count { case (pred, label) =>
      Math.abs(pred - label) <= tolerance
    }
    numCorrect.toDouble / predictions.length
  }


}