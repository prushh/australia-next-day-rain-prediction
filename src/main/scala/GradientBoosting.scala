/*
package it.unibo.andrp

import scala.math.exp


class GradientBoosting(val numIterations: Int, val learningRate: Double, val numFeatures: Int, val maxDepth: Int, numClasses: Int) {

  var trees: List[DecisionTree] = List()

  // Inizializzazione dei pesi delle istanze a 1/N
  var weights: Array[Array[Double]] = Array.fill(numClasses, numFeatures)(1.0 / numFeatures)

  def train(data: List[DataPoint]): Unit = {
    val n = data.length

    // Creazione del modello costante f0
    val f0 = Array.fill(n)(Array.fill(numClasses)(1.0 / numClasses))

    // Iterazione dell'algoritmo di Gradient Boosting
    for (i <- 0 until numIterations) {
      // Calcolo dei residui
      val r = Array.ofDim[Double](n, numClasses)
      for (j <- 0 until n) {
        for (k <- 0 until numClasses) {
          r(j)(k) = data(j).label - f0(j)(k)
        }
      }

      // Creazione del nuovo modello debolissimo
      val tree = new DecisionTreeGB(maxDepth ,numFeatures)
      tree.train(data.map(_.features).toArray, r)

      // Aggiornamento dei pesi
      val p = data.map(dp => tree.predict(dp.features.toArray)).toArray
      val eta = learningRate / (1.0 + i)
      for (j <- 0 until n) {
        for (k <- 0 until numClasses) {
          weights(k) = weights(k).updated(j, weights(k)(j) * exp(-eta * r(j)(k) * p(j)(k)))
        }
      }

      // Normalizzazione dei pesi
      for (j <- 0 until n) {
        val wsum = weights.map(_(j)).sum
        for (k <- 0 until numClasses) {
          weights(k) = weights(k).updated(j, weights(k)(j) / wsum)
        }
      }

      // Aggiunta del nuovo albero all'ensemble
      trees = trees :+ tree

      // Aggiornamento del modello f0
      for (j <- 0 until n) {
        for (k <- 0 until numClasses) {
          f0(j)(k) = 0.0
          for (t <- trees.indices) {
            f0(j)(k) += weights(k)(j) * trees(t).predict(data(j).features.toArray)(0)
          }
        }
      }
    }
  }

  def predict(data: List[DataPoint]): List[Double] = {
    val n = data.length
    val scores = Array.ofDim[Double](n, numClasses)
    for (i <- 0 until n) {
      for (j <- 0 until numClasses) {
        scores(i)(j) = 0.0
        for (t <- trees.indices) {
          scores(i)(j) += learningRate * trees(t).predict(data(i).features.toArray)(0)
        }
      }
    }
    scores.map(_.indexOf(scores.max)).toList
  }
}

*/
