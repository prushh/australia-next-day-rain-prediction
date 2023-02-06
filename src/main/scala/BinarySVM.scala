package it.unibo.andrp

import it.unibo.andrp.global.DataFrame

import scala.annotation.tailrec


object dotProduct {
  def apply(x: Vector[Double], w: Vector[Double]): Double = (x, w).zipped.map((a, b) => a * b).sum
}
/**
 * @param x Feature data.
 * @param labels Binary labels
 * @param eta Learning rate
 * @param epochs No. of training epochs
 */
class BinarySVM(x: DataFrame[Double], labels: Vector[Int], eta: Double=1, epochs: Int=10000) {

  // Add a bias term to the data.
  def prepare(x: DataFrame[Double]): DataFrame[Double] = x.map(_ :+ 1.0)

  // Prepared data.
  val df: DataFrame[Double] = prepare(x)


  // Weights initialization.
  var w :Vector[Double] = (for (_ <- 1 to df(0).length) yield 0.0).toVector


  def fit(): Unit = {
    // Will only be called if classification is wrong.
    def gradient(w: Vector[Double], data: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      (w, data).zipped.map((w, d) => w + eta * ((d * label) + (-2 * (1 / epoch) * w)))
    }

    def regularizationGradient(w: Vector[Double], label: Int, epoch: Int): Vector[Double] = {
      w.map(i => i + eta * (-2  * (1 / epoch) * i))
    }

    // Misclassification treshold.
    def misClassification(x: Vector[Double], w: Vector[Double], label: Int): Boolean = {
      dotProduct(x, w) * label < 1
    }

    def trainOneEpoch(w: Vector[Double], x: DataFrame[Double], labels: Vector[Int], epoch: Int): Vector[Double] = (x, labels) match {
      // If classification is wrong. Update weights with loss gradient
      case (xh +: xs, lh +: ls) if misClassification(xh, w, lh) => trainOneEpoch(gradient(w, xh, lh, epoch), xs, ls, epoch)
      // If classification is correct: update weights with regularizer gradient
      case (_ +: xs, lh +: ls) => trainOneEpoch(regularizationGradient(w, lh, epoch), xs, ls, epoch)
      case _ => w
    }

    def trainEpochs(w: Vector[Double], epochs: Int, epochCount: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => trainEpochs(trainOneEpoch(w, df, labels, epochCount), epochs - 1, epochCount + 1)
    }

    // Update weights
    w = trainEpochs(w, epochs)
  }

  def classification(x: Vector[Vector[Double]], w: Vector[Double] = w): Vector[Int] = x.map(dotProduct(_, w).signum)
}