package it.unibo.andrp
import scala.collection.mutable.ArrayBuffer

class svm_new_impl(C: Double, tol: Double, maxPasses: Int) {


  /**
   * Support Vector Machine for binary classification.
   *
   * @param C The regularization parameter.
   * @param tol The tolerance for stopping criterion.
   * @param maxPasses The maximum number of passes over the training data.
   */

    type AlphaPair = (Double, Double)
    /** The data set. */
    var data: Array[(Array[Double], Double)] = _
    /** The support vectors. */
    var sv: Array[(Array[Double], Double)] = _
    /** The target values. */
    var target: Array[Double] = _
    /** The alphas. */
    var alphas: Array[Double] = _
    /** The bias. */
    var b: Double = 0.0
    /** The kernel function. */
    var kernel: (Int, Int) => Double = (i, j) => {
      val x1 = data(i)._1
      val x2 = data(j)._1
      x1.zip(x2).map { case (a, b) => a * b }.sum
    }

    /** The number of iterations. */
    var passes: Int = 0

    /**
     * Trains the SVM.
     * @param data The training data set.
     */
    def train(data: Array[(Array[Double], Double)]): Unit = {
      this.data = data
      val m = data.length
      target = data.map(_._2)
      alphas = Array.fill(m)(0.0)
      b = 0.0
      var numChanged = 0
      var examineAll = true
      val errors = Array.fill(m)(0.0)
      passes = 0
      while (passes < maxPasses && (numChanged > 0 || examineAll)) {
        numChanged = 0
        if (examineAll) {
          for (i <- 0 until m) {
            numChanged += examineExample(i, errors)
          }
        } else {
          for (i <- 0 until m if alphas(i) > 0 && alphas(i) < C) {
            numChanged += examineExample(i, errors)
          }
        }
        passes += 1
        if (examineAll) {
          examineAll = false
        } else if (numChanged == 0) {
          examineAll = true
        }
      }
      val idx = (0 until m).filter(i => alphas(i) > 0).toArray
      sv = idx.map(i => (data(i)._1, data(i)._2))
      val alphaIdx = idx.zip(alphas.slice(idx.head, idx.last + 1))
      for (j <- 0 until m) {
        b += target(j)
        for ((i, alpha) <- alphaIdx) {
          b -= alpha * target(i) * kernel(i, j)
        }
      }
      b /= m
    }
    /**
     * Predicts the target value for a new instance.
     * @param x The instance to predict.
     * @return The predicted target value.
     */
    def predict(x: Array[Double]): Double = {
      var y = b
      for {
        (xi: ((Array[Double], Double), Double), ti: Double, alpha: Double) <- sv.zip(target).zip(alphas)
      }
        yield alpha * ti * kernel(xi._1, x)
    }

    /**
     * Examines one example.
     * @param i The index of the example to examine.
     * @param errors The cache of error values.
     * @return The number of changed variables.
     */
    def examineExample(i: Int, errors: Array[Double]): Int = {
      val y = target(i)
      errors(i) = predict(data(i)._1) - y
      val error = errors(i)
      val r = y * error
      if ((r < -tol && alphas(i) < C) || (r > tol && alphas(i) > 0)) {
        val j = selectJ(i, errors)
        if (update(i, j, errors)) {
          return 1
        }
      }
      0
    }

    /**
     * Selects the second variable to optimize.
     * @param i The index of the first variable.
     * @param errors The cache of error values.
     * @return The index of the second variable.
     */
    def selectJ(i: Int, errors: Array[Double]): Int = {
      val m = target.length
      var max = 0
      var j = 0
      for (k <- 0 until m if k != i) {
        val ek = errors(k)
        val delta = (errors(i) - ek).abs
        if (delta > max) {
          max = delta
          j = k
        }
      }
      j
    }

    /**
     * Updates the variables.
     * @param i The index of the first variable.
     * @param j The index of the second variable.
     * @param errors The cache of error values.
     * @return Whether any variable has been updated.
     */
    def update(i: Int, j: Int, errors: Array[Double]): Boolean = {
      val y1 = target(i)
      val y2 = target(j)
      val alpha1 = alphas(i)
      val alpha2 = alphas(j)
      val x1 = data(i)._1
      val x2 = data(j)._1
      val L = if (y1 != y2) {
        val L1 = 0.0 max (0.0 - alpha2 + alpha1)
        val L2 = C min (C + alpha2 - alpha1)
        (L1, L2)
      } else {
        val L1 = 0.0 max (alpha2 + alpha1 - C)
        val L2 = C min (alpha2 + alpha1)
        (L1, L2)
      }
      if (L._1 == L._2) {
        return false
      }
      val k11 = kernel(x1, x1)
      val k12 = kernel(x1, x2)
      val k22 = kernel(x2, x2)
      val eta = 2 * k12 - k11 - k22
      var a2 = alpha2 + y2 * (errors(i) - errors(j)) / eta
      if (a2 < L._1) {
        a2 = L._1
      } else if (a2 > L._2) {
        a2 = L._2
      }
      val delta = a2 - alpha2
      if (delta.abs < 1e-12) {
        return false
      }
      val a1 = alpha1 + y1 * y2 * (alpha2 - a2)
      alphas(i) = a1
      alphas(j) = a2
      sv = if (0 < a1 && a1 < C) {
        (data(i)._1, target(i)) :: sv
      } else {
        sv.filterNot(_._1 == data(i)._1)
      }
      sv = if (0 < a2 && a2 < C) {
        (data(j)._1, target(j)) :: sv
      } else {
        sv.filterNot(_._1 == data(j)._1)
      }
      val b1 = b - errors(i) - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
      val b2 = b - errors(j) - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
      b = if (0 < a1 && a1 < C) {
        b1
      } else if (0 < a2 && a2 < C) {
        b2
      } else {
        (b1 + b2) / 2
      }
      true
    }





}
