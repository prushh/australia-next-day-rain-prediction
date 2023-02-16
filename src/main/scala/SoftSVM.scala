package it.unibo.andrp

import scala.util.Random
import scala.collection.mutable.ArrayBuffer
import scala.math.exp

class SoftSVM {

    def train_soft_margin(X: Array[Array[Double]], y: Array[Double], gamma: Double, C: Double, max_iter: Int): Array[Double] = {
        val n = X.length

        val numTrainingExamples = y.length
        val alpha = Array.fill(numTrainingExamples)(0.0)
        alpha(0) = 1.0 / numTrainingExamples

        val eps = 1e-3
        val tol = 1e-3
        val max_passes = 5
        var passes = 0

        val K = Array.ofDim[Double](n, n)
        for {
            i <- 0 until n
            j <- 0 until n
        } K(i)(j) = rbf_kernel(X(i), X(j), gamma)

        while (passes < max_passes) {
            var num_changed_alphas = 0

            for (i <- 0 until n) {
                val Ei = margin_error(X, y, K, alpha, i) - y(i)

                if (((y(i) * Ei < -tol) && (alpha(i) < C)) || ((y(i) * Ei > tol) && (alpha(i) > 0))) {
                    val j = select_second_alpha(i, n)
                    val Ej = margin_error(X, y, K, alpha, j) - y(j)

                    val old_alpha_i = alpha(i)
                    val old_alpha_j = alpha(j)

                    val (l, h) = compute_bounds(y(i), y(j), C, old_alpha_i, old_alpha_j)
                    if (l == h) {
                        // Skip if the bounds are the same
                        //println("L == H")
                        ()
                    } else {
                        val eta = 2.0 * K(i)(j) - K(i)(i) - K(j)(j)
                        if (eta >= 0) {
                            // Skip if the kernel is non-positive definite
                            //println("eta >= 0")
                            ()
                        } else {
                            alpha(j) -= y(j) * (Ei - Ej) / eta
                            alpha(j) = clip_alpha(alpha(j), l, h)
                            if (math.abs(alpha(j) - old_alpha_j) < eps) {
                                // Skip if the change in alpha is too small
                                //println("j not moving enough")
                                ()
                            } else {
                                alpha(i) += y(i) * y(j) * (old_alpha_j - alpha(j))
                                num_changed_alphas += 1
                            }
                        }
                    }
                }
            }

            passes = if (num_changed_alphas == 0) passes + 1 else 0
        }

        alpha
    }

    def rbf_kernel(x: Array[Double], z: Array[Double], gamma: Double): Double = {
        val dist = x.zip(z).map(p => math.pow(p._1 - p._2, 2)).sum
        exp(-gamma * dist)
    }

    def margin_error(X: Array[Array[Double]], y: Array[Double], K: Array[Array[Double]], alpha: Array[Double], i: Int): Double = {
        val wx = (for {
            j <- X.indices
        } yield alpha(j) * y(j) * K(i)(j)).sum
        wx + y(i)
    }

    def select_second_alpha(i: Int, n: Int): Int = {
        val j = Random.nextInt(n)
        if (j == i) {
            // Try again if j is the same
            select_second_alpha(i, n)
        } else {
            j
        }
    }

    def compute_bounds(yi: Double, yj: Double, C: Double, alpha_i: Double, alpha_j: Double): (Double, Double) = {
        if (yi != yj) {
            val L = math.max(0, alpha_j - alpha_i)
            val H = math.min(C, C + alpha_j - alpha_i)
            (L, H)
        } else {
            val L = math.max(0, alpha_i + alpha_j - C)
            val H = math.min(C, alpha_i + alpha_j)
            (L, H)
        }
    }

    def clip_alpha(alpha: Double, L: Double, H: Double): Double = {
        math.max(L, math.min(alpha, H))
    }

    def predict(X: Array[Array[Double]], y: Array[Double], alpha: Array[Double], gamma: Double, x: Array[Double]): Double = {
        val wx = (for {
            i <- X.indices
        } yield alpha(i) * y(i) * rbf_kernel(X(i), x, gamma)).sum
        if (wx > 0) 1.0 else -1.0
    }

    def accuracy(X: Array[Array[Double]], y: Array[Double], alpha: Array[Double], gamma: Double, X_test: Array[Array[Double]], y_test: Array[Double]): Double = {
        val y_pred = for {
            x <- X_test
        } yield predict(X, y, alpha, gamma, x)
        val correct = y_test.zip(y_pred).count(p => p._1 == p._2)
        correct.toDouble / y_test.length
    }
}