package it.unibo.andrp
package utils

import breeze.plot.*
import it.unibo.andrp.global.DataFrame


object Plotter {
  def apply(x: DataFrame[Double], labels: Vector[Int], weights: Vector[Double]): Unit = {
    val f = Figure()
    f.width = 800
    f.height = 600
    val p = f.subplot(0)
    p.title = "Decision boundary SVM"
    p.xlabel = "x1"
    p.ylabel = "x2"

    val x1 = (x, labels).zipped.filter((_, b) => b == -1)._1

    val x2 = (x, labels).zipped.filter((_, b) => b == 1)._1

    p += plot(x1.map(_(0)), x1.map(_(1)), '.', "b")

    p += plot(x2.map(_(0)), x2.map(_(1)), '.', "r")

    val sorted_x = x.sortWith((i1, i2) => i1(0) > i2(0))
    val w_ = weights.patch(1, Vector(0.0), 1)
    p += plot(sorted_x.map(_(0)), sorted_x.map(x => -dotProduct(x, w_) /  weights(1)))
    f.saveas("t.png")
  }
}
