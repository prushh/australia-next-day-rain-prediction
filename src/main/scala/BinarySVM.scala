package it.unibo.andrp

import scala.annotation.tailrec

/**
 * @param x feature vector,
 * @param y target vector,
 */
// eta: Double=1, it_num: Int = 10000  param eta learning rate, param it_num iteration number
class BinarySVM(x: Vector[Vector[Double]], y:Vector[Double], epochs: Int = 1000, eta: Int = 1) {

  // adding the bias to the x vector
  def add_bias(x: Vector[Vector[Double]]): Vector[Vector[Double]] = x.map(_ :+ 1.0)

  val x_bias: Vector[Vector[Double]] = add_bias(x)

  var weights: Vector[Double] = (for (_ <- 1 to x_bias(0).length) yield 0.0).toVector

  // dot product
  def dot_product(x: Vector[Double], w: Vector[Double]): Double = {
    x.lazyZip(w).map((a, b) => a * b).sum
  }

  def predict(x: Vector[Vector[Double]], w: Vector[Double] = weights): Vector[Double] = {
    // y = sign(x.w + b)
    x.map(dot_product(_, w).sign)
  }

  // fit
  def fit(): Unit = {
    def reg_gradient(w: Vector[Double], x_i: Vector[Double], y_i: Double, epoch: Int): Vector[Double] = {
      // w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
      w.lazyZip(x_i).map((w, x_i) => w + eta * ((x_i * y_i) + (-2 * (1 / epoch) * w)))
    }

    def gradient(w: Vector[Double], epoch: Int): Vector[Double] = {
      // w = w + eta * (-2  *(1/epoch)* w)
      w.map(w_i => w_i + eta * (-2 * (1 / epoch) * w_i))
    }

    @tailrec
    def iteration(w: Vector[Double], x: Vector[Vector[Double]], y: Vector[Double], epoch: Int) : Vector[Double] = (x, y) match {
      // miss-classification condition: (Y[i]*np.dot(X[i], w)) < 1
      case (xh +: xs, yh +: ys) if (yh * dot_product(xh, w)) < 1 =>
        iteration(reg_gradient(w, xh, yh, epoch), xs, ys, epoch)
      case (_ +: xs, _ +: ys) =>
        iteration(gradient(w, epoch), xs, ys, epoch)
      case _ => w
    }

    @tailrec
    def train(w: Vector[Double], epochs: Int, t: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => train(iteration(w, x, y, t), epochs - 1, t + 1)
    }

    // update weight
    weights = train(weights, epochs)
  }

  // get_support_vectors

  // get hypothesis
}