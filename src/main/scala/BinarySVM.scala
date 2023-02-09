package it.unibo.andrp

import scala.annotation.tailrec

/**
 * @param x feature vector,
 * @param y target vector,
 */
// eta: Double=1, it_num: Int = 10000  param eta learning rate, param it_num iteration number
class BinarySVM(x: Vector[Vector[Double]], y:Vector[Double], epochs: Int = 100, eta: Double = 0.001) {

  // adding the bias to the x vector
  def add_bias(x: Vector[Vector[Double]]): Vector[Vector[Double]] = x.map(_ :+ -1.0)

  val x_bias: Vector[Vector[Double]] = add_bias(x)
  var weights: Vector[Double] = (for (_ <- 1 to x_bias(0).length) yield 0.0).toVector

  // dot product
  def dot_product(x: Vector[Double], w: Vector[Double]): Double = {
    x.lazyZip(w).map((a, b) => a * b).sum
  }

  // fit
  def fit(): Unit = {
    def reg_gradient(w: Vector[Double], x_i: Vector[Double], y_i: Double, epoch: Int): Vector[Double] = {
      // w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
      //self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
      //self.b -= self.lr * y_[idx]
      w.zip(x_i).map((w_, x_i_) => (w_, x_i_) match{
        case (_,-1.0)=>
          w_ - eta * y_i
        case (_,_) =>w_ - eta * ((2 * (1 / epoch) * w_) - (x_i_ * y_i) )}
      )

    }

    def gradient(w: Vector[Double], epoch: Int): Vector[Double] = {
      // w = w + eta * (-2  *(1/epoch)* w)
      //self.w -= self.lr * (2 * self.lambda_param * self.w)
      w.map(w_i => w_i + eta * (-2 * (1 / epoch) * w_i))
    }

    @tailrec
    def iteration(w: Vector[Double], x: Vector[Vector[Double]], y: Vector[Double], epoch: Int) : Vector[Double] = (x, y) match {
      // miss-classification condition: (Y[i]*np.dot(X[i], w)) < 1
      // y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
      case (xh +: xs, yh +: ys) if (yh * (dot_product(xh, w))) < 1 =>
        iteration(reg_gradient(w, xh, yh, epoch), xs, ys, epoch)
      case (_ +: xs, _ +: ys) =>
        iteration(gradient(w, epoch), xs, ys, epoch)
      case _ => w
    }

    @tailrec
    def train(w: Vector[Double], epochs: Int, t: Int = 1): Vector[Double] = epochs match {
      case 0 => w
      case _ => train(iteration(w, x_bias, y, t), epochs - 1, t + 1)
    }

    // update weight

    weights = train(weights, epochs)

  }

  // functions to call after fitting

  def predict(x: Vector[Vector[Double]], w: Vector[Double] = weights): Vector[Double] = {
    // y = sign(x.w + b)
    x.map(dot_product(_, w).sign)
  }

  def get_support_vectors(w: Vector[Double]= weights):Unit =
    /*this.x.map(x_i => dot_product(x_i, w) match {
      case 1 => x_i
      case -1 => x_i
    })*/
    //val support = this.x_bias.dropWhile(x_i=>{dot_product(x_i, w) <= 0.95 || dot_product(x_i, w) >= 1.05 || dot_product(x_i, w) >= -0.95 || dot_product(x_i, w) <= -1.05})
    //support
    println(this.x_bias.map(dot_product(_,w)))

  // get hypothesis function
  def get_hypothesis(): Vector[Double] => Double = {
    (x: Vector[Double]) => dot_product(x, weights)
  }
  /*def print_graph(): Unit={
    val f = Figure()
    val p = f.subplot(0)
    p.title = "Decision boundary SVM"
    p.xlabel = "x1"
    p.ylabel = "x2"

    val x1 = (svm.x_bias, y).zipped.filter((_, b) => b == -1)._1
    val x2 = (svm.x_bias, y).zipped.filter((_, b) => b == 1)._1

    p += plot(x1.map(_(0)), x1.map(_(1)), '.', "b")
    p += plot(x2.map(_(0)), x2.map(_(1)), '.', "r")

    val sorted_x = x.sortWith((i1, i2) => i1(0) > i2(0))
    val w_ = weights.patch(1, Vector(0.0), 1)
    p += plot(sorted_x.map(_(0)), sorted_x.map(x => -dotProduct(x, w_) /  weights(1)))
    f.saveas("fig.png")
  }*/
}