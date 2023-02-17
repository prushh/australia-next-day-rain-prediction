package it.unibo.andrp
import scala.util.Random





import utils.{ReadCSV, readCSVIris}

package object global {
    type DataFrame[A] = Vector[Vector[A]]
}

object Main {
    def main(args: Array[String]): Unit = {
        // load data
        val (df, target) = readCSVIris("data/iris.csv")
        //val (df, target) = ReadCSV("data/weatherAUS-final.csv")

        val data = df zip target
        val data_shuffled = Random.shuffle(data).toArray
        val (trainData, testData) = data_shuffled.splitAt((data.length * 0.8).toInt)

        val (x_train, y_train) = trainData.unzip
        val (x_test, y_test) = testData.unzip

        val clf = SoftSVM()
        val alpha = clf.train_soft_margin(x_train, y_train, 0.0001, 1.0, 10000)
        println(clf.accuracy(x_train, y_train, alpha, 0.0001, x_test, y_test))
        plotDataAndSupportVectors(trainData,alpha)
        // initialize new SVM object
//        val svm = new BinarySVM(df, labels)
//        // train svm
//        svm.fit()
//
//        println("Classification accuracy:" +
//          (svm.classification(svm.df), labels).zipped.count(i => i._1 == i._2).toDouble / svm.df.length)
//        println("Weigths:" + svm.w)
//        println(svm.get_support_vectors(tol = 0.005).sorted)


        //Plotter(svm.df, labels, svm.w)

    }

  import breeze.linalg._
  import breeze.plot._
  import java.awt.{Color, Paint}


  def plotDataAndSupportVectors(data: Array[(Array[Double], Double)], alphas: Array[Double]): Unit = {

    // filter support vectors
    val svIndices = alphas.indices.filter(i => alphas(i) > 1e-12)
    val sv = svIndices.map(data(_)).toArray

    // create plot
    val f = Figure()
    val p = f.subplot(0)

    // plot data points
    val xs = data.map(_._1(0))
    val ys = data.map(_._1(1))
    val labels = data.map(_._2)
    // plot data points
    val markers = data.map(p => if (p._2 == 1.0) '+' else 'o')
    val colors = data.map(p => if (p._2 == 1.0) Color.blue else Color.red)
    p += scatter(xs, ys, colors = (i => colors(i)), size = (i => 0.01))

    // plot support vectors
    val svxs = sv.map(_._1(0))
    val svys = sv.map(_._1(1))
    p += scatter(svxs, svys, colors=(i => Color.yellow), size= (i => 0.2))

    // show plot
    f.refresh()
  }


}








