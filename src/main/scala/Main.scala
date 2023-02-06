package it.unibo.andrp

package object global {
    type DataFrame[A] = Vector[Vector[A]]
}
import breeze.plot._
import global._
import utils._

object Main {
    def main(args: Array[String]): Unit = {
        // load data
        val (df, labels) = ReadCSV("data/weatherAUSfinal.csv")

        // initialize new SVM object
        val svm = new BinarySVM(df.take(1000), labels.take(1000))
        // train svm
        svm.fit()

        println("Classification accuracy:"+
            (svm.classification(svm.df), labels).zipped.count(i => i._1 == i._2).toDouble / svm.df.length)
        println("Weigths:"+ svm.w)

        Plotter(svm.df, labels, svm.w)

    }
}








