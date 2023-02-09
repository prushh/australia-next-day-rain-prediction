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
        //val (df, labels) = readCSVIris("data/iris.csv")
        val (df, labels) = ReadCSV("data/weatherAUS-final.csv")

        // initialize new SVM object
        val svm = new BinarySVM(df, labels)
        // train svm
        svm.fit()

        println("Classification accuracy:"+
            (svm.classification(svm.df), labels).zipped.count(i => i._1 == i._2).toDouble / svm.df.length)
        println("Weigths:"+ svm.w)
        println(svm.get_support_vectors(tol=0.005).sorted)


        //Plotter(svm.df, labels, svm.w)

    }
}








