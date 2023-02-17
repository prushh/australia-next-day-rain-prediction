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
        println(clf.accuracy(x_train, y_train, alpha, 0.001, x_test, y_test))

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
}








