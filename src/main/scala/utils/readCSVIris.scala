package it.unibo.andrp
package utils

import global.DataFrame


object readCSVIris {
  /**
   * Reads the iris dataset and return the 'setosa' and 'versicolor' class and takes the first two feature columns.
   *
   * @param path Path of the iris.csv file
   * @return a dataframe containing the features and the labels.
   */
  def apply(path: String): (DataFrame[Double], Vector[Int]) = {
    val bufferedSource = scala.io.Source.fromFile(path)

    // rename the setosa and versicolor class to -1 and 1
    def flowerClass(v: String): Double = v.trim() match {
      case "setosa" => -1
      case "versicolor" => 1
      case "virginica" => 10
      case x => x.toDouble
    }

    // read lines
    val lines = bufferedSource.getLines.toVector.tail

    // split lines and map the flowerClass function discarding the virginica flower.
    val rows = lines.map(_.split(",").map(flowerClass)).filter(i => i.last < 2)
    val labels = rows.map(_.last.toInt)
    val data = rows.map(_.init)
    bufferedSource.close()

    (data.map(i => Vector(i(0), i(1), i(2), i(3))), labels)
  }
}
