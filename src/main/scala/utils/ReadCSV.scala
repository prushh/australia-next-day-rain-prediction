package it.unibo.andrp
package utils

object ReadCSV {

  def apply(path: String): Vector[Vector[Double]] = {

    def toDouble(v: String): Double = v.trim() match {
      case x => x.toDouble
    }

    val source = scala.io.Source.fromFile(path);

    // all the lines except the header
    val lines = source.getLines.toVector.tail;

    // split lines
    val rows = lines.map(_.split(",").map(toDouble));

    
    // dividing the target column from the feature column // TODO: bisogna farlo dopo divisione in chunks
    /*val target = rows.map(_.last)
    val features = rows.map(_.init)
    
    // creating the pair of matrix of doubles and vector of doubles
    (features.map(i => Vector(i(0), i(1))), target)*/

    source.close()
    
    rows.map(i => Vector(i(0), i(1)))
  }


}
