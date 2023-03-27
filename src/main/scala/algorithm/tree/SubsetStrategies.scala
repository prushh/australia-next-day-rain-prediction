package it.unibo.andrp
package algorithm.tree

import scala.util.Random

object SubsetStrategies extends Enumeration {
    type SubsetStrategy = Value

    val All, Sqrt, Log2 = Value

    def all(features: Seq[Int]): Seq[Int] =
        features

    def sqrt(features: Seq[Int]): Seq[Int] = {
        Random.shuffle(features) take {
            Math.sqrt(features.length).ceil.toInt
        }
    }

    def log2(features: Seq[Int]): Seq[Int] = {
        Random.shuffle(features) take {
            (Math.log(features.length) / Math.log(2.0)).ceil.toInt
        }
    }

}
