package it.unibo.andrp
package algorithm.tree

object Impurities extends Enumeration {
    type ImpurityFunc = (Seq[Double], Seq[Double]) => Double
    
    val Gini, Entropy = Value

    def gini(labels: Seq[Double], weights: Seq[Double]): Double = {
        val weightedCounts = (labels zip weights).groupBy(_._1).mapValues {
            _.map(_._2).sum
        }
        val totalWeight = weights.sum

        weightedCounts.values.map { count =>
            val weight = count / totalWeight
            weight * (1.0 - weight)
        }.sum
    }

    def entropy(labels: Seq[Double], weights: Seq[Double]): Double = {
        val totalWeight = weights.sum
        val probabilities = labels.groupBy(identity).mapValues {
            _.size.toDouble / labels.size
        }

        val entropy = probabilities.values.map {
            p => -1 * p * Math.log(p)
        }.sum * totalWeight

        if (entropy == 0.0) {
            0.0
        } else {
            val intrinsicValue = probabilities.values.map {
                p => -1 * p * Math.log(p)
            }.sum * -1

            entropy / intrinsicValue
        }
    }
}
