package it.unibo.andrp
package algorithm.knn

import model.DataPoint

object Distance extends Enumeration {
    type Distance = Value
    type DistanceFunc = (DataPoint, DataPoint) => Double
    
    val Euclidean, Manhattan = Value
    def euclidean(a: DataPoint, b: DataPoint): Double = {
        math.sqrt {
            (a.features zip b.features).map {
                case (x, y) => math.pow(x - y, 2)
            }.sum
        }
    }

    def manhattan(a: DataPoint, b: DataPoint): Double = {
        (a.features zip b.features).map {
            case (ai, bi) => math.abs(ai - bi)
        }.sum
    }
}
